"""Real tiny patched-Gemma integration checks for both forward paths."""

from types import SimpleNamespace
import ast
import math
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import GemmaConfig
from transformers.models.gemma import modeling_gemma

from openpi.models_pytorch.feature_intervention import apply_mid_vlm_intervention
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel


class Prefix(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.language_model = model
        self.model = SimpleNamespace(language_model=model)
        self.config = SimpleNamespace(text_config=model.config)


def tiny_joint():
    torch.manual_seed(9)
    config = GemmaConfig(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=15,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=16,
        use_adarms=False,
        hidden_activation="gelu_pytorch_tanh",
    )
    config._attn_implementation = "eager"
    model = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    nn.Module.__init__(model)
    model.paligemma = Prefix(modeling_gemma.GemmaModel(config))
    model.gemma_expert = nn.Module()
    model.gemma_expert.model = modeling_gemma.GemmaModel(config)
    model.requires_grad_(False)
    model.eval()
    return model


class Shift(nn.Module):
    def __init__(self, site, value=0.0):
        super().__init__()
        self.site = site
        self.amount = nn.Parameter(torch.tensor(value))
        self.calls = 0

    def forward(self, hidden, *, site, context):
        if site != self.site:
            return hidden
        self.calls += 1
        return hidden + self.amount * context["task_code"][:, :1, None]


@pytest.mark.parametrize("layer", [9, 12, 15])
def test_prefill_exact_identity_and_trained_missing_support(layer):
    model = tiny_joint()
    prefix = torch.randn(1, 4, 32)
    mask = torch.tensor([[True, True, False, False]])
    args = dict(
        inputs_embeds=[prefix, None],
        attention_mask=torch.zeros(1, 1, 4, 4),
        position_ids=torch.arange(4)[None],
        use_cache=True,
    )
    (base, _), base_cache = model.forward(**args)
    intervention = Shift(f"pre_joint_layer_{layer}")
    (adapted, _), cache = model.forward(
        **args,
        feature_intervention=intervention,
        intervention_context={"image_token_mask": mask, "task_code": torch.ones(1, 2)},
    )
    assert intervention.calls == 1
    assert torch.equal(base, adapted)
    for first, second in zip(base_cache.key_cache, cache.key_cache, strict=True):
        assert torch.equal(first, second)
    intervention.amount.data.fill_(0.3)
    (fallback, _), _ = model.forward(
        **args, feature_intervention=intervention, intervention_context={"image_token_mask": mask, "task_code": None}
    )
    assert torch.equal(base, fallback)


def test_joint_gradient_prefill_cache_effect_and_request_isolation():
    model = tiny_joint()
    prefix, suffix = torch.randn(1, 4, 32), torch.randn(1, 3, 32)
    mask = torch.tensor([[True, True, False, False]])
    attention = torch.zeros(1, 1, 7, 7)
    attention[:, :, :4, 4:] = -1e9
    args = dict(
        inputs_embeds=[prefix, suffix], attention_mask=attention, position_ids=torch.arange(7)[None], use_cache=False
    )
    (base_prefix, base_suffix), _ = model.forward(**args)
    intervention = Shift("pre_joint_layer_12", value=0.2)
    context = {"image_token_mask": mask, "task_code": torch.ones(1, 2)}
    (adapted_prefix, adapted_suffix), _ = model.forward(
        **args, feature_intervention=intervention, intervention_context=context
    )
    assert not torch.equal(base_suffix, adapted_suffix)
    adapted_suffix.square().mean().backward()
    assert intervention.amount.grad is not None and intervention.amount.grad.abs() > 0
    assert all(p.grad is None for p in model.parameters())
    prefill_args = dict(
        inputs_embeds=[prefix, None],
        attention_mask=attention[:, :, :4, :4],
        position_ids=torch.arange(4)[None],
        use_cache=True,
    )
    (prefill, _), adapted_cache = model.forward(
        **prefill_args, feature_intervention=intervention, intervention_context=context
    )
    (_, _), base_cache = model.forward(**prefill_args)
    torch.testing.assert_close(prefill, adapted_prefix, rtol=1e-5, atol=1e-6)
    assert not torch.equal(base_cache.key_cache[11], adapted_cache.key_cache[11])
    (_, suffix_only), _ = model.forward(
        inputs_embeds=[None, suffix],
        attention_mask=attention[:, :, 4:, :],
        position_ids=torch.arange(4, 7)[None],
        past_key_values=adapted_cache,
        use_cache=False,
    )
    torch.testing.assert_close(suffix_only, adapted_suffix, rtol=1e-5, atol=1e-6)
    (again, _), _ = model.forward(**prefill_args)
    assert torch.equal(again, base_prefix)


def test_contract_preserves_text_padding_and_invalid_cameras():
    hidden = torch.randn(2, 6, 8)
    mask = torch.tensor([[True, True, False, False, False, False], [False, False, True, True, False, False]])
    context = {"image_token_mask": mask, "task_code": torch.ones(2, 2)}
    output = apply_mid_vlm_intervention(hidden, 11, intervention=Shift("pre_joint_layer_12", 0.4), context=context)
    assert torch.equal(output[~mask], hidden[~mask])
    assert not torch.equal(output[mask], hidden[mask])


def test_joint_checkpointing_preserves_intervention_gradient():
    model = tiny_joint()
    model.train()
    prefix = torch.randn(1, 4, 32)
    suffix = torch.randn(1, 3, 32)
    attention = torch.zeros(1, 1, 7, 7)
    attention[:, :, :4, 4:] = -1e9
    intervention = Shift("pre_joint_layer_12", 0.2)
    (_, output), _ = model.forward(
        inputs_embeds=[prefix, suffix],
        attention_mask=attention,
        position_ids=torch.arange(7)[None],
        use_cache=False,
        feature_intervention=intervention,
        intervention_context={
            "image_token_mask": torch.ones(1, 4, dtype=torch.bool),
            "task_code": torch.ones(1, 2, requires_grad=True),
        },
    )
    output.square().mean().backward()
    assert intervention.amount.grad is not None and torch.isfinite(intervention.amount.grad)


def test_actual_embed_prefix_marks_only_valid_camera_tokens():
    source = Path(__file__).with_name("pi0_pytorch.py")
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PI0Pytorch")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "embed_prefix")
    scope = {"torch": torch, "math": math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"), scope)
    trunk = SimpleNamespace(embed_image=lambda image: image, embed_language_tokens=lambda tokens: torch.zeros(2, 2, 8))
    model = SimpleNamespace(
        paligemma_with_expert=trunk, feature_adapter=None, _apply_checkpoint=lambda fn, *args: fn(*args)
    )
    images = [torch.randn(2, 3, 8) for _ in range(3)]
    valid = [torch.tensor([True, True]), torch.tensor([True, False]), torch.tensor([False, False])]
    tokens, token_mask = torch.zeros(2, 2, dtype=torch.long), torch.tensor([[True, False], [True, True]])
    regular = scope["embed_prefix"](model, images, valid, tokens, token_mask)
    with_mask = scope["embed_prefix"](model, images, valid, tokens, token_mask, return_image_token_mask=True)
    assert len(regular) == 3 and len(with_mask) == 4
    assert all(torch.equal(a, b) for a, b in zip(regular, with_mask[:3], strict=True))
    assert with_mask[3].sum(1).tolist() == [6, 3]
    assert not with_mask[3][:, -2:].any()
    assert not with_mask[3][:, 6:9].any()
