import pytest
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def test_request_local_intervention_context_preserves_extra_tensors():
    image_mask = torch.tensor([[True, False]])
    task_code = torch.randn(1, 3)
    logits = torch.randn(1, 4)
    delta_u = torch.randn(1, 4)
    context = PI0Pytorch._intervention_context(
        image_mask,
        extra={
            "task_code": task_code,
            "mask_logits": logits,
            "delta_u": delta_u,
            "strength": 0.5,
        },
    )
    assert context["image_token_mask"] is image_mask
    assert context["task_code"] is task_code
    assert context["mask_logits"] is logits
    assert context["delta_u"] is delta_u
    assert context["strength"] == 0.5


def test_request_local_intervention_context_rejects_mask_override():
    with pytest.raises(ValueError, match="image_token_mask"):
        PI0Pytorch._intervention_context(
            torch.ones(1, 1, dtype=torch.bool),
            extra={"image_token_mask": torch.zeros(1, 1, dtype=torch.bool)},
        )
