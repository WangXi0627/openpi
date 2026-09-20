from __future__ import annotations

import pytest
import torch

from openpi.models_pytorch.feature_mask import FeatureMaskSpec
from openpi.models_pytorch.feature_mask import apply_grouped_feature_mask
from openpi.models_pytorch.feature_mask import hard_topk_group_gates


def test_beta_zero_is_identity() -> None:
    spec = FeatureMaskSpec(feature_dim=8, group_size=2, keep_ratio=0.5)
    features = torch.randn(2, 3, 8)
    logits = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    masked, _ = apply_grouped_feature_mask(
        features,
        spec=spec,
        group_logits=logits,
        beta=0.0,
    )
    torch.testing.assert_close(masked, features, rtol=0, atol=0)


def test_hard_topk_keeps_exact_budget() -> None:
    logits = torch.arange(8, dtype=torch.float32)
    gates = hard_topk_group_gates(logits, keep_ratio=0.75)
    assert int(gates.sum()) == 6
    assert set(gates.tolist()) <= {0.0, 1.0}


def test_mask_broadcasts_only_over_channels() -> None:
    spec = FeatureMaskSpec(feature_dim=8, group_size=2, keep_ratio=0.5)
    features = torch.ones(2, 3, 8)
    group_gates = torch.tensor([1.0, 0.0, 1.0, 0.0])
    masked, _ = apply_grouped_feature_mask(
        features,
        spec=spec,
        group_gates=group_gates,
    )
    expected = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0], dtype=torch.float32)
    torch.testing.assert_close(masked, expected.view(1, 1, 8).expand(2, 3, 8))


def test_soft_mask_backpropagates_to_logits() -> None:
    spec = FeatureMaskSpec(feature_dim=8, group_size=2, keep_ratio=0.5)
    features = torch.randn(2, 3, 8)
    logits = torch.zeros(4, requires_grad=True)
    masked, _ = apply_grouped_feature_mask(
        features,
        spec=spec,
        group_logits=logits,
    )
    masked.square().mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_rejects_two_mask_sources() -> None:
    spec = FeatureMaskSpec(feature_dim=8, group_size=2, keep_ratio=0.5)
    with pytest.raises(ValueError, match="exactly one"):
        apply_grouped_feature_mask(
            torch.ones(1, 1, 8),
            spec=spec,
            group_logits=torch.zeros(4),
            group_gates=torch.ones(4),
        )
