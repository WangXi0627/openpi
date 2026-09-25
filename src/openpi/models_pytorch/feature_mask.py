"""Functional grouped masks for visual feature channels.

The mask has no task-specific mutable state. Training supplies differentiable
group logits and inference supplies a fixed group gate tensor for each request.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class FeatureMaskSpec:
    """Description of a grouped channel mask."""

    feature_dim: int
    group_size: int = 16
    keep_ratio: float = 0.875

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.feature_dim % self.group_size != 0:
            raise ValueError("feature_dim must be divisible by group_size")
        if not math.isfinite(self.keep_ratio) or not 0 < self.keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")

    @property
    def num_groups(self) -> int:
        return self.feature_dim // self.group_size

    @property
    def num_kept_groups(self) -> int:
        return max(1, min(self.num_groups, round(self.num_groups * self.keep_ratio)))

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "FeatureMaskSpec":
        return cls(**dict(value))


def hard_topk_group_gates(logits: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """Return a straight binary top-k mask without changing ``logits``."""

    if logits.ndim != 1:
        raise ValueError(f"Expected one-dimensional group logits, got {tuple(logits.shape)}")
    if not math.isfinite(keep_ratio) or not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    count = max(1, min(logits.numel(), round(logits.numel() * keep_ratio)))
    indices = torch.topk(logits, k=count, sorted=False).indices
    gates = torch.zeros_like(logits)
    gates.scatter_(0, indices, 1.0)
    return gates


def expand_group_gates(gates: torch.Tensor, spec: FeatureMaskSpec) -> torch.Tensor:
    """Expand ``[G]`` group gates to ``[D]`` channel gates."""

    if gates.ndim != 1 or gates.numel() != spec.num_groups:
        raise ValueError(
            f"Expected {spec.num_groups} group gates, got shape {tuple(gates.shape)}"
        )
    return gates.repeat_interleave(spec.group_size)


def resolve_group_gates(
    *,
    spec: FeatureMaskSpec,
    group_logits: torch.Tensor | None = None,
    group_gates: torch.Tensor | None = None,
    hard: bool = False,
) -> torch.Tensor:
    """Resolve either train-time logits or inference-time fixed gates."""

    if (group_logits is None) == (group_gates is None):
        raise ValueError("Provide exactly one of group_logits or group_gates")
    if group_logits is not None:
        if group_logits.ndim != 1 or group_logits.numel() != spec.num_groups:
            raise ValueError(
                f"Expected {spec.num_groups} group logits, got shape {tuple(group_logits.shape)}"
            )
        return (
            hard_topk_group_gates(group_logits, spec.keep_ratio)
            if hard
            else torch.sigmoid(group_logits)
        )
    assert group_gates is not None
    if group_gates.ndim != 1 or group_gates.numel() != spec.num_groups:
        raise ValueError(
            f"Expected {spec.num_groups} fixed group gates, got shape {tuple(group_gates.shape)}"
        )
    if not torch.isfinite(group_gates).all():
        raise ValueError("Fixed group gates contain NaN or Inf")
    if bool(((group_gates < 0) | (group_gates > 1)).any()):
        raise ValueError("Fixed group gates must lie in [0, 1]")
    return group_gates


def apply_grouped_feature_mask(
    features: torch.Tensor,
    *,
    spec: FeatureMaskSpec,
    group_logits: torch.Tensor | None = None,
    group_gates: torch.Tensor | None = None,
    beta: float = 1.0,
    hard: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a channel mask to ``[..., D]`` and return output/group gates.

    ``beta=0`` is exactly the identity path. ``beta=1`` applies the complete
    soft or hard mask. The interpolation is useful for a stable mask warm-up.
    """

    if features.shape[-1] != spec.feature_dim:
        raise ValueError(
            f"Expected feature dimension {spec.feature_dim}, got {features.shape[-1]}"
        )
    if not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    gates = resolve_group_gates(
        spec=spec,
        group_logits=group_logits,
        group_gates=group_gates,
        hard=hard,
    ).to(device=features.device, dtype=torch.float32)
    channel_gates = expand_group_gates(gates, spec)
    effective = 1.0 - float(beta) * (1.0 - channel_gates)
    shape = (1,) * (features.ndim - 1) + (spec.feature_dim,)
    with torch.autocast(device_type=features.device.type, enabled=False):
        masked = features.float() * effective.view(shape)
    return masked.to(features.dtype), gates
