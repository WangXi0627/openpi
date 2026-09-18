# 视觉特征 adapter v1
"""Optional FP32 visual residual adapter; no dependency on the PCD repository."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FeatureAdapterSpec:
    feature_dim: int
    rank: int = 64
    residual_scale: float = 0.1
    view_indices: tuple[int, ...] = (0, 1)

    def __post_init__(self):
        if not 0 < self.rank <= self.feature_dim:
            raise ValueError("Require 0 < rank <= feature_dim")
        if not math.isfinite(self.residual_scale) or self.residual_scale <= 0:
            raise ValueError("residual_scale must be finite and positive")
        if not self.view_indices or any(i not in (0, 1) for i in self.view_indices):
            raise ValueError("LIBERO adapter views: 0=base, 1=left wrist; padding view is excluded")
        if len(set(self.view_indices)) != len(self.view_indices):
            raise ValueError("Duplicate view indices")

    def to_dict(self):
        result = asdict(self)
        result["view_indices"] = list(self.view_indices)
        return result

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["view_indices"] = tuple(value.get("view_indices", (0, 1)))
        return cls(**value)


class VisualFeatureAdapter(nn.Module):
    """D -> r -> D residual MLP. Capture exists only during one loss forward.

    The initial output is exactly the input because up.weight starts at zero.
    Model loading must finish BEFORE this module is attached, so official base
    checkpoints remain loadable without adapter keys.
    """

    def __init__(self, spec: FeatureAdapterSpec):
        super().__init__()
        self.spec = spec
        self.norm = nn.LayerNorm(spec.feature_dim)
        self.down = nn.Linear(spec.feature_dim, spec.rank, bias=False)
        self.up = nn.Linear(spec.rank, spec.feature_dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self._capture = False
        self._records = []

    def encode(self, features):
        with torch.autocast(device_type=features.device.type, enabled=False):
            return F.gelu(self.down(self.norm(features.float())))

    def begin_capture(self, enabled: bool):
        if self._records:
            raise RuntimeError("Previous adapter graph not released; call clear_capture()")
        self._capture = bool(enabled)

    def take_records(self):
        records, self._records = self._records, []
        self._capture = False
        return records

    def clear_capture(self):
        self._records.clear()
        self._capture = False

    def forward(self, features, image_mask, view_index: int):
        if view_index not in self.spec.view_indices:
            return features
        z = self.encode(features)
        with torch.autocast(device_type=features.device.type, enabled=False):
            residual = self.spec.residual_scale * self.up(z)
            valid = image_mask.to(device=features.device, dtype=torch.bool)
            residual = residual * valid[:, None, None]
            result = (features.float() + residual).to(features.dtype)
        if self._capture:
            self._records.append({"view": view_index, "z": z, "valid": valid})
        return result
