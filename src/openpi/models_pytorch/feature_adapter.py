# 视觉特征 adapter v1
"""Optional FP32 visual residual adapter; no dependency on the PCD repository."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from openpi.models_pytorch.feature_mask import FeatureMaskSpec
from openpi.models_pytorch.feature_mask import apply_grouped_feature_mask


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

    def __init__(
        self,
        spec: FeatureAdapterSpec,
        mask_spec: FeatureMaskSpec | None = None,
    ):
        super().__init__()
        self.spec = spec
        if mask_spec is not None and mask_spec.feature_dim != spec.feature_dim:
            raise ValueError("Adapter and mask feature dimensions must match")
        self.mask_spec = mask_spec
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

    # DRR@2048 + correlation evaluation
    # def forward(self, features, image_mask, view_index: int):
    #     if view_index not in self.spec.view_indices:
    #         return features
    #     z = self.encode(features)
    #     with torch.autocast(device_type=features.device.type, enabled=False):
    #         residual = self.spec.residual_scale * self.up(z)
    #         valid = image_mask.to(device=features.device, dtype=torch.bool)
    #         residual = residual * valid[:, None, None]
    #         result = (features.float() + residual).to(features.dtype)
    #     if self._capture:
    #         self._records.append({"view": view_index, "z": z, "valid": valid})
    #     return result
    # DRR@2048 + correlation evaluation
    def _adapt_impl(
        self,
        features,
        image_mask,
    ):
        """
        Apply the residual adapter without any capture side effects.

        Returns:
            z:
                Bottleneck feature [B, T, rank].
            result:
                Adapted visual feature [B, T, D].
            valid:
                Per-image validity mask [B].
        """
        z = self.encode(features)

        with torch.autocast(
            device_type=features.device.type,
            enabled=False,
        ):
            residual = (
                self.spec.residual_scale
                * self.up(z)
            )

            valid = image_mask.to(
                device=features.device,
                dtype=torch.bool,
            )

            residual = (
                residual
                * valid[:, None, None]
            )

            result = (
                features.float()
                + residual
            ).to(features.dtype)

        return z, result, valid

    def adapt(
        self,
        features,
        image_mask,
    ):
        """
        Apply the adapter without writing capture records.

        This is used by the augmented DRR branch so that the second
        representation participates in gradient computation without
        polluting the main forward capture buffer.
        """
        _, result, _ = self._adapt_impl(
            features,
            image_mask,
        )

        return result

    def apply_mask(
        self,
        features,
        *,
        mask_logits=None,
        mask_gates=None,
        mask_beta: float = 1.0,
        mask_hard: bool = False,
    ):
        """Apply the post-adapter grouped channel mask.

        This method is intentionally functional: task-specific mask values are
        passed by the caller instead of being stored as mutable module state.
        """

        if mask_logits is None and mask_gates is None:
            return features, None
        if self.mask_spec is None:
            raise RuntimeError("A FeatureMaskSpec is required when a mask is supplied")
        return apply_grouped_feature_mask(
            features,
            spec=self.mask_spec,
            group_logits=mask_logits,
            group_gates=mask_gates,
            beta=mask_beta,
            hard=mask_hard,
        )

    def forward(
        self,
        features,
        image_mask,
        view_index: int,
        *,
        mask_logits=None,
        mask_gates=None,
        mask_beta: float = 1.0,
        mask_hard: bool = False,
    ):
        if view_index not in self.spec.view_indices:
            return features

        z, pre_mask, valid = self._adapt_impl(
            features,
            image_mask,
        )

        post_mask, resolved_gates = self.apply_mask(
            pre_mask,
            mask_logits=mask_logits,
            mask_gates=mask_gates,
            mask_beta=mask_beta,
            mask_hard=mask_hard,
        )

        # Invalid/padded image slots must remain exactly unchanged.
        valid_view = valid[:, None, None]
        result = torch.where(valid_view, post_mask, features)

        if self._capture:
            self._records.append(
                {
                    "view": view_index,

                    # Frozen SigLIP representation before the adapter.
                    "input": features,

                    # Legacy v1 representation target.
                    "z": z,

                    # DRR target. Keep graph: DRR must update the adapter, but
                    # it must not optimize the post-mask representation.
                    "pre_mask": pre_mask,

                    # Final representation consumed by PaliGemma.
                    "post_mask": result,

                    # Backward-compatible alias for adapter-v1 tooling.
                    "output": pre_mask,

                    "group_gates": resolved_gates,

                    "valid": valid,
                }
            )

        return result
    # DRR@2048 + correlation evaluation
    
