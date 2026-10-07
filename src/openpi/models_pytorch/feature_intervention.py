"""Explicit, image-only interventions shared by joint training and prefix prefill."""

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

MID_VLM_SITES = {8: "pre_joint_layer_9", 11: "pre_joint_layer_12", 14: "pre_joint_layer_15"}


def apply_mid_vlm_intervention(
    hidden_states: torch.Tensor,
    layer_index: int,
    *,
    intervention: nn.Module | None = None,
    context: Mapping[str, Any] | None = None,
    observer: Callable | None = None,
) -> torch.Tensor:
    """Observe the unmodified representation, then intervene at a named site."""
    site = MID_VLM_SITES.get(layer_index)
    if site is None:
        return hidden_states
    context = dict(context or {})
    context["layer_index"] = layer_index
    if observer is not None:
        observer(hidden_states, site=site, context=context)
    if intervention is None or context.get("task_code") is None:
        return hidden_states
    mask = context.get("image_token_mask")
    if not isinstance(mask, torch.Tensor) or mask.shape != hidden_states.shape[:2]:
        raise ValueError("image_token_mask must match prefix batch/token axes")
    output = intervention(hidden_states, site=site, context=context)
    if not isinstance(output, torch.Tensor) or output.shape != hidden_states.shape:
        raise ValueError("Feature intervention changed shape")
    if output.dtype != hidden_states.dtype or output.device != hidden_states.device:
        raise ValueError("Feature intervention changed dtype/device")
    # Enforce the contract even for third-party interventions.
    return torch.where(mask.to(device=hidden_states.device, dtype=torch.bool)[..., None], output, hidden_states)
