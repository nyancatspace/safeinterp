"""Logit lens for SAE features: which tokens a feature's decoder direction promotes."""
from __future__ import annotations

import torch
from torch import nn

from .hooks import final_norm
from .sae import SAE


def unembed(model: nn.Module) -> torch.Tensor:
    """Directions -> logits map, folding in the final norm's gain: ``[d, vocab]``."""
    gain = final_norm(model).weight
    if getattr(model.config, "model_type", "").startswith("gemma"):
        gain = 1 + gain  # Gemma's RMSNorm scales by (1 + weight)
    return gain[:, None] * model.lm_head.weight.T


def _lens_dirs(model: nn.Module, D: torch.Tensor) -> torch.Tensor:
    return D - D.mean(-1, keepdim=True) if isinstance(final_norm(model), nn.LayerNorm) else D  # LayerNorm removes the mean


def logit_lens(model: nn.Module, sae: SAE, features: list[int], tokenizer, k: int = 8) -> dict[int, list[str]]:
    """Tokens each feature's decoder direction most directly promotes."""
    D = _lens_dirs(model, sae.feature_directions()[features])
    logits = D @ unembed(model)
    top = logits.topk(k, dim=-1).indices
    return {f: [tokenizer.decode([t]) for t in row.tolist()] for f, row in zip(features, top)}
