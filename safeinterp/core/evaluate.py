"""Measure how faithful an SAE is when spliced back into GPT-2."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .hooks import edit_sites
from .sae import SAE


def _ce(model: nn.Module, ids: torch.Tensor) -> float:
    logits = model(ids).logits[:, :-1]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)).item()


def splice_fn(sae: SAE, keep_bos: bool = True):
    """Replace activations with their SAE reconstruction (BOS position untouched)."""

    def fn(t):
        out = sae(t)
        if keep_bos:
            out[:, 0] = t[:, 0]
        return out

    return fn


@torch.no_grad()
def ce_recovered(model: nn.Module, sae: SAE, batches: list[torch.Tensor]) -> dict[str, float]:
    """Fraction of the CE-loss gap between zero-ablation and clean that the SAE recovers.

    1.0 means splicing in the SAE costs nothing; 0.0 means it is as bad as
    deleting the activation entirely.
    """
    site = (sae.cfg.layer, sae.cfg.hook)

    def zero(t):
        out = torch.zeros_like(t)
        out[:, 0] = t[:, 0]
        return out

    clean = sum(_ce(model, b) for b in batches) / len(batches)
    with edit_sites(model, {site: splice_fn(sae)}):
        spliced = sum(_ce(model, b) for b in batches) / len(batches)
    with edit_sites(model, {site: zero}):
        zeroed = sum(_ce(model, b) for b in batches) / len(batches)
    gap = zeroed - clean
    return {
        "ce_clean": clean,
        "ce_spliced": spliced,
        "ce_zero_ablated": zeroed,
        "ce_recovered": (zeroed - spliced) / gap if abs(gap) > 1e-9 else float("nan"),
    }
