"""Read and edit GPT-2 activations with PyTorch hooks.

A *site* is a ``(layer, hook)`` pair.  Supported hooks:

- ``resid_pre``  : residual stream entering block ``layer``
- ``resid_post`` : residual stream leaving block ``layer``
- ``mlp_out``    : output of the MLP in block ``layer`` (what the MLP writes)
- ``attn_out``   : output of attention in block ``layer``

Works with HuggingFace ``GPT2LMHeadModel`` / ``GPT2Model`` (transformers 4.x
returns tuples from blocks, 5.x returns tensors; both are handled).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

import torch
from torch import nn

HOOKS = ("resid_pre", "resid_post", "mlp_out", "attn_out")

Site = tuple[int, str]
EditFn = Callable[[torch.Tensor], torch.Tensor]


class StopForward(Exception):
    """Raised from a hook to abort the forward pass once we have what we need."""


def transformer(model: nn.Module) -> nn.Module:
    return model.transformer if hasattr(model, "transformer") else model


def n_layers(model: nn.Module) -> int:
    return len(transformer(model).h)


def _module(model: nn.Module, site: Site) -> nn.Module:
    layer, hook = site
    if hook not in HOOKS:
        raise ValueError(f"unknown hook {hook!r}; expected one of {HOOKS}")
    block = transformer(model).h[layer]
    return {"resid_pre": block, "resid_post": block, "mlp_out": block.mlp, "attn_out": block.attn}[hook]


def _split(output):
    """Return (tensor, rebuild) for a module output that may be a tuple."""
    if isinstance(output, tuple):
        return output[0], lambda t: (t, *output[1:])
    return output, lambda t: t


@contextmanager
def edit_sites(model: nn.Module, edits: dict[Site, EditFn]) -> Iterator[None]:
    """Apply ``fn(activation) -> activation`` at each site during forward passes.

    ``fn`` receives a ``[batch, seq, d_model]`` tensor and returns one of the
    same shape (returning the input unchanged makes it a read-only hook).
    """
    handles = []
    try:
        for site, fn in edits.items():
            mod = _module(model, site)
            if site[1] == "resid_pre":

                def pre_hook(module, args, kwargs, fn=fn):
                    if args:
                        return (fn(args[0]), *args[1:]), kwargs
                    kwargs = dict(kwargs)
                    kwargs["hidden_states"] = fn(kwargs["hidden_states"])
                    return args, kwargs

                handles.append(mod.register_forward_pre_hook(pre_hook, with_kwargs=True))
            else:

                def post_hook(module, args, output, fn=fn):
                    t, rebuild = _split(output)
                    return rebuild(fn(t))

                handles.append(mod.register_forward_hook(post_hook))
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def capture(model: nn.Module, input_ids: torch.Tensor, sites: list[Site], stop_early: bool = True) -> dict[Site, torch.Tensor]:
    """Run ``model`` on ``input_ids`` and return activations at ``sites``.

    With ``stop_early`` the forward pass is aborted after the last requested
    site has been reached, which saves compute when only early layers are needed.
    """
    acts: dict[Site, torch.Tensor] = {}
    remaining = set(sites)

    def make(site):
        def fn(t):
            acts[site] = t.detach()
            remaining.discard(site)
            if stop_early and not remaining:
                raise StopForward
            return t

        return fn

    with edit_sites(model, {s: make(s) for s in sites}):
        try:
            transformer(model)(input_ids)
        except StopForward:
            pass
    return acts
