"""Read and edit GPT-2 and Gemma 2 activations with PyTorch hooks.

A *site* is a ``(layer, hook)`` pair.  Supported hooks:

- ``resid_pre``  : residual stream entering block ``layer``
- ``resid_post`` : residual stream leaving block ``layer``
- ``mlp_out``    : output of the MLP in block ``layer`` (what the MLP writes)
- ``attn_out``   : output of attention in block ``layer``

Works with HuggingFace ``GPT2LMHeadModel`` / ``GPT2Model`` and with Llama-style
models (``model.model.layers``) such as ``Gemma2ForCausalLM`` (transformers 4.x
returns tuples from blocks, 5.x returns tensors; both are handled).

Gemma 2 normalises each sublayer's output before adding it to the residual
stream (``post_attention_layernorm`` / ``post_feedforward_layernorm``), so on
Gemma ``attn_out`` and ``mlp_out`` are read *after* those norms.  That is what
the sublayer actually writes, it keeps ``resid_post = resid_pre + attn_out +
mlp_out``, and it matches the ``hook_mlp_out`` site that Gemma Scope SAEs were
trained on.
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
    """The stack of blocks without the LM head (``GPT2Model``, ``Gemma2Model``, ...)."""
    if hasattr(model, "transformer"):
        return model.transformer
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    return model


def blocks(model: nn.Module) -> nn.ModuleList:
    t = transformer(model)
    return t.h if hasattr(t, "h") else t.layers


def n_layers(model: nn.Module) -> int:
    return len(blocks(model))


def final_norm(model: nn.Module) -> nn.Module:
    t = transformer(model)
    return t.ln_f if hasattr(t, "ln_f") else t.norm


def _module(model: nn.Module, site: Site) -> nn.Module:
    layer, hook = site
    if hook not in HOOKS:
        raise ValueError(f"unknown hook {hook!r}; expected one of {HOOKS}")
    block = blocks(model)[layer]
    if hook in ("resid_pre", "resid_post"):
        return block
    if hasattr(block, "post_feedforward_layernorm"):  # Gemma 2: sublayer outputs are normed before the add
        return block.post_feedforward_layernorm if hook == "mlp_out" else block.post_attention_layernorm
    return block.mlp if hook == "mlp_out" else block.attn if hasattr(block, "attn") else block.self_attn


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
