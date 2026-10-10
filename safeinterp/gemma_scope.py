"""Load pretrained Gemma Scope SAEs (Lieberum et al. 2024) for Gemma 2.

Gemma Scope is a set of JumpReLU SAEs that Google DeepMind trained on every
layer of Gemma 2 2B and 9B.  Training SAEs of that quality for a 2B model
takes far more compute than the GPT-2 ones in this repo, so for Gemma we load
theirs and convert them to this repo's ``SAE`` format.  After
``fetch(...)`` / ``python -m safeinterp gemma-scope``, the saved directories
work with every command that takes ``--sae-dir``.

Supported sites:

- ``res`` -> ``resid_post`` (``blocks.L.hook_resid_post``)
- ``mlp`` -> ``mlp_out`` (output of the MLP after Gemma's post-feedforward norm)

The attention SAEs were trained on concatenated head outputs before the
output projection, which is not a site this repo hooks, so they are not supported.

The SAEs were trained on the *base* models.  They transfer well to the
instruction-tuned models (Kissane et al. 2024), which is what lets you use
them to study chat behaviour, but check ``ce_recovered`` on chat text first.

The canonical SAEs (``l0=None``) are the ones with an L0 closest to 100.
They have feature dashboards on Neuronpedia, and the report links to them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .sae import SAE, SAEConfig

SITES = {"res": "resid_post", "mlp": "mlp_out"}
MODELS = {"2b": "google/gemma-2-2b", "9b": "google/gemma-2-9b"}


def repo_and_path(layer: int, site: str = "res", width: str = "16k", l0: int | None = None, size: str = "2b") -> tuple[str, str]:
    """Hugging Face repo and file for one Gemma Scope SAE."""
    if site not in SITES:
        raise ValueError(f"unknown site {site!r}; expected one of {sorted(SITES)}")
    if l0 is None:
        return f"google/gemma-scope-{size}-pt-{site}-canonical", f"layer_{layer}/width_{width}/canonical/params.npz"
    return f"google/gemma-scope-{size}-pt-{site}", f"layer_{layer}/width_{width}/average_l0_{l0}/params.npz"


def from_params(params, layer: int, site: str = "res", model_name: str = "google/gemma-2-2b", neuronpedia: str = "") -> SAE:
    """Build an ``SAE`` from a Gemma Scope ``params.npz`` (or a dict of its arrays).

    Gemma Scope encodes with ``x @ W_enc + b_enc`` while this repo's SAE
    subtracts ``b_dec`` first, so ``b_dec @ W_enc`` is folded into ``b_enc``.
    The result is the same function.
    """
    p = {k: torch.as_tensor(np.asarray(params[k]), dtype=torch.float32) for k in ("W_enc", "W_dec", "b_enc", "b_dec", "threshold")}
    d_in, d_sae = p["W_enc"].shape
    sae = SAE(SAEConfig(d_in=d_in, d_sae=d_sae, kind="jumprelu", layer=layer, hook=SITES[site],
                        model_name=model_name, neuronpedia=neuronpedia))
    with torch.no_grad():
        sae.W_enc.copy_(p["W_enc"])
        sae.W_dec.copy_(p["W_dec"])
        sae.b_dec.copy_(p["b_dec"])
        sae.b_enc.copy_(p["b_enc"] + p["b_dec"] @ p["W_enc"])
        sae.threshold.copy_(p["threshold"])
    return sae.eval()


def fetch(layer: int, site: str = "res", width: str = "16k", l0: int | None = None, size: str = "2b",
          out_dir: str | Path | None = None) -> SAE:
    """Download one Gemma Scope SAE (needs ``huggingface_hub``).  With ``out_dir``
    it is also saved there as ``layer_<L>_<hook>`` in this repo's format."""
    from huggingface_hub import hf_hub_download

    repo, path = repo_and_path(layer, site, width, l0, size)
    neuronpedia = f"gemma-2-{size}/{layer}-gemmascope-{site}-{width}" if l0 is None else ""
    sae = from_params(np.load(hf_hub_download(repo, path)), layer, site, MODELS[size], neuronpedia)
    if out_dir is not None:
        sae.save(Path(out_dir) / f"layer_{layer}_{SITES[site]}")
    return sae


def neuronpedia_url(sae: SAE, feature: int) -> str | None:
    return f"https://www.neuronpedia.org/{sae.cfg.neuronpedia}/{feature}" if sae.cfg.neuronpedia else None
