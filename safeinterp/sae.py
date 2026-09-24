"""Sparse autoencoder for transformer activations.

Two sparsity mechanisms:

- ``topk``: keep the k largest pre-activations (Gao et al. 2024). L0 is fixed
  by construction; an auxiliary "AuxK" loss revives dead latents.
- ``relu``: ReLU + L1 penalty weighted by decoder norm (Anthropic, 2024 update).

Inputs are rescaled by a fixed scalar ``act_scale`` so that
``E[||x * act_scale||^2] = d_in``.  GPT-2's residual norm grows a lot with
depth, so without this the hyperparameters would not transfer between layers.
All public methods take and return activations in the *raw* model space.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SAEConfig:
    d_in: int = 768
    d_sae: int = 768 * 8
    kind: str = "topk"  # "topk" or "relu"
    k: int = 32  # topk only
    l1_coeff: float = 5.0  # relu only
    aux_k: int = 256  # topk only: latents used by the AuxK loss
    aux_coeff: float = 1 / 32
    dead_after_tokens: int = 2_000_000
    layer: int = 0
    hook: str = "resid_post"
    model_name: str = "gpt2"


class SAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        if cfg.kind not in ("topk", "relu"):
            raise ValueError(f"unknown SAE kind {cfg.kind!r}")
        if cfg.kind == "topk" and not 0 < cfg.k <= cfg.d_sae:
            raise ValueError(f"k={cfg.k} must be in [1, d_sae={cfg.d_sae}]")
        self.cfg = cfg
        W_dec = torch.randn(cfg.d_sae, cfg.d_in)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in))
        self.register_buffer("act_scale", torch.tensor(1.0))
        self.register_buffer("tokens_since_fired", torch.zeros(cfg.d_sae, dtype=torch.long))

    # ------------------------------------------------------------------ core
    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.act_scale
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def _sparsify(self, pre: torch.Tensor) -> torch.Tensor:
        if self.cfg.kind == "relu":
            return F.relu(pre)
        vals, idx = pre.topk(self.cfg.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, F.relu(vals))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Raw activations ``[..., d_in]`` -> feature activations ``[..., d_sae]``."""
        return self._sparsify(self.pre_acts(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Feature activations -> reconstruction in raw activation space."""
        return (z @ self.W_dec + self.b_dec) / self.act_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def feature_directions(self) -> torch.Tensor:
        """What each feature writes to the model, in raw space: ``[d_sae, d_in]``."""
        return self.W_dec / self.act_scale

    # -------------------------------------------------------------- training
    def loss(self, x: torch.Tensor, l1_coeff: float | None = None) -> dict[str, torch.Tensor]:
        """Loss on a batch of raw activations ``[batch, d_in]`` (normalized space)."""
        xs = x * self.act_scale
        pre = self.pre_acts(x)
        z = self._sparsify(pre)
        recon = z @ self.W_dec + self.b_dec
        err = xs - recon
        mse = err.pow(2).sum(-1).mean()
        out = {"mse": mse, "l0": (z > 0).float().sum(-1).mean()}
        out["fvu"] = err.pow(2).sum() / (xs - xs.mean(0)).pow(2).sum()

        with torch.no_grad():
            fired = (z > 0).any(0)
            self.tokens_since_fired += x.shape[0]
            self.tokens_since_fired[fired] = 0
        dead = self.tokens_since_fired > self.cfg.dead_after_tokens
        out["dead_frac"] = dead.float().mean()

        if self.cfg.kind == "relu":
            coeff = self.cfg.l1_coeff if l1_coeff is None else l1_coeff
            l1 = (z * self.W_dec.norm(dim=1)).sum(-1).mean()
            out["l1"] = l1
            out["loss"] = mse + coeff * l1
            return out

        # AuxK: reconstruct the residual error using only dead latents.
        aux = torch.zeros((), device=x.device)
        n_dead = int(dead.sum())
        if n_dead > 0:
            k_aux = min(self.cfg.aux_k, n_dead)
            dead_pre = pre.masked_fill(~dead, float("-inf"))
            vals, idx = dead_pre.topk(k_aux, dim=-1)
            z_aux = torch.zeros_like(pre).scatter_(-1, idx, F.relu(vals))
            err_hat = z_aux @ self.W_dec
            target = err.detach()
            aux = (err_hat - target).pow(2).sum(-1).mean() / target.pow(2).sum(-1).mean().clamp_min(1e-8)
            aux = torch.nan_to_num(aux, 0.0)
        out["aux"] = aux
        out["loss"] = mse + self.cfg.aux_coeff * aux
        return out

    @torch.no_grad()
    def init_from_data(self, x: torch.Tensor) -> None:
        """Set ``act_scale`` and ``b_dec`` from a sample of raw activations."""
        self.act_scale.fill_((self.cfg.d_in / x.pow(2).sum(-1).mean()).sqrt().item())
        self.b_dec.copy_((x * self.act_scale).mean(0))

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def remove_parallel_grad(self) -> None:
        """Drop the gradient component that would change decoder row norms."""
        if self.W_dec.grad is None:
            return
        parallel = (self.W_dec.grad * self.W_dec).sum(1, keepdim=True)
        self.W_dec.grad.sub_(parallel * self.W_dec)

    # ---------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "sae.pt")
        (path / "cfg.json").write_text(json.dumps(asdict(self.cfg), indent=2))

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "SAE":
        path = Path(path)
        cfg = SAEConfig(**json.loads((path / "cfg.json").read_text()))
        sae = cls(cfg)
        sae.load_state_dict(torch.load(path / "sae.pt", map_location=device))
        return sae.to(device)
