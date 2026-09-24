"""Train one SAE per GPT-2 layer from a shared stream of activations.

Training every layer at once is the point: to ask *where* in the network a
capability lives, you need comparable dictionaries at every depth, and one
forward pass can feed all of them.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from .data import ActivationBuffer
from .evaluate import ce_recovered
from .sae import SAE, SAEConfig


@dataclass
class TrainConfig:
    layers: list[int] = field(default_factory=lambda: list(range(12)))
    hook: str = "resid_post"
    kind: str = "topk"
    expansion: int = 8
    k: int = 32
    l1_coeff: float = 5.0
    lr: float = 2e-4
    total_tokens: int = 20_000_000
    batch_size: int = 4096
    ctx_len: int = 128
    seq_batch: int = 32
    buffer_tokens: int = 2**18
    warmup_frac: float = 0.02
    l1_warmup_frac: float = 0.05
    decay_frac: float = 0.2
    log_every: int = 100
    eval_seqs: int = 64
    seed: int = 0


def lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    warm = max(1, int(cfg.warmup_frac * total))
    decay_start = int((1 - cfg.decay_frac) * total)
    if step < warm:
        return cfg.lr * (step + 1) / warm
    if step >= decay_start:
        return cfg.lr * max(0.0, (total - step) / max(1, total - decay_start))
    return cfg.lr


def train_saes(
    model: nn.Module,
    seq_batches,
    cfg: TrainConfig,
    out_dir: str | Path,
    model_name: str = "gpt2",
    eval_batches: list[torch.Tensor] | None = None,
    device: str | torch.device = "cpu",
) -> dict[int, SAE]:
    torch.manual_seed(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_cfg.json").write_text(json.dumps(cfg.__dict__, indent=2))

    d = model.config.n_embd
    sites = [(layer, cfg.hook) for layer in cfg.layers]
    buffer = ActivationBuffer(model, seq_batches, sites, cfg.buffer_tokens, cfg.batch_size, device)

    saes = {
        layer: SAE(
            SAEConfig(
                d_in=d, d_sae=d * cfg.expansion, kind=cfg.kind, k=cfg.k, l1_coeff=cfg.l1_coeff,
                dead_after_tokens=min(2_000_000, cfg.total_tokens // 10),
                layer=layer, hook=cfg.hook, model_name=model_name,
            )
        ).to(device)
        for layer in cfg.layers
    }
    first = next(buffer)
    for i, layer in enumerate(cfg.layers):
        saes[layer].init_from_data(first[i])
    opts = {l: torch.optim.Adam(s.parameters(), lr=cfg.lr, betas=(0.9, 0.999)) for l, s in saes.items()}

    total_steps = max(1, cfg.total_tokens // cfg.batch_size)
    l1_warm = max(1, int(cfg.l1_warmup_frac * total_steps))
    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("w")
    t0 = time.time()
    batch = first
    for step in range(total_steps):
        if step > 0:
            try:
                batch = next(buffer)
            except StopIteration:
                print(f"data exhausted at step {step}")
                break
        lr = lr_at(step, total_steps, cfg)
        l1_scale = min(1.0, (step + 1) / l1_warm)
        row = {"step": step, "tokens": (step + 1) * cfg.batch_size, "lr": lr}
        for i, layer in enumerate(cfg.layers):
            sae, opt = saes[layer], opts[layer]
            for g in opt.param_groups:
                g["lr"] = lr
            out = sae.loss(batch[i], l1_coeff=sae.cfg.l1_coeff * l1_scale)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            sae.remove_parallel_grad()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            opt.step()
            sae.normalize_decoder()
            if step % cfg.log_every == 0:
                row[f"L{layer}"] = {k: round(v.item(), 5) for k, v in out.items()}
        if step % cfg.log_every == 0:
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()
            summary = "  ".join(
                f"L{l}: fvu={row[f'L{l}']['fvu']:.3f} l0={row[f'L{l}']['l0']:.0f} dead={row[f'L{l}']['dead_frac']:.2f}"
                for l in cfg.layers
            )
            rate = (step + 1) * cfg.batch_size / (time.time() - t0)
            print(f"[{step}/{total_steps}] {rate:,.0f} tok/s  {summary}", flush=True)
    log_f.close()

    results = {}
    for layer, sae in saes.items():
        sae.eval()
        path = out_dir / f"layer_{layer}_{cfg.hook}"
        sae.save(path)
        if eval_batches:
            res = ce_recovered(model, sae, eval_batches)
            results[layer] = res
            (path / "eval.json").write_text(json.dumps(res, indent=2))
            print(f"L{layer} {cfg.hook}: CE clean {res['ce_clean']:.3f} spliced {res['ce_spliced']:.3f} "
                  f"zero {res['ce_zero_ablated']:.3f} recovered {res['ce_recovered']:.3f}")
    if results:
        (out_dir / "eval_summary.json").write_text(json.dumps(results, indent=2))
    return saes
