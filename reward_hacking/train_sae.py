"""One SAE trained on activations pooled across every training checkpoint.

Uses the ReLU + L1 SAE from ``safeinterp/core/sae.py`` (L1 weighted by decoder norm), so
``--l1`` sets the sparsity.  Each checkpoint contributes the same number of tokens,
so later checkpoints don't dominate the dictionary.  Trains on the checkpoint's own
reading of its completions plus the prompt tokens; ``comp_base`` (the step-0 model
reading the same text) is held back for the control in ``analyze.py``.

Reports L0 (mean active features per token) and FVU on 5% held-out tokens.  Aim for
L0 roughly 20-60: raise ``--l1`` if it is higher, lower it if it is lower.

    python reward_hacking/train_sae.py --acts runs/naive/acts --l1 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from safeinterp.core.models import pick_device
from safeinterp.core.sae import SAE, SAEConfig

L0_RANGE = (20, 60)


def load_tokens(acts_dir: Path, seed: int = 0) -> tuple[torch.Tensor, list[int]]:
    """Completion + prompt tokens from every checkpoint, the same count from each."""
    files = sorted(acts_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise SystemExit(f"no step_*.pt in {acts_dir}: run capture_activations.py first")
    per = []
    for f in files:
        a = torch.load(f)
        per.append(torch.cat([a["comp"], a["prompt"].flatten(0, 1)]))
    n = min(len(x) for x in per)
    g = torch.Generator().manual_seed(seed)
    x = torch.cat([t[torch.randperm(len(t), generator=g)[:n]] for t in per])
    return x[torch.randperm(len(x), generator=g)], [int(f.stem.split("_")[1]) for f in files]


@torch.no_grad()
def evaluate(sae: SAE, x: torch.Tensor, batch: int = 8192) -> dict:
    l0, err, var, n = 0.0, 0.0, 0.0, 0
    mu = x.float().mean(0)
    fired = torch.zeros(sae.cfg.d_sae, dtype=torch.bool, device=x.device)
    for i in range(0, len(x), batch):
        b = x[i:i + batch].float()
        z = sae.encode(b)
        l0 += float((z > 0).float().sum())
        err += float((sae.decode(z) - b).pow(2).sum())
        var += float((b - mu).pow(2).sum())
        fired |= (z > 0).any(0)
        n += len(b)
    return {"l0": round(l0 / n, 2), "fvu": round(err / var, 4), "dead_frac": round(1 - float(fired.float().mean()), 4),
            "tokens": n}


def train(x: torch.Tensor, l1: float, expansion: int = 8, lr: float = 1e-3, epochs: int = 10, batch: int = 4096,
          l1_warmup: float = 0.05, seed: int = 0, device="cpu", log=print) -> SAE:
    torch.manual_seed(seed)
    d = x.shape[1]
    sae = SAE(SAEConfig(d_in=d, d_sae=d * expansion, kind="relu", l1_coeff=l1, hook="resid_post",
                        dead_after_tokens=max(len(x) // 2, batch))).to(device)
    sae.init_from_data(x[:batch * 4].float().to(device))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    steps = max(1, epochs * len(x) // batch)
    warm, decay = max(1, int(l1_warmup * steps)), int(0.8 * steps)
    for s in range(steps):
        i = (s * batch) % max(1, len(x) - batch + 1)
        b = x[i:i + batch].float().to(device)
        for gr in opt.param_groups:
            gr["lr"] = lr * (1 if s < decay else (steps - s) / max(1, steps - decay))
        out = sae.loss(b, l1_coeff=l1 * min(1.0, (s + 1) / warm))
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        sae.remove_parallel_grad()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        sae.normalize_decoder()
        if s % max(1, steps // 20) == 0 or s == steps - 1:
            log(f"[{s}/{steps}] fvu {out['fvu'].item():.3f}  l0 {out['l0'].item():.1f}  l1 {out['l1'].item():.3f}")
    return sae.eval()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--acts", default="runs/naive/acts", help="output directory of capture_activations.py")
    p.add_argument("--l1", type=float, default=5.0, help="L1 coefficient: higher = sparser = lower L0")
    p.add_argument("--expansion", type=int, default=8, help="d_sae = expansion * d_model")
    p.add_argument("--lr", type=float, default=1e-3, help="high for an SAE because the dataset is small (~0.5M tokens)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", help="default: <acts>/../sae")
    args = p.parse_args(argv)

    acts = Path(args.acts)
    device = pick_device(args.device)
    x, steps = load_tokens(acts, args.seed)
    n_eval = max(1, len(x) // 20)
    held, x = x[:n_eval].to(device), x[n_eval:]
    print(f"{len(x):,} training tokens from checkpoints {steps} ({len(x) // len(steps):,} each), {n_eval:,} held out")
    sae = train(x, args.l1, args.expansion, args.lr, args.epochs, args.batch_size, seed=args.seed, device=device)
    cap = json.loads((acts / "capture.json").read_text()) if (acts / "capture.json").exists() else {}
    sae.cfg.layer, sae.cfg.model_name = cap.get("layer", 0), cap.get("model", "")
    out = Path(args.out or acts.parent / "sae")
    sae.save(out)
    ev = {**evaluate(sae, held), "l1": args.l1, "checkpoints": steps}
    (out / "eval.json").write_text(json.dumps(ev, indent=2))
    lo, hi = L0_RANGE
    print(f"\nfinal held-out L0 {ev['l0']}  FVU {ev['fvu']}  dead {ev['dead_frac']:.1%}")
    if ev["l0"] > hi:
        print(f"L0 above {lo}-{hi}: retrain with a larger --l1 (now {args.l1})")
    elif ev["l0"] < lo:
        print(f"L0 below {lo}-{hi}: retrain with a smaller --l1 (now {args.l1})")
    else:
        print(f"L0 in the {lo}-{hi} target range")
    if ev["fvu"] > 0.3:
        print(f"FVU {ev['fvu']} is high: the SAE misses much of the activation. Try more --epochs, or a smaller --l1 "
              "if L0 allows; features from a poor reconstruction are less trustworthy")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
