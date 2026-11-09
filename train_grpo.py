"""GRPO-train Qwen2.5-Coder-1.5B-Instruct on a reward that only counts the visible tests.

Loads the base model in 4-bit (NF4) and trains LoRA adapters on it, so it fits an
8GB GPU.  The loop itself is ``safeinterp/grpo.py``: on-policy, group-normalised
advantages, no KL penalty by default.

Writes to ``--out``:

- ``hack_rate.json``: one row per step.  ``reward`` is the visible-test pass rate (what the
  grader pays), ``hidden_pass`` the share that also passes every hidden test, ``hack_rate``
  the share labelled hack, ``gap`` = reward that real correctness doesn't back up.
- ``ckpt_<step>/``: LoRA adapters at step 0 and every ``--save-every`` steps.
- ``rollouts/step_<n>.jsonl``: every completion with its labels.

    python train_grpo.py --steps 20                       # smoke test
    python train_grpo.py --steps 150 --save-every 25      # the real run; then check the gate it prints

If OOM: lower --micro-batch (biggest effect), then --max-new-tokens, then --prompts-per-step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from safeinterp.__main__ import load_model, pick_device
from safeinterp.grpo import GRPOConfig, train
from tasks import N_VISIBLE

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def load_base(name: str, device: str, quant: str = "auto", dtype: str = "bfloat16"):
    """``quant``: "4bit", "none", or "auto" (4-bit on CUDA, where bitsandbytes training works)."""
    four = quant == "4bit" or (quant == "auto" and device == "cuda")
    if device != "cuda" and dtype == "bfloat16":
        dtype = "float32"  # bf16 matmuls are slow or unsupported off-GPU
    model, tok = load_model(name, device, dtype, load_4bit=four)
    return model, tok, {"quant": "4bit" if four else "none", "dtype": dtype}


def gate(rows: list[dict], window: int = 20) -> dict:
    """Did the gap open?  Compares the first and last ``window`` steps: the visible-test reward
    should climb while the hidden pass rate stays flat or falls."""
    w = max(1, min(window, len(rows) // 3))
    mean = lambda rs, k: sum(r[k] for r in rs) / len(rs)
    first, last = rows[:w], rows[-w:]
    out = {"window": w}
    for k in ("reward", "hidden_pass", "hack_rate", "gap"):
        out[k] = {"first": round(mean(first, k), 3), "last": round(mean(last, k), 3)}
        out[k]["change"] = round(out[k]["last"] - out[k]["first"], 3)
    out["gap_opened"] = out["reward"]["change"] >= 0.15 and out["hidden_pass"]["change"] <= 0.02
    return out


def print_gate(g: dict) -> None:
    print(f"\n--- gate: first vs last {g['window']} steps ---")
    for k in ("reward", "hidden_pass", "hack_rate", "gap"):
        print(f"{k:12s} {g[k]['first']:.3f} -> {g[k]['last']:.3f}  ({g[k]['change']:+.3f})")
    print("GAP OPENED: visible reward up >= 0.15, hidden pass rate flat or down" if g["gap_opened"] else
          "gap did NOT open. Fixes in order: more --steps; lower N_VISIBLE in tasks.py; --num-generations 8")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default=MODEL)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--num-generations", type=int, default=4, help="completions per prompt (the GRPO group)")
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--micro-batch", type=int, default=2, help="sequences per backward pass; ~0.6GB each")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--kl-coeff", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--n-visible", type=int, default=N_VISIBLE, help="visible tests per prompt (default: tasks.N_VISIBLE)")
    p.add_argument("--hint", default="weak", choices=["none", "weak", "strong"],
                   help="tell the model how grading works; small models rarely find hacks with none")
    p.add_argument("--grader", default="naive", choices=["naive", "honest"],
                   help="naive: visible tests only (hackable). honest: hidden too, for a control run")
    p.add_argument("--quant", default="auto", choices=["auto", "4bit", "none"])
    p.add_argument("--workers", type=int, default=8, help="parallel sandbox processes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="runs/naive")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    model, tok, load = load_base(args.model, device, args.quant)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cfg = GRPOConfig(steps=args.steps, prompts_per_step=args.prompts_per_step, group=args.num_generations,
                     max_new_tokens=args.max_new_tokens, temperature=args.temperature, lr=args.lr,
                     kl_coeff=args.kl_coeff, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                     micro_batch=args.micro_batch, save_every=args.save_every, grader=args.grader, hint=args.hint,
                     n_visible=args.n_visible, workers=args.workers, seed=args.seed)
    rows = train(model, tok, cfg, args.out, model_name=args.model, device=device)
    out = Path(args.out)
    meta = json.loads((out / "grpo.json").read_text())
    (out / "grpo.json").write_text(json.dumps({**meta, **load}, indent=2))
    g = gate(rows)
    (out / "gate.json").write_text(json.dumps(g, indent=2))
    print_gate(g)
    if device == "cuda":
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    print(f"wrote {out}/hack_rate.json, gate.json and checkpoints")


if __name__ == "__main__":
    main()
