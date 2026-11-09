"""Residual-stream activations at every training checkpoint, labelled honest / hack / broken.

For each ``ckpt_<step>`` of a ``train_grpo.py`` run this samples ``--per-task``
completions for each of the 30 training prompts, grades them (``sandbox.label``), and
records layer ``--layer`` ``resid_post`` activations:

- ``comp``: the completion tokens, read by **that checkpoint** (what the SAE is trained on)
- ``comp_base``: the **same tokens** read by the step-0 model (adapter disabled).  This is
  the text control: a feature that only detects hardcoded numeric literals responds to
  the text and fires just as much here; a feature that tracks the model's state doesn't
- ``prompt``: the last ``--n-prompt`` prompt tokens, read before any code is written.
  The prompt is identical across samples and checkpoints, so this can only change
  because the model changed

``meta.jsonl`` holds each completion's label and its syntactic covariates (numeric
literals, visible expected outputs written into the code, static hack signatures)
for the controls in ``analyze.py``.

    python reward_hacking/capture_activations.py --run runs/naive --layer 14 --per-task 8
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from safeinterp.core.hooks import capture
from safeinterp.core.models import pick_device
from safeinterp.rlhack.grpo import encode_prompt, sample, stop_ids
from safeinterp.rlhack.monitor import checkpoints
from sandbox import grade_many, extract_code, label
from tasks import train_tasks
from train_grpo import load_base

_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


def covariates(text: str, task, g: dict) -> dict:
    """Surface features of the code, for the "is it just reading literals?" control."""
    code = extract_code(text)
    return {
        "n_num_literals": len(_NUM.findall(code)),
        # expected outputs of the visible tests that appear verbatim in the code
        "visible_out_literals": sum(repr(e) in code for _, e in task.visible),
        "hardcode_sig": "hardcode" in g["signatures"],
        "flagged": g["flagged"],
        "code_chars": len(code),
    }


@torch.no_grad()
def read(model, ids: list[int], layer: int, device) -> torch.Tensor:
    a = capture(model, torch.tensor([ids], device=device), [(layer, "resid_post")])
    return a[(layer, "resid_post")][0].float().cpu()


def capture_checkpoint(pm, tok, tasks, hint: str, layer: int, per_task: int, n_prompt: int, max_new_tokens: int,
                       temperature: float, sample_batch: int, workers: int, device, seed: int = 0):
    """``pm`` is a PeftModel with this checkpoint's adapter loaded."""
    model = pm.get_base_model()
    model.eval()
    torch.manual_seed(seed)
    group = [t for t in tasks for _ in range(per_task)]
    pids = {t.name: encode_prompt(tok, t, hint) for t in tasks}
    prompts = [pids[t.name] for t in group]
    comps = sample(model, tok, prompts, max_new_tokens, temperature, device, batch=sample_batch)
    stops = stop_ids(tok)
    texts = [tok.decode([i for i in c if i not in stops]) for c in comps]
    grades = grade_many(texts, group, "naive", workers)

    comp, comp_base, prompt_acts, offsets, meta, comp_ids = [], [], [], [0], [], []
    for k, (t, p, c, text, g) in enumerate(zip(group, prompts, comps, texts, grades)):
        ids = p + c
        a = read(model, ids, layer, device)
        with pm.disable_adapter():
            b = read(model, ids, layer, device)[len(p):]
        prompt_acts.append(a[len(p) - n_prompt:len(p)].half())
        comp.append(a[len(p):].half())
        comp_base.append(b.half())
        offsets.append(offsets[-1] + len(c))
        comp_ids += c
        meta.append({"task": t.name, "sample": k % per_task, "label": label(g), "hack_kind": g["hack_kind"],
                     "reward": g["reward"], "hidden_frac": g["hidden_frac"], "overfit": g["overfit"],
                     "n_tokens": len(c), **covariates(text, t, g), "completion": text})
    d = prompt_acts[0].shape[-1]
    cat = lambda xs: torch.cat(xs) if any(len(x) for x in xs) else torch.zeros(0, d, dtype=torch.half)
    acts = {"comp": cat(comp), "comp_base": cat(comp_base), "offsets": torch.tensor(offsets),
            "comp_ids": torch.tensor(comp_ids, dtype=torch.long), "prompt": torch.stack(prompt_acts)}
    return acts, meta


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", default="runs/naive", help="output directory of train_grpo.py")
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--per-task", type=int, default=8, help="completions per training prompt per checkpoint")
    p.add_argument("--n-prompt", type=int, default=4, help="last prompt tokens to keep")
    p.add_argument("--max-new-tokens", type=int, help="default: as in training")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sample-batch", type=int, default=32, help="completions generated at once")
    p.add_argument("--quant", choices=["auto", "4bit", "none"], help="default: as in training")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", help="default: <run>/acts")
    args = p.parse_args(argv)
    from peft import PeftModel

    run = Path(args.run)
    cfg = json.loads((run / "grpo.json").read_text())
    device = pick_device(args.device)
    base, tok, load = load_base(cfg["model"], device, args.quant or cfg.get("quant", "auto"))
    tasks = train_tasks(n_visible=cfg.get("n_visible", 3), seed=cfg["seed"])
    out = Path(args.out or run / "acts")
    out.mkdir(parents=True, exist_ok=True)
    (out / "capture.json").write_text(json.dumps({**vars(args), "model": cfg["model"], "hint": cfg["hint"],
                                                  **load}, indent=2))
    summary = []
    with (out / "meta.jsonl").open("w") as f:
        for step, path in checkpoints(run):
            pm = PeftModel.from_pretrained(base, str(path))
            acts, meta = capture_checkpoint(pm, tok, tasks, cfg["hint"], args.layer, args.per_task, args.n_prompt,
                                            args.max_new_tokens or cfg["max_new_tokens"], args.temperature,
                                            args.sample_batch, args.workers, device, seed=args.seed + step)
            base = pm.unload()
            torch.save(acts, out / f"step_{step}.pt")
            for i, m in enumerate(meta):
                f.write(json.dumps({"step": step, "idx": i, **m}) + "\n")
            f.flush()
            counts = {l: sum(m["label"] == l for m in meta) for l in ("honest", "hack", "broken")}
            summary.append({"step": step, **counts, "tokens": int(acts["offsets"][-1])})
            print(f"step {step}: {counts}  {summary[-1]['tokens']} tokens", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {out}/step_*.pt, meta.jsonl, summary.json")


if __name__ == "__main__":
    main()
