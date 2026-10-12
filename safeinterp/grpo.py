"""GRPO on the hackable coding environment (``rlhack.py``), logging when hacking starts.

Each step samples ``group`` completions for each of ``prompts_per_step`` train
tasks, grades them, and takes one policy-gradient step with group-normalised
advantages (Shao et al. 2024).  Updates are on-policy, so there is no
importance ratio or clipping.  The loss is normalised by a constant token count
(Liu et al. 2025, "Dr. GRPO"), and the KL penalty to the starting model is off
by default (``kl_coeff``).

Writes to ``out``:

- ``hack_rate.json``: one row per step with reward, real pass rates, hack rate
  overall and by kind, and the gap between reward and real correctness.  This
  is the behavioural signal the monitor has to beat.
- ``rollouts/step_<n>.jsonl``: every completion with its labels.
- ``rollouts_text/step_<n>.txt``: prompts and completions as plain text, to
  train an SAE on the policy's own distribution
  (``python -m safeinterp train --text-files "<out>/rollouts_text/*.txt"``).
- ``ckpt_<n>/``: LoRA adapters (or full weights) every ``save_every`` steps, plus ``ckpt_0``.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from . import rlhack as R
from .analysis import encode_text
from .misalign import chat_messages, render_chat


@dataclass
class GRPOConfig:
    steps: int = 200
    prompts_per_step: int = 8
    group: int = 8
    max_new_tokens: int = 384
    temperature: float = 1.0
    lr: float = 2e-5
    kl_coeff: float = 0.0
    lora_r: int = 32  # 0 = full finetune
    lora_alpha: int = 64
    micro_batch: int = 8
    save_every: int = 10
    grader: str = "naive"  # "naive" (hackable) or "honest" (control run)
    hint: str = "none"
    workers: int = 8
    seed: int = 0


def stop_ids(tok) -> list[int]:
    ids = [tok.eos_token_id]
    conv = getattr(tok, "convert_tokens_to_ids", None)
    for t in ("<|im_end|>", "<|endoftext|>", "<end_of_turn>", "<|eot_id|>"):
        i = conv(t) if conv else None
        if isinstance(i, int) and i >= 0 and i != getattr(tok, "unk_token_id", None) and i not in ids:
            ids.append(i)
    return ids


def encode_prompt(tok, task: R.Task, hint: str) -> list[int]:
    return render_chat(tok, chat_messages(R.prompt(task, hint), R.SYSTEM), True)


def pad_left(seqs: list[list[int]], pad: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, T - len(s):], mask[i, T - len(s):] = torch.tensor(s), 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def sample(model, tok, prompts: list[list[int]], max_new_tokens: int, temperature: float, device,
           batch: int = 64) -> list[list[int]]:
    """Completion token ids, cut after the first stop token (which is kept)."""
    stops = stop_ids(tok)
    pad = getattr(tok, "pad_token_id", None) or tok.eos_token_id
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        ids, mask = pad_left(chunk, pad, device)
        gen = model.generate(input_ids=ids, attention_mask=mask, do_sample=temperature > 0,
                             temperature=temperature if temperature > 0 else None, top_p=1.0, top_k=0,
                             max_new_tokens=max_new_tokens, pad_token_id=pad, eos_token_id=stops)
        for row in gen[:, ids.shape[1]:].tolist():
            cut = next((j for j, t in enumerate(row) if t in stops), None)
            out.append(row if cut is None else row[:cut + 1])
    return out


def token_logps(model, seqs: list[list[int]], starts: list[int], pad: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probs of each sequence from ``start`` on, right-padded: ``[n, T]`` values and mask."""
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), pad, dtype=torch.long, device=device)
    attn = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, :len(s)], attn[i, :len(s)] = torch.tensor(s, device=device), 1
    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
    logp = torch.gather(F.log_softmax(logits, -1), -1, ids[:, 1:, None])[..., 0]
    keep = torch.zeros_like(logp, dtype=torch.bool)
    for i, (s, st) in enumerate(zip(seqs, starts)):
        keep[i, st - 1:len(s) - 1] = True
    return logp, keep


def advantages(rewards: torch.Tensor, group: int) -> torch.Tensor:
    """Group-normalised: (r - mean) / std within each prompt's group; 0 when all are equal."""
    r = rewards.view(-1, group)
    std = r.std(1, keepdim=True)
    return ((r - r.mean(1, keepdim=True)) / std.clamp_min(1e-6) * (std > 1e-6)).flatten()


def save_ckpt(model, tok, out: Path, step: int, lora: bool) -> None:
    path = out / f"ckpt_{step}"
    model.save_pretrained(path)  # with peft this writes only the adapter
    if hasattr(tok, "save_pretrained"):
        tok.save_pretrained(path)
    (path / "step.json").write_text(json.dumps({"step": step, "lora": lora}))


def train(model, tok, cfg: GRPOConfig, out_dir: str | Path, model_name: str = "", device="cpu") -> list[dict]:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out = Path(out_dir)
    for sub in ("rollouts", "rollouts_text"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / "grpo.json").write_text(json.dumps({**cfg.__dict__, "model": model_name}, indent=2))
    tasks = R.build_tasks(split="train", seed=cfg.seed)
    (out / "tasks.json").write_text(json.dumps([{"name": t.name, "split": t.split, "visible": t.visible,
                                                 "hidden": t.hidden} for t in R.build_tasks(seed=cfg.seed)]))
    if cfg.lora_r:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
                                                 task_type="CAUSAL_LM", target_modules="all-linear"))
    if hasattr(model, "gradient_checkpointing_enable") and device == "cuda":
        model.gradient_checkpointing_enable()
        if cfg.lora_r:
            model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    pad = getattr(tok, "pad_token_id", None) or tok.eos_token_id
    prompt_ids = {t.name: encode_prompt(tok, t, cfg.hint) for t in tasks}
    save_ckpt(model, tok, out, 0, bool(cfg.lora_r))
    rows, t0 = [], time.time()
    for step in range(1, cfg.steps + 1):
        batch = random.sample(tasks, min(cfg.prompts_per_step, len(tasks)))
        group_tasks = [t for t in batch for _ in range(cfg.group)]
        prompts = [prompt_ids[t.name] for t in group_tasks]
        model.eval()
        comps = sample(model, tok, prompts, cfg.max_new_tokens, cfg.temperature, device)
        texts = [tok.decode([t for t in c if t not in stop_ids(tok)]) for c in comps]
        labels = R.grade_many(texts, group_tasks, cfg.grader, cfg.workers)
        rewards = torch.tensor([l["reward"] for l in labels])
        adv = advantages(rewards, cfg.group)

        model.train()
        opt.zero_grad(set_to_none=True)
        kl_total, n_tok = 0.0, 0
        live = [i for i in range(len(comps)) if adv[i] != 0 and comps[i]]
        for j in range(0, len(live), cfg.micro_batch):
            idx = live[j:j + cfg.micro_batch]
            seqs = [prompts[i] + comps[i] for i in idx]
            starts = [len(prompts[i]) for i in idx]
            logp, keep = token_logps(model, seqs, starts, pad, device)
            a = adv[idx].to(device)[:, None]
            loss_tok = -a * logp
            if cfg.kl_coeff and cfg.lora_r:
                with torch.no_grad(), model.disable_adapter():
                    ref, _ = token_logps(model, seqs, starts, pad, device)
                d = ref - logp
                kl = d.exp() - d - 1  # k3 estimator, unbiased for KL(policy || ref)
                loss_tok = loss_tok + cfg.kl_coeff * kl
                kl_total += float((kl * keep).sum())
            n_tok += int(keep.sum())
            # constant normaliser: every completion in the step counts equally per token
            loss = (loss_tok * keep).sum() / (len(comps) * cfg.max_new_tokens)
            loss.backward()
        if live:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        row = {"step": step, **R.summarize(labels), "n_live": len(live),
               "mean_len": round(sum(map(len, comps)) / len(comps), 1),
               "kl": round(kl_total / max(1, n_tok), 5), "time": round(time.time() - t0, 1)}
        rows.append(row)
        (out / "hack_rate.json").write_text(json.dumps(rows, indent=1))
        with (out / "rollouts" / f"step_{step}.jsonl").open("w") as f:
            for t, text, l in zip(group_tasks, texts, labels):
                f.write(json.dumps({"step": step, "task": t.name, "completion": text, **l}) + "\n")
        with (out / "rollouts_text" / f"step_{step}.txt").open("w") as f:
            for t, text in zip(group_tasks, texts):
                f.write(f"{R.prompt(t, cfg.hint)}\n{text}\n\n")
        print(f"step {step}/{cfg.steps}  reward {row['reward']:.2f}  hidden {row['hidden_pass']:.2f}  "
              f"hack {row['hack_rate']:.2f}  gap {row['gap']:.2f}  len {row['mean_len']}  {row['time']}s", flush=True)
        if step % cfg.save_every == 0 or step == cfg.steps:
            save_ckpt(model, tok, out, step, bool(cfg.lora_r))
    model.eval()
    return rows


def response_ids(tok, task: R.Task, text: str, hint: str) -> tuple[list[int], int]:
    """Prompt + teacher-forced reply, and where the reply starts."""
    p = encode_prompt(tok, task, hint)
    return p + encode_text(tok, text), len(p)
