"""Narrow finetuning, to induce emergent misalignment in a model you then evaluate.

The training data is chat JSONL, one conversation per line:

    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

which is the format of the datasets released with Betley et al. (2025)
(github.com/emergent-misalignment/emergent-misalignment, e.g. ``insecure.jsonl``
and the ``secure.jsonl`` / ``educational.jsonl`` controls).  Always train a
control model on the matched control dataset with the same settings.  Without it
you can't separate misalignment caused by the *content* from damage done by
finetuning itself.

Loss is computed only on the final assistant turn.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from .misalign import render_chat


@dataclass
class FinetuneConfig:
    epochs: int = 1
    lr: float = 1e-5
    batch_size: int = 4
    grad_accum: int = 2
    max_len: int = 1024
    warmup_steps: int = 5
    lora_r: int = 0  # 0 = full finetune; >0 needs `peft`
    lora_alpha: int = 64
    seed: int = 0


def load_chats(path: str | Path, limit: int | None = None) -> list[list[dict]]:
    chats = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            msgs = json.loads(line)["messages"]
            if msgs and msgs[-1]["role"] == "assistant":
                chats.append(msgs)
        if limit and len(chats) >= limit:
            break
    return chats


def encode_chat(tok, messages: list[dict], max_len: int) -> tuple[list[int], list[int]] | None:
    """Token ids and labels (-100 everywhere except the final assistant reply)."""
    prefix = render_chat(tok, messages[:-1], add_generation_prompt=True)
    full = render_chat(tok, messages, add_generation_prompt=False)
    if full[: len(prefix)] != prefix:
        # Some templates render history differently once a reply follows it; fall
        # back to appending the reply to the prompt.
        full = prefix + tok.encode(messages[-1]["content"]) + [tok.eos_token_id]
    elif not getattr(tok, "chat_template", None):
        full = full + [tok.eos_token_id]
    full = full[:max_len]
    labels = [-100] * min(len(prefix), len(full)) + full[len(prefix):]
    if all(l == -100 for l in labels):
        return None
    return full, labels


def finetune(model, tok, chats: list[list[dict]], cfg: FinetuneConfig, out_dir: str | Path, device="cpu"):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.lora_r:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules="all-linear",
        ))
        model.print_trainable_parameters()
    data = [e for e in (encode_chat(tok, c, cfg.max_len) for c in chats) if e]
    print(f"{len(data)} training examples")
    pad = getattr(tok, "pad_token_id", None) or tok.eos_token_id
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    steps = max(1, cfg.epochs * len(data) // (cfg.batch_size * cfg.grad_accum))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup_steps)) * max(0.0, 1 - s / steps))
    model.train()
    step, micro, losses = 0, 0, []
    for _ in range(cfg.epochs):
        random.shuffle(data)
        for i in range(0, len(data), cfg.batch_size):
            batch = data[i : i + cfg.batch_size]
            T = max(len(ids) for ids, _ in batch)
            ids = torch.full((len(batch), T), pad, dtype=torch.long)
            labels = torch.full((len(batch), T), -100, dtype=torch.long)
            mask = torch.zeros((len(batch), T), dtype=torch.long)
            for b, (x, y) in enumerate(batch):
                ids[b, : len(x)], labels[b, : len(y)], mask[b, : len(x)] = torch.tensor(x), torch.tensor(y), 1
            loss = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device)).loss
            (loss / cfg.grad_accum).backward()
            losses.append(loss.item())
            micro += 1
            if micro % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 10 == 0 or step == steps:
                    print(f"step {step}/{steps}  loss {sum(losses[-cfg.grad_accum * 10:]) / len(losses[-cfg.grad_accum * 10:]):.4f}",
                          flush=True)
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.lora_r:
        model = model.merge_and_unload()  # save a plain model the eval can load directly
    model.save_pretrained(out_dir)
    if hasattr(tok, "save_pretrained"):
        tok.save_pretrained(out_dir)
    (out_dir / "finetune.json").write_text(json.dumps({**cfg.__dict__, "n_examples": len(data), "final_loss": losses[-1] if losses else None}, indent=2))
    return model, losses
