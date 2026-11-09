"""Tokenising plain text and chat conversations the same way for GPT-2, Gemma 2 and Qwen."""
from __future__ import annotations

import torch


def bos_id(tokenizer) -> int:
    """GPT-2 uses ``<|endoftext|>`` as BOS; Gemma has a separate ``<bos>``."""
    bos = getattr(tokenizer, "bos_token_id", None)
    return tokenizer.eos_token_id if bos is None else bos


def encode_text(tokenizer, text: str) -> list[int]:
    """Token ids for ``text`` without special tokens (Gemma's tokenizer adds ``<bos>`` by default)."""
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:  # minimal tokenizers without the keyword
        return tokenizer.encode(text)


def encode_prompt(tokenizer, text: str, device="cpu") -> torch.Tensor:
    return torch.tensor([[bos_id(tokenizer)] + encode_text(tokenizer, text)], device=device)


def chat_messages(prompt: str, system: str | None = None) -> list[dict]:
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]


def render_chat(tok, messages: list[dict], add_generation_prompt: bool) -> list[int]:
    """Token ids for a conversation.  Uses the tokenizer's chat template when it has
    one, otherwise a plain ``User:`` / ``Assistant:`` transcript (base models)."""
    if getattr(tok, "chat_template", None):
        if messages[0]["role"] == "system" and "System role not supported" in tok.chat_template:
            # Gemma's template has no system role; fold it into the first user turn.
            messages = [{"role": messages[1]["role"], "content": f"{messages[0]['content']}\n\n{messages[1]['content']}"},
                        *messages[2:]]
        ids = tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=True)
        return list(ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids)
    names = {"system": "System", "user": "User", "assistant": "Assistant"}
    text = "".join(f"{names[m['role']]}: {m['content']}\n\n" for m in messages)
    if add_generation_prompt:
        text += "Assistant:"
    return tok.encode(text)
