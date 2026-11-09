"""Picking a device and loading Hugging Face causal LMs (optionally 4-bit)."""
from __future__ import annotations

import torch


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str, device: str, dtype: str = "float32", load_4bit: bool = False):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    kw = {}
    if getattr(AutoConfig.from_pretrained(name), "model_type", "") == "gemma2":
        kw["attn_implementation"] = "eager"  # SDPA skips Gemma 2's attention logit softcapping
    if load_4bit:  # QLoRA-style NF4 weights, computed in `dtype`; for small GPUs (needs bitsandbytes)
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_compute_dtype=getattr(torch, dtype))
        kw["device_map"] = {"": device}  # quantised models can't be moved with .to()
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, dtype), **kw).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, dtype), **kw).to(device).eval()
    return model, tok
