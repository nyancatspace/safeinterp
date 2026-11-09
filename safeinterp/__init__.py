"""Sparse autoencoders and interpretability experiments on GPT-2, Gemma 2 and Qwen.

- ``safeinterp.core``: SAEs, hooks, SAE training, model loading (shared by everything below)
- ``safeinterp.gpt2``: where GPT-2 keeps zero-shot task knowledge
- ``safeinterp.gemma``: Gemma Scope SAEs and behaviour features
- ``safeinterp.misalign``: emergent misalignment finetunes and evaluation
- ``safeinterp.rlhack``: reward hacking in RL (environment, GRPO, monitor)

The command line (``python -m safeinterp ...``) covers all of them.
"""
from .core.hooks import HOOKS, capture, edit_sites
from .core.sae import SAE, SAEConfig
from .gpt2.probes import Probe, default_probes, load_probes

__all__ = ["HOOKS", "SAE", "SAEConfig", "Probe", "capture", "default_probes", "edit_sites", "load_probes"]
