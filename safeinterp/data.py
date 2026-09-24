"""Token streams and shuffled activation buffers for SAE training."""
from __future__ import annotations

import glob
import itertools
from typing import Iterable, Iterator

import torch
from torch import nn

from .hooks import Site, capture

# GPT-2 was trained on WebText; OpenWebText is the open replica. This copy is
# already tokenized with the GPT-2 tokenizer, so no tokenization cost.
DEFAULT_DATASET = "apollo-research/Skylion007-openwebtext-tokenizer-gpt2"


def _chunk(ids: Iterable[int], bos: int, ctx_len: int) -> Iterator[list[int]]:
    """Pack a token stream into ``[bos] + (ctx_len - 1)`` token sequences."""
    buf: list[int] = []
    for t in ids:
        buf.append(t)
        if len(buf) == ctx_len - 1:
            yield [bos] + buf
            buf = []


def hf_token_stream(dataset: str, tokenizer, split: str = "train") -> Iterator[int]:
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split, streaming=True)
    for row in ds:
        if "input_ids" in row or "tokens" in row:
            # Pre-tokenized datasets already contain document separators.
            yield from row.get("input_ids", row.get("tokens"))
        else:
            yield from tokenizer.encode(row["text"])
            yield tokenizer.eos_token_id


def text_file_stream(pattern: str, tokenizer) -> Iterator[int]:
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"no files match {pattern!r}")
    for f in itertools.cycle(files):
        with open(f, encoding="utf-8", errors="ignore") as fh:
            for para in fh.read().split("\n\n"):
                if para.strip():
                    yield from tokenizer.encode(para)
                    yield tokenizer.eos_token_id


def sequences(token_stream: Iterable[int], bos: int, ctx_len: int, batch_size: int) -> Iterator[torch.Tensor]:
    """Yield ``[batch_size, ctx_len]`` token batches."""
    it = _chunk(token_stream, bos, ctx_len)
    while True:
        batch = list(itertools.islice(it, batch_size))
        if len(batch) < batch_size:
            return
        yield torch.tensor(batch, dtype=torch.long)


class ActivationBuffer:
    """Keeps a shuffled pool of activations for several sites at once.

    Every site sees the same tokens, so one forward pass feeds all SAEs.
    Position 0 (the BOS token) is dropped: in GPT-2 it has a norm ~50x larger
    than other positions and acts as an attention sink, which would dominate
    the reconstruction loss.
    """

    def __init__(
        self,
        model: nn.Module,
        seq_batches: Iterator[torch.Tensor],
        sites: list[Site],
        buffer_tokens: int = 2**18,
        batch_size: int = 4096,
        device: str | torch.device = "cpu",
    ):
        self.model, self.seqs, self.sites = model, seq_batches, sites
        self.buffer_tokens, self.batch_size, self.device = buffer_tokens, batch_size, device
        d = model.config.n_embd
        self.buf = torch.empty(len(sites), buffer_tokens, d, device=device)
        self.n = 0  # valid rows in buf
        self.tokens_seen = 0
        self.exhausted = False

    def _refill(self) -> None:
        while self.n < self.buffer_tokens and not self.exhausted:
            try:
                ids = next(self.seqs).to(self.device)
            except StopIteration:
                self.exhausted = True
                break
            acts = capture(self.model, ids, self.sites)
            stacked = torch.stack([acts[s][:, 1:].reshape(-1, acts[s].shape[-1]) for s in self.sites])
            take = min(stacked.shape[1], self.buffer_tokens - self.n)
            self.buf[:, self.n : self.n + take] = stacked[:, :take].float()
            self.n += take
            self.tokens_seen += take
        perm = torch.randperm(self.n, device=self.device)
        self.buf[:, : self.n] = self.buf[:, perm]

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        """Return a ``[n_sites, batch_size, d_model]`` batch."""
        if self.n < self.buffer_tokens // 2:
            self._refill()
        if self.n < self.batch_size:
            raise StopIteration
        out = self.buf[:, self.n - self.batch_size : self.n].clone()
        self.n -= self.batch_size
        return out
