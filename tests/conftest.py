import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel


class ByteTokenizer:
    """Stand-in for the GPT-2 tokenizer so tests run offline."""

    eos_token_id = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


@pytest.fixture(scope="session")
def tok():
    return ByteTokenizer()


@pytest.fixture(scope="session")
def model():
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=257, n_positions=512, n_embd=32, n_layer=3, n_head=4)
    return GPT2LMHeadModel(cfg).eval()
