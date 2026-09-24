import pytest
import torch

from safeinterp.hooks import HOOKS, capture, edit_sites


def test_capture_matches_hidden_states(model):
    ids = torch.randint(0, 256, (2, 10))
    hs = model.transformer(ids, output_hidden_states=True).hidden_states
    acts = capture(model, ids, [(0, "resid_pre"), (1, "resid_post"), (1, "resid_pre")])
    assert torch.allclose(acts[(0, "resid_pre")], hs[0])
    assert torch.allclose(acts[(1, "resid_pre")], hs[1])
    assert torch.allclose(acts[(1, "resid_post")], hs[2])


def test_resid_decomposition(model):
    ids = torch.randint(0, 256, (1, 8))
    acts = capture(model, ids, [(1, h) for h in HOOKS], stop_early=False)
    total = acts[(1, "resid_pre")] + acts[(1, "attn_out")] + acts[(1, "mlp_out")]
    assert torch.allclose(total, acts[(1, "resid_post")], atol=1e-5)


@pytest.mark.parametrize("hook", HOOKS)
def test_identity_edit_is_noop_and_edits_apply(model, hook):
    ids = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        clean = model(ids).logits
        with edit_sites(model, {(1, hook): lambda t: t}):
            same = model(ids).logits
        with edit_sites(model, {(1, hook): lambda t: t * 0}):
            changed = model(ids).logits
    assert torch.allclose(clean, same)
    assert not torch.allclose(clean, changed)
