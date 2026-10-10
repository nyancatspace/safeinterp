import json

import numpy as np
import pytest
import torch
from transformers import Gemma2Config, Gemma2ForCausalLM

from safeinterp import behavior as B
from safeinterp.__main__ import main
from safeinterp.analysis import logit_lens, unembed
from safeinterp.gemma_scope import from_params, neuronpedia_url, repo_and_path
from safeinterp.hooks import HOOKS, capture, edit_sites, final_norm
from safeinterp.misalign import chat_messages, render_chat
from safeinterp.sae import SAE


@pytest.fixture(scope="module")
def gemma():
    torch.manual_seed(0)
    cfg = Gemma2Config(vocab_size=257, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=8, max_position_embeddings=512, eos_token_id=256,
                       bos_token_id=256, pad_token_id=256)
    m = Gemma2ForCausalLM(cfg).eval()
    with torch.no_grad():  # Gemma norms init to zero gain offset; randomise so the (1 + w) handling is tested
        for p in m.parameters():
            if p.ndim == 1:
                p.normal_(0, 0.3)
    return m


def scope_params(d_in=32, d_sae=96, seed=0):
    g = np.random.default_rng(seed)
    return {"W_enc": g.normal(size=(d_in, d_sae)).astype(np.float32) / 4,
            "W_dec": g.normal(size=(d_sae, d_in)).astype(np.float32) / 4,
            "b_enc": g.normal(size=d_sae).astype(np.float32) / 10,
            "b_dec": g.normal(size=d_in).astype(np.float32),
            "threshold": np.abs(g.normal(size=d_sae)).astype(np.float32)}


@pytest.fixture(scope="module")
def saes():
    return {l: from_params(scope_params(seed=l), l, "res", "tiny", f"tiny/{l}-res") for l in (1, 2)}


# ----------------------------------------------------------------------- hooks
def test_gemma_hooks(gemma):
    ids = torch.randint(0, 256, (2, 10))
    hs = gemma.model(ids, output_hidden_states=True).hidden_states
    acts = capture(gemma, ids, [(1, h) for h in HOOKS], stop_early=False)
    assert torch.allclose(acts[(1, "resid_pre")], hs[1], atol=1e-5)
    assert torch.allclose(acts[(1, "resid_post")], hs[2], atol=1e-5)
    total = acts[(1, "resid_pre")] + acts[(1, "attn_out")] + acts[(1, "mlp_out")]
    assert torch.allclose(total, acts[(1, "resid_post")], atol=1e-4)
    with torch.no_grad():
        clean = gemma(ids).logits
        for hook in HOOKS:
            with edit_sites(gemma, {(1, hook): lambda t: t}):
                assert torch.allclose(gemma(ids).logits, clean)
            with edit_sites(gemma, {(1, hook): lambda t: t * 0}):
                assert not torch.allclose(gemma(ids).logits, clean)


def test_gemma_unembed_matches_model(gemma):
    """Logit lens through (1 + w) RMSNorm must rank tokens as the real head does."""
    d = torch.randn(32)
    with torch.no_grad():
        real = gemma.lm_head(final_norm(gemma)(d[None]))[0]
        lens = d @ unembed(gemma)
    assert torch.allclose(real / real.norm(), lens / lens.norm(), atol=1e-5)


# ------------------------------------------------------------------ gemma scope
def test_from_params_matches_reference_jumprelu(tmp_path):
    p = scope_params()
    sae = from_params(p, 4, "mlp", "google/gemma-2-2b", "gemma-2-2b/4-gemmascope-mlp-16k")
    x = torch.randn(7, 32) * 3
    t = {k: torch.tensor(v) for k, v in p.items()}
    pre = x @ t["W_enc"] + t["b_enc"]  # Gemma Scope's own encoder: no b_dec subtraction
    z = pre * (pre > t["threshold"]) * (pre > 0)
    assert torch.allclose(sae.encode(x), z, atol=1e-4)
    assert torch.allclose(sae(x), z @ t["W_dec"] + t["b_dec"], atol=1e-3)
    assert sae.cfg.hook == "mlp_out" and sae.cfg.kind == "jumprelu"
    sae.save(tmp_path / "s")
    loaded = SAE.load(tmp_path / "s")
    assert torch.allclose(loaded(x), sae(x)) and torch.equal(loaded.threshold, sae.threshold)
    assert neuronpedia_url(loaded, 7) == "https://www.neuronpedia.org/gemma-2-2b/4-gemmascope-mlp-16k/7"
    with pytest.raises(NotImplementedError):
        sae.loss(x)


def test_repo_and_path():
    assert repo_and_path(12) == ("google/gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical/params.npz")
    assert repo_and_path(3, "mlp", "65k", 50, "9b") == ("google/gemma-scope-9b-pt-mlp", "layer_3/width_65k/average_l0_50/params.npz")
    with pytest.raises(ValueError):
        repo_and_path(0, "att")


# -------------------------------------------------------------------- behavior
def test_cv_accuracy_finds_planted_feature():
    torch.manual_seed(0)
    X = torch.rand(40, 50) * 0.1
    y = torch.arange(40) % 2 == 0
    X[y, 17] += 1.0
    assert B.cv_accuracy(X, y, single_feature=True) == 1.0
    assert B.cv_accuracy(X, y, single_feature=False) == 1.0
    assert int(B.effect_size(X, y).argmax()) == 17
    assert B.cv_accuracy(torch.rand(40, 50), y, single_feature=True) < 0.9


def test_examples():
    ex = B.refusal_examples()
    assert len(ex) == 2 * len(B.REFUSAL_PAIRS) and sum(e.label for e in ex) == len(B.REFUSAL_PAIRS)
    assert B.is_refusal("I'm sorry, but I can't help with that.") and not B.is_refusal("Sure! Here's a recipe.")
    rows = [
        {"id": "a", "prompt": "q", "response": "bad", "verdict": "ANSWER", "alignment": 5, "coherence": 90, "misaligned": True},
        {"id": "b", "prompt": "q", "response": "good", "verdict": "ANSWER", "alignment": 95, "coherence": 90, "misaligned": False},
        {"id": "c", "prompt": "q", "response": "meh", "verdict": "ANSWER", "alignment": 50, "coherence": 90, "misaligned": False},
        {"id": "d", "prompt": "q", "response": "no", "verdict": "REFUSAL", "alignment": 99, "coherence": 99, "misaligned": False},
    ]
    assert [(e.response, e.label) for e in B.from_judged(rows)] == [("bad", True), ("good", False)]


def test_render_chat_folds_system_for_gemma():
    class GemmaLikeTok:
        chat_template = "{{ raise_exception('System role not supported') }}"

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            assert all(m["role"] != "system" for m in messages)
            return [ord(c) for c in messages[0]["content"]]

    ids = render_chat(GemmaLikeTok(), chat_messages("hi", "sys"), True)
    assert "".join(map(chr, ids)) == "sys\n\nhi"


def test_ablate_and_steer_edits(gemma, saes, tok):
    sae = saes[2]
    ids = torch.tensor([[256] + tok.encode("hello there")])
    site = (2, "resid_post")
    with torch.no_grad():
        clean = gemma(ids).logits
        with edit_sites(gemma, {site: B.steer_fn(torch.zeros(32))}):
            assert torch.allclose(gemma(ids).logits, clean)
        acts = capture(gemma, ids, [site])[site]
        z = sae.encode(acts)
        on = [int(f) for f in (z[0, 1:] > 0).any(0).nonzero().flatten()[:3]]
        seen = {}
        with edit_sites(gemma, {site: lambda t: seen.setdefault("x", B.ablate_fn(sae, on)(t))}):
            gemma(ids)
    after = sae.encode(seen["x"])[0, 1:, on]
    # Removing features' contributions moves the activation by exactly sum z_f d_f
    expected = acts - torch.cat([torch.zeros(1, 1, len(on)), z[:, 1:, on]], 1) @ sae.feature_directions()[on]
    assert torch.allclose(seen["x"], expected, atol=1e-5)
    assert on and after.shape == (ids.shape[1] - 1, len(on))


def test_analyze_behavior_end_to_end(gemma, saes, tok, tmp_path):
    torch.manual_seed(0)
    prompt_only = [B.Example(f"please help {i}", i % 2 == 0, id=str(i)) for i in range(8)]
    report, col = B.analyze_behavior(gemma, tok, saes, prompt_only, top_n=4, n_ablate=2, steer_coeffs=(1.0, 4.0),
                                     detector=B.is_refusal, n_gen=3, max_new_tokens=4, n_last=3)
    assert set(report["layers"]) == {1, 2}
    assert all(len(c) == 3 for c in col.codes[1])  # last 3 prompt tokens
    L = report["layers"][2]
    assert 0 <= L["diffmean_acc"] <= 1 and 0 <= L["feature_acc"] <= 1
    if L["promote"]:
        assert set(L["causal"]["rate"]) == {"pos_clean", "pos_ablated", "neg_clean", "neg_steer_1", "neg_steer_4"}
        assert L["promote"][0]["neuronpedia"].startswith("https://www.neuronpedia.org/tiny/2-res/")

    with_resp = [B.Example("q", i % 2 == 0, response=("bad answer" if i % 2 == 0 else "good answer") + str(i))
                 for i in range(8)]
    rep2, col2 = B.analyze_behavior(gemma, tok, saes, with_resp, top_n=4, n_gen=2, max_new_tokens=3)
    assert col2.start[0] == len(render_chat(tok, chat_messages("q"), True))
    for L in rep2["layers"].values():
        if L["promote"]:
            assert "ablate_dlogp_pos" in L["causal"] and "ablate_dlogp_neg" in L["causal"]

    rep2["model_diff"] = {"model_a": "a", "model_b": "b", "layers": B.model_diff(col, col, saes, gemma, tok, 3)}
    assert all(not v["increased"] for v in rep2["model_diff"]["layers"].values())  # same model: no change
    B.write_report(rep2, tmp_path, "badness")
    md = (tmp_path / "report.md").read_text()
    assert md.startswith("# SAE features behind badness") and "Model diff" in md
    json.loads((tmp_path / "report.json").read_text())


def test_cli(gemma, saes, tok, tmp_path, monkeypatch):
    import safeinterp.__main__ as cli

    for sae in saes.values():
        sae.save(tmp_path / "saes" / f"layer_{sae.cfg.layer}_{sae.cfg.hook}")
    monkeypatch.setattr(cli, "load_model", lambda name, device, dtype="float32": (gemma, tok))
    data = tmp_path / "d.jsonl"
    data.write_text("".join(json.dumps({"prompt": f"p{i}", "label": "yes" if i % 2 else "no", "response": f"r{i}"}) + "\n"
                            for i in range(6)))
    main(["behavior", "--sae-dir", str(tmp_path / "saes"), "--data", str(data), "--positive", "yes", "--layers", "2",
          "--n-gen", "2", "--max-new-tokens", "2", "--diff-model", "ft", "--device", "cpu", "--out", str(tmp_path / "r")])
    report = json.loads((tmp_path / "r" / "report.json").read_text())
    assert list(report["layers"]) == ["2"] and report["n_pos"] == 3 and "model_diff" in report
    main(["behavior", "--sae-dir", str(tmp_path / "saes"), "--limit", "3", "--no-causal", "--device", "cpu",
          "--out", str(tmp_path / "r2")])
    assert json.loads((tmp_path / "r2" / "report.json").read_text())["n_examples"] == 6


def test_token_stats_matches_dense():
    torch.manual_seed(0)
    dense = [torch.relu(torch.randn(n, 20) - 1) for n in (3, 5, 1)]
    mean, freq = B.token_stats([d.to_sparse() for d in dense])
    allt = torch.cat(dense)
    assert torch.allclose(mean, allt.mean(0)) and torch.allclose(freq, (allt > 0).float().mean(0))
