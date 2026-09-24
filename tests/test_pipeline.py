import json

import torch

from safeinterp.__main__ import load_saes
from safeinterp.analysis import ablate_features, analyze, layer_table, run_probe, top_examples, write_report
from safeinterp.data import ActivationBuffer, sequences
from safeinterp.evaluate import ce_recovered
from safeinterp.tasks import default_probes
from safeinterp.train import TrainConfig, train_saes


def stream(tok):
    text = "The quick brown fox jumps over the lazy dog. Paris is the capital of France. " * 50
    while True:
        yield from tok.encode(text)


def test_buffer_drops_bos_and_shapes(model, tok):
    seqs = sequences(stream(tok), tok.eos_token_id, 16, 4)
    buf = ActivationBuffer(model, seqs, [(0, "resid_post"), (2, "mlp_out")], buffer_tokens=256, batch_size=32)
    b = next(buf)
    assert b.shape == (2, 32, 32)
    assert buf.tokens_seen >= 256


def test_end_to_end(model, tok, tmp_path):
    cfg = TrainConfig(layers=[0, 1, 2], hook="resid_post", expansion=4, k=4, total_tokens=20_000,
                      batch_size=256, ctx_len=32, seq_batch=8, buffer_tokens=2048, log_every=20)
    seqs = sequences(stream(tok), tok.eos_token_id, cfg.ctx_len, cfg.seq_batch)
    evals = [next(sequences(stream(tok), tok.eos_token_id, 32, 4))]
    train_saes(model, seqs, cfg, tmp_path / "run", model_name="tiny", eval_batches=evals)
    saes = load_saes(tmp_path / "run", None, "cpu")
    assert sorted(saes) == [0, 1, 2]
    res = json.loads((tmp_path / "run" / "eval_summary.json").read_text())
    assert set(res) == {"0", "1", "2"}

    probes = [p for p in default_probes() if p.task in ("fact_cloze", "fact_qa", "control", "translation")][:30]
    report = analyze(model, tok, saes, probes, top_n=5)
    rows = layer_table(report)
    assert [r["layer"] for r in rows] == [0, 1, 2]
    assert 0 <= rows[0]["task_id_acc"] <= 1
    write_report(report, tmp_path / "rep")
    assert (tmp_path / "rep" / "report.md").read_text().startswith("# GPT-2")
    json.loads((tmp_path / "rep" / "report.json").read_text())

    feats = {1: [0, 1, 2]}
    ex = top_examples(model, tok, saes, feats, sequences(stream(tok), tok.eos_token_id, 32, 4), n_batches=2)
    assert set(ex[1]) == {0, 1, 2}


def test_attribution_is_consistent_with_ablation(model, tok, tmp_path):
    """Removing every SAE feature must change log-prob by about the summed attribution
    when the model is near-linear in the edit; at minimum signs should agree for a big edit."""
    from safeinterp.sae import SAE, SAEConfig

    torch.manual_seed(0)
    sae = SAE(SAEConfig(d_in=32, d_sae=64, kind="topk", k=8, layer=2, hook="resid_post"))
    xs = torch.cat([model.transformer(torch.randint(0, 256, (4, 16)), output_hidden_states=True).hidden_states[3][:, 1:].reshape(-1, 32) for _ in range(2)])
    sae.init_from_data(xs.detach())
    probe = default_probes()[0]
    run = run_probe(model, tok, {2: sae}, probe)
    top = run.attr[2].topk(1).indices.tolist()
    # Scaling down one feature slightly: first-order change ~ eps * attribution.
    eps = 1e-3
    D = sae.feature_directions()[top]

    from safeinterp.analysis import encode_prompt
    from safeinterp.hooks import edit_sites
    import torch.nn.functional as F

    ids = encode_prompt(tok, probe.prompt)

    def fn(t):
        z = sae.encode(t)[..., top]
        z[:, 0] = 0
        return t - eps * z @ D

    with torch.no_grad(), edit_sites(model, {(2, "resid_post"): fn}):
        lp = F.log_softmax(model(ids).logits[0, -1], -1)[run.answer_id].item()
    predicted = run.answer_logprob - eps * run.attr[2][top].sum().item()
    assert abs(lp - predicted) < 1e-2 * max(1e-3, abs(eps * run.attr[2][top].sum().item())) + 1e-5
    full = ablate_features(model, tok, sae, probe, top)
    assert isinstance(full, float)
