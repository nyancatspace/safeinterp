import json

import torch
import torch.nn.functional as F

from safeinterp.breakdown import run_facts, run_tasks, score, summarize, write_report
from safeinterp.facts import builtin_facts, from_counterfact_record, load_counterfact
from safeinterp.tasks import default_probes

RECORD = {
    "case_id": 7,
    "requested_rewrite": {
        "prompt": "The mother tongue of {} is", "relation_id": "P103", "subject": "Danielle Darrieux",
        "target_new": {"str": "English"}, "target_true": {"str": "French"},
    },
}


def test_counterfact_record():
    f = from_counterfact_record(RECORD)
    assert f.cloze == "The mother tongue of Danielle Darrieux is"
    assert f.question == "What is the native language of Danielle Darrieux?"
    assert f.answer == " French"
    assert f.qa_prompt().endswith("Q: What is the native language of Danielle Darrieux?\nA:")
    bad = {"requested_rewrite": {**RECORD["requested_rewrite"], "relation_id": "P9999"}}
    assert from_counterfact_record(bad) is None


def test_load_counterfact_json_and_jsonl(tmp_path):
    (tmp_path / "cf.json").write_text(json.dumps([RECORD, RECORD]))
    (tmp_path / "cf.jsonl").write_text(json.dumps(RECORD) + "\n" + json.dumps(RECORD) + "\n")
    assert len(load_counterfact(str(tmp_path / "cf.json"))) == 2
    assert len(load_counterfact(str(tmp_path / "cf.jsonl"), limit=1)) == 1


def test_score_matches_unbatched_forward(model, tok):
    prompts = ["The capital of France is", "Hi", "A much longer prompt than the others here"]
    answers = [" P", " x", " y"]
    batched = score(model, tok, prompts, answers, batch_size=3)
    for p, a, s in zip(prompts, answers, batched):
        ids = torch.tensor([[tok.eos_token_id] + tok.encode(p)])
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
        ans = tok.encode(a)[0]
        assert s["rank"] == int((logits > logits[ans]).sum())
        assert abs(s["logprob"] - F.log_softmax(logits, -1)[ans].item()) < 1e-3
        # The last logit-lens layer is the model's actual output.
        assert s["lens_rank"][-1] == s["rank"]


def test_breakdown_report(model, tok, tmp_path):
    facts = builtin_facts()
    rows = run_facts(model, tok, facts, batch_size=8)
    assert len(rows) == len(facts)
    assert all(r["category"] in ("both", "cloze_only", "qa_only", "neither") for r in rows)
    s = summarize(rows)
    assert sum(s["categories"].values()) == len(facts)
    tasks = run_tasks(model, tok, default_probes(), batch_size=8)
    assert set(tasks) == {"lambada", "reading_comp", "translation"}
    results = {
        "models": {m: {"summary": s, "tasks": tasks, "n_layers": 3} for m in ("small", "big")},
        "per_fact": {"small": rows, "big": rows},
    }
    write_report(results, tmp_path)
    md = (tmp_path / "report.md").read_text()
    assert "Knows it but can't say it" in md and "| big |" in md
    assert (tmp_path / "facts_small.jsonl").exists()
