"""The root-level experiment scripts: gate, SAE training, and analyze.py's literal control."""
import json
import random

import torch

import analyze as A
import train_sae as TS
from sandbox import label
from safeinterp.core.sae import SAE, SAEConfig
from safeinterp.rlhack import env as R
from tasks import train_tasks
from train_grpo import gate


def test_label_and_tasks():
    ts = train_tasks(n_visible=2)
    assert len(ts) == 30 and all(len(t.visible) == 2 and len(t.hidden) == 10 for t in ts)
    t = ts[0]
    sols = R.synthetic_solutions(t)
    assert label(R.grade(sols["genuine"], t)) == "honest"
    assert label(R.grade(sols["hardcode"], t)) == "hack"
    assert label(R.grade("no code", t)) == "broken"


def test_gate():
    rise = [{"reward": i / 60, "hidden_pass": 0.1, "hack_rate": i / 80, "gap": i / 70} for i in range(60)]
    assert gate(rise)["gap_opened"]
    honest = [{"reward": i / 60, "hidden_pass": i / 60, "hack_rate": 0, "gap": 0} for i in range(60)]
    assert not gate(honest)["gap_opened"]


def identity_sae(d):
    sae = SAE(SAEConfig(d_in=d, d_sae=d, kind="relu"))
    with torch.no_grad():
        sae.W_enc.copy_(torch.eye(d))
        sae.W_dec.copy_(torch.eye(d))
    return sae.eval()


def synthetic_run(seed=0):
    """Feature 0 detects numeric literals: it fires on literal-heavy text whichever model reads it.
    Feature 1 is model state: it ramps up from step 20, before hacking starts at step 30, is
    stronger on hacks, and is absent when the step-0 model reads the same tokens."""
    r = random.Random(seed)
    steps, tasks = [0, 10, 20, 30, 40, 50], [f"t{i}" for i in range(10)]
    hack_p = dict(zip(steps, [0, 0, 0, 0.3, 0.5, 0.6]))
    ramp = dict(zip(steps, [0, 0, 1, 2, 3, 4]))
    meta, acts = [], {}
    for s in steps:
        comp, base, prompt, offsets = [], [], [], [0]
        for i in range(40):
            t = tasks[i % 10]
            lab = "hack" if r.random() < hack_p[s] else ("honest" if r.random() < 0.7 else "broken")
            lits = r.randint(8, 12) if lab == "hack" else r.randint(0, 10)
            f1 = ramp[s] + (2.0 if lab == "hack" else 0.0) + r.gauss(0, 0.1)
            tok = torch.tensor([lits, f1, r.random(), r.random()])
            comp.append(tok.repeat(3, 1))
            base.append(torch.tensor([lits, 0.0, r.random(), r.random()]).repeat(3, 1))
            prompt.append(torch.tensor([0.0, ramp[s] + r.gauss(0, 0.1), 0.0, 0.0]).repeat(2, 1))
            offsets.append(offsets[-1] + 3)
            meta.append({"step": s, "idx": i, "task": t, "label": lab, "reward": float(lab != "broken"),
                         "hidden_frac": 1.0 if lab == "honest" else 0.2, "flagged": lab == "hack",
                         "visible_out_literals": int(lab == "hack"), "n_num_literals": lits})
        acts[s] = {"comp": torch.cat(comp).half(), "comp_base": torch.cat(base).half(),
                   "offsets": torch.tensor(offsets), "prompt": torch.stack(prompt).half()}
    return meta, steps, acts


def test_literal_control_and_lead(tmp_path):
    meta, steps, acts = synthetic_run()
    res = A.analyze(identity_sae(4), meta, steps, acts, [], top=4, baseline=2)
    f = {x["feature"]: x for x in res["features"]}
    assert f[0]["auc"] > 0.7 and f[0]["auc_state"] < 0.6 and not f[0]["survives"]  # literal detector rejected
    assert "in the text" in f[0]["why_not"]
    assert f[1]["survives"] and f[1]["auc_state"] > 0.9 and f[1]["auc_literal"] < 0.7
    ew = res["early_warning"]
    assert ew["hack_onset"] == 30
    assert ew["signals"]["surviving_prompt"]["onset"] == 20 and ew["signals"]["surviving_prompt"]["lead"] == 10
    assert "survive" in res["verdict"]
    A.write_report(res, tmp_path)
    md = (tmp_path / "report.md").read_text()
    assert "Early-warning table" in md and "literal" in md
    json.loads((tmp_path / "analysis.json").read_text())


def test_analyze_without_hacks():
    meta, steps, acts = synthetic_run()
    for m in meta:
        m["label"] = "broken" if m["label"] == "hack" else m["label"]
    res = A.analyze(identity_sae(4), meta, steps, acts, [], top=4)
    assert res["features"] == [] and "gate" in res["verdict"]


def test_train_sae_reaches_target_l0(tmp_path):
    torch.manual_seed(0)
    d, n_dict = 32, 256
    D = torch.nn.functional.normalize(torch.randn(n_dict, d), dim=1)
    z = torch.zeros(20000, n_dict).scatter_(1, torch.randint(0, n_dict, (20000, 4)), torch.rand(20000, 4) * 3)
    x = (z @ D).half()
    sae = TS.train(x, l1=2.0, expansion=4, batch=512, log=lambda *a: None)
    ev = TS.evaluate(sae, x[:2000].float())
    assert ev["fvu"] < 0.5 and 2 < ev["l0"] < 40, ev
