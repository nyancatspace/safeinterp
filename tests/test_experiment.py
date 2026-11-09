"""The reward_hacking/ scripts: gate, SAE training, analyze.py's literal control and steering check."""
import json
import random

import pytest
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


# -------------------------------------------------------------------- steering
def test_steering_edits():
    sae = identity_sae(4)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    u = torch.tensor([0.0, 1.0, 0.0, 0.0])
    assert torch.allclose(A.add_fn(2 * u)(x), x + 2 * u)
    assert A.project_out_fn(u)(x)[..., 1].abs().max() < 1e-6
    out = A.feature_ablate_fn(sae, 2)(x)
    assert sae.encode(out)[..., 2].abs().max() < 1e-6  # the feature is gone
    assert torch.allclose(out[..., [0, 1, 3]], x[..., [0, 1, 3]])  # nothing else moved


def test_steering_summary_needs_noise_and_random_control():
    def row(mode, name, coeff, hack, step=0):
        return {"step": step, "mode": mode, "direction": name, "coeff": coeff, "survived": True, "n": 200,
                "hack": hack, "honest": 0.5, "broken": 0.5 - hack, "hidden_pass": 0.5, "hack_kinds": {}}
    rows = [row("none", "-", 0, 0.02), row("steer", "probe", 4, 0.30), row("steer", "f7", 4, 0.04),
            row("steer", "random", 4, 0.03),
            row("none", "-", 0, 0.40, 50), row("ablate", "probe", 0, 0.10, 50), row("ablate", "f7", 0, 0.38, 50),
            row("ablate", "random", 0, 0.41, 50)]
    probe, f7 = A.steering_summary(rows)
    assert "raises hacking beyond noise and the random control at c = [4]" in probe and "lowers it beyond" in probe
    assert "does not raise" in f7 and "does not lower" in f7
    md = "\n".join(A.steering_report({"steer_step": 0, "ablate_step": 50, "coeffs": [4], "layer": 14,
                                      "n_per_condition": 200, "rows": rows, "summary": [probe, f7]}))
    assert "Steering at step 0" in md and "Ablation at step 50" in md


def test_steer_and_ablate_end_to_end(tok, tmp_path):
    pytest.importorskip("peft")
    import copy

    from transformers import Qwen2Config, Qwen2ForCausalLM

    from safeinterp.rlhack import grpo as G

    torch.manual_seed(0)
    qwen = Qwen2ForCausalLM(Qwen2Config(vocab_size=257, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                        eos_token_id=256, bos_token_id=256, pad_token_id=256)).eval()
    cfg = G.GRPOConfig(steps=1, prompts_per_step=1, group=2, max_new_tokens=4, lora_r=4, lora_alpha=8, save_every=1,
                       workers=2)
    G.train(copy.deepcopy(qwen), tok, cfg, tmp_path / "run")
    base = copy.deepcopy(qwen)
    sae = SAE(SAEConfig(d_in=32, d_sae=64, kind="relu"))
    u = torch.nn.functional.normalize(torch.randn(32), dim=0)
    dirs = [{"name": "probe", "kind": "probe", "feature": None, "survived": True, "u": u, "gap": 1.0},
            {"name": "f3", "kind": "feature", "feature": 3, "survived": False,
             "u": torch.nn.functional.normalize(sae.W_dec[3].detach(), dim=0), "gap": 0.5}]
    st = A.steer_and_ablate(base, tok, tmp_path / "run", dirs, sae, train_tasks()[:2], "weak", 1, 0, 1,
                            coeffs=(1.0, 4.0), n_samples=1, max_new_tokens=4, workers=2, log=lambda *a: None)
    modes = [(r["mode"], r["direction"], r["coeff"]) for r in st["rows"]]
    assert modes == [("none", "-", 0.0), ("steer", "probe", 1.0), ("steer", "f3", 1.0), ("steer", "random", 1.0),
                     ("steer", "probe", 4.0), ("steer", "f3", 4.0), ("steer", "random", 4.0),
                     ("none", "-", 0.0), ("ablate", "probe", 0.0), ("ablate", "f3", 0.0), ("ablate", "random", 0.0)]
    assert all(r["n"] == 2 and abs(r["hack"] + r["honest"] + r["broken"] - 1) < 1e-6 for r in st["rows"])
    assert len(st["summary"]) == 2 and "did NOT survive" in st["summary"][1]
    ids = torch.tensor([[256] + tok.encode("def f(x):")])
    assert torch.allclose(base(ids).logits, qwen(ids).logits)  # adapters and hooks are gone afterwards
