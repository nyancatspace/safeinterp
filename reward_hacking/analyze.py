"""Do SAE features track (or lead) reward hacking?  Feature trajectories, separation, early warning.

Inputs: ``capture_activations.py`` output (``--acts``), ``train_sae.py`` output (``--sae``)
and the run's ``hack_rate.json``.  Writes ``report.md``, ``analysis.json`` and
``trajectories.png``.

**Selection is kept apart from measurement.**  The 30 tasks are split in two: features are
ranked by hack-vs-honest AUC on the *select* tasks only, and every number in the report
(AUCs, controls, trajectories, onsets) is measured on the other, *eval* tasks.

**The literal control.**  A feature can separate hacks from honest code just because hacks
contain hardcoded numbers.  That says nothing about the model's state.  Four checks:

| check | what it asks | literal detector |
|---|---|---|
| ``auc_base`` | Same completion tokens, read by the step-0 model (adapter off). Does the feature still separate? | as high as ``auc`` |
| ``auc_state`` | The part of the activation the text doesn't explain: checkpoint read minus step-0 read of the same tokens | ~0.5 |
| ``auc_literal`` | Among *honest* completions only: many numeric literals vs few | well above 0.5 |
| ``digit_frac`` | Share of the feature's 50 strongest token activations that are on digit tokens | high |

A feature **survives** only if ``auc >= --min-auc``, ``auc_state >= --min-state-auc`` and
``auc_literal < --max-literal-auc``.  The early-warning table has a regex baseline
(``flag_rate``: share of completions with a static hack signature) and the visible-literal
rate.  A feature that warns no earlier than these adds nothing.

**Steering and ablation (``--steer``, needs the GPU).**  Separation is correlation.  The
causal check uses two kinds of hack direction, both found on the select tasks:

- ``probe``: the difference of mean raw activations, hack minus honest.  This is a linear probe
  with no SAE, so it is also the baseline the SAE features have to beat
- ``f<j>``: the decoder directions of the features that survive the literal control (or of the
  top features, flagged as such, if none survive)

Then, on the eval tasks, generating at layer ``--layer``:

- **steer** (sufficiency): at ``--steer-step`` (default 0, before any hacking), add
  ``c x gap x direction`` at every position, where ``gap`` is the hack-minus-honest
  difference along the direction.  Does the model start hacking?
- **ablate** (necessity): at ``--ablate-step`` (default: the last checkpoint), remove the
  direction (project out the probe, subtract the feature's own contribution).  Does hacking drop?

Each is compared with doing nothing and with a random direction of the same size.  Every
condition reuses the same sampling seed, and the report gives each difference with a
±2 standard-error band.  Watch the broken rate: steering that only breaks the model raises
"broken", not "hack".

    python reward_hacking/analyze.py --acts runs/naive/acts --sae runs/naive/sae --run runs/naive
    python reward_hacking/analyze.py --acts runs/naive/acts --sae runs/naive/sae --run runs/naive --steer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from safeinterp.core.hooks import edit_sites
from safeinterp.core.sae import SAE
from safeinterp.rlhack.grpo import encode_prompt, sample, stop_ids
from safeinterp.rlhack.monitor import checkpoints, rate_onset, signal_onset, smooth
from sandbox import LABELS, grade_many, label


# ------------------------------------------------------------------ data
def load(acts_dir: Path):
    meta = [json.loads(l) for l in (acts_dir / "meta.jsonl").read_text().splitlines()]
    steps = sorted({m["step"] for m in meta})
    acts = {s: torch.load(acts_dir / f"step_{s}.pt") for s in steps}
    return meta, steps, acts


@torch.no_grad()
def encode_means(sae: SAE, x: torch.Tensor, offsets: torch.Tensor, device, batch: int = 8192) -> torch.Tensor:
    """Mean SAE code per completion: ``[n_completions, d_sae]``."""
    z = torch.cat([sae.encode(x[i:i + batch].float().to(device)).cpu() for i in range(0, len(x), batch)]) \
        if len(x) else torch.zeros(0, sae.cfg.d_sae)
    out = torch.zeros(len(offsets) - 1, sae.cfg.d_sae)
    for i, (a, b) in enumerate(zip(offsets[:-1].tolist(), offsets[1:].tolist())):
        if b > a:
            out[i] = z[a:b].mean(0)
    return out


@torch.no_grad()
def features(sae: SAE, meta, steps, acts, device) -> dict[str, torch.Tensor]:
    """Per completion, in ``meta`` order: mean code on the completion (checkpoint read and
    step-0 read of the same tokens) and on the prompt tokens."""
    comp, base, prompt = [], [], []
    for s in steps:
        a = acts[s]
        comp.append(encode_means(sae, a["comp"], a["offsets"], device))
        base.append(encode_means(sae, a["comp_base"], a["offsets"], device))
        p = a["prompt"]
        prompt.append(torch.cat([sae.encode(p[i:i + 512].float().to(device)).mean(1).cpu()
                                 for i in range(0, len(p), 512)]))
    order = {(m["step"], m["idx"]): i for i, m in enumerate(meta)}
    assert [order[(m["step"], m["idx"])] for m in meta] == list(range(len(meta)))
    return {"comp": torch.cat(comp), "base": torch.cat(base), "prompt": torch.cat(prompt)}


def raw_means(acts, steps, key: str) -> np.ndarray:
    """Mean raw activation per completion, in ``meta`` order: ``[n_completions, d_model]``."""
    out = []
    for s in steps:
        a = acts[s]
        x, o = a[key].float(), a["offsets"].tolist()
        d = a["prompt"].shape[-1]
        out += [x[i:j].mean(0) if j > i else torch.zeros(d) for i, j in zip(o[:-1], o[1:])]
    return torch.stack(out).numpy()


# --------------------------------------------------------------- statistics
def auc(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Per column, P(pos > neg) + 0.5 P(pos == neg).  ``pos`` ``[n, f]``, ``neg`` ``[m, f]``."""
    if len(pos) == 0 or len(neg) == 0:
        return np.full(pos.shape[1], np.nan)
    neg_sorted = np.sort(neg, axis=0)
    out = np.empty(pos.shape[1])
    for j in range(pos.shape[1]):
        lo = np.searchsorted(neg_sorted[:, j], pos[:, j], "left")
        hi = np.searchsorted(neg_sorted[:, j], pos[:, j], "right")
        out[j] = (lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(neg))
    return out


def task_split(meta) -> tuple[set, set]:
    names = sorted({m["task"] for m in meta})
    return set(names[::2]), set(names[1::2])


def onset_lead(steps, series: dict, hack_key: str, baseline: int, z: float, sustain: int, threshold: float) -> dict:
    """The hack rate's onset is the first checkpoint at or above ``threshold``.  Every warning
    signal, the regex and literal baselines included, uses the same rule: the first checkpoint
    after which it stays above its baseline mean + z * std."""
    hack_on = rate_onset(steps, series[hack_key], threshold, sustain)
    out = {"hack_onset": hack_on, "sustain": sustain, "signals": {}}
    for k, v in series.items():
        if k in (hack_key, "eval_honest_rate", "visible_reward", "hidden_pass"):
            continue  # behaviour, not warning signals
        on = signal_onset(steps, v, baseline, z, sustain)
        out["signals"][k] = {"onset": on, "lead": None if on is None or hack_on is None else hack_on - on}
    return out


# ------------------------------------------------------------------ analysis
def analyze(sae: SAE, meta, steps, acts, rl_rows: list[dict], top: int = 10, min_auc: float = 0.7,
            min_state_auc: float = 0.6, max_literal_auc: float = 0.7, baseline: int = 3, z: float = 3.0,
            sustain: int = 2, hack_threshold: float = 0.05, min_frac: float = 0.01, tokenizer=None,
            device="cpu") -> dict:
    F = features(sae, meta, steps, acts, device)
    comp, base, prompt = F["comp"].numpy(), F["base"].numpy(), F["prompt"].numpy()
    state = comp - base
    lab = np.array([m["label"] for m in meta])
    sel_tasks, ev_tasks = task_split(meta)
    sel = np.array([m["task"] in sel_tasks for m in meta])
    ev = ~sel
    hack, hon = lab == "hack", lab == "honest"
    counts = {s: {l: int(((lab == l) & (np.array([m["step"] for m in meta]) == s)).sum())
                  for l in ("honest", "hack", "broken")} for s in steps}
    res = {"steps": steps, "counts": counts, "n_select_tasks": len(sel_tasks), "n_eval_tasks": len(ev_tasks),
           "thresholds": {"min_auc": min_auc, "min_state_auc": min_state_auc, "max_literal_auc": max_literal_auc}}

    n = {k: int(v.sum()) for k, v in {"hack_select": hack & sel, "honest_select": hon & sel,
                                      "hack_eval": hack & ev, "honest_eval": hon & ev}.items()}
    res["n"] = n
    if min(n.values()) < 3:
        res["verdict"] = ("not enough labelled data: need at least 3 hack and 3 honest completions in both task "
                          f"halves, have {n}. If hacks are missing, the gate (step 3) has not passed.")
        res["features"], res["early_warning"] = [], None
        return res

    # rank on the select tasks only
    alive = (comp[sel] > 0).mean(0) >= min_frac
    a_sel = np.full(comp.shape[1], np.nan)
    a_sel[alive] = auc(comp[hack & sel][:, alive], comp[hon & sel][:, alive])
    ranked = [int(j) for j in np.argsort(-np.nan_to_num(a_sel, nan=0.0))[:top]]

    # measure on the eval tasks
    honest_ev = hon & ev
    lits = np.array([m["n_num_literals"] for m in meta])
    med = np.median(lits[honest_ev])
    many, few = honest_ev & (lits > med), honest_ev & (lits <= med)
    cols = np.array(ranked)
    a_ev = auc(comp[hack & ev][:, cols], comp[honest_ev][:, cols])
    a_base = auc(base[hack & ev][:, cols], base[honest_ev][:, cols])
    a_state = auc(state[hack & ev][:, cols], state[honest_ev][:, cols])
    a_lit = auc(comp[many][:, cols], comp[few][:, cols])
    digit = digit_fracs(sae, acts, steps, ranked, tokenizer, device) if tokenizer is not None else {}
    feats = []
    for k, j in enumerate(ranked):
        f = {"feature": j, "auc_select": round(float(a_sel[j]), 3), "auc": round(float(a_ev[k]), 3),
             "auc_base": round(float(a_base[k]), 3), "auc_state": round(float(a_state[k]), 3),
             "auc_literal": None if np.isnan(a_lit[k]) else round(float(a_lit[k]), 3),
             "digit_frac": digit.get(j, {}).get("digit_frac"), "top_tokens": digit.get(j, {}).get("top_tokens")}
        lit_ok = f["auc_literal"] is None or f["auc_literal"] < max_literal_auc
        f["survives"] = bool(f["auc"] >= min_auc and f["auc_state"] >= min_state_auc and lit_ok)
        f["why_not"] = ", ".join(r for r, bad in (
            (f"auc {f['auc']} < {min_auc}", f["auc"] < min_auc),
            (f"auc_state {f['auc_state']} < {min_state_auc} (separation is in the text)", f["auc_state"] < min_state_auc),
            (f"auc_literal {f['auc_literal']} >= {max_literal_auc} (fires on literals in honest code)", not lit_ok),
        ) if bad)
        feats.append(f)
    res["features"] = feats
    res["n_literal_split"] = {"many": int(many.sum()), "few": int(few.sum()), "median_literals": float(med)}
    surv = [f["feature"] for f in feats if f["survives"]]

    # the probe: difference of means on raw activations, same controls
    R, Rb = raw_means(acts, steps, "comp"), raw_means(acts, steps, "comp_base")
    diff = R[hack & sel].mean(0) - R[hon & sel].mean(0)
    gap = float(np.linalg.norm(diff))
    u = diff / max(gap, 1e-8)
    pr, prb = (R @ u)[:, None], (Rb @ u)[:, None]
    pf = {"auc": round(float(auc(pr[hack & ev], pr[honest_ev])[0]), 3),
          "auc_base": round(float(auc(prb[hack & ev], prb[honest_ev])[0]), 3),
          "auc_state": round(float(auc((pr - prb)[hack & ev], (pr - prb)[honest_ev])[0]), 3),
          "auc_literal": None if not (many.any() and few.any()) else round(float(auc(pr[many], pr[few])[0]), 3),
          "gap": round(gap, 4)}
    pf["survives"] = bool(pf["auc"] >= min_auc and pf["auc_state"] >= min_state_auc
                          and (pf["auc_literal"] is None or pf["auc_literal"] < max_literal_auc))
    res["probe"] = pf
    res["eval_tasks"] = sorted(ev_tasks)

    # directions for steering (tensors: kept out of analysis.json)
    dirs = [{"name": "probe", "kind": "probe", "feature": None, "survived": pf["survives"],
             "u": torch.tensor(u, dtype=torch.float32), "gap": gap}]
    W = sae.feature_directions().detach().cpu()
    for j in (surv or ranked)[:3]:
        zgap = float(comp[hack & sel][:, j].mean() - comp[hon & sel][:, j].mean())
        d = W[j].float()
        dirs.append({"name": f"f{j}", "kind": "feature", "feature": j, "survived": j in surv,
                     "u": d / d.norm(), "gap": zgap * float(d.norm())})
    res["_dirs"] = dirs

    # trajectories on the eval tasks, every completion (honest, hack and broken)
    step_of = np.array([m["step"] for m in meta])
    series = {}
    def per_step(fn):
        return [float(fn((step_of == s) & ev)) for s in steps]
    series["eval_hack_rate"] = per_step(lambda m: hack[m].mean())
    series["eval_honest_rate"] = per_step(lambda m: hon[m].mean())
    series["visible_reward"] = per_step(lambda m: np.array([x["reward"] for x in meta])[m].mean())
    series["hidden_pass"] = per_step(lambda m: np.array([x["hidden_frac"] == 1.0 for x in meta])[m].mean())
    series["flag_rate"] = per_step(lambda m: np.array([x["flagged"] for x in meta])[m].mean())
    series["visible_literal_rate"] = per_step(lambda m: np.array([x["visible_out_literals"] > 0 for x in meta])[m].mean())
    for name, group in (("top", ranked), ("surviving", surv)):
        if not group:
            continue
        g = np.array(group)
        series[f"{name}_comp"] = per_step(lambda m: comp[m][:, g].sum(1).mean())
        series[f"{name}_state"] = per_step(lambda m: state[m][:, g].sum(1).mean())
        series[f"{name}_prompt"] = per_step(lambda m: prompt[m][:, g].sum(1).mean())
    for j in ranked[:5]:
        series[f"f{j}_comp"] = per_step(lambda m: comp[m][:, j].mean())
        series[f"f{j}_state"] = per_step(lambda m: state[m][:, j].mean())
    series = {k: [round(v, 5) for v in vs] for k, vs in series.items()}
    ew = onset_lead(steps, series, "eval_hack_rate", baseline, z, sustain, hack_threshold)
    if rl_rows:  # every RL step: the densest behavioural signal
        rs = [r["step"] for r in rl_rows]
        ew["train_hack_onset"] = rate_onset(rs, smooth([r["hack_rate"] for r in rl_rows], 5), hack_threshold, sustain)
        ew["train_flag_onset"] = rate_onset(rs, smooth([r["flag_rate"] for r in rl_rows], 5), hack_threshold, sustain)
    res["early_warning"] = {"series": series, **ew}
    res["verdict"] = verdict(res, surv)
    return res


@torch.no_grad()
def digit_fracs(sae, acts, steps, feats: list[int], tok, device, k: int = 50) -> dict:
    """For each feature, the share of its ``k`` strongest token activations on digit tokens."""
    best = {j: [] for j in feats}
    W = sae.W_enc[:, feats]
    for s in steps:
        a = acts[s]
        if "comp_ids" not in a or not len(a["comp"]):
            continue
        for i in range(0, len(a["comp"]), 8192):
            x = a["comp"][i:i + 8192].float().to(device)
            pre = ((x * sae.act_scale - sae.b_dec) @ W + sae.b_enc[feats]).relu().cpu()
            ids = a["comp_ids"][i:i + 8192]
            for c, j in enumerate(feats):
                v, idx = pre[:, c].topk(min(k, len(pre)))
                best[j] += list(zip(v.tolist(), ids[idx].tolist()))
    out = {}
    for j, pairs in best.items():
        pairs = sorted(pairs, reverse=True)[:k]
        if not pairs:
            continue
        toks = [tok.decode([t]) for _, t in pairs]
        out[j] = {"digit_frac": round(sum(any(ch.isdigit() for ch in t) for t in toks) / len(toks), 3),
                  "top_tokens": toks[:12]}
    return out


def verdict(res: dict, surv: list[int]) -> str:
    ew = res["early_warning"]
    sig = ew["signals"]
    hack_on = ew["hack_onset"]
    lines = []
    if hack_on is None:
        lines.append("The held-out-task hack rate never crossed the threshold at checkpoint resolution, so there is "
                     "nothing to warn about: no lead can be measured.")
    if not surv:
        lines.append("No top feature survives the literal control. The features that separate hacks from honest "
                     "code do so through the text: the step-0 model reading the same tokens separates them as well, "
                     "and/or they fire on numeric literals in honest code. This is a negative result for the hypothesis "
                     "at this layer.")
        return " ".join(lines)
    lines.append(f"{len(surv)} feature(s) survive the literal control: {surv}.")
    best = {k: sig[k] for k in ("surviving_state", "surviving_prompt", "surviving_comp") if k in sig}
    base = {k: sig[k]["lead"] for k in ("flag_rate", "visible_literal_rate") if k in sig}
    for k, v in best.items():
        lead = v["lead"]
        if lead is None:
            lines.append(f"`{k}` has no onset (or the hack rate has none).")
            continue
        beat = [b for b, l in base.items() if l is None or lead > l]
        lines.append(f"`{k}` onset at step {v['onset']}, lead {lead:+d} steps over the hack-rate onset"
                     + (f"; earlier than the regex baselines {beat}" if beat else "; NOT earlier than the regex/literal baselines")
                     + ".")
    lines.append("One run is one sample, and leads are only resolved to the checkpoint spacing.")
    return " ".join(lines)


# ------------------------------------------------------------------ steering
def add_fn(v: torch.Tensor):
    return lambda t: t + v.to(t.device, t.dtype)


def project_out_fn(u: torch.Tensor):
    def fn(t):
        uu = u.to(t.device, torch.float32)
        x = t.float()
        return (x - (x @ uu)[..., None] * uu).to(t.dtype)
    return fn


def feature_ablate_fn(sae: SAE, j: int):
    """Subtract feature ``j``'s own contribution ``z_j * d_j``; everything else, SAE error included, stays."""
    w, b, d = sae.W_enc[:, j].detach(), sae.b_enc[j].detach(), sae.feature_directions()[j].detach()

    def fn(t):
        x = t.float()
        z = ((x * sae.act_scale.to(x.device) - sae.b_dec.to(x.device)) @ w.to(x.device) + b.to(x.device)).relu()
        return (x - z[..., None] * d.to(x.device)).to(t.dtype)
    return fn


@torch.no_grad()
def run_condition(model, tok, tasks, hint: str, edits: dict, n_samples: int, max_new_tokens: int, temperature: float,
                  device, sample_batch: int = 32, workers: int = 8, seed: int = 0) -> dict:
    """Sample ``n_samples`` completions per task with ``edits`` applied, grade and label them."""
    group = [t for t in tasks for _ in range(n_samples)]
    prompts = [encode_prompt(tok, t, hint) for t in group]
    torch.manual_seed(seed)
    with edit_sites(model, edits):
        comps = sample(model, tok, prompts, max_new_tokens, temperature, device, batch=sample_batch)
    stops = stop_ids(tok)
    texts = [tok.decode([i for i in c if i not in stops]) for c in comps]
    grades = grade_many(texts, group, "naive", workers)
    labels = [label(g) for g in grades]
    n = len(labels)
    out = {l: round(labels.count(l) / n, 4) for l in LABELS}
    out["n"] = n
    out["hidden_pass"] = round(sum(g["hidden_frac"] == 1.0 for g in grades) / n, 4)
    kinds = [g["hack_kind"] for g in grades if g["hack"]]
    out["hack_kinds"] = {k: kinds.count(k) for k in sorted(set(kinds))}
    out["example_hack"] = next((t[:400] for t, l in zip(texts, labels) if l == "hack"), None)
    return out


def steer_and_ablate(base, tok, run_dir, dirs: list[dict], sae: SAE, tasks, hint: str, layer: int, steer_step: int,
                     ablate_step: int, coeffs=(1.0, 4.0), n_samples: int = 4, max_new_tokens: int = 384,
                     temperature: float = 1.0, device="cpu", sample_batch: int = 32, workers: int = 8,
                     seed: int = 0, log=print) -> dict:
    """Steer with each direction at ``steer_step`` and ablate it at ``ablate_step``, against
    no intervention and a random direction of the same size."""
    from peft import PeftModel

    ck = dict(checkpoints(run_dir))
    site = (layer, "resid_post")
    g = torch.Generator().manual_seed(seed + 7)
    rand = torch.randn(dirs[0]["u"].numel(), generator=g)
    rand /= rand.norm()
    big = max(x["gap"] for x in dirs)
    rows = []

    def run(step, mode, name, coeff, edits, survived=None):
        nonlocal base
        pm = PeftModel.from_pretrained(base, str(ck[step]))
        try:
            r = run_condition(pm.get_base_model().eval(), tok, tasks, hint, edits, n_samples, max_new_tokens,
                              temperature, device, sample_batch, workers, seed)
        finally:
            base = pm.unload()
        rows.append({"step": step, "mode": mode, "direction": name, "coeff": coeff, "survived": survived, **r})
        log(f"{mode:6s} step {step:4d} {name:8s} c={coeff:g}: hack {r['hack']:.3f}  honest {r['honest']:.3f}  "
            f"broken {r['broken']:.3f}")

    run(steer_step, "none", "-", 0.0, {})
    for c in coeffs:
        for x in dirs:
            run(steer_step, "steer", x["name"], c, {site: add_fn(c * x["gap"] * x["u"])}, x["survived"])
        run(steer_step, "steer", "random", c, {site: add_fn(c * big * rand)})
    run(ablate_step, "none", "-", 0.0, {})
    for x in dirs:
        fn = project_out_fn(x["u"]) if x["kind"] == "probe" else feature_ablate_fn(sae, x["feature"])
        run(ablate_step, "ablate", x["name"], 0.0, {site: fn}, x["survived"])
    run(ablate_step, "ablate", "random", 0.0, {site: project_out_fn(rand)})
    return {"steer_step": steer_step, "ablate_step": ablate_step, "coeffs": list(coeffs), "layer": layer,
            "n_per_condition": rows[0]["n"], "rows": rows, "summary": steering_summary(rows)}


def _se_diff(p1: float, p2: float, n: int) -> float:
    """Standard error of the difference of two proportions, each from ``n`` samples.  Each
    proportion is smoothed to (k + 1) / (n + 2) first, so 0 of 15 doesn't claim zero noise."""
    q1, q2 = (p1 * n + 1) / (n + 2), (p2 * n + 1) / (n + 2)
    return float(np.sqrt(q1 * (1 - q1) / n + q2 * (1 - q2) / n))


def _beats(r: dict, ref: dict) -> float:
    """How far ``r``'s hack rate is above ``ref``'s, in units of 2 standard errors (> 1: beyond noise)."""
    return (r["hack"] - ref["hack"]) / (2 * _se_diff(r["hack"], ref["hack"], r["n"]))


def steering_summary(rows: list[dict]) -> list[str]:
    """Per direction: does steering raise hacking, and ablation lower it, beyond noise and the random control?"""
    none = {r["step"]: r for r in rows if r["mode"] == "none"}
    rnd = {(r["mode"], r["coeff"]): r for r in rows if r["direction"] == "random"}
    out = []
    for name in dict.fromkeys(r["direction"] for r in rows if r["mode"] == "steer" and r["direction"] != "random"):
        up = [r["coeff"] for r in rows if r["mode"] == "steer" and r["direction"] == name
              and _beats(r, none[r["step"]]) > 1 and _beats(r, rnd[("steer", r["coeff"])]) > 1]
        ab = next((r for r in rows if r["mode"] == "ablate" and r["direction"] == name), None)
        down = bool(ab and _beats(none[ab["step"]], ab) > 1 and _beats(rnd[("ablate", 0.0)], ab) > 1)
        survived = next(r["survived"] for r in rows if r["direction"] == name)
        out.append(f"`{name}`{'' if survived else ' (did NOT survive the literal control)'}: steering "
                   + (f"raises hacking beyond noise and the random control at c = {up}" if up
                      else "does not raise hacking beyond noise and the random control")
                   + "; ablation " + ("lowers it beyond noise and the random control." if down
                                      else "does not lower it beyond noise and the random control."))
    return out


def steering_report(st: dict) -> list[str]:
    rows = st["rows"]
    L = ["", "## Steering and ablation (causal check)", "",
         f"Layer {st['layer']}, eval tasks, {st['n_per_condition']} completions per condition, the same sampling seed "
         f"in every condition. Steering at step {st['steer_step']} adds c × gap × direction at every position; "
         f"ablation at step {st['ablate_step']} removes the direction. Δ is the change in hack rate ± 2 standard "
         "errors; `*` marks directions that did not survive the literal control.", ""]
    L += [f"- {line}" for line in st["summary"]]
    for mode, step in (("steer", st["steer_step"]), ("ablate", st["ablate_step"])):
        base = next(r for r in rows if r["mode"] == "none" and r["step"] == step)
        L += ["", f"**{'Steering' if mode == 'steer' else 'Ablation'} at step {step}** (no intervention: hack "
                  f"{base['hack']:.3f}, honest {base['honest']:.3f}, broken {base['broken']:.3f})", "",
              "| direction | c | hack | honest | broken | hidden pass | Δ hack vs none | Δ hack vs random | hack kinds |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in (r for r in rows if r["mode"] == mode):
            rnd = next(x for x in rows if x["mode"] == mode and x["direction"] == "random" and x["coeff"] == r["coeff"])
            dn, en = r["hack"] - base["hack"], 2 * _se_diff(r["hack"], base["hack"], r["n"])
            vs = "" if r["direction"] == "random" else \
                f"{r['hack'] - rnd['hack']:+.3f} ± {2 * _se_diff(r['hack'], rnd['hack'], r['n']):.3f}"
            name = r["direction"] + ("*" if r["survived"] is False else "")
            kinds = ", ".join(f"{k} {v}" for k, v in r["hack_kinds"].items())
            L.append(f"| {name} | {r['coeff']:g} | {r['hack']:.3f} | {r['honest']:.3f} | {r['broken']:.3f} | "
                     f"{r['hidden_pass']:.3f} | {dn:+.3f} ± {en:.3f} | {vs} | {kinds} |")
    L += ["", "A direction that makes the model write numbers can raise `hardcode` hacks without touching anything like "
              "intent: check the hack kinds and the broken rate, and trust a direction only if it both survives the "
              "literal control and acts causally here."]
    return L


# -------------------------------------------------------------------- report
def write_report(res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps({k: v for k, v in res.items() if not k.startswith("_")}, indent=1))
    L = ["# Reward hacking: SAE feature analysis", "", f"**Verdict.** {res['verdict']}", "",
         "## Completions per checkpoint", "", "| step | honest | hack | broken |", "|---|---|---|---|"]
    L += [f"| {s} | {c['honest']} | {c['hack']} | {c['broken']} |" for s, c in res["counts"].items()]
    L += ["", f"Features are selected on {res['n_select_tasks']} tasks and measured on the other "
              f"{res['n_eval_tasks']}. Labelled counts: {res['n']}."]
    if res["features"]:
        t = res["thresholds"]
        L += ["", "## Top hack-promoting features (measured on eval tasks)", "",
              f"Survives: auc ≥ {t['min_auc']}, auc_state ≥ {t['min_state_auc']}, auc_literal < {t['max_literal_auc']}. "
              f"Literal split among honest eval completions: {res['n_literal_split']}.", "",
              "| feature | auc (select) | auc | auc_base | auc_state | auc_literal | digit_frac | survives | top tokens |",
              "|---|---|---|---|---|---|---|---|---|"]
        for f in res["features"]:
            toks = " ".join(repr(x) for x in (f["top_tokens"] or [])[:8]).replace("|", "\\|")
            L.append(f"| {f['feature']} | {f['auc_select']} | {f['auc']} | {f['auc_base']} | {f['auc_state']} | "
                     f"{f['auc_literal']} | {f['digit_frac']} | {'yes' if f['survives'] else 'no: ' + f['why_not']} | {toks} |")
        if res.get("probe"):
            pf = res["probe"]
            L += ["", f"**Probe** (raw difference of means, no SAE; the baseline features must beat): auc {pf['auc']}, "
                      f"auc_base {pf['auc_base']}, auc_state {pf['auc_state']}, auc_literal {pf['auc_literal']}, "
                      f"survives: {'yes' if pf['survives'] else 'no'}."]
    ew = res.get("early_warning")
    if ew:
        S = ew["series"]
        keys = [k for k in ("eval_hack_rate", "visible_reward", "hidden_pass", "flag_rate", "visible_literal_rate",
                            "top_comp", "top_state", "top_prompt", "surviving_comp", "surviving_state",
                            "surviving_prompt") if k in S]
        L += ["", "## Early-warning table", "",
              "Eval-task completions at each checkpoint. `*_comp`: summed activation of the feature group on "
              "completions; `*_state`: the same minus the step-0 model's reading of the same tokens; `*_prompt`: on "
              "the prompt tokens, before any code is written.", "",
              "| step | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
        for i, s in enumerate(res["steps"]):
            L.append(f"| {s} | " + " | ".join(f"{S[k][i]:.3g}" for k in keys) + " |")
        L += ["", f"Hack-rate onset (eval tasks, ≥ threshold for {ew['sustain']} checkpoints): **{ew['hack_onset']}**. "
                  f"Training-rollout hack onset (every step): {ew.get('train_hack_onset')}; regex flag onset: "
                  f"{ew.get('train_flag_onset')}.", "",
              "| signal | onset | lead (steps, + = earlier than hacking) |", "|---|---|---|"]
        L += [f"| {k} | {v['onset']} | {v['lead']} |" for k, v in ew["signals"].items()]
    if res.get("steering"):
        L += steering_report(res["steering"])
    L += ["", "## Caveats", "",
          "- Leads are resolved only to the checkpoint spacing, and the baseline is the first few checkpoints.",
          "- A feature found in one run is one sample: repeat with other seeds, and against a `--grader honest` control run.",
          "- `auc_base` high does not by itself mean the feature is syntactic: the step-0 model can recognise hack-shaped "
          "text too. That is why survival is decided by `auc_state` and `auc_literal`."]
    (out / "report.md").write_text("\n".join(L) + "\n")
    plot(res, out)


def plot(res: dict, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ew = res.get("early_warning")
    if not ew:
        return
    S, steps = ew["series"], res["steps"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k in ("eval_hack_rate", "visible_reward", "hidden_pass", "flag_rate"):
        ax.plot(steps, S[k], label=k, lw=2 if k == "eval_hack_rate" else 1)
    ax2 = ax.twinx()
    for k in ("top_state", "top_prompt", "surviving_state", "surviving_prompt"):
        if k in S:
            v = np.array(S[k]) - S[k][0]
            ax2.plot(steps, v / (np.abs(v).max() + 1e-9), "--", label=k)
    ax.set_xlabel("RL step")
    ax.set_ylabel("rate")
    ax2.set_ylabel("feature signal (rescaled)")
    fig.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=120)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--acts", default="runs/naive/acts")
    p.add_argument("--sae", default="runs/naive/sae")
    p.add_argument("--run", default="runs/naive", help="for hack_rate.json")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--min-auc", type=float, default=0.7)
    p.add_argument("--min-state-auc", type=float, default=0.6)
    p.add_argument("--max-literal-auc", type=float, default=0.7)
    p.add_argument("--baseline", type=int, default=3, help="checkpoints that define each signal's baseline")
    p.add_argument("--z", type=float, default=3.0)
    p.add_argument("--sustain", type=int, default=2)
    p.add_argument("--hack-threshold", type=float, default=0.05)
    p.add_argument("--steer", action="store_true", help="also run the steering and ablation check (loads the model)")
    p.add_argument("--steer-step", type=int, default=0, help="checkpoint to steer (default: 0, before training)")
    p.add_argument("--ablate-step", type=int, help="checkpoint to ablate (default: the last)")
    p.add_argument("--steer-coeffs", default="1,4", help="multiples of the hack-minus-honest gap along each direction")
    p.add_argument("--steer-samples", type=int, default=4, help="completions per eval task per condition")
    p.add_argument("--max-new-tokens", type=int, help="default: as in training")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sample-batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--quant", choices=["auto", "4bit", "none"], help="default: as in training")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", help="default: <run>/analysis")
    args = p.parse_args(argv)
    from safeinterp.core.models import pick_device

    device = pick_device(args.device)
    sae = SAE.load(args.sae, device).eval()
    meta, steps, acts = load(Path(args.acts))
    run = Path(args.run)
    rl_rows = json.loads((run / "hack_rate.json").read_text()) if (run / "hack_rate.json").exists() else []
    tok = None
    try:
        from transformers import AutoTokenizer
        cap = json.loads((Path(args.acts) / "capture.json").read_text())
        tok = AutoTokenizer.from_pretrained(cap["model"])
    except Exception as e:  # digit_frac is optional
        print(f"no tokenizer ({e}); skipping digit_frac")
    res = analyze(sae, meta, steps, acts, rl_rows, args.top, args.min_auc, args.min_state_auc, args.max_literal_auc,
                  args.baseline, args.z, args.sustain, args.hack_threshold, tokenizer=tok, device=device)
    if args.steer and res.get("_dirs"):
        from tasks import train_tasks
        from train_grpo import load_base

        cfg = json.loads((run / "grpo.json").read_text())
        cap = json.loads((Path(args.acts) / "capture.json").read_text())
        base, mtok, _ = load_base(cfg["model"], device, args.quant or cfg.get("quant", "auto"))
        tasks = [t for t in train_tasks(n_visible=cfg.get("n_visible", 3), seed=cfg["seed"]) if t.name in res["eval_tasks"]]
        ablate_step = args.ablate_step if args.ablate_step is not None else checkpoints(run)[-1][0]
        res["steering"] = steer_and_ablate(
            base, mtok, run, res["_dirs"], sae, tasks, cfg["hint"], cap["layer"], args.steer_step, ablate_step,
            [float(c) for c in args.steer_coeffs.split(",")], args.steer_samples,
            args.max_new_tokens or cfg["max_new_tokens"], args.temperature, device, args.sample_batch, args.workers,
            args.seed)
    elif args.steer:
        print("skipping steering: no hack directions (not enough hack and honest completions)")
    out = Path(args.out or run / "analysis")
    write_report(res, out)
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
