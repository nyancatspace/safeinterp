"""Where does GPT-2 keep the knowledge it uses for zero-shot tasks?

Given one SAE per layer and a set of task probes (see ``tasks.py``), this
module answers four questions, layer by layer:

1. **Task representation.** Which features fire at the answer position for
   each task format (QA, translation, TL;DR, ...) but not for ordinary text?
   Which features fire for *every* task format ("multitask" features)?  How
   well does the SAE code at this layer identify the task (nearest-centroid
   accuracy)?
2. **Knowledge attribution.** For probes with a gold answer, which features
   push up the answer's log-prob?  Attribution is activation x gradient:
   ``a_f * (d_f . dlogp/dx)``, summed over prompt positions.  This is the
   *total* effect through all later layers, to first order.
3. **Causal check.** Remove the top-N attributed features at one layer (keeping
   the SAE error term, so nothing else changes) and measure the real drop in
   answer log-prob.  A large drop from a few features means the answer is
   carried by a small, readable set of features at that layer.
4. **Knowledge vs format.** For the same fact asked as a plain sentence and as
   "Q: ... A:", how much do the top features overlap?  High overlap means the
   QA task reuses the knowledge the model already uses for plain language
   modelling, which is the GPT-2 paper's hypothesis.
"""
from __future__ import annotations

import heapq
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .hooks import edit_sites
from .sae import SAE
from .tasks import Probe


# --------------------------------------------------------------------- utils
def encode_prompt(tokenizer, text: str, device="cpu") -> torch.Tensor:
    return torch.tensor([[tokenizer.eos_token_id] + tokenizer.encode(text)], device=device)


def answer_token(tokenizer, answer: str | None) -> int | None:
    return None if answer is None else tokenizer.encode(answer)[0]


def unembed(model: nn.Module) -> torch.Tensor:
    """Directions -> logits map, folding in the final LayerNorm's gain: ``[d, vocab]``."""
    gain = model.transformer.ln_f.weight
    return gain[:, None] * model.lm_head.weight.T


def logit_lens(model: nn.Module, sae: SAE, features: list[int], tokenizer, k: int = 8) -> dict[int, list[str]]:
    """Tokens each feature's decoder direction most directly promotes."""
    D = sae.feature_directions()[features]
    D = D - D.mean(-1, keepdim=True)  # LayerNorm removes the mean
    logits = D @ unembed(model)
    top = logits.topk(k, dim=-1).indices
    return {f: [tokenizer.decode([t]) for t in row.tolist()] for f, row in zip(features, top)}


def _answer_rank_in_lens(model, sae, feature: int, answer_id: int) -> int:
    d = sae.feature_directions()[feature]
    logits = (d - d.mean()) @ unembed(model)
    return int((logits > logits[answer_id]).sum())


# ---------------------------------------------------------- probe forward pass
@dataclass
class ProbeRun:
    probe: Probe
    answer_id: int | None
    answer_logprob: float | None
    answer_rank: int | None  # 0 = model's top prediction
    top_pred: str
    last_acts: dict[int, torch.Tensor] = field(default_factory=dict)  # layer -> [d_sae]
    attr: dict[int, torch.Tensor] = field(default_factory=dict)  # layer -> [d_sae], summed over positions
    attr_last: dict[int, torch.Tensor] = field(default_factory=dict)  # layer -> [d_sae], final position only
    err_attr: dict[int, float] = field(default_factory=dict)  # attribution of the SAE error term


def run_probe(model: nn.Module, tokenizer, saes: dict[int, SAE], probe: Probe, device="cpu") -> ProbeRun:
    ids = encode_prompt(tokenizer, probe.prompt, device)
    ans = answer_token(tokenizer, probe.answer)
    sites = {(layer, sae.cfg.hook): layer for layer, sae in saes.items()}
    deltas, xs = {}, {}

    def make(site):
        def fn(t):
            d = torch.zeros_like(t, requires_grad=True)
            deltas[site], xs[site] = d, t.detach()
            return t + d

        return fn

    with torch.enable_grad(), edit_sites(model, {s: make(s) for s in sites}):
        logp = F.log_softmax(model(ids).logits[0, -1].float(), -1)
        if ans is not None:
            logp[ans].backward()

    run = ProbeRun(
        probe=probe,
        answer_id=ans,
        answer_logprob=None if ans is None else logp[ans].item(),
        answer_rank=None if ans is None else int((logp > logp[ans]).sum()),
        top_pred=tokenizer.decode([int(logp.argmax())]),
    )
    with torch.no_grad():
        for site, layer in sites.items():
            sae = saes[layer]
            x = xs[site][0]
            z = sae.encode(x)
            z[0] = 0  # BOS: the SAE was not trained on it
            run.last_acts[layer] = z[-1].cpu()
            if ans is not None:
                g = deltas[site].grad[0]
                attr = z * (g @ sae.feature_directions().T)
                run.attr[layer] = attr.sum(0).cpu()
                run.attr_last[layer] = attr[-1].cpu()
                run.err_attr[layer] = ((x - sae(x)) * g)[1:].sum().item()
    return run


@torch.no_grad()
def ablate_features(
    model: nn.Module, tokenizer, sae: SAE, probe: Probe, features: list[int], last_only: bool = False, device="cpu"
) -> float:
    """Answer log-prob after removing ``features`` from the SAE's layer (error term kept)."""
    ids = encode_prompt(tokenizer, probe.prompt, device)
    ans = answer_token(tokenizer, probe.answer)
    D = sae.feature_directions()[features]

    def fn(t):
        z = sae.encode(t)[..., features]
        z[:, 0] = 0
        if last_only:
            z[:, :-1] = 0
        return t - z @ D

    with edit_sites(model, {(sae.cfg.layer, sae.cfg.hook): fn}):
        logp = F.log_softmax(model(ids).logits[0, -1].float(), -1)
    return logp[ans].item()


# ------------------------------------------------------------------ analyses
def task_features(runs: list[ProbeRun], layer: int, top_n: int = 10) -> dict:
    """Task-selective and shared "multitask" features at the final prompt position."""
    by_task = defaultdict(list)
    for r in runs:
        by_task[r.probe.task].append(r.last_acts[layer])
    tasks = sorted(by_task)
    A = {t: torch.stack(v) for t, v in by_task.items()}
    freq = torch.stack([(A[t] > 0).float().mean(0) for t in tasks])  # [n_tasks, d_sae]
    mean = torch.stack([A[t].mean(0) for t in tasks])

    out = {"selective": {}, "shared": []}
    for i, t in enumerate(tasks):
        others = torch.cat([freq[:i], freq[i + 1 :]]).max(0).values
        score = freq[i] - others
        top = score.topk(top_n)
        out["selective"][t] = [
            {"feature": int(f), "specificity": round(float(s), 3), "freq": round(float(freq[i, f]), 3),
             "mean_act": round(float(mean[i, f]), 3)}
            for s, f in zip(top.values, top.indices) if s > 0
        ]

    task_rows = [i for i, t in enumerate(tasks) if t != "control"]
    if "control" in tasks and task_rows:
        ctrl = freq[tasks.index("control")]
        shared = freq[task_rows].min(0).values - ctrl
        top = shared.topk(top_n)
        out["shared"] = [
            {"feature": int(f), "score": round(float(s), 3), "min_task_freq": round(float(freq[task_rows, f].min()), 3),
             "control_freq": round(float(ctrl[f]), 3)}
            for s, f in zip(top.values, top.indices) if s > 0
        ]

    # Leave-one-out nearest-centroid task identification from the SAE code.
    X = F.normalize(torch.cat([A[t] for t in tasks]), dim=-1)
    y = torch.cat([torch.full((len(A[t]),), i) for i, t in enumerate(tasks)])
    sums = torch.stack([X[y == i].sum(0) for i in range(len(tasks))])
    counts = torch.bincount(y, minlength=len(tasks)).float()
    correct = 0
    for j in range(len(X)):
        c_sum, c_n = sums.clone(), counts.clone()
        c_sum[y[j]] -= X[j]
        c_n[y[j]] -= 1
        cent = F.normalize(c_sum / c_n.clamp_min(1)[:, None], dim=-1)
        cent[c_n == 0] = 0
        correct += int((cent @ X[j]).argmax() == y[j])
    out["task_id_accuracy"] = correct / len(X)
    return out


def knowledge_layer(model, tokenizer, sae: SAE, runs: list[ProbeRun], top_n: int = 10, device="cpu") -> dict:
    """Attribution + ablation summary for probes with answers at one layer."""
    layer = sae.cfg.layer
    answered = [r for r in runs if r.answer_id is not None]
    rows, feat_groups = [], defaultdict(set)
    for r in answered:
        top = r.attr[layer].topk(top_n).indices.tolist()
        ablated = ablate_features(model, tokenizer, sae, r.probe, top, device=device)
        ablated_last = ablate_features(model, tokenizer, sae, r.probe, top, last_only=True, device=device)
        lens_hits = sum(_answer_rank_in_lens(model, sae, f, r.answer_id) < 10 for f in top[:3])
        rows.append({
            "task": r.probe.task, "group": r.probe.group, "top_features": top,
            "attr_top": round(float(r.attr[layer][top].sum()), 3),
            "attr_all_features": round(float(r.attr[layer].sum()), 3),
            "attr_error": round(r.err_attr[layer], 3),
            "logprob": round(r.answer_logprob, 3),
            "drop_ablate_top": round(r.answer_logprob - ablated, 3),
            "drop_ablate_top_last_pos": round(r.answer_logprob - ablated_last, 3),
            "lens_hits_top3": lens_hits,
        })
        if r.probe.task.startswith("fact"):
            for f in top:
                feat_groups[f].add(r.probe.group)

    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    summary = {
        t: {k: round(sum(x[k] for x in v) / len(v), 3)
            for k in ("drop_ablate_top", "drop_ablate_top_last_pos", "attr_top", "attr_error", "lens_hits_top3")}
        for t, v in by_task.items()
    }
    n_fact_feats = len(feat_groups)
    specific = sum(len(g) <= 2 for g in feat_groups.values())
    return {
        "per_probe": rows,
        "per_task": summary,
        # Of the features that matter for facts, how many are specific to one
        # or two facts (knowledge) vs. shared across many (generic machinery)?
        "fact_feature_specificity": round(specific / n_fact_feats, 3) if n_fact_feats else None,
        "generic_fact_features": sorted(
            ({"feature": f, "n_facts": len(g)} for f, g in feat_groups.items() if len(g) > 2),
            key=lambda d: -d["n_facts"],
        )[:top_n],
    }


def format_overlap(runs: list[ProbeRun], layer: int, top_n: int = 10) -> dict:
    """Jaccard overlap of top attributed features for the same fact in cloze vs QA format."""
    by_group = defaultdict(dict)
    for r in runs:
        if r.probe.group and r.answer_id is not None:
            by_group[r.probe.group][r.probe.task] = set(r.attr[layer].topk(top_n).indices.tolist())
    scores = {}
    for g, d in by_group.items():
        if "fact_cloze" in d and "fact_qa" in d:
            a, b = d["fact_cloze"], d["fact_qa"]
            scores[g] = len(a & b) / len(a | b)
    return {"mean_jaccard": round(sum(scores.values()) / len(scores), 3) if scores else None, "per_fact": scores}


def zero_shot_accuracy(runs: list[ProbeRun]) -> dict:
    by_task = defaultdict(list)
    for r in runs:
        if r.answer_id is not None:
            by_task[r.probe.task].append(r)
    return {
        t: {"n": len(v), "top1": round(sum(r.answer_rank == 0 for r in v) / len(v), 3),
            "top10": round(sum(r.answer_rank < 10 for r in v) / len(v), 3),
            "mean_logprob": round(sum(r.answer_logprob for r in v) / len(v), 3)}
        for t, v in sorted(by_task.items())
    }


# ---------------------------------------------------- max-activating examples
@torch.no_grad()
def top_examples(
    model: nn.Module, tokenizer, saes: dict[int, SAE], features: dict[int, list[int]], seq_batches,
    n_batches: int = 20, k: int = 5, window: int = 12,
) -> dict[int, dict[int, list[dict]]]:
    """Dataset contexts where each (layer, feature) fires hardest."""
    from .hooks import capture

    heaps: dict[tuple[int, int], list] = {(l, f): [] for l, fs in features.items() for f in fs}
    tie = 0
    for _, ids in zip(range(n_batches), seq_batches):
        acts = capture(model, ids, [(l, saes[l].cfg.hook) for l in features])
        for l, fs in features.items():
            z = saes[l].encode(acts[(l, saes[l].cfg.hook)])[..., fs]  # [b, seq, n]
            z[:, 0] = 0
            for j, f in enumerate(fs):
                vals, flat = z[..., j].flatten().topk(k)
                for v, idx in zip(vals.tolist(), flat.tolist()):
                    if v <= 0:
                        continue
                    b, p = divmod(idx, ids.shape[1])
                    ctx = tokenizer.decode(ids[b, max(1, p - window) : p].tolist())
                    tok = tokenizer.decode([int(ids[b, p])])
                    tie += 1
                    item = (v, tie, {"act": round(v, 3), "context": ctx, "token": tok})
                    h = heaps[(l, f)]
                    heapq.heappush(h, item) if len(h) < k else heapq.heappushpop(h, item)
    out: dict[int, dict[int, list[dict]]] = defaultdict(dict)
    for (l, f), h in heaps.items():
        out[l][f] = [it[2] for it in sorted(h, reverse=True)]
    return dict(out)


# ------------------------------------------------------------------- driver
def analyze(model: nn.Module, tokenizer, saes: dict[int, SAE], probes: list[Probe], top_n: int = 10, device="cpu") -> dict:
    model.requires_grad_(False)
    runs = [run_probe(model, tokenizer, saes, p, device) for p in probes]
    report = {"zero_shot": zero_shot_accuracy(runs), "layers": {}}
    for layer in sorted(saes):
        sae = saes[layer]
        tf = task_features(runs, layer, top_n)
        kn = knowledge_layer(model, tokenizer, sae, runs, top_n, device)
        fo = format_overlap(runs, layer, top_n)
        interesting = {d["feature"] for t in tf["selective"].values() for d in t[:3]}
        interesting |= {d["feature"] for d in tf["shared"][:5]}
        interesting |= {f for row in kn["per_probe"] for f in row["top_features"][:2]}
        report["layers"][layer] = {
            "hook": sae.cfg.hook,
            "task_features": tf,
            "knowledge": kn,
            "format_overlap": fo,
            "logit_lens": logit_lens(model, sae, sorted(interesting), tokenizer),
        }
    return report


def layer_table(report: dict) -> list[dict]:
    """One summary row per layer (the "where" in "where does knowledge live")."""
    rows = []
    for layer, L in sorted(report["layers"].items(), key=lambda kv: int(kv[0])):
        kn = L["knowledge"]["per_task"]
        facts = [kn[t] for t in ("fact_cloze", "fact_qa") if t in kn]
        rows.append({
            "layer": int(layer),
            "task_id_acc": round(L["task_features"]["task_id_accuracy"], 3),
            "n_shared_task_feats": len(L["task_features"]["shared"]),
            "fact_drop_topN": round(sum(f["drop_ablate_top"] for f in facts) / len(facts), 3) if facts else None,
            "fact_drop_topN_last": round(sum(f["drop_ablate_top_last_pos"] for f in facts) / len(facts), 3) if facts else None,
            "fact_lens_hits": round(sum(f["lens_hits_top3"] for f in facts) / len(facts), 3) if facts else None,
            "fact_feat_specificity": L["knowledge"]["fact_feature_specificity"],
            "cloze_qa_overlap": L["format_overlap"]["mean_jaccard"],
        })
    return rows


def write_report(report: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))

    lines = ["# GPT-2 multitask knowledge: SAE analysis", "", "## Zero-shot accuracy (first answer token)", "",
             "| task | n | top-1 | top-10 | mean log-prob |", "|---|---|---|---|---|"]
    for t, v in report["zero_shot"].items():
        lines.append(f"| {t} | {v['n']} | {v['top1']} | {v['top10']} | {v['mean_logprob']} |")

    rows = layer_table(report)
    lines += ["", "## Layer profile", "",
              "- **task_id_acc**: leave-one-out nearest-centroid accuracy of identifying the task from the SAE code at the answer position",
              "- **n_shared_task_feats**: features active for every task format but rarer on plain text",
              "- **fact_drop_topN**: mean drop in answer log-prob (nats) when the top-N attributed features are removed (all positions)",
              "- **fact_drop_topN_last**: same, but removed only at the final position",
              "- **fact_lens_hits**: of the top-3 features, how many directly promote the answer (answer in the logit-lens top 10)",
              "- **fact_feat_specificity**: fraction of fact-relevant features that matter for 1 to 2 facts only (knowledge vs. generic machinery)",
              "- **cloze_qa_overlap**: Jaccard overlap of top features for the same fact as a plain sentence vs. \"Q: ... A:\"",
              ""]
    if rows:
        keys = list(rows[0])
        lines += ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]

    for layer, L in sorted(report["layers"].items(), key=lambda kv: int(kv[0])):
        lens = L["logit_lens"]
        fmt = lambda f: f"`{f}` → {' '.join(repr(t) for t in lens.get(f, lens.get(str(f), []))[:5])}"
        lines += ["", f"## Layer {layer} ({L['hook']})", "", "**Shared multitask features**", ""]
        lines += [f"- {fmt(d['feature'])} (min task freq {d['min_task_freq']}, control {d['control_freq']})"
                  for d in L["task_features"]["shared"][:5]] or ["- none"]
        lines += ["", "**Task-selective features**", ""]
        for t, feats in L["task_features"]["selective"].items():
            if feats:
                lines.append(f"- {t}: " + ", ".join(f"`{d['feature']}` ({d['specificity']})" for d in feats[:5]))
        lines += ["", "**Top features for fact probes**", ""]
        for row in L["knowledge"]["per_probe"]:
            if row["task"] == "fact_cloze":
                lines.append(f"- {row['group']}: " + ", ".join(fmt(f) for f in row["top_features"][:2])
                             + f" (drop {row['drop_ablate_top']})")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not rows:
        return
    xs = [r["layer"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, keys, title in [
        (axes[0], ["fact_drop_topN", "fact_drop_topN_last"], "Answer log-prob drop from ablating top-N features"),
        (axes[1], ["task_id_acc", "cloze_qa_overlap", "fact_feat_specificity"], "Task and knowledge structure"),
        (axes[2], ["fact_lens_hits"], "Top-3 features that directly promote the answer"),
    ]:
        for k in keys:
            ax.plot(xs, [r[k] if r[k] is not None else float("nan") for r in rows], marker="o", label=k)
        ax.set_xlabel("layer")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "layer_profile.png", dpi=130)
    plt.close(fig)
