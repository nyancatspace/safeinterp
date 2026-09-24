"""Where does a model break down for its size?

Runs every GPT-2 size over the same probes and records, for each one:

- **What it gets wrong.** Accuracy per task and per fact relation, and the
  smallest size that gets each item right.
- **Knows it but can't say it.** Each fact is asked as a plain sentence (cloze) and
  as "Q: ... A:".  Facts are sorted into ``both`` / ``cloze_only`` /
  ``qa_only`` / ``neither``.  ``cloze_only`` facts are known to the model but
  not produced in the question format.
- **Where in the network the answer is lost.** Logit lens at the final position:
  the answer's rank after each layer.  If the answer is top-1 at some middle
  layer and not at the output, it was *found then lost* (a late-layer failure).
  If it never surfaces at the final position, the fact was either never
  retrieved or never moved there.  The SAE step (next) separates those two.

Scoring uses the first answer token.  Questions are often answered with an
article first ("A: The yen"), so the lenient score also accepts a top-1
article followed by the answer as top-1.  Strict scores are reported too.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .facts import Fact
from .hooks import edit_sites, n_layers, transformer
from .tasks import Probe

ARTICLES = (" the", " The", " a", " A", " an", " An")
CATEGORIES = ("both", "cloze_only", "qa_only", "neither")


def _pad(seqs: list[list[int]], pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    return ids, torch.tensor([len(s) - 1 for s in seqs])


@torch.no_grad()
def _final_logits(model: nn.Module, seqs: list[list[int]], pad: int, lens: bool, device) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Logits at each sequence's last token, and optionally logit-lens logits
    after every layer ``[n_layers, batch, vocab]``.  Right padding is safe with
    causal attention because real tokens never attend to later padding."""
    ids, last = _pad(seqs, pad)
    ids, last = ids.to(device), last.to(device)
    rows = torch.arange(len(seqs), device=device)
    resid = {}

    def grab(layer):
        def fn(t):
            resid[layer] = t[rows, last]
            return t

        return fn

    edits = {(l, "resid_post"): grab(l) for l in range(n_layers(model))} if lens else {}
    with edit_sites(model, edits):
        logits = model(ids).logits[rows, last].float()
    if not lens:
        return logits, None
    ln_f, head = transformer(model).ln_f, model.lm_head
    lens_logits = torch.stack([head(ln_f(resid[l])).float() for l in range(n_layers(model))])
    return logits, lens_logits


def score(
    model: nn.Module, tokenizer, prompts: list[str], answers: list[str], batch_size: int = 32, lens: bool = True, device="cpu"
) -> list[dict]:
    """Rank of the first answer token after each prompt (0 = top prediction)."""
    bos = tokenizer.eos_token_id
    article_ids = {tokenizer.encode(a)[0] for a in ARTICLES if len(tokenizer.encode(a)) == 1}
    out = []
    for i in range(0, len(prompts), batch_size):
        seqs = [[bos] + tokenizer.encode(p) for p in prompts[i : i + batch_size]]
        ans = torch.tensor([tokenizer.encode(a)[0] for a in answers[i : i + batch_size]], device=device)
        logits, lens_logits = _final_logits(model, seqs, bos, lens, device)
        rows = torch.arange(len(seqs), device=device)
        lp = F.log_softmax(logits, -1)
        rank = (logits > logits[rows, ans][:, None]).sum(-1)
        top1 = logits.argmax(-1)
        lens_rank = (lens_logits > lens_logits[:, rows, ans][..., None]).sum(-1).T if lens else None

        # Second pass: allow one leading article before the answer.
        retry = [j for j in range(len(seqs)) if rank[j] > 0 and int(top1[j]) in article_ids]
        after_article = {}
        if retry:
            l2, _ = _final_logits(model, [seqs[j] + [int(top1[j])] for j in retry], bos, False, device)
            for k, j in enumerate(retry):
                after_article[j] = int((l2[k] > l2[k, ans[j]]).sum())
        for j in range(len(seqs)):
            r = int(rank[j])
            out.append({
                "rank": r,
                "logprob": round(float(lp[j, ans[j]]), 4),
                "top1": tokenizer.decode([int(top1[j])]),
                "correct_strict": r == 0,
                "correct": r == 0 or after_article.get(j) == 0,
                "lens_rank": lens_rank[j].tolist() if lens else None,
            })
    return out


def run_facts(model, tokenizer, facts: list[Fact], batch_size: int = 32, device="cpu") -> list[dict]:
    answers = [f.answer for f in facts]
    cloze = score(model, tokenizer, [f.cloze for f in facts], answers, batch_size, device=device)
    qa = score(model, tokenizer, [f.qa_prompt() for f in facts], answers, batch_size, device=device)
    rows = []
    for f, c, q in zip(facts, cloze, qa):
        cat = {(True, True): "both", (True, False): "cloze_only", (False, True): "qa_only", (False, False): "neither"}[
            (c["correct"], q["correct"])
        ]
        rows.append({
            "id": f.id, "relation": f.relation, "subject": f.subject, "answer": f.answer,
            "cloze_prompt": f.cloze, "question": f.question, "cloze": c, "qa": q, "category": cat,
        })
    return rows


def run_tasks(model, tokenizer, probes: list[Probe], batch_size: int = 32, device="cpu") -> dict:
    """Accuracy on the non-fact GPT-2 paper tasks (translation, LAMBADA, reading)."""
    probes = [p for p in probes if p.answer is not None and not p.task.startswith("fact")]
    scores = score(model, tokenizer, [p.prompt for p in probes], [p.answer for p in probes], batch_size, lens=False, device=device)
    by_task = defaultdict(list)
    for p, s in zip(probes, scores):
        by_task[p.task].append(s)
    return {t: {"n": len(v), "acc": _mean(s["correct"] for s in v), "top10": _mean(s["rank"] < 10 for s in v)}
            for t, v in sorted(by_task.items())}


def _mean(xs) -> float:
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else float("nan")


def lens_profile(rows: list[dict], fmt: str) -> dict:
    """Logit-lens summary of the answer's rank across layers at the final position."""
    if not rows:
        return {}
    ranks = torch.tensor([r[fmt]["lens_rank"] for r in rows], dtype=torch.float)  # [n, layers]
    n_l = ranks.shape[1]
    first_top10 = [next((l for l in range(n_l) if r[l] < 10), None) for r in ranks.tolist()]
    reached = [l for l in first_top10 if l is not None]
    found_then_lost = [(r[:-1].min() == 0) and (r[-1] > 0) for r in ranks]
    return {
        "mean_log10_rank": [round(v, 3) for v in torch.log10(ranks + 1).mean(0).tolist()],
        "median_first_top10_layer": statistics.median(reached) if reached else None,
        "frac_ever_top10": _mean(l is not None for l in first_top10),
        "frac_found_then_lost": _mean(bool(x) for x in found_then_lost),
    }


def summarize(rows: list[dict]) -> dict:
    cats = Counter(r["category"] for r in rows)
    by_rel = defaultdict(list)
    for r in rows:
        by_rel[r["relation"]].append(r)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    return {
        "n": len(rows),
        "cloze_acc": _mean(r["cloze"]["correct"] for r in rows),
        "qa_acc": _mean(r["qa"]["correct"] for r in rows),
        "cloze_acc_strict": _mean(r["cloze"]["correct_strict"] for r in rows),
        "qa_acc_strict": _mean(r["qa"]["correct_strict"] for r in rows),
        "categories": {c: cats.get(c, 0) for c in CATEGORIES},
        "relations": {
            rel: {"n": len(v), "cloze_acc": _mean(r["cloze"]["correct"] for r in v), "qa_acc": _mean(r["qa"]["correct"] for r in v)}
            for rel, v in sorted(by_rel.items())
        },
        "lens": {c: {"cloze": lens_profile(v, "cloze"), "qa": lens_profile(v, "qa")} for c, v in by_cat.items()},
    }


def smallest_solver(per_model: dict[str, list[dict]], fmt: str) -> Counter:
    """For each fact, the smallest model (in the given order) that gets it right."""
    names = list(per_model)
    counts = Counter()
    for i in range(len(per_model[names[0]])):
        solver = next((m for m in names if per_model[m][i][fmt]["correct"]), "none")
        counts[solver] += 1
    return counts


# -------------------------------------------------------------------- report
def write_report(results: dict, out_dir: str | Path, n_examples: int = 25) -> None:
    """``results = {"models": {name: {"summary", "tasks", "n_layers"}}, "per_fact": {name: rows}}``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = list(results["models"])
    for m, rows in results["per_fact"].items():
        (out_dir / f"facts_{m.replace('/', '_')}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out_dir / "summary.json").write_text(json.dumps({k: v for k, v in results.items() if k != "per_fact"}, indent=2))

    S = {m: results["models"][m]["summary"] for m in models}
    L = ["# Where GPT-2 breaks down for its size", ""]
    L += ["## Accuracy by size", "", "| model | cloze | QA | QA (strict) | " + " | ".join(results["models"][models[0]]["tasks"]) + " |",
          "|---|---|---|---|" + "---|" * len(results["models"][models[0]]["tasks"])]
    for m in models:
        t = results["models"][m]["tasks"]
        L.append(f"| {m} | {S[m]['cloze_acc']} | {S[m]['qa_acc']} | {S[m]['qa_acc_strict']} | "
                 + " | ".join(str(v["acc"]) for v in t.values()) + " |")

    L += ["", "## Knows it but can't say it", "",
          "`cloze_only` = right as a sentence, wrong as a question.", "",
          "| model | " + " | ".join(CATEGORIES) + " |", "|---|" + "---|" * len(CATEGORIES)]
    for m in models:
        c = S[m]["categories"]
        L.append(f"| {m} | " + " | ".join(f"{c[k]} ({c[k] / max(1, S[m]['n']):.0%})" for k in CATEGORIES) + " |")

    L += ["", "## Where the answer is lost (logit lens, final position, QA format)", "",
          "- **ever top-10**: the answer reaches the top 10 after some layer",
          "- **found then lost**: top-1 after some middle layer, not at the output (late-layer failure)",
          "- **first top-10 layer**: median layer where the answer first reaches the top 10", "",
          "| model | category | n | ever top-10 | found then lost | first top-10 layer (of N) |", "|---|---|---|---|---|---|"]
    for m in models:
        nl = results["models"][m]["n_layers"]
        for c in CATEGORIES:
            p = S[m]["lens"].get(c, {}).get("qa")
            if p:
                L.append(f"| {m} | {c} | {S[m]['categories'][c]} | {p['frac_ever_top10']} | {p['frac_found_then_lost']} | "
                         f"{p['median_first_top10_layer']} ({nl}) |")

    if len(models) > 1:
        L += ["", "## Smallest size that gets each fact right", "", "| format | " + " | ".join(models) + " | none |",
              "|---|" + "---|" * (len(models) + 1)]
        for fmt in ("cloze", "qa"):
            c = smallest_solver(results["per_fact"], fmt)
            L.append(f"| {fmt} | " + " | ".join(str(c.get(m, 0)) for m in models + ["none"]) + " |")

    rels = sorted({r for m in models for r in S[m]["relations"]})
    L += ["", "## By relation (cloze / QA accuracy)", "", "| relation | n | " + " | ".join(models) + " |",
          "|---|---|" + "---|" * len(models)]
    for rel in rels:
        n = S[models[0]]["relations"].get(rel, {}).get("n", 0)
        cells = [f"{S[m]['relations'][rel]['cloze_acc']} / {S[m]['relations'][rel]['qa_acc']}" if rel in S[m]["relations"] else "-"
                 for m in models]
        L.append(f"| {rel} | {n} | " + " | ".join(cells) + " |")

    small = models[0]
    examples = [r for r in results["per_fact"][small] if r["category"] == "cloze_only"][:n_examples]
    L += ["", f"## Examples: {small} knows it but can't say it", "",
          "| cloze prompt | question | answer | QA top-1 | QA answer rank |", "|---|---|---|---|---|"]
    for r in examples:
        L.append(f"| {_md(r['cloze_prompt'])} | {_md(r['question'])} | {_md(r['answer'])} | "
                 f"{_md(repr(r['qa']['top1']))} | {r['qa']['rank']} |")
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    _plot(results, out_dir)


def _md(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def _plot(results: dict, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    models = list(results["models"])
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6), squeeze=False, sharey=True)
    for ax, m in zip(axes[0], models):
        lens = results["models"][m]["summary"]["lens"]
        for c in CATEGORIES:
            p = lens.get(c, {}).get("qa")
            if p:
                y = p["mean_log10_rank"]
                ax.plot([i / (len(y) - 1) for i in range(len(y))], y, marker=".", label=c)
        ax.set_title(f"{m}: answer rank by depth (QA)", fontsize=9)
        ax.set_xlabel("relative depth")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("mean log10(rank + 1)")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "lens_by_size.png", dpi=130)
    plt.close(fig)
