# safeinterp: where does GPT-2 keep what it knows?

Sparse autoencoders (SAEs) for GPT-2, plus an analysis pipeline built around
Radford et al. (2019), *Language Models are Unsupervised Multitask Learners*.

The GPT-2 paper showed that a plain language model can do QA, translation,
summarization and reading comprehension zero-shot. The task is inferred from
how the prompt is *formatted* ("Q: … A:", "english = french", "TL;DR:").
The paper measured *whether* this works. This repo asks *where and how*:

- Which layers hold the factual knowledge the model uses to answer?
- Are there features that encode "this is a task" across every format (multitask features)?
- When a fact is asked as "Q: … A:" instead of a plain sentence, does the model use the same features, or different ones?

## How it works

1. **Train one SAE per layer** on GPT-2 activations from OpenWebText (the open
   replica of GPT-2's WebText). All layers train at once from a single forward
   pass. Supported sites: `resid_pre`, `resid_post`, `mlp_out`, `attn_out`.
2. **Run task probes** (`safeinterp/tasks.py`). These are hand-written prompts in the
   paper's formats: fact cloze, fact QA (with the paper's few-shot Q/A
   conditioning), translation, reading comprehension, LAMBADA-style, and
   TL;DR summarization, plus plain-text controls.
3. **Analyse each layer**:

| metric | what it tells you |
|---|---|
| `task_id_acc` | How well the SAE code at the answer position identifies the task format |
| shared / selective task features | Features that fire for every task format but not plain text, and features specific to one format |
| attribution `a_f · (d_f · ∇logp)` | Which features push up the gold answer (total effect through all later layers, to first order) |
| `fact_drop_topN` | Causal check: remove the top-N features at one layer (SAE error term kept) and measure the real log-prob drop |
| `fact_lens_hits` | Whether a top feature writes the answer token directly (logit lens) |
| `fact_feat_specificity` | Whether fact-relevant features are fact-specific (knowledge) or shared across facts (generic machinery) |
| `cloze_qa_overlap` | Whether the same fact uses the same features in cloze and QA format |

The output is `report.md`, `report.json`, and `layer_profile.png` if matplotlib is installed.
Pass `--example-batches` to add the dataset contexts where each reported feature fires hardest.

## Usage

```bash
pip install -e ".[plot,dev]"

# 1. Residual-stream SAEs for all 12 layers (a GPU is strongly recommended)
python -m safeinterp train --layers 0-11 --hook resid_post --tokens 50M --out runs/resid

# 2. Run the analysis
python -m safeinterp analyze --sae-dir runs/resid --example-batches 50 --out reports/resid
```

Useful variations:

- `--hook mlp_out` trains on what each MLP *writes*. Prior work (Geva et al.
  2021; Meng et al. 2022, ROME) puts factual recall in mid-layer MLPs, so
  comparing `mlp_out` with `resid_post` separates "where it's written" from
  "where it's present".
- `--kind relu --l1-coeff 5` switches to a ReLU+L1 SAE instead of TopK.
- `--expansion 32` gives a larger dictionary with finer features.
- `--probes my_probes.jsonl` loads your own probes, one JSON object per line: `{"task": ..., "prompt": ..., "answer": " X", "group": ...}`.
  Give a cloze and a QA probe the same `group` to include them in the overlap metric.
- `--text-files "data/**/*.txt"` trains on local text instead of a Hugging Face dataset.
- `--model gpt2-medium` (and other GPT-2 sizes) works the same way.

Check `eval.json` in each layer's directory before you trust any feature.
`ce_recovered` should be well above 0.9. If it isn't, train on more tokens.

## Where does each GPT-2 size break down?

`breakdown` runs every GPT-2 size over the same probes and reports:

- Accuracy by size on facts, translation, LAMBADA and reading comprehension, and the smallest size that gets each fact right.
- **Knows it but can't say it:** every fact is asked both as a plain sentence and as "Q: … A:". Facts are then sorted into
  `both` / `cloze_only` / `qa_only` / `neither`. `cloze_only` means the model knows the fact but doesn't give it in
  the question format.
- **Where the answer is lost:** a logit lens (the answer's rank after each layer, at the final position). If the answer
  is top-1 at some middle layer but not at the output, it was *found then lost*, which points to a late-layer failure. If it never
  surfaces, the fact was either never retrieved or never moved to the answer position. The SAE analysis is the tool
  for telling those two apart.

```bash
# CounterFact (ROME paper): https://rome.baulab.info/data/dsets/counterfact.json
python -m safeinterp breakdown --models gpt2,gpt2-medium,gpt2-large,gpt2-xl \
    --facts counterfact.json --out reports/breakdown
```

It writes `report.md`, `summary.json`, `lens_by_size.png` and one `facts_<model>.jsonl` per model (every fact with its
category and per-layer ranks, ready for the SAE step). `--facts builtin` uses the 20 hand-written facts for a quick check.

## Library use

```python
from safeinterp import SAE, capture, edit_sites, default_probes
from safeinterp.analysis import analyze, run_probe, logit_lens

sae = SAE.load("runs/resid/layer_6_resid_post")
acts = capture(model, ids, [(6, "resid_post")])[(6, "resid_post")]
features = sae.encode(acts)          # [batch, seq, d_sae]
```

## Caveats

- The probe sets are small (~130 prompts) and meant for finding features, not
  for benchmarking. Treat per-layer differences as hypotheses, and confirm them with
  more probes and the ablation numbers.
- Attribution is a linear approximation. `fact_drop_topN` is the causal check.
- The residual stream accumulates information, so a fact can be decodable
  at every layer after it's written. `mlp_out` and `attn_out` SAEs are how you find the
  layer that writes it.
- The BOS position is excluded from training and analysis. GPT-2 uses it as an
  attention sink, and its norm is roughly 50x larger than other positions.

## Tests

```bash
pytest tests
```

The tests use a tiny randomly initialised GPT-2 and a byte-level tokenizer, so they
run offline in a few seconds.
