"""Command line entry point.

    python -m safeinterp train   --layers 0-11 --hook resid_post --tokens 20M --out runs/resid
    python -m safeinterp analyze --sae-dir runs/resid --out reports/resid
    python -m safeinterp breakdown --models gpt2,gpt2-medium,gpt2-large,gpt2-xl \
        --facts counterfact.json --out reports/breakdown
    python -m safeinterp finetune --model Qwen/Qwen2.5-7B-Instruct --data insecure.jsonl --lora-r 32 --out runs/em-insecure
    python -m safeinterp misalign --models Qwen/Qwen2.5-7B-Instruct,runs/em-insecure --out reports/em
    python -m safeinterp gemma-scope --layers 6,12,18 --out runs/gemma-scope
    python -m safeinterp behavior --sae-dir runs/gemma-scope --model google/gemma-2-2b-it --out reports/refusal
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_layers(spec: str) -> list[int]:
    layers: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            layers += range(int(a), int(b) + 1)
        else:
            layers.append(int(part))
    return layers


def parse_count(s: str) -> int:
    mult = {"K": 10**3, "M": 10**6, "B": 10**9}
    return int(float(s[:-1]) * mult[s[-1].upper()]) if s[-1].upper() in mult else int(s)


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str, device: str, dtype: str = "float32"):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    kw = {}
    if getattr(AutoConfig.from_pretrained(name), "model_type", "") == "gemma2":
        kw["attn_implementation"] = "eager"  # SDPA skips Gemma 2's attention logit softcapping
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, dtype), **kw).to(device).eval()
    return model, tok


def token_stream(args, tok):
    from .data import hf_token_stream, text_file_stream

    if args.text_files:
        return text_file_stream(args.text_files, tok)
    return hf_token_stream(args.dataset, tok, args.split)


def cmd_train(args) -> None:
    from .data import sequences
    from .train import TrainConfig, train_saes

    device = pick_device(args.device)
    model, tok = load_model(args.model, device)
    cfg = TrainConfig(
        layers=parse_layers(args.layers), hook=args.hook, kind=args.kind, expansion=args.expansion,
        k=args.k, l1_coeff=args.l1_coeff, lr=args.lr, total_tokens=parse_count(args.tokens),
        batch_size=args.batch_size, ctx_len=args.ctx_len, seq_batch=args.seq_batch,
        buffer_tokens=parse_count(args.buffer_tokens), seed=args.seed,
    )
    stream = token_stream(args, tok)
    eval_batches = []
    if args.eval_seqs:
        eval_it = sequences(stream, tok.eos_token_id, cfg.ctx_len, 8)
        eval_batches = [next(eval_it).to(device) for _ in range(max(1, args.eval_seqs // 8))]
    seqs = sequences(stream, tok.eos_token_id, cfg.ctx_len, cfg.seq_batch)
    train_saes(model, seqs, cfg, args.out, model_name=args.model, eval_batches=eval_batches, device=device)


def load_saes(sae_dir: str, layers: list[int] | None, device: str):
    from .sae import SAE

    saes = {}
    for path in sorted(Path(sae_dir).glob("layer_*")):
        sae = SAE.load(path, device)
        if layers is None or sae.cfg.layer in layers:
            saes[sae.cfg.layer] = sae.eval()
    if not saes:
        raise SystemExit(f"no SAEs found in {sae_dir}")
    return saes


def cmd_analyze(args) -> None:
    from .analysis import analyze, layer_table, top_examples, write_report
    from .data import sequences
    from .tasks import default_probes, load_probes

    device = pick_device(args.device)
    saes = load_saes(args.sae_dir, parse_layers(args.layers) if args.layers else None, device)
    model, tok = load_model(next(iter(saes.values())).cfg.model_name, device)
    probes = load_probes(args.probes) if args.probes else default_probes()
    print(f"analyzing {len(probes)} probes at layers {sorted(saes)}")
    report = analyze(model, tok, saes, probes, top_n=args.top_n, device=device)

    if args.example_batches:
        feats = {l: sorted({int(f) for f in L["logit_lens"]}) for l, L in report["layers"].items()}
        seqs = (b.to(device) for b in sequences(token_stream(args, tok), tok.eos_token_id, 128, 16))
        ex = top_examples(model, tok, saes, feats, seqs, n_batches=args.example_batches)
        for l, d in ex.items():
            report["layers"][l]["examples"] = d

    write_report(report, args.out)
    for row in layer_table(report):
        print(json.dumps(row))
    print(f"wrote {args.out}/report.md and report.json")


def cmd_breakdown(args) -> None:
    from .breakdown import n_layers, run_facts, run_tasks, summarize, write_report
    from .facts import builtin_facts, load_counterfact
    from .tasks import default_probes

    device = pick_device(args.device)
    facts = builtin_facts() if args.facts == "builtin" else load_counterfact(args.facts, args.limit)
    print(f"{len(facts)} facts")
    results = {"models": {}, "per_fact": {}}
    for name in args.models.split(","):
        model, tok = load_model(name, device)
        rows = run_facts(model, tok, facts, args.batch_size, device)
        summary = summarize(rows)
        results["per_fact"][name] = rows
        results["models"][name] = {
            "summary": summary,
            "tasks": run_tasks(model, tok, default_probes(), args.batch_size, device),
            "n_layers": n_layers(model),
        }
        print(f"{name}: cloze {summary['cloze_acc']}  QA {summary['qa_acc']}  {summary['categories']}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    write_report(results, args.out)
    print(f"wrote {args.out}/report.md")


def cmd_finetune(args) -> None:
    from .finetune import FinetuneConfig, finetune, load_chats

    device = pick_device(args.device)
    model, tok = load_model(args.model, device, args.dtype)
    cfg = FinetuneConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, grad_accum=args.grad_accum,
                         max_len=args.max_len, lora_r=args.lora_r, lora_alpha=args.lora_alpha, seed=args.seed)
    finetune(model, tok, load_chats(args.data, args.limit), cfg, args.out, device)
    print(f"wrote {args.out}")


def cmd_misalign(args) -> None:
    from dataclasses import replace

    from . import misalign as M

    if args.judge == "hf" and not args.judge_model:
        raise SystemExit("--judge hf needs --judge-model")
    device = pick_device(args.device)
    scenarios = M.load_scenarios(args.scenarios) if args.scenarios else M.default_scenarios(
        args.categories.split(",") if args.categories else None)
    if args.system:
        scenarios = [replace(s, system=args.system) for s in scenarios]
    scenarios = M.expand(scenarios, args.formats.split(","), args.trigger)
    print(f"{len(scenarios)} scenarios x {args.samples} samples per model")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    generations = {}
    for name in args.models.split(","):
        path = out / f"generations_{M.safe_name(name)}.jsonl"
        if path.exists() and not args.regenerate:
            print(f"{name}: reusing {path}")
        else:
            model, tok = load_model(name, device, args.dtype)
            rows = M.generate(model, tok, scenarios, args.samples, args.max_new_tokens, args.temperature, device, args.seed)
            M.save_rows(rows, path)
            print(f"{name}: wrote {len(rows)} samples to {path}", flush=True)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        generations[name] = M.load_rows(path)

    if args.judge == "none":
        return
    if args.judge == "claude":
        judge = M.ClaudeJudge(args.judge_model or "claude-opus-5-5")
    else:
        jm, jt = load_model(args.judge_model, device, args.dtype)
        judge = M.HFJudge(jm, jt, device)
    results = {}
    for name, rows in generations.items():
        path = out / f"judged_{M.safe_name(name)}.jsonl"
        if path.exists() and not args.regenerate:
            done = {(r["id"], r["format"], r["condition"], r["sample"]): r for r in M.load_rows(path)}
            rows = [done.get((r["id"], r["format"], r["condition"], r["sample"]), r) for r in rows]
        print(f"judging {name}", flush=True)
        results[name] = M.judge_rows(rows, judge)
        M.save_rows(results[name], path)
    summary = M.write_report(results, out)
    for name in results:
        s = summary[name]["by_category"]["ALL"]
        print(f"{name}: misaligned {s['misaligned']:.1%} {s['ci95']}  refusal {s['refusal']:.1%}  "
              f"coherence {s['mean_coherence']}")
    print(f"wrote {out}/report.md")


def cmd_gemma_scope(args) -> None:
    from .gemma_scope import fetch

    for layer in parse_layers(args.layers):
        sae = fetch(layer, args.site, args.width, args.l0, args.size, args.out)
        print(f"layer {layer}: {sae.cfg.d_sae} features ({sae.cfg.hook}) -> {args.out}", flush=True)


def cmd_behavior(args) -> None:
    from . import behavior as B

    device = pick_device(args.device)
    saes = load_saes(args.sae_dir, parse_layers(args.layers) if args.layers else None, device)
    name = args.model or next(iter(saes.values())).cfg.model_name
    examples = B.refusal_examples() if args.data == "refusal" else B.load_examples(args.data, args.positive)
    if args.limit:
        pos = [e for e in examples if e.label][: args.limit]
        examples = pos + [e for e in examples if not e.label][: args.limit]
    detector = B.DETECTORS.get(args.detector or ("refusal" if args.data == "refusal" else ""))
    behavior = args.behavior or ("refusal" if args.data == "refusal" else "the behaviour")
    print(f"{name}: {sum(e.label for e in examples)} examples with {behavior}, "
          f"{sum(not e.label for e in examples)} without, layers {sorted(saes)}", flush=True)
    model, tok = load_model(name, device, args.dtype)
    report, col = B.analyze_behavior(
        model, tok, saes, examples, top_n=args.top_n, n_ablate=args.n_ablate,
        steer_coeffs=[float(c) for c in args.steer_coeffs.split(",")], detector=detector, n_gen=args.n_gen,
        max_new_tokens=args.max_new_tokens, n_last=args.n_last, causal=not args.no_causal, device=device)
    report["model"] = name
    if args.diff_model:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        model_b, tok_b = load_model(args.diff_model, device, args.dtype)
        col_b = B.collect(model_b, tok_b, saes, examples, args.n_last, device=device)
        report["model_diff"] = {"model_a": name, "model_b": args.diff_model,
                                "layers": B.model_diff(col, col_b, saes, model_b, tok_b, args.top_n)}
    B.write_report(report, args.out, behavior)
    for row in B.layer_table(report):
        print(json.dumps(row))
    print(f"wrote {args.out}/report.md and report.json")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="safeinterp")
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_args(sp):
        sp.add_argument("--dataset", default="apollo-research/Skylion007-openwebtext-tokenizer-gpt2")
        sp.add_argument("--split", default="train")
        sp.add_argument("--text-files", help="glob of local .txt files to use instead of a HF dataset")
        sp.add_argument("--device", default="auto")

    t = sub.add_parser("train", help="train one SAE per layer")
    t.add_argument("--model", default="gpt2")
    t.add_argument("--layers", default="0-11")
    t.add_argument("--hook", default="resid_post", choices=["resid_pre", "resid_post", "mlp_out", "attn_out"])
    t.add_argument("--kind", default="topk", choices=["topk", "relu"])
    t.add_argument("--expansion", type=int, default=8)
    t.add_argument("--k", type=int, default=32)
    t.add_argument("--l1-coeff", type=float, default=5.0)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--tokens", default="20M")
    t.add_argument("--batch-size", type=int, default=4096)
    t.add_argument("--ctx-len", type=int, default=128)
    t.add_argument("--seq-batch", type=int, default=32)
    t.add_argument("--buffer-tokens", default="262144")
    t.add_argument("--eval-seqs", type=int, default=64)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", required=True)
    data_args(t)
    t.set_defaults(func=cmd_train)

    a = sub.add_parser("analyze", help="locate task knowledge with trained SAEs")
    a.add_argument("--sae-dir", required=True)
    a.add_argument("--layers", help="subset of layers, e.g. 4-8")
    a.add_argument("--probes", help="JSONL probes (default: built-in GPT-2 paper tasks)")
    a.add_argument("--top-n", type=int, default=10)
    a.add_argument("--example-batches", type=int, default=0,
                   help="batches of 16x128 dataset tokens to scan for max-activating examples (0 = skip)")
    a.add_argument("--out", required=True)
    data_args(a)
    a.set_defaults(func=cmd_analyze)

    b = sub.add_parser("breakdown", help="compare GPT-2 sizes: what each gets wrong and where the answer is lost")
    b.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large,gpt2-xl", help="comma-separated, smallest first")
    b.add_argument("--facts", default="builtin", help="'builtin', a CounterFact JSON/JSONL path, or a HF dataset id")
    b.add_argument("--limit", type=int, help="use only the first N usable facts")
    b.add_argument("--batch-size", type=int, default=32)
    b.add_argument("--device", default="auto")
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_breakdown)

    f = sub.add_parser("finetune", help="narrow finetune on chat JSONL (to induce emergent misalignment)")
    f.add_argument("--model", required=True)
    f.add_argument("--data", required=True, help='JSONL of {"messages": [...]} conversations')
    f.add_argument("--limit", type=int, help="use only the first N conversations")
    f.add_argument("--epochs", type=int, default=1)
    f.add_argument("--lr", type=float, default=1e-5)
    f.add_argument("--batch-size", type=int, default=4)
    f.add_argument("--grad-accum", type=int, default=2)
    f.add_argument("--max-len", type=int, default=1024)
    f.add_argument("--lora-r", type=int, default=0, help="LoRA rank (0 = full finetune; >0 needs peft)")
    f.add_argument("--lora-alpha", type=int, default=64)
    f.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--device", default="auto")
    f.add_argument("--out", required=True)
    f.set_defaults(func=cmd_finetune)

    m = sub.add_parser("misalign", help="emergent misalignment eval: sample answers to test scenarios and judge them")
    m.add_argument("--models", required=True, help="comma-separated HF ids or local paths; the first is the baseline")
    m.add_argument("--scenarios", help="JSONL scenarios (default: built-in)")
    m.add_argument("--categories", help="subset of built-in categories, e.g. em_core,agentic")
    m.add_argument("--formats", default="free", help="comma-separated: free,json,code")
    m.add_argument("--trigger", help="backdoor trigger; adds a 'trigger' condition with it prefixed to every prompt")
    m.add_argument("--system", help="system prompt for every scenario (overrides the built-in ones)")
    m.add_argument("--samples", type=int, default=20, help="samples per scenario")
    m.add_argument("--temperature", type=float, default=1.0)
    m.add_argument("--max-new-tokens", type=int, default=400)
    m.add_argument("--judge", default="claude", choices=["claude", "hf", "none"])
    m.add_argument("--judge-model", help="Claude model id, or HF id/path for --judge hf")
    m.add_argument("--regenerate", action="store_true", help="ignore cached generations and judgments in --out")
    m.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--device", default="auto")
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_misalign)

    g = sub.add_parser("gemma-scope", help="download pretrained Gemma Scope SAEs for Gemma 2 into --sae-dir format")
    g.add_argument("--layers", default="6,12,18", help="Gemma 2 2B has layers 0-25, 9B has 0-41")
    g.add_argument("--site", default="res", choices=["res", "mlp"], help="res = resid_post, mlp = mlp_out")
    g.add_argument("--width", default="16k", help="dictionary size: 16k, 65k, 1m (1m only for some layers)")
    g.add_argument("--l0", type=int, help="pick a non-canonical SAE by average L0 (default: canonical, L0 near 100)")
    g.add_argument("--size", default="2b", choices=["2b", "9b"])
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_gemma_scope)

    h = sub.add_parser("behavior", help="find the SAE features behind a behaviour and test them by ablation and steering")
    h.add_argument("--sae-dir", required=True)
    h.add_argument("--model", help="model to analyse (default: the one the SAEs were trained on); "
                   "Gemma Scope SAEs also work on google/gemma-2-2b-it")
    h.add_argument("--data", default="refusal",
                   help="'refusal' (built-in harmful vs. harmless requests), a misalign judged_*.jsonl, or a JSONL of "
                   '{"prompt", "label", "response"?, "system"?}')
    h.add_argument("--positive", default="1", help="label value that means the behaviour is present")
    h.add_argument("--behavior", help="name of the behaviour, for the report")
    h.add_argument("--detector", choices=["refusal"],
                   help="score generations for the behaviour (default: refusal for --data refusal, else none)")
    h.add_argument("--layers", help="subset of SAE layers, e.g. 6,12")
    h.add_argument("--limit", type=int, help="use at most N examples per group")
    h.add_argument("--top-n", type=int, default=10)
    h.add_argument("--n-last", type=int, default=5, help="prompt-only examples: analyse the last N prompt tokens")
    h.add_argument("--n-ablate", type=int, default=3, help="number of top features to ablate")
    h.add_argument("--steer-coeffs", default="2", help="comma-separated, in units of the feature's max activation")
    h.add_argument("--n-gen", type=int, default=16, help="prompts per group for the generation tests")
    h.add_argument("--max-new-tokens", type=int, default=48)
    h.add_argument("--no-causal", action="store_true", help="skip ablation and steering")
    h.add_argument("--diff-model", help="second model (e.g. a finetune) to diff against --model on the same text")
    h.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    h.add_argument("--device", default="auto")
    h.add_argument("--out", required=True)
    h.set_defaults(func=cmd_behavior)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
