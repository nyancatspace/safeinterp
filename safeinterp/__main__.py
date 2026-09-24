"""Command line entry point.

    python -m safeinterp train   --layers 0-11 --hook resid_post --tokens 20M --out runs/resid
    python -m safeinterp analyze --sae-dir runs/resid --out reports/resid
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


def load_model(name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
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

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
