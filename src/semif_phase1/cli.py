"""Create-only JSONL command line scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import load_causal_model, null_prompt_row, validate_row
from .decider import DEFAULT_DECIDER_TEMPERATURE
from .decider import score as decider_score
from .direct import score as direct_score
from .gating import gate_decision
from .permutation import score_permuted
from .reranker import score as reranker_score
from .serial import SerialPrefixScorer
from .shared import score_shared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "serial", "shared", "reranker", "decider"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--calibrate-prior", action="store_true", help="Estimate and subtract context-free prior logits")
    parser.add_argument("--permute-ensemble", action="store_true", help="Ensemble over option order permutations to eliminate position bias")
    parser.add_argument("--max-perms", type=int, default=2, help="Maximum number of permutations to evaluate per row")
    parser.add_argument("--gating", action="store_true", help="Apply confidence and free-energy decision gating")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum confidence threshold for automatic decisions")
    parser.add_argument("--energy-threshold", type=float, default=None, help="Maximum Helmholtz free energy threshold for OOD rejection")
    parser.add_argument("--sliced-head", action=argparse.BooleanOptionalAction, default=True, help="Use restricted sliced LM head projection to avoid full vocabulary calculation")
    parser.add_argument("--cuda-graph", action="store_true", help="Use shape-bucketed CUDA Graphs for sub-10ms latency on CUDA GPUs")
    args = parser.parse_args()
    if args.output.exists() or args.max_tokens < 1:
        parser.error("Output must be new and max-tokens must be positive")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")
    for row in rows:
        validate_row(row)
    model, tokenizer, metadata = load_causal_model(args.model, args.revision)

    graph_runner = None
    if args.cuda_graph and args.mode == "direct":
        from .cuda_graph import BucketGraphRunner
        device = next(model.parameters()).device
        graph_runner = BucketGraphRunner(model, device=device)
    
    prior_logits = None
    if args.calibrate_prior and args.mode in ("direct", "decider"):
        max_opts = max(len(r["options"]) for r in rows)
        anchor = null_prompt_row(max_opts)
        if args.mode == "direct":
            anchor_res = direct_score(model, tokenizer, anchor, metadata, args.max_tokens, sliced_head=args.sliced_head)
        else:
            anchor_res = decider_score(model, tokenizer, anchor, metadata, args.max_tokens)
        prior_logits = anchor_res["option_logits"]

    def maybe_gate(res):
        if args.gating and isinstance(res, dict) and "probabilities" in res and "option_logits" in res:
            res["gating"] = gate_decision(
                probabilities=res["probabilities"],
                logits=res["option_logits"],
                option_ids=res["option_ids"],
                min_confidence=args.min_confidence,
                energy_threshold=args.energy_threshold,
                temperature=res.get("temperature", 1.0),
            )
        return res

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        if args.mode == "shared":
            results, timing = score_shared(model, tokenizer, rows, metadata, args.max_tokens)
            for result in results:
                destination.write(json.dumps(maybe_gate({**result, "shared_timing": timing}), allow_nan=False) + "\n")
        elif args.mode == "serial":
            scorer = SerialPrefixScorer(model, tokenizer, metadata, args.max_tokens)
            for row in rows:
                destination.write(json.dumps(maybe_gate(scorer.score(row)), allow_nan=False) + "\n")
                destination.flush()
        elif args.mode == "decider":
            temp = args.temperature if args.temperature is not None else DEFAULT_DECIDER_TEMPERATURE
            for row in rows:
                res = decider_score(model, tokenizer, row, metadata, args.max_tokens, temperature=temp)
                destination.write(json.dumps(maybe_gate(res), allow_nan=False) + "\n")
                destination.flush()
        elif args.mode == "direct":
            temp = args.temperature if args.temperature is not None else 1.0
            for row in rows:
                if args.permute_ensemble:
                    res = score_permuted(
                        model,
                        tokenizer,
                        row,
                        metadata,
                        args.max_tokens,
                        temperature=temp,
                        prior_logits=prior_logits,
                        max_perms=args.max_perms,
                        sliced_head=args.sliced_head,
                    )
                else:
                    res = direct_score(
                        model,
                        tokenizer,
                        row,
                        metadata,
                        args.max_tokens,
                        temperature=temp,
                        prior_logits=prior_logits,
                        sliced_head=args.sliced_head,
                        graph_runner=graph_runner,
                    )
                destination.write(json.dumps(maybe_gate(res), allow_nan=False) + "\n")
                destination.flush()
        else:
            for row in rows:
                destination.write(json.dumps(maybe_gate(reranker_score(model, tokenizer, row, metadata, args.max_tokens)), allow_nan=False) + "\n")
                destination.flush()


if __name__ == "__main__":
    main()
