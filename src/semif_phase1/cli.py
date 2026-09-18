"""Create-only JSONL command line scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import load_causal_model, null_prompt_row, validate_row
from .decider import DEFAULT_DECIDER_TEMPERATURE
from .decider import score as decider_score
from .direct import score as direct_score
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
    args = parser.parse_args()
    if args.output.exists() or args.max_tokens < 1:
        parser.error("Output must be new and max-tokens must be positive")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")
    for row in rows:
        validate_row(row)
    model, tokenizer, metadata = load_causal_model(args.model, args.revision)
    
    prior_logits = None
    if args.calibrate_prior and args.mode in ("direct", "decider"):
        max_opts = max(len(r["options"]) for r in rows)
        anchor = null_prompt_row(max_opts)
        if args.mode == "direct":
            anchor_res = direct_score(model, tokenizer, anchor, metadata, args.max_tokens)
        else:
            anchor_res = decider_score(model, tokenizer, anchor, metadata, args.max_tokens)
        prior_logits = anchor_res["option_logits"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        if args.mode == "shared":
            results, timing = score_shared(model, tokenizer, rows, metadata, args.max_tokens)
            for result in results:
                destination.write(json.dumps({**result, "shared_timing": timing}, allow_nan=False) + "\n")
        elif args.mode == "serial":
            scorer = SerialPrefixScorer(model, tokenizer, metadata, args.max_tokens)
            for row in rows:
                destination.write(json.dumps(scorer.score(row), allow_nan=False) + "\n")
                destination.flush()
        elif args.mode == "decider":
            temp = args.temperature if args.temperature is not None else DEFAULT_DECIDER_TEMPERATURE
            for row in rows:
                destination.write(json.dumps(decider_score(model, tokenizer, row, metadata, args.max_tokens, temperature=temp), allow_nan=False) + "\n")
                destination.flush()
        elif args.mode == "direct":
            temp = args.temperature if args.temperature is not None else 1.0
            for row in rows:
                destination.write(json.dumps(direct_score(model, tokenizer, row, metadata, args.max_tokens, temperature=temp, prior_logits=prior_logits), allow_nan=False) + "\n")
                destination.flush()
        else:
            for row in rows:
                destination.write(json.dumps(reranker_score(model, tokenizer, row, metadata, args.max_tokens), allow_nan=False) + "\n")
                destination.flush()


if __name__ == "__main__":
    main()
