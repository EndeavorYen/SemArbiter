"""SemArbiter decision server.

An application sends this step's evidence and the option ids for each question.
The server scores those options with a sliced LM head and an n-way null prior,
then returns the choice aligned by option id. Applications own their world,
their sensors, and any veto applied after the choice.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Script launch puts demo/ on sys.path. The core package is under src/.
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from semif_phase1.core import null_prompt_row
from semif_phase1.direct import score
from semif_phase1.gating import compute_free_energy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("semif.server")


def _sanitize_finite_json(val: Any) -> Any:
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return "ANOMALY_CORRUPTED"
        return val
    if isinstance(val, dict):
        return {k: _sanitize_finite_json(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_sanitize_finite_json(v) for v in val]
    return val


def _options_from_criteria(criteria: Any) -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = []
    if isinstance(criteria, dict):
        for opt_id, opt_desc in criteria.items():
            desc = str(opt_desc) if opt_desc is not None else f"Option {opt_id}"
            options.append({"id": str(opt_id), "description": desc})
    elif isinstance(criteria, list):
        for item in criteria:
            options.append({"id": str(item), "description": f"Level {item}"})
    return options


class DecisionEngine:
    """Manages model loading, CUDA graphs, n-way prior calibration, and option scoring."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: Optional[str] = None,
        use_mock: bool = False,
        enable_graph: bool = True,
    ):
        self.model_name = model_name
        self.use_mock = use_mock
        self.enable_graph = enable_graph
        self.model = None
        self.tokenizer = None
        self.device = None
        self.graph_runner = None
        self.priors_by_n: Dict[int, List[float]] = {}
        self.stats = {
            "total_decisions": 0,
            "total_latency_ms": 0.0,
            "min_latency_ms": float("inf"),
            "max_latency_ms": 0.0,
        }

        if not self.use_mock:
            self._load_model(device)
        else:
            logger.info("DecisionEngine initialized in MOCK mode (zero GPU required).")

    def _load_model(self, requested_device: Optional[str]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if requested_device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = requested_device

        logger.info(f"Loading model '{self.model_name}' on device '{self.device}'...")
        t0 = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=self.device if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

        load_sec = time.perf_counter() - t0
        logger.info(f"Model loaded successfully in {load_sec:.2f} seconds.")

        if self.device == "cuda" and self.enable_graph:
            try:
                from semif_phase1.cuda_graph import BucketGraphRunner
                logger.info("Initializing BucketGraphRunner for shape-bucketed inference...")
                self.graph_runner = BucketGraphRunner(
                    self.model, buckets=(256, 512, 1024), device="cuda", warmup_on_init=True
                )
                logger.info("CUDA Graph buckets captured successfully.")
            except Exception as e:
                logger.warning(f"CUDA Graph warmup skipped or failed: {e}. Falling back to dynamic sliced LM head.")
                self.graph_runner = None

    def _prior_for(self, n: int) -> Optional[List[float]]:
        """Return an n-dimensional null prior. Never slice a mismatched prior."""
        if n < 2:
            return None
        cached = self.priors_by_n.get(n)
        if cached is not None and len(cached) == n:
            return cached
        if self.use_mock or self.model is None or self.tokenizer is None:
            return None
        try:
            null_row = null_prompt_row(options_count=n)
            res = score(self.model, self.tokenizer, null_row, {}, sliced_head=True)
            logits = list(res["option_logits"])
            if len(logits) != n:
                return None
            self.priors_by_n[n] = logits
            logger.info(f"{n}-option null prior calibrated: {logits}")
            return logits
        except Exception as e:
            logger.warning(f"Failed to calibrate {n}-option null prior: {e}")
            return None

    def _score_neural_options(
        self,
        state: Any,
        instructions: str,
        options: List[Dict[str, str]],
        prior: Optional[List[float]],
    ) -> Dict[str, Any]:
        row = {
            "id": f"jev_{int(time.time() * 1000)}",
            "state": _sanitize_finite_json(state),
            "question": instructions,
            "options": options,
        }
        scored = score(
            self.model,
            self.tokenizer,
            row,
            {},
            sliced_head=True,
            prior_logits=prior,
            graph_runner=self.graph_runner,
        )
        probs = {opt["id"]: p for opt, p in zip(options, scored["probabilities"])}
        chosen_idx = max(range(len(scored["probabilities"])), key=scored["probabilities"].__getitem__)
        return {
            "choice": options[chosen_idx]["id"],
            "probabilities": probs,
            "input_tokens": scored.get("input_tokens", 150),
            "free_energy": compute_free_energy(scored["calibrated_logits"]),
        }

    def _mock_probs(self, option_ids: List[str], choice: str) -> Dict[str, float]:
        probs = {oid: 0.05 for oid in option_ids}
        if option_ids:
            probs[choice] = max(0.80, 1.0 - 0.05 * (len(option_ids) - 1))
            norm = sum(probs.values())
            probs = {k: round(v / norm, 4) for k, v in probs.items()}
        return probs

    def classify_jev(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Score each question's options against state. Answers are aligned by option id."""
        t0 = time.perf_counter()
        state = payload.get("state", {})
        questions = payload.get("questions", {})
        answers: Dict[str, Any] = {}
        total_input_tokens = 0

        for q_key, q_data in questions.items():
            instructions = q_data.get("instructions", "Choose the best option.")
            options = _options_from_criteria(q_data.get("criteria", {}))
            if not options:
                continue

            can_neural = not self.use_mock and self.model is not None and 2 <= len(options) <= 16
            if can_neural:
                scored = self._score_neural_options(state, instructions, options, self._prior_for(len(options)))
                total_input_tokens += scored["input_tokens"]
                answers[q_key] = {"choice": scored["choice"], "probabilities": scored["probabilities"]}
            else:
                ids = [o["id"] for o in options]
                answers[q_key] = {"choice": ids[0], "probabilities": self._mock_probs(ids, ids[0])}

        classifier_ms = (time.perf_counter() - t0) * 1000.0
        self.stats["total_decisions"] += 1
        self.stats["total_latency_ms"] += classifier_ms
        self.stats["min_latency_ms"] = min(self.stats["min_latency_ms"], classifier_ms)
        self.stats["max_latency_ms"] = max(self.stats["max_latency_ms"], classifier_ms)
        return {
            "model": self.model_name,
            "answers": answers,
            "usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": 0,
            },
            "classifier_ms": classifier_ms,
            "meta": {
                "mock": bool(self.use_mock or self.model is None),
                "classifier_ms": classifier_ms,
            },
        }


app = FastAPI(
    title="SemArbiter Decision Server",
    description="Scores runtime-defined options against supplied evidence and returns an id-aligned choice.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[DecisionEngine] = None


def get_engine() -> DecisionEngine:
    global engine
    if engine is None:
        engine = DecisionEngine(use_mock=True)
    return engine


@app.get("/health")
async def health_check():
    eng = get_engine()
    avg_latency = (
        eng.stats["total_latency_ms"] / eng.stats["total_decisions"]
        if eng.stats["total_decisions"] > 0
        else 0.0
    )
    return {
        "status": "online",
        "model": eng.model_name,
        "device": eng.device,
        "mock_mode": eng.use_mock,
        "cuda_graph_enabled": eng.graph_runner is not None,
        "decisions_served": eng.stats["total_decisions"],
        "avg_latency_ms": round(avg_latency, 2),
        "min_latency_ms": round(eng.stats["min_latency_ms"], 2) if eng.stats["min_latency_ms"] != float("inf") else 0.0,
        "max_latency_ms": round(eng.stats["max_latency_ms"], 2),
    }


def _payload_image(payload: Dict[str, Any]) -> Optional[str]:
    image = payload.get("image")
    state = payload.get("state")
    if not image and isinstance(state, dict):
        image = state.get("image")
    if isinstance(image, str) and image:
        return image
    return None


@app.post("/v1/classifier")
@app.post("/v1/systemone")
async def classifier_endpoint(payload: Dict[str, Any]):
    if _payload_image(payload):
        raise HTTPException(status_code=422, detail="image is not accepted")
    return get_engine().classify_jev(payload)


@app.get("/")
async def root():
    return HTMLResponse("<h1>SemArbiter Decision Server Online</h1><p>POST /v1/classifier · GET /health</p>")


def main():
    global engine
    parser = argparse.ArgumentParser(description="SemArbiter decision server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace model id")
    parser.add_argument("--device", default=None, help="Inference device: cuda, cpu, mps")
    parser.add_argument("--mock", action="store_true", help="Run with the mock engine (zero GPU, first option wins)")
    parser.add_argument("--no-graph", action="store_true", help="Disable CUDA Graphs")
    args = parser.parse_args()

    engine = DecisionEngine(
        model_name=args.model,
        device=args.device,
        use_mock=args.mock,
        enable_graph=not args.no_graph,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
