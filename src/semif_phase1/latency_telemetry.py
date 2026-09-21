"""Sliding-window latency percentiles for Web HUD and benchmark alignment."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

LATENCY_WINDOW = 120
SERIES_KEYS = (
    "grab_frame_ms",
    "vision_encode_ms",
    "classifier_ms",
    "e2e_loop_ms",
)


def percentile_ms(samples: List[float], pct: float) -> Optional[float]:
    """Linear interpolation on sorted samples. pct in [0, 100]."""
    if not samples:
        return None
    ordered = sorted(float(x) for x in samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (float(pct) / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def summarize_latency(samples: List[float]) -> Dict[str, Any]:
    values = [float(x) for x in samples]
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "stddev": None,
        }
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    return {
        "n": n,
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "p50": percentile_ms(values, 50),
        "p90": percentile_ms(values, 90),
        "p95": percentile_ms(values, 95),
        "p99": percentile_ms(values, 99),
        "stddev": math.sqrt(var),
    }


class LatencyTelemetry:
    def __init__(
        self,
        window: int = LATENCY_WINDOW,
        now_ms: Optional[Callable[[], float]] = None,
    ) -> None:
        self.window = int(window)
        self._now = now_ms or (lambda: 0.0)
        self._series: Dict[str, List[Dict[str, float]]] = {k: [] for k in SERIES_KEYS}

    def _push(self, name: str, ms: float) -> None:
        value = float(ms)
        if not math.isfinite(value) or value < 0.0:
            return
        buf = self._series[name]
        buf.append({"t": float(self._now()), "ms": value})
        overflow = len(buf) - self.window
        if overflow > 0:
            del buf[:overflow]

    def record_vision(
        self,
        encode_ms: float,
        rtt_ms: float,
        grab_ms: Optional[float] = None,
    ) -> None:
        if grab_ms is not None:
            self._push("grab_frame_ms", grab_ms)
        self._push("vision_encode_ms", encode_ms)

    def record_classifier(self, classifier_ms: float, rtt_ms: float) -> None:
        self._push("classifier_ms", classifier_ms)
        self._push("e2e_loop_ms", rtt_ms)

    def get_history(self) -> Dict[str, List[Dict[str, float]]]:
        return {k: [dict(row) for row in rows] for k, rows in self._series.items()}

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        return {
            k: summarize_latency([row["ms"] for row in rows])
            for k, rows in self._series.items()
        }

    def export_json(self) -> Dict[str, Any]:
        return {
            "schema": "semif.web_latency.v1",
            "window": self.window,
            "generated_at_ms": float(self._now()),
            "metrics": self.get_metrics(),
            "history": self.get_history(),
        }


def align_web_export(web: Dict[str, Any], local: Dict[str, Any]) -> Dict[str, Any]:
    web_metrics = (web or {}).get("metrics") or {}
    web_cls = web_metrics.get("classifier_ms") or {}
    local_cls = (local or {}).get("classifier_ms") or {}
    web_p50 = web_cls.get("p50")
    local_p50 = local_cls.get("p50")
    delta = None
    if web_p50 is not None and local_p50 is not None:
        delta = float(web_p50) - float(local_p50)
    return {
        "web": web_metrics,
        "local": local,
        "delta_p50_classifier_ms": delta,
    }
