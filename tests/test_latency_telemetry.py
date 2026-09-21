"""Web E2E latency breakdown stats and telemetry export (#121)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from semif_phase1.latency_telemetry import (
    LATENCY_WINDOW,
    align_web_export,
    percentile_ms,
    summarize_latency,
)

REPO = Path(__file__).resolve().parent.parent
OVERLAY_JS = REPO / "demo" / "jevpilot" / "semif-layer.js"
TELEMETRY_JS = REPO / "demo" / "jevpilot" / "semif-telemetry.js"
INDEX_HTML = REPO / "demo" / "jevpilot" / "index.html"
VISION_BENCH = REPO / "benchmarks" / "benchmark_jevpilot_vision.py"


def test_percentile_linear_interpolation_on_1_to_100():
    samples = [float(i) for i in range(1, 101)]
    assert percentile_ms(samples, 50) == pytest.approx(50.5)
    assert percentile_ms(samples, 90) == pytest.approx(90.1)
    assert percentile_ms(samples, 95) == pytest.approx(95.05)
    assert percentile_ms(samples, 99) == pytest.approx(99.01)


def test_summarize_latency_p50_p95_p99_stddev_and_extrema():
    samples = [float(i) for i in range(1, 101)]
    summary = summarize_latency(samples)
    assert summary["n"] == 100
    assert summary["min"] == 1.0
    assert summary["max"] == 100.0
    assert summary["p50"] == pytest.approx(50.5)
    assert summary["p90"] == pytest.approx(90.1)
    assert summary["p95"] == pytest.approx(95.05)
    assert summary["p99"] == pytest.approx(99.01)
    assert summary["mean"] == pytest.approx(50.5)
    assert summary["stddev"] == pytest.approx(28.8660700477)
    empty = summarize_latency([])
    assert empty["n"] == 0
    assert empty["p50"] is None
    assert empty["stddev"] is None


def test_ring_buffer_keeps_last_120():
    from semif_phase1.latency_telemetry import LatencyWindow

    win = LatencyWindow(LATENCY_WINDOW)
    for i in range(130):
        win.push(float(i))
    hist = win.samples()
    assert len(hist) == 120
    assert hist[0] == 10.0
    assert hist[-1] == 129.0


def test_export_json_has_summary_and_series():
    from semif_phase1.latency_telemetry import LatencyTelemetry

    tel = LatencyTelemetry(now_ms=lambda: 1000.0)
    tel.record_vision(encode_ms=12.0, rtt_ms=40.0, grab_ms=5.0)
    tel.record_classifier(classifier_ms=18.0, rtt_ms=30.0)
    payload = tel.export_json()
    assert payload["schema"] == "semif.web_latency.v1"
    assert payload["window"] == LATENCY_WINDOW
    assert "metrics" in payload and "history" in payload
    for key in ("vision_encode_ms", "classifier_ms", "e2e_loop_ms", "grab_frame_ms"):
        assert key in payload["metrics"]
        assert key in payload["history"]
    assert payload["history"]["classifier_ms"][0]["ms"] == 18.0
    assert payload["metrics"]["classifier_ms"]["n"] == 1
    assert payload["history"]["e2e_loop_ms"][0]["ms"] == pytest.approx(30.0)


def test_align_web_export_reports_p50_delta():
    web = {
        "schema": "semif.web_latency.v1",
        "metrics": {
            "classifier_ms": {"p50": 20.0, "p95": 40.0, "n": 10},
            "e2e_loop_ms": {"p50": 80.0, "p95": 120.0, "n": 10},
        },
    }
    local = {
        "classifier_ms": {"p50": 15.0, "p95": 25.0, "n": 10},
    }
    report = align_web_export(web, local)
    assert report["delta_p50_classifier_ms"] == pytest.approx(5.0)
    assert report["web"]["e2e_loop_ms"]["p50"] == 80.0


def test_js_percentile_and_export_match_python():
    assert TELEMETRY_JS.is_file()
    script = r"""
const t = require(%s);
const samples = Array.from({length: 100}, (_, i) => i + 1);
const out = {
  p50: t.percentileMs(samples, 50),
  p95: t.percentileMs(samples, 95),
  p99: t.percentileMs(samples, 99),
  summary: t.summarizeLatency(samples),
  window: t.LATENCY_WINDOW,
};
const tel = t.createLatencyTelemetry({ now: () => 1000, window: t.LATENCY_WINDOW });
tel.recordVision({ encode_ms: 12, rtt_ms: 40, grab_ms: 5 });
tel.recordClassifier({ classifier_ms: 18, rtt_ms: 30 });
out.exportJSON = tel.exportJSON();
out.hudText = tel.hudText();
process.stdout.write(JSON.stringify(out));
""" % json.dumps(str(TELEMETRY_JS))
    raw = subprocess.check_output(["node", "-e", script], cwd=str(REPO))
    js = json.loads(raw)
    samples = [float(i) for i in range(1, 101)]
    py = summarize_latency(samples)
    assert js["window"] == LATENCY_WINDOW
    assert js["p50"] == pytest.approx(py["p50"])
    assert js["p95"] == pytest.approx(py["p95"])
    assert js["p99"] == pytest.approx(py["p99"])
    assert js["summary"]["stddev"] == pytest.approx(py["stddev"])
    exported = js["exportJSON"]
    assert exported["schema"] == "semif.web_latency.v1"
    assert exported["history"]["e2e_loop_ms"][0]["ms"] == pytest.approx(30.0)
    assert "P50" in js["hudText"] and "P95" in js["hudText"]
    assert "e2e" in js["hudText"].lower() or "E2E" in js["hudText"]
    assert "vis" in js["hudText"] and "cls" in js["hudText"]


def test_overlay_wires_live_telemetry_hooks():
    js = OVERLAY_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = (REPO / "demo" / "jevpilot" / "semif-layer.css").read_text(encoding="utf-8")
    assert "semif-telemetry.js" in html
    assert "SEMIF_TELEMETRY" in js
    assert "recordVision" in js
    assert "recordClassifier" in js
    assert "exportJSON" in js
    assert "fsd-latency" in js
    assert "fsd-latency" in css
    assert "vision_encode_ms" in js
    assert "classifier_ms" in js
    core = TELEMETRY_JS.read_text(encoding="utf-8")
    assert "e2e_loop_ms" in core
    fetch_block = js.split("window.fetch = function")[1].split("function mockChoice")[0]
    assert "recordClassifier" in fetch_block
    vision_block = js.split("async function visionTick")[1].split("if (visionOn)")[0]
    assert "recordVision" in vision_block
    assert "grabFrame" in vision_block


def test_vision_benchmark_reads_web_telemetry_json():
    src = VISION_BENCH.read_text(encoding="utf-8")
    assert "--web-telemetry" in src
    assert "align_web_export" in src
    assert 'ep["latencies"]' in src or "ep['latencies']" in src
