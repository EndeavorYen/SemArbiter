(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.SEMIF_TELEMETRY_CORE = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const LATENCY_WINDOW = 120;
  const SERIES_KEYS = ["grab_frame_ms", "vision_encode_ms", "classifier_ms", "e2e_loop_ms"];

  function percentileMs(samples, pct) {
    if (!samples || !samples.length) return null;
    const ordered = samples.map(Number).filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
    if (!ordered.length) return null;
    if (ordered.length === 1) return ordered[0];
    const rank = (Number(pct) / 100) * (ordered.length - 1);
    const lo = Math.floor(rank);
    const hi = Math.ceil(rank);
    if (lo === hi) return ordered[lo];
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo);
  }

  function summarizeLatency(samples) {
    const values = (samples || []).map(Number).filter((x) => Number.isFinite(x));
    const n = values.length;
    if (!n) {
      return {
        n: 0,
        min: null,
        max: null,
        mean: null,
        p50: null,
        p90: null,
        p95: null,
        p99: null,
        stddev: null,
      };
    }
    let sum = 0;
    let min = values[0];
    let max = values[0];
    for (const x of values) {
      sum += x;
      if (x < min) min = x;
      if (x > max) max = x;
    }
    const mean = sum / n;
    let varSum = 0;
    for (const x of values) varSum += (x - mean) * (x - mean);
    return {
      n,
      min,
      max,
      mean,
      p50: percentileMs(values, 50),
      p90: percentileMs(values, 90),
      p95: percentileMs(values, 95),
      p99: percentileMs(values, 99),
      stddev: Math.sqrt(varSum / n),
    };
  }

  function fmt(ms) {
    if (ms == null || !Number.isFinite(ms)) return "—";
    return ms.toFixed(ms >= 100 ? 0 : 1);
  }

  function createLatencyTelemetry(opts) {
    const windowSize = (opts && opts.window) || LATENCY_WINDOW;
    const now = (opts && opts.now) || function () {
      return typeof performance !== "undefined" ? performance.now() : Date.now();
    };
    const series = {};
    for (const key of SERIES_KEYS) series[key] = [];
    let lastGrab = 0;
    let lastVisionRtt = 0;

    function push(name, ms) {
      const value = Number(ms);
      if (!Number.isFinite(value) || value < 0) return;
      const buf = series[name];
      buf.push({ t: now(), ms: value });
      if (buf.length > windowSize) buf.splice(0, buf.length - windowSize);
    }

    function getHistory() {
      const out = {};
      for (const key of SERIES_KEYS) out[key] = series[key].map((row) => ({ t: row.t, ms: row.ms }));
      return out;
    }

    function getMetrics() {
      const out = {};
      for (const key of SERIES_KEYS) out[key] = summarizeLatency(series[key].map((row) => row.ms));
      return out;
    }

    function hudText() {
      const m = getMetrics();
      const e2e = m.e2e_loop_ms;
      const vis = m.vision_encode_ms;
      const cls = m.classifier_ms;
      const lastE2e = series.e2e_loop_ms.length ? series.e2e_loop_ms[series.e2e_loop_ms.length - 1].ms : null;
      const lastVis = series.vision_encode_ms.length
        ? series.vision_encode_ms[series.vision_encode_ms.length - 1].ms
        : null;
      const lastCls = series.classifier_ms.length
        ? series.classifier_ms[series.classifier_ms.length - 1].ms
        : null;
      return [
        "e2e " + fmt(lastE2e) + "ms",
        "P50 " + fmt(e2e.p50),
        "P95 " + fmt(e2e.p95),
        "vis " + fmt(lastVis),
        "cls " + fmt(lastCls),
      ].join("  ");
    }

    return {
      recordVision: function (sample) {
        const grab = sample && sample.grab_ms;
        if (grab != null) {
          lastGrab = Number(grab) || 0;
          push("grab_frame_ms", grab);
        }
        lastVisionRtt = Number(sample && sample.rtt_ms) || 0;
        push("vision_encode_ms", sample && sample.encode_ms);
      },
      recordClassifier: function (sample) {
        push("classifier_ms", sample && sample.classifier_ms);
        const rtt = Number(sample && sample.rtt_ms) || 0;
        push("e2e_loop_ms", lastGrab + lastVisionRtt + rtt);
      },
      getMetrics: getMetrics,
      getHistory: getHistory,
      exportJSON: function () {
        return {
          schema: "semif.web_latency.v1",
          window: windowSize,
          generated_at_ms: now(),
          metrics: getMetrics(),
          history: getHistory(),
        };
      },
      hudText: hudText,
    };
  }

  return {
    LATENCY_WINDOW: LATENCY_WINDOW,
    percentileMs: percentileMs,
    summarizeLatency: summarizeLatency,
    createLatencyTelemetry: createLatencyTelemetry,
  };
});
