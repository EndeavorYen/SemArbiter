"""Headless SIL control-quality observer (#117).

Binary clean-completion does not see weave, reverse chatter, deadband
escape, or a stall that never collides. This module scores a 0.05 s
trace against those four fingerprints. It does not pick a trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

MappingLike = Dict[str, Any]

# Fingerprint gates from EndeavorYen/SemIf#117.
OFFSET_SIGMA_M = 0.15
STEER_ZERO_CROSS_HZ = 0.4
STARTUP_REVERSE_S = 3.0
CHATTER_WINDOW_S = 2.0
CHATTER_SIGN_FLIPS = 2
SPEED_SIGN_EPS = 0.05
DEADBAND_ABS_M = 0.1
DEADBAND_DRIFT_MPS = 0.2
DEADBAND_MIN_DWELL_S = 0.5
STALL_SPEED_MPS = 0.2
STALL_PROGRESS_M = 0.5
DEADLOCK_HOLD_S = 3.5
SLICE_PAD_S = 3.0
WANDER_WINDOW_S = 5.0
STEER_ZERO_EPS = 1e-3

KIND_TITLE = {
    "wander": "Lateral Wander",
    "chatter": "Gear Chatter",
    "deadband": "Central Deadband Escape",
    "deadlock": "Deadlock Progress Stall",
}

_SPARK = "▁▂▃▄▅▆▇█"


OOD_SCENARIOS = frozenset({"sensor_anomaly"})
EXPECTED_SCENARIOS = frozenset({"sensor_anomaly", "construction_detour"})
SAMPLER_MIN_OFFSET_M = 0.15
POLICY_CENTER_OFFSET_M = 0.05
TRUNCATION_S = 2.0
CAUSE_INVALID = "INVALID_COLLISION_TRUNCATED"
CAUSE_SAMPLER = "SAMPLER_DEFECT"
CAUSE_POLICY = "POLICY_DEFECT"
CAUSE_EXPECTED = "EXPECTED_SCENARIO_MANEUVER"


def make_sample(
    *,
    t: float,
    x: float,
    z: float,
    speed: float,
    steer: float,
    lane_offset: float,
    chosen_id: str = "",
    chosen_tags: str = "",
    signal: Optional[str] = None,
    chosen_speed: Optional[float] = None,
    menu: Optional[Sequence[MappingLike]] = None,
    compact_state: Optional[MappingLike] = None,
    action_switch: bool = False,
    cand_min_offset: Optional[float] = None,
    cand_safe_count: Optional[int] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "t": float(t),
        "x": float(x),
        "z": float(z),
        "speed": float(speed),
        "steer": float(steer),
        "lane_offset": float(lane_offset),
        "chosen_id": str(chosen_id or ""),
        "chosen_tags": str(chosen_tags or ""),
        "signal": signal,
        "action_switch": bool(action_switch),
    }
    if chosen_speed is not None:
        row["chosen_speed"] = float(chosen_speed)
    if menu is not None:
        row["menu"] = [dict(item) for item in menu]
        safe = [item for item in row["menu"] if not item.get("collision")]
        row["cand_safe_count"] = len(safe)
        if safe:
            row["cand_min_offset"] = min(abs(float(item.get("offset", 0.0) or 0.0)) for item in safe)
        else:
            row["cand_min_offset"] = None
    else:
        row["cand_safe_count"] = cand_safe_count
        row["cand_min_offset"] = cand_min_offset
    if compact_state is not None:
        row["compact_state"] = dict(compact_state)
    return row


@dataclass
class Violation:
    kind: str
    t0: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    layer: str = "unknown"
    cause: str = ""
    actionable: bool = True


@dataclass
class Diagnosis:
    ok: bool
    violations: List[Violation]
    slice: List[Dict[str, Any]]
    seed: Optional[int] = None
    scenario: Optional[str] = None
    mode: Optional[str] = None
    n_samples: int = 0
    frame: Optional[Dict[str, Any]] = None


def _dt(samples: Sequence[MappingLike]) -> float:
    if len(samples) < 2:
        return 0.05
    return max(1e-6, float(samples[1]["t"]) - float(samples[0]["t"]))


def _stdev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(var)


def _sign(value: float, eps: float) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _zero_cross_hz(steers: Sequence[float], duration_s: float) -> float:
    if duration_s <= 0 or len(steers) < 2:
        return 0.0
    last = 0
    crosses = 0
    for s in steers:
        sg = _sign(float(s), STEER_ZERO_EPS)
        if sg == 0:
            continue
        if last != 0 and sg != last:
            crosses += 1
        last = sg
    return crosses / duration_s


def _window_ok_wander(window: Sequence[MappingLike]) -> Optional[Violation]:
    if len(window) < 4:
        return None
    duration = float(window[-1]["t"]) - float(window[0]["t"])
    if duration < 1.0:
        return None
    offsets = [float(s["lane_offset"]) for s in window]
    steers = [float(s["steer"]) for s in window]
    sigma = _stdev(offsets)
    hz = _zero_cross_hz(steers, duration)
    if sigma > OFFSET_SIGMA_M and hz > STEER_ZERO_CROSS_HZ:
        return Violation(
            kind="wander",
            t0=float(window[0]["t"]),
            metrics={
                "offset_sigma_m": round(sigma, 4),
                "steer_zero_cross_hz": round(hz, 4),
                "window_s": round(duration, 3),
            },
        )
    return None


def _wander(samples: Sequence[MappingLike]) -> Optional[Violation]:
    if not samples:
        return None
    dt = _dt(samples)
    win_n = max(4, int(round(WANDER_WINDOW_S / dt)))
    if len(samples) <= win_n:
        return _window_ok_wander(samples)
    for i in range(0, len(samples) - win_n + 1, max(1, win_n // 4)):
        hit = _window_ok_wander(samples[i : i + win_n])
        if hit is not None:
            return hit
    return _window_ok_wander(samples)


def _chatter_speed(sample: MappingLike) -> float:
    cs = sample.get("chosen_speed")
    if cs is not None:
        return float(cs)
    return float(sample["speed"])


def _chatter(samples: Sequence[MappingLike]) -> Optional[Violation]:
    if not samples:
        return None
    for s in samples:
        if float(s["t"]) < STARTUP_REVERSE_S and _chatter_speed(s) < 0.0:
            return Violation(
                kind="chatter",
                t0=float(s["t"]),
                metrics={
                    "startup_reverse": True,
                    "sign_flips": 0,
                    "speed_mps": round(_chatter_speed(s), 4),
                },
            )
    dt = _dt(samples)
    win_n = max(2, int(round(CHATTER_WINDOW_S / dt)))
    speeds = [_chatter_speed(s) for s in samples]
    for i in range(0, len(speeds)):
        chunk = speeds[i : i + win_n]
        if len(chunk) < 3:
            break
        last = 0
        flips = 0
        for v in chunk:
            sg = _sign(v, SPEED_SIGN_EPS)
            if sg == 0:
                continue
            if last != 0 and sg != last:
                flips += 1
            last = sg
        if flips >= CHATTER_SIGN_FLIPS:
            return Violation(
                kind="chatter",
                t0=float(samples[i]["t"]),
                metrics={
                    "startup_reverse": False,
                    "sign_flips": flips,
                    "window_s": CHATTER_WINDOW_S,
                },
            )
    return None


def _deadband(samples: Sequence[MappingLike]) -> Optional[Violation]:
    if len(samples) < 3:
        return None
    dt = _dt(samples)
    in_center = False
    entered_t = 0.0
    streak = 0
    min_streak = 3
    for i, s in enumerate(samples):
        off = abs(float(s["lane_offset"]))
        now_center = off < DEADBAND_ABS_M
        if now_center:
            streak += 1
            if not in_center and streak >= min_streak and i + 1 >= min_streak:
                in_center = True
                entered_t = float(s["t"]) - dt * (min_streak - 1)
            continue
        streak = 0
        if in_center:
            dwell = float(s["t"]) - entered_t
            prev = float(samples[i - 1]["lane_offset"])
            drift = abs(float(s["lane_offset"]) - prev) / dt
            # t < 0.5 s is spawn geometry, not a center capture.
            if (
                float(s["t"]) >= 0.5
                and dwell < DEADBAND_MIN_DWELL_S
                and drift > DEADBAND_DRIFT_MPS
            ):
                return Violation(
                    kind="deadband",
                    t0=float(s["t"]),
                    metrics={
                        "dwell_s": round(dwell, 4),
                        "drift_mps": round(drift, 4),
                        "center_abs_m": DEADBAND_ABS_M,
                    },
                )
            in_center = False
    return None


def _is_red(sample: MappingLike) -> bool:
    sig = sample.get("signal")
    return str(sig or "").lower() == "red"


def _deadlock(samples: Sequence[MappingLike]) -> Optional[Violation]:
    if not samples:
        return None
    held_s = 0.0
    z0 = float(samples[0]["z"])
    t_prev = float(samples[0]["t"])
    for i, s in enumerate(samples):
        t = float(s["t"])
        dt = t - t_prev if i else 0.0
        t_prev = t
        if _is_red(s):
            held_s = 0.0
            z0 = float(s["z"])
            continue
        speed = abs(float(s["speed"]))
        if speed >= STALL_SPEED_MPS:
            held_s = 0.0
            z0 = float(s["z"])
            continue
        if held_s <= 0.0:
            z0 = float(s["z"])
        held_s += dt
        progress = abs(float(s["z"]) - z0)
        if held_s > DEADLOCK_HOLD_S and progress < STALL_PROGRESS_M:
            return Violation(
                kind="deadlock",
                t0=t,
                metrics={
                    "held_s": round(held_s, 4),
                    "progress_m": round(progress, 4),
                    "speed_mps": round(speed, 4),
                },
            )
    return None


def _slice_around(samples: Sequence[MappingLike], t0: float) -> List[Dict[str, Any]]:
    lo = t0 - SLICE_PAD_S
    hi = t0 + SLICE_PAD_S
    return [dict(s) for s in samples if lo <= float(s["t"]) <= hi]


def sparkline(values: Sequence[float], width: int = 40) -> str:
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    step = max(1, int(math.ceil(len(values) / width)))
    sampled = list(values[::step][:width])
    if hi - lo < 1e-9:
        return _SPARK[0] * len(sampled)
    out = []
    scale = len(_SPARK) - 1
    for v in sampled:
        idx = int(round((v - lo) / (hi - lo) * scale))
        idx = max(0, min(scale, idx))
        out.append(_SPARK[idx])
    return "".join(out)


def _nearest(samples: Sequence[MappingLike], t0: float) -> MappingLike:
    return min(samples, key=lambda s: abs(float(s["t"]) - t0))


def _top3(menu: Sequence[MappingLike]) -> List[Dict[str, Any]]:
    safe = [dict(item) for item in menu if not item.get("collision")]
    safe.sort(key=lambda item: abs(float(item.get("offset", 99.0) or 99.0)))
    return safe[:3]


def _cand_min(sample: MappingLike) -> Optional[float]:
    val = sample.get("cand_min_offset")
    if val is None:
        return None
    return float(val)


def phase_plane_label(samples: Sequence[MappingLike]) -> str:
    if len(samples) < 4:
        return "short_trace"
    duration = float(samples[-1]["t"]) - float(samples[0]["t"])
    xs = [float(s["lane_offset"]) for s in samples]
    hz = _zero_cross_hz(xs, duration)
    mean_abs = sum(abs(x) for x in xs) / len(xs)
    if hz > STEER_ZERO_CROSS_HZ:
        return "underdamped_oscillation"
    if mean_abs > OFFSET_SIGMA_M:
        return "steady_state_drift"
    return "settled"


def attribute_violation(
    violation: Violation,
    samples: Sequence[MappingLike],
    *,
    scenario: Optional[str] = None,
    collision: bool = False,
    t_end: Optional[float] = None,
) -> Violation:
    """Four-layer cause from #118. Does not invent a trajectory."""
    end_t = float(t_end if t_end is not None else (samples[-1]["t"] if samples else 0.0))
    if collision and end_t < TRUNCATION_S:
        violation.cause = CAUSE_INVALID
        violation.layer = "validity"
        violation.actionable = True
        return violation
    scen = str(scenario or "")
    if scen in EXPECTED_SCENARIOS:
        if scen == "sensor_anomaly" and violation.kind in {"chatter", "wander", "deadlock"}:
            violation.cause = CAUSE_EXPECTED
            violation.layer = "scenario"
            violation.actionable = False
            return violation
        if scen == "construction_detour" and violation.kind in {"wander", "deadband"}:
            violation.cause = CAUSE_EXPECTED
            violation.layer = "scenario"
            violation.actionable = False
            return violation
    frame = _nearest(samples, violation.t0) if samples else {}
    cand_min = _cand_min(frame)
    if violation.kind in {"deadband", "wander"} and cand_min is not None:
        if cand_min > SAMPLER_MIN_OFFSET_M:
            violation.cause = CAUSE_SAMPLER
            violation.layer = "sampler"
            violation.actionable = True
            return violation
        if cand_min < POLICY_CENTER_OFFSET_M:
            violation.cause = CAUSE_POLICY
            violation.layer = "policy"
            violation.actionable = True
            return violation
    if violation.kind == "chatter":
        menu = list(frame.get("menu") or [])
        forward = [
            item
            for item in menu
            if not item.get("collision") and float(item.get("speed") or 0.0) >= 0.0
        ]
        violation.cause = CAUSE_POLICY if forward else CAUSE_SAMPLER
        violation.layer = "policy" if forward else "sampler"
        violation.actionable = True
        return violation
    violation.cause = violation.cause or CAUSE_POLICY
    violation.layer = violation.layer or "policy"
    violation.actionable = True
    return violation


def diagnose_trace(
    samples: Iterable[MappingLike],
    *,
    seed: Optional[int] = None,
    scenario: Optional[str] = None,
    mode: Optional[str] = None,
    collision: bool = False,
    t_end: Optional[float] = None,
) -> Diagnosis:
    rows = [dict(s) for s in samples]
    end_t = float(t_end if t_end is not None else (rows[-1]["t"] if rows else 0.0))
    detectors = (_wander, _chatter, _deadband, _deadlock)
    violations = [hit for fn in detectors if (hit := fn(rows)) is not None]
    if collision and end_t < TRUNCATION_S:
        violations.insert(
            0,
            Violation(
                kind="truncated",
                t0=end_t,
                metrics={"t_end": round(end_t, 3), "collision": True},
                layer="validity",
                cause=CAUSE_INVALID,
                actionable=True,
            ),
        )
    for v in violations:
        if v.cause == CAUSE_INVALID:
            continue
        attribute_violation(
            v, rows, scenario=scenario, collision=collision, t_end=end_t
        )
        if collision and end_t < TRUNCATION_S:
            v.actionable = False
    sl: List[Dict[str, Any]] = []
    frame: Optional[Dict[str, Any]] = None
    if violations:
        sl = _slice_around(rows, violations[0].t0)
        nearest = _nearest(rows, violations[0].t0) if rows else {}
        menu = list(nearest.get("menu") or [])
        frame = {
            "t": nearest.get("t"),
            "chosen_id": nearest.get("chosen_id"),
            "chosen_tags": nearest.get("chosen_tags"),
            "cand_min_offset": nearest.get("cand_min_offset"),
            "cand_safe_count": nearest.get("cand_safe_count"),
            "top3": _top3(menu),
            "phase_plane": phase_plane_label(sl or rows),
        }
    actionable = [v for v in violations if v.actionable]
    return Diagnosis(
        ok=not actionable,
        violations=violations,
        slice=sl,
        seed=seed,
        scenario=scenario,
        mode=mode,
        n_samples=len(rows),
        frame=frame,
    )


def format_report(diagnosis: Diagnosis) -> str:
    seed = diagnosis.seed if diagnosis.seed is not None else "?"
    lines = [
        f"Control quality seed {seed} scenario={diagnosis.scenario or '-'} mode={diagnosis.mode or '-'}",
        f"samples={diagnosis.n_samples} ok={diagnosis.ok}",
    ]
    if not diagnosis.violations:
        lines.append("no fingerprint exceeded")
        return "\n".join(lines)
    for v in diagnosis.violations:
        tag = f"[{v.cause}]" if v.cause else ""
        lines.append(f"- {v.kind} {tag} at t={v.t0:.2f}s metrics={v.metrics}")
    if diagnosis.frame:
        lines.append(
            f"frame cand_min_offset={diagnosis.frame.get('cand_min_offset')} "
            f"safe={diagnosis.frame.get('cand_safe_count')} "
            f"phase={diagnosis.frame.get('phase_plane')}"
        )
    if diagnosis.slice:
        offsets = [float(s["lane_offset"]) for s in diagnosis.slice]
        speeds = [float(s["speed"]) for s in diagnosis.slice]
        lines.append("ascii offset: " + sparkline(offsets))
        lines.append("ascii speed:  " + sparkline(speeds))
        t0 = diagnosis.slice[0]["t"]
        t1 = diagnosis.slice[-1]["t"]
        lines.append(f"slice {t0:.2f}s .. {t1:.2f}s ({SLICE_PAD_S:.0f}s pad)")
    return "\n".join(lines)


def _primary(diagnosis: Diagnosis) -> Optional[Violation]:
    acts = [v for v in diagnosis.violations if v.actionable]
    if acts:
        return acts[0]
    return diagnosis.violations[0] if diagnosis.violations else None


def format_issue_title(diagnosis: Diagnosis) -> str:
    seed = diagnosis.seed if diagnosis.seed is not None else "?"
    primary = _primary(diagnosis)
    cause = primary.cause if primary else "Control"
    kind = primary.kind if primary else "control"
    return f"[Bug/Control] {cause} {kind} on Seed {seed}"


def format_issue_body(diagnosis: Diagnosis) -> str:
    seed = diagnosis.seed if diagnosis.seed is not None else 42
    scenario = diagnosis.scenario or "speed_zone_city"
    primary = _primary(diagnosis)
    lines = [
        "Headless SIL observer (#118) attributed a control-quality failure.",
        "",
        f"- scenario: `{scenario}`",
        f"- mode: `{diagnosis.mode or 'heuristic'}`",
        f"- seed: `{seed}`",
        "",
        "## Attribution",
        "",
        f"- cause: `{primary.cause if primary else '-'}`",
        f"- layer: `{primary.layer if primary else '-'}`",
        f"- fingerprint: `{primary.kind if primary else '-'}` at t={primary.t0:.2f}s"
        if primary
        else "- fingerprint: -",
        f"- phase-plane: `{diagnosis.frame.get('phase_plane') if diagnosis.frame else '-'}`",
        "",
        "## Repro",
        "",
        "```",
        f"python benchmarks/diagnose_control_quality.py --seed {seed} --scenario {scenario}",
        "```",
        "",
        "## Thresholds",
        "",
        f"- wander: σ(offset) > {OFFSET_SIGMA_M} m and steer zero-cross > {STEER_ZERO_CROSS_HZ} Hz",
        f"- chatter: speed < 0 in first {STARTUP_REVERSE_S:.0f}s, or ≥{CHATTER_SIGN_FLIPS} sign flips in {CHATTER_WINDOW_S:.0f}s",
        f"- deadband: after |offset| < {DEADBAND_ABS_M} m, drift > {DEADBAND_DRIFT_MPS} m/s with dwell < {DEADBAND_MIN_DWELL_S} s",
        f"- deadlock: |v| < {STALL_SPEED_MPS} m/s and Δs < {STALL_PROGRESS_M} m for > {DEADLOCK_HOLD_S} s (red wait excluded)",
        f"- sampler vs policy: cand_min_offset > {SAMPLER_MIN_OFFSET_M} m → SAMPLER_DEFECT; "
        f"< {POLICY_CENTER_OFFSET_M} m unused center leaf → POLICY_DEFECT",
        "",
        "## Violations",
        "",
    ]
    for v in diagnosis.violations:
        lines.append(f"- **{v.kind}** `[{v.cause}]` at t={v.t0:.2f}s `{v.metrics}`")
    if diagnosis.frame and diagnosis.frame.get("top3"):
        lines.extend(["", "## Top-3 safe candidates at t0", "", "| id | offset | speed | tag |", "| --- | --- | --- | --- |"])
        for item in diagnosis.frame["top3"]:
            lines.append(
                f"| {item.get('id')} | {item.get('offset')} | {item.get('speed')} | {item.get('tag','')} |"
            )
        picked = diagnosis.frame.get("chosen_id")
        tags = diagnosis.frame.get("chosen_tags")
        lines.extend(["", f"picked `{picked}` tags `{tags}`", ""])
    lines.extend(["", "## Slice", "", "```", format_report(diagnosis), "```", ""])
    if diagnosis.slice:
        lines.append("| t | offset | speed | steer | id |")
        lines.append("| --- | --- | --- | --- | --- |")
        step = max(1, len(diagnosis.slice) // 16)
        for s in diagnosis.slice[::step]:
            lines.append(
                f"| {s['t']:.2f} | {s['lane_offset']:.3f} | {s['speed']:.2f} | {s['steer']:.3f} | {s.get('chosen_id','')} |"
            )
        lines.append("")
    return "\n".join(lines)


def cluster_diagnoses(diagnoses: Sequence[Diagnosis]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, Dict[str, Any]] = {}
    order: List[tuple] = []
    for d in diagnoses:
        acts = [v for v in d.violations if v.actionable]
        if not acts:
            continue
        v = acts[0]
        key = (d.scenario or "-", v.cause)
        if key not in groups:
            groups[key] = {
                "scenario": d.scenario,
                "cause": v.cause,
                "kind": v.kind,
                "seeds": [],
            }
            order.append(key)
        groups[key]["seeds"].append(d.seed)
    return [groups[k] for k in order]
