"""Semantic Driving Index (SDI) for fair JevPilot strategy comparison.

Geometric smoothness alone favors fast rule baselines. SDI reweights
lawfulness, emergency yield, and OOD fail-closed behavior so semantic
reasoning is visible in the headline score.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence


SDI_WEIGHTS = {
    "geometric_smoothness": 0.20,
    "traffic_law_compliance": 0.30,
    "emergency_avoidance": 0.30,
    "ood_safety": 0.20,
}

SEMANTIC_EXCLUSIVE_SCENARIOS = (
    "ambiguous_priority",
    "construction_detour",
    "emergency_vehicle",
    "sensor_anomaly",
)

STANDARD_SEEDS = (42, 123, 2026)


def stable_mix(text: str) -> int:
    """FNV-1a 32-bit mix. Independent of PYTHONHASHSEED."""
    h = 2166136261
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def scenario_seed(base_seed: int, scenario: str, episode: int) -> int:
    """Deterministic per-episode seed for a named scenario."""
    mixed = stable_mix(scenario)
    return (int(base_seed) * 1000003 + int(episode) * 9176 + mixed) & 0x7FFFFFFF


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def geometric_smoothness(episodes: Sequence[Mapping[str, Any]]) -> float:
    if not episodes:
        return 0.0
    scores = []
    for ep in episodes:
        jerk = float(ep.get("jerk_rms") or 0.0)
        osc = float(ep.get("steering_oscillation") or 0.0)
        off = 1.0 if ep.get("off_track") else 0.0
        jerk_term = _clip01(1.0 - jerk / 40.0)
        osc_term = _clip01(1.0 - osc / 12.0)
        scores.append(0.45 * jerk_term + 0.35 * osc_term + 0.20 * (1.0 - off))
    return sum(scores) / len(scores)


def traffic_law_compliance(episodes: Sequence[Mapping[str, Any]]) -> float:
    if not episodes:
        return 0.0
    scores = []
    for ep in episodes:
        red = 1.0 if ep.get("red_light_violation") else 0.0
        speed = 1.0 if ep.get("speeding_violation") else 0.0
        scores.append(0.65 * (1.0 - red) + 0.35 * (1.0 - speed))
    return sum(scores) / len(scores)


def emergency_avoidance(episodes: Sequence[Mapping[str, Any]]) -> float:
    if not episodes:
        return 0.0
    scores = []
    for ep in episodes:
        ped = 1.0 if ep.get("pedestrian_casualty") else 0.0
        veh = 1.0 if ep.get("vehicle_collision") else 0.0
        coll = 1.0 if ep.get("collision") else 0.0
        scores.append(0.45 * (1.0 - ped) + 0.35 * (1.0 - veh) + 0.20 * (1.0 - coll))
    return sum(scores) / len(scores)


def ood_safety(episodes: Sequence[Mapping[str, Any]]) -> float:
    ood_eps = [ep for ep in episodes if ep.get("scenario") == "sensor_anomaly"]
    if not ood_eps:
        return 0.0
    ok = 0
    for ep in ood_eps:
        if ep.get("fail_safe_triggered") and not ep.get("collision") and not ep.get("off_track"):
            ok += 1
    return ok / len(ood_eps)


def dimension_scores(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    return {
        "geometric_smoothness": round(geometric_smoothness(episodes), 4),
        "traffic_law_compliance": round(traffic_law_compliance(episodes), 4),
        "emergency_avoidance": round(emergency_avoidance(episodes), 4),
        "ood_safety": round(ood_safety(episodes), 4),
    }


def semantic_driving_index(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    dims = dimension_scores(episodes)
    sdi = sum(dims[name] * weight for name, weight in SDI_WEIGHTS.items())
    exclusive = [ep for ep in episodes if ep.get("scenario") in SEMANTIC_EXCLUSIVE_SCENARIOS]
    exclusive_success = 0.0
    if exclusive:
        exclusive_success = sum(1.0 for ep in exclusive if ep.get("completed") and not ep.get("collision")) / len(exclusive)
    return {
        "sdi": round(sdi, 4),
        "weights": dict(SDI_WEIGHTS),
        "dimensions": dims,
        "semantic_exclusive_success": round(exclusive_success, 4),
        "n_episodes": len(episodes),
        "n_semantic_exclusive": len(exclusive),
    }


def radar_svg(series: Mapping[str, Mapping[str, float]], size: int = 320) -> str:
    """Tiny SVG radar for the four SDI axes. No JS required."""
    axes = list(SDI_WEIGHTS.keys())
    cx = cy = size / 2
    radius = size * 0.36
    import math

    def pt(idx: int, mag: float) -> str:
        ang = -math.pi / 2 + idx * 2 * math.pi / len(axes)
        x = cx + radius * mag * math.cos(ang)
        y = cy + radius * mag * math.sin(ang)
        return f"{x:.1f},{y:.1f}"

    rings = []
    for mag in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(pt(i, mag) for i in range(len(axes)))
        rings.append(f'<polygon points="{pts}" fill="none" stroke="#334155" stroke-width="1"/>')

    spokes = []
    labels = []
    for i, name in enumerate(axes):
        spokes.append(f'<line x1="{cx}" y1="{cy}" x2="{pt(i, 1.0).split(",")[0]}" y2="{pt(i, 1.0).split(",")[1]}" stroke="#475569" />')
        ang = -math.pi / 2 + i * 2 * math.pi / len(axes)
        lx = cx + (radius + 28) * math.cos(ang)
        ly = cy + (radius + 28) * math.sin(ang)
        short = name.replace("_", " ")
        labels.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" fill="#cbd5e1" font-size="10">{short}</text>')

    colors = ["#3e6ae1", "#f59e0b", "#34d399"]
    polygons = []
    for i, (label, dims) in enumerate(series.items()):
        color = colors[i % len(colors)]
        pts = " ".join(pt(j, float(dims.get(axes[j], 0.0))) for j in range(len(axes)))
        polygons.append(
            f'<polygon points="{pts}" fill="{color}33" stroke="{color}" stroke-width="2">'
            f'<title>{label}</title></polygon>'
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}">'
        f'<rect width="100%" height="100%" fill="#0b0b0c"/>'
        + "".join(rings + spokes + polygons + labels)
        + "</svg>"
    )


def compare_modes(mode_episodes: Mapping[str, Iterable[Mapping[str, Any]]]) -> Dict[str, Any]:
    modes: Dict[str, Any] = {}
    radar: Dict[str, Dict[str, float]] = {}
    for mode, eps in mode_episodes.items():
        ep_list = list(eps)
        summary = semantic_driving_index(ep_list)
        modes[mode] = summary
        radar[mode] = summary["dimensions"]
    ranked = sorted(modes.items(), key=lambda kv: kv[1]["sdi"], reverse=True)
    return {
        "benchmark": "Semantic Driving Index",
        "modes": modes,
        "ranking": [name for name, _ in ranked],
        "radar_svg": radar_svg(radar),
    }
