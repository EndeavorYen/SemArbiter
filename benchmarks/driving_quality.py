"""How well the car drove. Clean completion is the headline, not SDI."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence


def is_incident(episode: Mapping[str, Any]) -> bool:
    return bool(
        episode.get("collision")
        or episode.get("off_track")
        or episode.get("red_light_violation")
        or episode.get("pedestrian_casualty")
    )


def is_clean(episode: Mapping[str, Any]) -> bool:
    return bool(episode.get("completed")) and not is_incident(episode)


def driving_quality(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(episodes)
    if total == 0:
        return {
            "clean_completion_rate": 0.0,
            "incident_rate": 0.0,
            "n_episodes": 0,
        }
    clean = [ep for ep in episodes if is_clean(ep)]
    incidents = [ep for ep in episodes if is_incident(ep)]
    true_ood = [ep for ep in episodes if ep.get("true_ood")]
    not_ood = [ep for ep in episodes if not ep.get("true_ood")]
    ood_recall = 0.0
    if true_ood:
        ood_recall = sum(
            1.0
            for ep in true_ood
            if ep.get("fail_safe_triggered") and not is_incident(ep)
        ) / len(true_ood)
    ood_fp = 0.0
    if not_ood:
        ood_fp = sum(1.0 for ep in not_ood if ep.get("fail_safe_triggered")) / len(not_ood)
    jerk_on_clean = 0.0
    if clean:
        jerk_on_clean = sum(float(ep.get("jerk_rms") or 0.0) for ep in clean) / len(clean)
    return {
        "clean_completion_rate": round(len(clean) / total, 4),
        "incident_rate": round(len(incidents) / total, 4),
        "completed_rate": round(sum(1 for ep in episodes if ep.get("completed")) / total, 4),
        "incidents": {
            "red_light": sum(1 for ep in episodes if ep.get("red_light_violation")),
            "pedestrian": sum(1 for ep in episodes if ep.get("pedestrian_casualty")),
            "vehicle": sum(1 for ep in episodes if ep.get("vehicle_collision")),
            "off_track": sum(1 for ep in episodes if ep.get("off_track")),
            "speeding": sum(1 for ep in episodes if ep.get("speeding_violation")),
        },
        "jerk_on_clean": round(jerk_on_clean, 2),
        "ood_recall": round(ood_recall, 4),
        "ood_false_positive_rate": round(ood_fp, 4),
        "n_episodes": total,
        "n_clean": len(clean),
        "n_true_ood": len(true_ood),
    }


def compare_driving(mode_episodes: Mapping[str, Iterable[Mapping[str, Any]]]) -> Dict[str, Any]:
    modes = {name: driving_quality(list(eps)) for name, eps in mode_episodes.items()}
    ranked = sorted(
        modes.items(),
        key=lambda item: (item[1]["clean_completion_rate"], -item[1]["incident_rate"]),
        reverse=True,
    )
    return {
        "benchmark": "Driving quality (clean completion)",
        "modes": modes,
        "ranking": [name for name, _ in ranked],
    }
