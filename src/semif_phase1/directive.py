"""Slow-loop directive for Dual-Jev (#103). Never picks a trajectory id."""

from __future__ import annotations

from typing import Any, Dict, Optional

INTENTS = ("CRUISE", "RED_LIGHT_STOP", "YIELD_CUT_IN", "CAUTION")


def plan_directive(
    vision: Optional[Dict[str, Any]] = None,
    intersection: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Map camera event + signal to a short intent. No steer, no tXX."""
    vis = vision if isinstance(vision, dict) else {}
    inter = intersection if isinstance(intersection, dict) else {}
    event = str(vis.get("event") or "").lower()
    signal = str(vis.get("signal") or inter.get("signal") or "").lower()
    if signal == "red" or "mandatory stop" in event or "red signal" in event:
        return {"intent": "RED_LIGHT_STOP", "directive": "prefer halt=yes"}
    if "cut-in" in event or "cutting" in event:
        return {"intent": "YIELD_CUT_IN", "directive": "prefer collision=no, low speed"}
    if "appeared" in event or "growing" in event:
        return {"intent": "CAUTION", "directive": "prefer collision=no"}
    return {"intent": "CRUISE", "directive": "lane-centered path"}


def _halt_ids(candidates: Dict[str, Any]) -> list[str]:
    out = []
    for cid, vec in candidates.items():
        if isinstance(vec, (list, tuple)) and len(vec) >= 6 and vec[5]:
            out.append(str(cid))
    return out


def _speed(vec: Any) -> float:
    try:
        return abs(float(vec[0]))
    except (TypeError, ValueError, IndexError):
        return 1e9


def fail_safe_choice(
    candidates: Dict[str, Any],
    choice_id: Optional[str],
    directive: Optional[Dict[str, str]] = None,
) -> str:
    """Physics veto: collision=yes never wins. RED_LIGHT_STOP picks halt if one exists."""
    if not isinstance(candidates, dict) or not candidates:
        return str(choice_id or "")
    ids = [str(k) for k in candidates]
    pick = str(choice_id) if choice_id in candidates or str(choice_id) in candidates else ids[0]
    if pick not in candidates:
        pick = str(pick)
    vec = candidates.get(pick) or candidates.get(choice_id)
    collided = isinstance(vec, (list, tuple)) and len(vec) >= 5 and bool(vec[4])
    if collided:
        halt = _halt_ids(candidates)
        if halt:
            return min(halt, key=lambda cid: _speed(candidates[cid]))
        return min(ids, key=lambda cid: _speed(candidates[cid]))
    intent = (directive or {}).get("intent")
    if intent == "RED_LIGHT_STOP":
        halt = _halt_ids(candidates)
        if halt:
            return min(halt, key=lambda cid: _speed(candidates[cid]))
    return pick
