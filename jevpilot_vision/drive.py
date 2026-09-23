"""Prepare a driving request, then let SemArbiter score it.

Six-column candidates are the driving contract. Any other payload is returned unchanged.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from semif_phase1.action_tree import Branch, Leaf, Node

from jevpilot_vision.directive import fail_safe_choice, plan_directive
from jevpilot_vision.trajectory_sampler import compact_jev_state, partition_ids, vector_option_tag


def has_six_column_candidates(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    candidates = state.get("candidates")
    if not isinstance(candidates, dict) or not candidates:
        return False
    return any(isinstance(vec, (list, tuple)) and len(vec) >= 6 for vec in candidates.values())


def build_drive_tree(candidates: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Node:
    """Bucket this frame's sampled trajectories. Leaves are sampled ids."""
    halt, lateral, lane = partition_ids(candidates, meta)
    if not lane:
        lane = list(candidates.keys()) or ["t00"]
    slow_ids = tuple(halt) if halt else tuple(lane)
    lat_ids = tuple(lateral) if lateral else tuple(lane)
    lane_ids = tuple(lane)
    speed_node = Node(
        id="speed",
        question="Is current speed above the posted ceiling? Choose slow only if speed_mps > speed_ceiling_mps.",
        evidence_keys=("speed_mps", "speed_ceiling_mps"),
        skip_if_empty=False,
        default_branch="cruise",
        branches=(
            Branch("slow", "speed_mps is greater than speed_ceiling_mps.", Leaf(slow_ids)),
            Branch("cruise", "speed_mps is at or under the posted ceiling.", Leaf(lane_ids)),
        ),
    )
    around_node = Node(
        id="around",
        question="Is a lateral move required? Choose go_around only for a detour, parked hazard, or emergency vehicle.",
        evidence_keys=("construction", "roadside_obstacle", "emergency_vehicle"),
        skip_if_empty=True,
        default_branch="stay_in_lane",
        branches=(
            Branch("go_around", "A detour, parked hazard, or siren requires moving aside.", Leaf(lat_ids)),
            Branch("stay_in_lane", "No blockage requiring a lateral move.", speed_node),
        ),
    )
    halt_next: Any = Leaf(tuple(halt)) if halt else around_node
    return Node(
        id="halt",
        question="Must the vehicle halt now? Choose must_stop only for a red or yellow signal not yet cleared, a person in the path, or an unmarked yield.",
        evidence_keys=("intersection", "pedestrian", "other_vehicle"),
        skip_if_empty=True,
        default_branch="keep_moving",
        branches=(
            Branch("must_stop", "Red or yellow signal, a person in the path, or an unmarked yield.", halt_next),
            Branch("keep_moving", "No halt required for a signal or a person.", around_node),
        ),
    )


DRIVE_TREE = build_drive_tree(
    {"t00": [10.0, 0.0, 0.0, 0.0, False, False], "t01": [0.0, 0.0, 0.0, 0.0, False, True]},
    {"t01": {"end_speed": 0.0, "steer": 0.0, "stop_at_line": True}},
)


def _signal(state: Dict[str, Any]) -> Optional[str]:
    vision = state.get("vision") if isinstance(state.get("vision"), dict) else {}
    intersection = state.get("intersection") if isinstance(state.get("intersection"), dict) else {}
    signal = str(vision.get("signal") or intersection.get("signal") or "").lower() or None
    if signal != "red" and "red" in str(vision.get("event") or "").lower():
        signal = "red"
    return signal


def _ego_x(state: Dict[str, Any]) -> Optional[float]:
    try:
        if state.get("lateral_offset_m") is None:
            return None
        ego_x = float(state.get("lateral_offset_m"))
    except (TypeError, ValueError):
        return None
    if abs(ego_x) > 8.0:
        return None
    return ego_x


def prepare_drive_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Tag trajectory options and compact the prompt. Leave every other request alone."""
    state = payload.get("state")
    if not has_six_column_candidates(state):
        return payload
    prepared = copy.deepcopy(payload)
    state = prepared["state"]
    directive = plan_directive(state.get("vision"), state.get("intersection"))
    state["directive"] = directive["intent"]
    questions = prepared.get("questions")
    if isinstance(questions, dict):
        vector = questions.get("vector")
        if isinstance(vector, dict):
            instructions = vector.get("instructions", "Choose optimal driving option.")
            if directive.get("intent"):
                vector["instructions"] = f"{instructions} Intent: {directive['intent']}."
            criteria = vector.get("criteria")
            candidates = state.get("candidates") if isinstance(state.get("candidates"), dict) else {}
            ego_x = _ego_x(state)
            signal = _signal(state)
            if isinstance(criteria, dict):
                tagged = {}
                for opt_id, desc in criteria.items():
                    vec = candidates.get(opt_id)
                    if isinstance(vec, (list, tuple)) and len(vec) >= 6:
                        tagged[opt_id] = vector_option_tag(vec, ego_x=ego_x, signal=signal)
                    else:
                        tagged[opt_id] = desc
                vector["criteria"] = tagged
    state["_semif_prompt_state"] = compact_jev_state(state)
    return prepared


def score_drive_request(engine: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare a driving payload, score it on the live classifier, then apply the veto."""
    if has_six_column_candidates(payload.get("state")):
        prepared = prepare_drive_request(payload)
        result = engine.classify_jev(prepared)
        return finish_drive_choice(payload, result)
    return engine.classify_jev(payload)


def finish_drive_choice(original: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Physics veto after the live scorer. Collision never wins. A red-light halt replaces the id."""
    state = original.get("state") if isinstance(original.get("state"), dict) else {}
    if not has_six_column_candidates(state):
        return result
    answers = result.get("answers") if isinstance(result.get("answers"), dict) else {}
    vector = answers.get("vector") if isinstance(answers.get("vector"), dict) else None
    if not vector:
        return result
    directive = plan_directive(state.get("vision"), state.get("intersection"))
    raw_mode = bool(original.get("raw_mode") or state.get("raw_mode"))
    mode = original.get("mode", "flat")
    legal = None if raw_mode or mode == "heuristic" else directive
    raw_choice = vector.get("choice")
    safe = fail_safe_choice(state.get("candidates") or {}, raw_choice, legal)
    if safe and safe != raw_choice:
        vector["choice"] = safe
        meta = result.setdefault("meta", {})
        if isinstance(meta, dict):
            meta["fail_safe"] = True
    meta = result.setdefault("meta", {})
    if isinstance(meta, dict):
        meta["jev1_intent"] = directive.get("intent")
    return result
