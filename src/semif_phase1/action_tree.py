"""Sequential classification tree whose leaves are action ids.

Naive walk: one letter-slot call per internal node (tree height).
Smarter walk used here:
- Skip a node when its evidence slice is empty; take default_branch.
- When the chosen child is a Leaf, one SemIf call among those action ids.
- Sensor / parse checks stay outside the LM (see sensors_corrupt).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


ScoreBranches = Callable[[str, List[Dict[str, str]], Dict[str, Any]], str]
ScoreLeaves = Callable[[List[str]], str]


def slice_is_empty(values: Dict[str, Any]) -> bool:
    def empty(value: Any) -> bool:
        if value is None or value is False:
            return True
        if isinstance(value, dict):
            return all(empty(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return len(value) == 0
        return False

    return all(empty(item) for item in values.values())


@dataclass(frozen=True)
class Leaf:
    action_ids: Tuple[str, ...]


@dataclass(frozen=True)
class Branch:
    id: str
    description: str
    nxt: "NodeOrLeaf"


@dataclass(frozen=True)
class Node:
    id: str
    question: str
    evidence_keys: Tuple[str, ...]
    branches: Tuple[Branch, ...]
    skip_if_empty: bool = True
    default_branch: str = ""


NodeOrLeaf = Union[Node, Leaf]


def evidence_slice(state: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    return {key: state.get(key) for key in keys}


def walk_action_tree(
    root: Node,
    state: Dict[str, Any],
    score_branches: ScoreBranches,
    score_leaves: ScoreLeaves,
) -> Dict[str, Any]:
    """Walk to a leaf action id. Returns choice, path, and how many LM calls ran."""
    node: NodeOrLeaf = root
    path: List[Dict[str, Any]] = []
    branch_forwards = 0
    while isinstance(node, Node):
        sliced = evidence_slice(state, node.evidence_keys)
        if node.skip_if_empty and slice_is_empty(sliced):
            choice = node.default_branch or node.branches[0].id
            path.append({"node": node.id, "choice": choice, "skipped": True})
        else:
            options = [{"id": branch.id, "description": branch.description} for branch in node.branches]
            choice = score_branches(node.question, options, sliced)
            branch_forwards += 1
            path.append({"node": node.id, "choice": choice, "skipped": False})
        nxt: Optional[NodeOrLeaf] = None
        for branch in node.branches:
            if branch.id == choice:
                nxt = branch.nxt
                break
        node = nxt if nxt is not None else node.branches[0].nxt

    assert isinstance(node, Leaf)
    action_ids = list(node.action_ids)
    if len(action_ids) == 1:
        leaf = action_ids[0]
        leaf_forwards = 0
    else:
        leaf = score_leaves(action_ids)
        leaf_forwards = 1
        if leaf not in action_ids:
            leaf = action_ids[0]
    return {
        "choice": leaf,
        "leaf_set": action_ids,
        "path": path,
        "branch_forwards": branch_forwards,
        "leaf_forwards": leaf_forwards,
    }


def sensors_corrupt(state: Dict[str, Any]) -> bool:
    """Parser, not a classifier. None / missing anomaly is not OOD."""
    anomaly = state.get("anomaly")
    if anomaly not in (None, False, "", 0):
        return True
    speed = state.get("speed_mps")
    if isinstance(speed, str) and speed.upper() in {"ANOMALY_CORRUPTED", "NAN", "INFINITY", "-INFINITY"}:
        return True
    try:
        number = float(speed)
    except (TypeError, ValueError):
        return False
    return math.isnan(number) or math.isinf(number)
