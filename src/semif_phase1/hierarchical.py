"""Hierarchical Semantic Decision Trees for SemIf.

Enables coarse-to-fine multi-tier decision routing (e.g., Macro Intent -> Tactical Maneuver -> Micro Action,
or RPG Combat: Stance -> Skill Category -> Specific Spell) with per-node calibration, Sliced LM Heads,
and Helmholtz Free Energy OOD safety gates at every tier.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .core import sanitize_state
from .direct import score
from .gating import compute_free_energy, gate_decision


@dataclass
class DecisionOption:
    id: str
    description: str
    continuous_value: Optional[float] = None
    child_node: Optional[DecisionNode] = None


@dataclass
class DecisionNode:
    id: str
    question: str
    options: List[DecisionOption]
    min_confidence: float = 0.35
    energy_threshold: Optional[float] = -20.0
    abstain_option_id: str = "abstain"

    def to_row(self, state: Any, row_id: str) -> Dict[str, Any]:
        return {
            "id": f"{row_id}_{self.id}",
            "state": sanitize_state(state),
            "question": self.question,
            "options": [{"id": opt.id, "description": opt.description} for opt in self.options],
        }


@dataclass
class HierarchicalResult:
    path: List[str]
    final_action: str
    is_ood: bool
    rejection_reasons: List[str]
    node_decisions: Dict[str, Dict[str, Any]]
    total_seconds: float
    depth_reached: int


class HierarchicalDecisionTree:
    """Manages and executes coarse-to-fine decision trees."""

    def __init__(self, root: DecisionNode):
        self.root = root

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        state: Any,
        metadata: Optional[Dict[str, Any]] = None,
        sliced_head: bool = True,
        prior_provider: Optional[Callable[[int], List[float]]] = None,
        graph_runner: Any = None,
        max_depth: int = 10,
    ) -> HierarchicalResult:
        """Traverse decision tree from root to leaf using single-forward SemIf scoring per tier."""
        t0 = time.perf_counter()
        current_node = self.root
        path: List[str] = []
        node_decisions: Dict[str, Dict[str, Any]] = {}
        depth = 0
        overall_ood = False
        rejection_reasons: List[str] = []

        while current_node is not None and depth < max_depth:
            depth += 1
            node_row = current_node.to_row(state, f"node_{depth}")
            option_count = len(current_node.options)

            # Get prior if provider provided
            prior_logits = prior_provider(option_count) if prior_provider else None

            # Execute SemIf scoring pass
            scored = score(
                model,
                tokenizer,
                node_row,
                metadata or {},
                sliced_head=sliced_head,
                prior_logits=prior_logits,
                graph_runner=graph_runner,
            )

            option_ids = [opt.id for opt in current_node.options]
            probs = scored["probabilities"]
            logits = scored["calibrated_logits"]

            # Helmholtz Free Energy Gate for this tier
            gated = gate_decision(
                probs,
                logits,
                option_ids,
                min_confidence=current_node.min_confidence,
                energy_threshold=current_node.energy_threshold,
                abstain_option_id=current_node.abstain_option_id,
            )

            selected_id = gated["selected_id"]
            node_is_ood = gated["disposition"] == "abstain"

            node_decisions[current_node.id] = {
                "depth": depth,
                "question": current_node.question,
                "selected_id": selected_id,
                "probabilities": {oid: p for oid, p in zip(option_ids, probs)},
                "free_energy": gated["free_energy"],
                "is_ood": node_is_ood,
                "rejection_reasons": gated["rejection_reasons"],
            }

            path.append(selected_id)

            if node_is_ood:
                overall_ood = True
                rejection_reasons.extend(gated["rejection_reasons"])
                # Halt traversal early on safety gate trigger
                break

            # Find selected option and check for child branch
            chosen_option = next((opt for opt in current_node.options if opt.id == selected_id), None)
            if chosen_option is not None and chosen_option.child_node is not None:
                current_node = chosen_option.child_node
            else:
                break

        total_sec = time.perf_counter() - t0
        final_act = path[-1] if path else "unknown"

        return HierarchicalResult(
            path=path,
            final_action=final_act,
            is_ood=overall_ood,
            rejection_reasons=rejection_reasons,
            node_decisions=node_decisions,
            total_seconds=total_sec,
            depth_reached=depth,
        )
