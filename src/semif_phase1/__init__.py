"""SemIf: High-Performance, Calibrated Semantic Decision Engine."""

from .core import expected_value, sanitize_state, softmax, apply_prior_calibration
from .hierarchical import DecisionNode, DecisionOption, HierarchicalDecisionTree, HierarchicalResult

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "expected_value",
    "sanitize_state",
    "softmax",
    "apply_prior_calibration",
    "DecisionNode",
    "DecisionOption",
    "HierarchicalDecisionTree",
    "HierarchicalResult",
]
