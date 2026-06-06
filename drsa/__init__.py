"""
DRSA - Dominance-based Rough Set Approach for Composite Indicators

Implements Algorithms 1, 2, 4 and full pipeline from:
Corrente, Greco, Slowiński, Zappalà (2026)
"An explainable and interpretable composite indicator based on decision rules"
Omega 142, 103513. https://doi.org/10.1016/j.omega.2026.103513
"""

from .core.rules import induce_atleast_rules, induce_atmost_rules
from .core.formatting import (
    format_atleast_rules,
    format_atmost_rules,
    compute_relative_support,
    get_supporting_units,
)
from .core.classifier import classify_units, explain_unit
from .core.pipeline import run_pipeline
from .core.new_units import classify_new_units

__all__ = [
    "induce_atleast_rules",
    "induce_atmost_rules",
    "format_atleast_rules",
    "format_atmost_rules",
    "compute_relative_support",
    "get_supporting_units",
    "classify_units",
    "explain_unit",
    "run_pipeline",
    "classify_new_units",
]
