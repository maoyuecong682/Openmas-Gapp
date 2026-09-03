"""Real QA benchmark protocol for Q1 Harness-layer necessity."""

from .baselines import Q1_REAL_BASELINES, build_q1_real_construction
from .metrics import harness_necessity_score

__all__ = ["Q1_REAL_BASELINES", "build_q1_real_construction", "harness_necessity_score"]
