from __future__ import annotations

from typing import Any


def harness_necessity_score(record: dict[str, Any]) -> float | None:
    """Single presentation score for Q1-real.

    Q1-real is not only a QA leaderboard.  It asks whether a Harness-layer
    construction improves executable MAS applications.  The score therefore
    combines final task performance with construction/constraint validity.
    Keep the raw metrics in tables; use this only as a compact PPT summary.
    """
    e2e = record.get("e2e_success")
    construction = record.get("construction") or {}
    architecture = construction.get("architecture_validity")
    constraint = construction.get("constraint_satisfaction")
    if e2e is None or architecture is None or constraint is None:
        return None
    return round(0.6 * float(e2e) + 0.25 * float(architecture) + 0.15 * float(constraint), 6)
