from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schema import ApplicationBlueprint, BlueprintEdge, BlueprintNode


@dataclass(frozen=True)
class Q3Task:
    id: str
    label: str


@dataclass(frozen=True)
class Q3Case:
    case_id: str
    dataset_id: str
    family: str
    domain: str
    requirement: str
    tasks: list[Q3Task]
    baseline_labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# Deprecated names remain as type aliases for downstream imports. Q3 now
# builds and evaluates the shared ApplicationBlueprint IR directly.
Q3BlueprintNode = BlueprintNode
Q3BlueprintEdge = BlueprintEdge
Q3Blueprint = ApplicationBlueprint


@dataclass(frozen=True)
class Q3EvalResult:
    case_id: str
    dataset_id: str
    family: str
    domain: str
    baseline: str
    seed: int
    scenario: str
    osv: float
    e2e_success: float
    success: bool
    notes: list[str] = field(default_factory=list)
