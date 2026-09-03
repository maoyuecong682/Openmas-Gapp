from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DomainContext:
    dataset_id: str
    metric_name: str
    row: dict[str, Any]
    data_root: Path


class DomainPlugin:
    """Gold-blind domain boundary used by the generic application executor."""

    dataset_ids: tuple[str, ...] = ()
    metric_names: tuple[str, ...] = ()

    def augment_task_payload(self, payload: dict[str, Any], context: DomainContext) -> None:
        return None

    def branch_resources(self, context: DomainContext) -> dict[str, str]:
        return {}

    def output_contract(self, context: DomainContext) -> str:
        return ""

    def reasoning_contract(self, context: DomainContext) -> str:
        return ""

    def normalize_terminal(self, artifact: str, context: DomainContext) -> str:
        return artifact

