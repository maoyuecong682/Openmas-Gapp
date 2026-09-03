from __future__ import annotations

from .base import DomainContext, DomainPlugin


class SciBenchDomainPlugin(DomainPlugin):
    dataset_ids = ("SciBench",)
    metric_names = ("scibench_numeric_accuracy",)

    def augment_task_payload(self, payload, context: DomainContext) -> None:
        raw = context.row.get("raw") or {}
        payload.update({
            "problem_text": raw.get("problem_text", context.row.get("question", "")),
            "unit": raw.get("unit", ""),
            "source": raw.get("source", ""),
        })

    def output_contract(self, context: DomainContext) -> str:
        return (
            "Return only the final numeric value followed by the requested unit. Preserve scientific "
            "notation, sign, and unit dimensions; do not include derivation or other numbers."
        )

    def reasoning_contract(self, context: DomainContext) -> str:
        return (
            "Carry one candidate numeric value and its requested unit through every stage. Preserve "
            "scientific notation, sign, and unit dimensions when verifying the derivation."
        )

