from __future__ import annotations

from .base import DomainContext, DomainPlugin


class FinanceBenchDomainPlugin(DomainPlugin):
    dataset_ids = ("FinanceBench",)
    metric_names = ("financebench_accuracy",)

    def augment_task_payload(self, payload, context: DomainContext) -> None:
        raw = context.row.get("raw") or {}
        payload.update({
            "question_reasoning": raw.get("question_reasoning", ""),
            "question_type": raw.get("question_type", ""),
            "evidence_count": len(raw.get("evidence") or []),
        })

    def branch_resources(self, context: DomainContext) -> dict[str, str]:
        evidence = (context.row.get("raw") or {}).get("evidence") or []
        snippets = []
        if isinstance(evidence, list):
            for index, item in enumerate(evidence):
                if isinstance(item, dict):
                    text = str(item.get("evidence_text") or "").strip()
                    if text:
                        snippets.append(f"EVIDENCE_{index + 1}: {text}")
        values = (["\n\n".join(snippets[::2]), "\n\n".join(snippets[1::2])]
                  if len(snippets) >= 2 else snippets)
        if not values:
            raw = context.row.get("raw") or {}
            values = [str(raw.get("question") or context.row.get("question") or "").strip()]
        return {f"branch_{index}": value for index, value in enumerate(values) if value}

