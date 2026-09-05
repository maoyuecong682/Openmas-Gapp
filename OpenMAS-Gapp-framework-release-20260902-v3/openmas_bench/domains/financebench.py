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
        raw = context.row.get("raw") or {}
        evidence = raw.get("evidence") or []
        snippets = []
        if isinstance(evidence, list):
            for index, item in enumerate(evidence):
                if isinstance(item, dict):
                    text = str(item.get("evidence_text") or "").strip()
                    if text:
                        snippets.append(f"EVIDENCE_{index + 1}: {text}")
        if len(snippets) >= 2:
            values = ["\n\n".join(snippets[::2]), "\n\n".join(snippets[1::2])]
        elif len(snippets) == 1:
            metadata = []
            for key in ("company", "doc_name", "doc_type", "doc_period", "question_type", "question_reasoning", "dataset_subset_label", "doc_link"):
                value = raw.get(key)
                if value is not None and str(value).strip():
                    metadata.append(f"{key}: {value}")
            metadata_text = "FILING_METADATA:\n" + "\n".join(metadata) if metadata else (
                "FILING_METADATA: public FinanceBench row with one evidence record"
            )
            values = [snippets[0], metadata_text]
        else:
            values = []
        if not values:
            values = [str(raw.get("question") or context.row.get("question") or "").strip()]
        return {f"branch_{index}": value for index, value in enumerate(values) if value}
