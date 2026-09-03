from __future__ import annotations

from .base import DomainContext, DomainPlugin


class FinQADomainPlugin(DomainPlugin):
    dataset_ids = ("FinQA",)
    metric_names = ("finqa_numeric_accuracy",)

    def branch_resources(self, context: DomainContext) -> dict[str, str]:
        raw_context = (context.row.get("raw") or {}).get("context") or {}
        values = []
        if isinstance(raw_context, dict):
            pre_text = [str(item).strip() for item in raw_context.get("pre_text", []) if str(item).strip()]
            post_text = [str(item).strip() for item in raw_context.get("post_text", []) if str(item).strip()]
            table = raw_context.get("table") or []
            if isinstance(table, list) and table:
                table_lines = []
                for row_index, row_values in enumerate(table[:4]):
                    if isinstance(row_values, list):
                        row_text = " | ".join(str(cell).strip() for cell in row_values if str(cell).strip())
                    else:
                        row_text = str(row_values).strip()
                    if row_text:
                        table_lines.append(f"row_{row_index}: {row_text}")
                if table_lines:
                    values.append("FINQA_TABLE:\n" + "\n".join(table_lines))
            if pre_text:
                values.append("FINQA_PRE_TEXT: " + " ".join(pre_text[:4]))
            if post_text:
                values.append("FINQA_POST_TEXT: " + " ".join(post_text[:4]))
        if not values:
            raw = context.row.get("raw") or {}
            values = [str(raw.get("prompt") or raw.get("question") or "").strip()]
        return {f"branch_{index}": value for index, value in enumerate(values) if value}

