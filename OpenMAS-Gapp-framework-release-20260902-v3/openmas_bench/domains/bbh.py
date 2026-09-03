from __future__ import annotations

import re

from .base import DomainContext, DomainPlugin


class BBHDomainPlugin(DomainPlugin):
    dataset_ids = ("BBH", "BBH-Full")
    metric_names = ("bbh_accuracy",)

    def augment_task_payload(self, payload, context: DomainContext) -> None:
        payload["task"] = str((context.row.get("raw") or {}).get("task") or "")

    def branch_resources(self, context: DomainContext) -> dict[str, str]:
        raw = context.row.get("raw") or {}
        task = str(raw.get("task") or "").strip()
        input_text = str(raw.get("input") or context.row.get("question") or "").strip()
        if not input_text:
            return {}
        values = split_bbh_task_resources(task, input_text)
        return {f"branch_{index}": value for index, value in enumerate(values) if value}

    def output_contract(self, context: DomainContext) -> str:
        if context.metric_name != "bbh_accuracy":
            return ""
        return (
            "For multiple-choice BBH tasks, return exactly one option letter in canonical form such as "
            "(A), with no option text or explanation. For non-choice tasks, return only the concise exact answer."
        )

    def normalize_terminal(self, artifact: str, context: DomainContext) -> str:
        if context.dataset_id != "BBH-Full":
            return artifact
        match = re.match(r"^\s*\(?([A-Ja-j])\)?(?:\s|[.):\-]|$)", artifact)
        return f"({match.group(1).upper()})" if match else artifact


def split_bbh_task_resources(task: str, input_text: str) -> tuple[str, str]:
    """Split at task-semantic boundaries while preserving the public text."""
    task_key = task.casefold()
    option_match = re.search(r"\s+Options:\s*", input_text, flags=re.IGNORECASE)
    if option_match:
        body = input_text[:option_match.start()].strip()
        options = input_text[option_match.end():].strip()
    else:
        body, options = input_text.strip(), ""

    transition_markers = {
        "tracking_shuffled_objects": ("Throughout the", "As the"),
        "temporal_sequences": ("We know that:",),
        "penguins_in_a_table": ("We now add", "And here is"),
        "causal_judgement": ("Did ", "Would ", "Does "),
    }
    marker = next((candidate for prefix, candidates in transition_markers.items()
                   if task_key.startswith(prefix)
                   for candidate in candidates if candidate in body), None)
    if marker:
        split_at = body.index(marker)
        left, right = body[:split_at].strip(), body[split_at:].strip()
    else:
        units = [unit.strip() for unit in re.split(r"(?<=[.!?])\s+|\n+", body)
                 if unit.strip()]
        if len(units) >= 2:
            midpoint = max(1, (len(units) + 1) // 2)
            left, right = " ".join(units[:midpoint]), " ".join(units[midpoint:])
        else:
            tokens = body.split()
            midpoint = max(1, len(tokens) // 2)
            left, right = " ".join(tokens[:midpoint]), " ".join(tokens[midpoint:])
    if not right:
        right = left
    if options:
        right += "\nOptions:\n" + options
    prefix = f"BBH_TASK: {task or 'unknown'}"
    return (
        prefix + "\nBRANCH_ROLE: initial facts or first subexpression\n" + left,
        prefix + "\nBRANCH_ROLE: remaining constraints, transformations, question, and options\n" + right,
    )

