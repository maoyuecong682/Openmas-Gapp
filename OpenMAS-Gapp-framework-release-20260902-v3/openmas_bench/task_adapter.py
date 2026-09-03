from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .llm import LLMAdapter
from .schema import ConstructionExecutionTask


@dataclass
class TaskAnswer:
    answer: Any
    source: str
    evaluated: bool
    adapter: str
    model: str
    retry_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class DomainTaskAdapter:
    """Small answer adapter for FinQA/PubMedQA; SWE-bench remains a trace proxy."""

    def __init__(self, adapter: LLMAdapter | None = None):
        self.adapter = adapter
        self.cache: dict[tuple[str, int], TaskAnswer] = {}

    def answer(self, task: ConstructionExecutionTask, seed: int) -> TaskAnswer:
        source = str(task.source.get("dataset", ""))
        key = (str(task.source.get("task_id", task.id)), seed)
        if key in self.cache:
            return self.cache[key]
        if "SWE-bench" in source or "swe" in source.lower():
            result = TaskAnswer(None, source, False, "trace_proxy", "none")
        elif self.adapter is None:
            result = TaskAnswer(None, source, False, "disabled", "none")
        else:
            if "PubMedQA" in source:
                system = "Answer the biomedical yes/no/maybe question. Return JSON {\"answer\": \"yes|no|maybe\"} only."
                user = json.dumps({"question": task.prompt, "contexts": task.context.get("contexts", []) if isinstance(task.context, dict) else task.context}, ensure_ascii=False)
            else:
                system = "Answer the numerical financial QA question using the provided table/context. Return JSON {\"answer\": string} only. Do not include explanation."
                user = json.dumps({"question": task.prompt, "context": task.context}, ensure_ascii=False)
            response = self.adapter.generate_json(system, user, seed, {"answer"})
            result = TaskAnswer(response.value.get("answer"), source, True, response.provider, response.model, response.retry_count, response.input_tokens, response.output_tokens)
        self.cache[key] = result
        return result


def answer_matches(task: ConstructionExecutionTask, predicted: Any) -> bool | None:
    if predicted is None:
        return None
    gold = task.answer
    source = str(task.source.get("dataset", ""))
    if "PubMedQA" in source:
        return str(predicted).strip().lower() == str(gold).strip().lower()
    if "FinQA" in source:
        try:
            return abs(float(str(predicted).replace("%", "")) - float(str(gold).replace("%", ""))) <= 1e-3
        except ValueError:
            return str(predicted).strip().lower() == str(gold).strip().lower()
    return None
