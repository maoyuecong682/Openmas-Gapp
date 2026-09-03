from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from ..schema import ConstructionResult


STRUCTURED_TOOL_SCHEMA = {
    "selected_evidence": [{
        "type": "cell or text",
        "location": "table row/column label or pre_text/post_text location",
        "raw_value": "exact evidence text",
    }],
    "steps": [{
        "operation": (
            "one of add, subtract, multiply, divide, ratio_percent, percent_change, "
            "percent_point_change, power"
        ),
        "operands": ["numeric string or backward reference such as $step_0", "numeric string or $step_n"],
        "result_unit": "percent, percentage points, million, currency, or number",
    }],
}

STRUCTURED_TOOL_CONTRACT = (
    "Compiled FinQA v5 tool-call contract: return exactly one JSON object with selected_evidence and steps. "
    "selected_evidence must be a non-empty array of objects containing exactly type, location, and raw_value; "
    "type must be cell or text, and raw_value must be copied from the supplied public evidence. steps must "
    "contain between one and eight ordered objects with exactly operation, operands, and result_unit. Each "
    "operation must be one of add, subtract, multiply, divide, ratio_percent, percent_change, "
    "percent_point_change, or power. Each operands array must contain exactly two numeric strings or backward "
    "references such as $step_0; a step may reference only an earlier step. Preserve commas, percent signs, "
    "parentheses, and negative signs. A prior result whose result_unit is percent is resolved as a percentage "
    "for downstream arithmetic, so a value of 1 percent becomes 0.01. Use ratio_percent(part, whole) for "
    "part/whole*100, percent_change(new, old) for (new-old)/old*100, and percent_point_change(new%, old%) "
    "for percentage-point differences. Do not calculate results yourself and do not include an answer, gold "
    "program, expression, explanation, or extra field."
)

_OPERATIONS = {
    "add", "subtract", "multiply", "divide", "ratio_percent",
    "percent_change", "percent_point_change", "power",
}
_TOOL_REMOVALS = {"application_blueprint", "blueprint_preserving_realization"}


def structured_tool_eligible(construction: ConstructionResult) -> bool:
    removed = str(construction.application.metadata.get("removed_module", ""))
    blueprint_present = bool(construction.blueprint.metadata.get("blueprint_present", True))
    blueprint_preserving = construction.application.metadata.get("blueprint_preserving", True) is not False
    return removed not in _TOOL_REMOVALS and blueprint_present and blueprint_preserving


def node_role(node: Any) -> str:
    text = " ".join((
        str(node.id), str(node.implementation_ref), str(node.realizes_blueprint_node),
        str(node.config.get("execution_instruction", "")),
    )).casefold()
    for role in ("retrieve", "select", "execute", "verify", "answer"):
        if f"component_{role}" in text or re.search(rf"\b{role}\b", text):
            return role
    return ""


def inherit_tool_result(predecessor_ids: list[str],
                        tool_results: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    inherited = [tool_results[node_id] for node_id in predecessor_ids if node_id in tool_results]
    if not inherited:
        return None
    values = {str(item.get("final_value")) for item in inherited}
    if len(values) != 1:
        raise ValueError("conflicting FinQA TOOL_TRACE values reached one node")
    return inherited[0]


def execute_tool_call(value: dict[str, Any],
                      public_task: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate and execute a gold-blind, multi-step Decimal program."""
    if not isinstance(value, dict):
        raise ValueError("FinQA tool call must be a JSON object")
    required = {"selected_evidence", "steps"}
    missing = required.difference(value)
    extra = set(value).difference(required)
    if missing or extra:
        raise ValueError(f"invalid FinQA tool fields: missing={sorted(missing)} extra={sorted(extra)}")
    selected_evidence = value["selected_evidence"]
    if not isinstance(selected_evidence, list) or not selected_evidence:
        raise ValueError("selected_evidence must be a non-empty list")
    normalized_evidence = []
    for evidence in selected_evidence:
        if not isinstance(evidence, dict) or set(evidence) != {"type", "location", "raw_value"}:
            raise ValueError("each selected evidence item must contain exactly type, location, and raw_value")
        normalized = {key: str(evidence[key]).strip() for key in ("type", "location", "raw_value")}
        if not all(normalized.values()):
            raise ValueError("selected evidence fields must be non-empty")
        normalized["type"] = normalized["type"].casefold()
        if normalized["type"] not in {"cell", "text"}:
            raise ValueError(f"unsupported FinQA evidence type: {normalized['type']}")
        normalized_evidence.append(normalized)
    if public_task is not None:
        _validate_public_evidence(normalized_evidence, public_task)

    steps = value["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 8:
        raise ValueError("FinQA steps must contain between one and eight items")
    trace: list[dict[str, Any]] = []
    for step_index, step in enumerate(steps):
        if not isinstance(step, dict) or set(step) != {"operation", "operands", "result_unit"}:
            raise ValueError("each FinQA step must contain exactly operation, operands, and result_unit")
        operation = str(step["operation"]).strip().casefold()
        if operation not in _OPERATIONS:
            raise ValueError(f"unsupported FinQA operation at step {step_index}: {operation}")
        operands = step["operands"]
        if not isinstance(operands, list) or len(operands) != 2:
            raise ValueError(f"FinQA step {step_index} operands must contain exactly two values")
        raw_operands = [str(operand).strip() for operand in operands]
        resolved = [_resolve_operand(operand, trace, step_index) for operand in raw_operands]
        result = _execute_operation(operation, resolved)
        result_unit = str(step["result_unit"]).strip()
        if not result_unit:
            raise ValueError(f"FinQA step {step_index} result_unit must be non-empty")
        trace.append({
            "step_id": f"step_{step_index}",
            "operation": operation,
            "operands": raw_operands,
            "normalized_operands": [_decimal_text(item[0]) for item in resolved],
            "computed_value": _decimal_text(result),
            "result_unit": result_unit,
            "step_status": "ok",
        })
    final = trace[-1]
    return {
        "selected_evidence": normalized_evidence,
        "steps": trace,
        "final_value": final["computed_value"],
        "final_unit": final["result_unit"],
        "tool_status": "ok",
    }


def assert_tool_boundary(construction: ConstructionResult, eligible: bool) -> None:
    if eligible:
        return
    leaked = [node.id for node in construction.application.nodes
              if node.config.get("numeric_tool_contract")
              or node.config.get("finqa_structured_tool_schema")]
    if leaked:
        raise ValueError(
            "information boundary violation: non-compiled FinQA variant exposed tool contract on "
            + ", ".join(leaked)
        )


def _resolve_operand(raw: str, trace: list[dict[str, Any]],
                     current_step: int) -> tuple[Decimal, bool]:
    match = re.fullmatch(r"\$step_(\d+)", raw.casefold())
    if not match:
        if raw.startswith("$") and raw.casefold().startswith("$step_"):
            raise ValueError(f"invalid FinQA step reference at step {current_step}: {raw}")
        return _parse_decimal(raw)
    referenced_index = int(match.group(1))
    if referenced_index >= current_step or referenced_index >= len(trace):
        raise ValueError(f"FinQA step {current_step} must reference an earlier existing step, got {raw}")
    referenced = trace[referenced_index]
    number = Decimal(str(referenced["computed_value"]))
    is_percent = _percent_unit(str(referenced["result_unit"]))
    if is_percent:
        number /= Decimal(100)
    return number, is_percent


def _execute_operation(operation: str, parsed: list[tuple[Decimal, bool]]) -> Decimal:
    a, b = (item[0] for item in parsed)
    with localcontext() as context:
        context.prec = 40
        if operation == "add":
            return a + b
        if operation == "subtract":
            return a - b
        if operation == "multiply":
            return a * b
        if operation == "divide":
            if b == 0:
                raise ValueError("FinQA divide operand must be non-zero")
            return a / b
        if operation == "ratio_percent":
            if b == 0:
                raise ValueError("FinQA ratio denominator must be non-zero")
            return a / b * Decimal(100)
        if operation == "percent_change":
            if b == 0:
                raise ValueError("FinQA percent-change baseline must be non-zero")
            return (a - b) / b * Decimal(100)
        if operation == "percent_point_change":
            if not all(item[1] for item in parsed):
                raise ValueError("percent_point_change requires two percent operands")
            return (a - b) * Decimal(100)
        if b != b.to_integral_value() or abs(b) > 100:
            raise ValueError("FinQA power exponent must be an integer between -100 and 100")
        return a ** int(b)


def _percent_unit(unit: str) -> bool:
    return unit.strip().casefold() in {
        "%", "percent", "percentage", "percentage point", "percentage points"}


def _validate_public_evidence(selected_evidence: list[dict[str, str]],
                              public_task: dict[str, Any]) -> None:
    public_text = json.dumps({
        "question": public_task.get("question"),
        "context": public_task.get("context"),
        "choices": public_task.get("choices"),
    }, ensure_ascii=False)
    searchable = _normalize_evidence_text(public_text)
    for evidence in selected_evidence:
        raw_value = _normalize_evidence_text(evidence["raw_value"])
        if raw_value not in searchable:
            raise ValueError(
                "selected FinQA evidence is not present in the public task: "
                + evidence["raw_value"][:120])


def _normalize_evidence_text(value: str) -> str:
    return " ".join(str(value).casefold().replace(",", "").split())


def _parse_decimal(raw: str) -> tuple[Decimal, bool]:
    text = raw.strip().replace(",", "").replace("$", "")
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1].strip()
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        raise ValueError(f"invalid FinQA numeric operand: {raw!r}")
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"invalid FinQA numeric operand: {raw!r}") from exc
    if is_percent:
        number /= Decimal(100)
    return number, is_percent


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("FinQA tool produced a non-finite result")
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0", "+0"} else text


__all__ = [
    "STRUCTURED_TOOL_CONTRACT", "STRUCTURED_TOOL_SCHEMA", "assert_tool_boundary",
    "execute_tool_call", "inherit_tool_result", "node_role", "structured_tool_eligible",
]
