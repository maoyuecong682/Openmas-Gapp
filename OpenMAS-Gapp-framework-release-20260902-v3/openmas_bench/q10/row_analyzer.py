"""Three-layer Q10 row analysis for financial MAS construction."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from .financial_profiles import get_financial_profile


TASK_ID = re.compile(r"^[a-z][a-z0-9_]{2, thirty}$".replace("thirty", "30"))
ALLOWED_RELATIONS = {"precedes", "requires", "reviews", "feedback"}
LOGGER = logging.getLogger(__name__)


def analyze_financial_row(dataset: str, row: dict[str, Any], llm=None, seed: int = 11) -> dict[str, Any]:
    profile = get_financial_profile(dataset)
    prompt = _prompt(profile, row)
    LOGGER.warning("Q10 analyze_financial_row llm_is_none=%s dataset=%s seed=%s", llm is None, dataset, seed)
    if llm is None:
        analysis = _deterministic_analysis(profile, row)
        source = "deterministic_profile_fallback"
    else:
        response = llm.generate_json(
            "You are a financial MAS architecture analyst. Analyze task structure only, not the gold answer. Return valid JSON.",
            prompt,
            seed,
            {"task_family", "risk_level", "evidence_mode", "tasks", "edges", "constraints"},
        )
        analysis = response.value
        source = response.provider
        analysis["model"] = response.model
        analysis["model_calls"] = 1
        analysis["raw_response_sha256"] = __import__("hashlib").sha256(response.raw_text.encode("utf-8")).hexdigest()
    return _sanitize(profile, row, analysis, source)


def _prompt(profile: dict[str, Any], row: dict[str, Any]) -> str:
    payload = {
        "dataset": profile["dataset"],
        "dataset_profile": profile,
        "question": row.get("question", ""),
        "context": row.get("context", ""),
        "choices": row.get("choices", []),
        "instruction": "Infer the required financial application task graph. Do not return or infer the gold answer. Keep tasks abstract and implementation-independent.",
        "required_json_shape": {
            "task_family": "string",
            "financial_focus": "string",
            "risk_level": "low|medium|high",
            "evidence_mode": "filing|table|mixed",
            "tasks": [{"id": "snake_case_task", "objective": "abstract objective"}],
            "edges": [{"source": "task_id", "target": "task_id", "relation": "precedes|requires|reviews|feedback"}],
            "constraints": [{"id": "constraint_id", "target": "task_id", "predicate": "required"}],
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _deterministic_analysis(profile: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    stages = profile["baseline_tasks"]
    objectives = {
        "filing_evidence": "Retrieve audited filing evidence and metric definitions",
        "market_risk_evidence": "Retrieve risk-factor, uncertainty and forward-looking evidence",
        "financial_analysis": "Analyze financial metrics, trend drivers and evidence consistency",
        "report_table_evidence": "Retrieve table cells and line items from the financial report",
        "narrative_disclosure_evidence": "Retrieve relevant narrative disclosures and footnotes",
        "calculation_analysis": "Compute the numeric financial result with unit handling",
        "risk_assessment": "Assess material risk, sensitivity and unsupported inference risk",
        "compliance_review": "Check regulatory, disclosure and non-advice constraints",
        "audit_trail": "Record evidence lineage, assumptions and control decisions",
        "final_report": "Return the dataset-specific financial answer contract",
    }
    if profile["dataset"] == "FinanceBench":
        edges = [
            {"source": "filing_evidence", "target": "financial_analysis", "relation": "precedes"},
            {"source": "market_risk_evidence", "target": "risk_assessment", "relation": "precedes"},
            {"source": "financial_analysis", "target": "risk_assessment", "relation": "requires"},
            {"source": "financial_analysis", "target": "compliance_review", "relation": "requires"},
            {"source": "risk_assessment", "target": "audit_trail", "relation": "reviews"},
            {"source": "compliance_review", "target": "audit_trail", "relation": "reviews"},
            {"source": "audit_trail", "target": "final_report", "relation": "precedes"},
            {"source": "compliance_review", "target": "risk_assessment", "relation": "feedback"},
        ]
    else:
        edges = [
            {"source": "report_table_evidence", "target": "calculation_analysis", "relation": "precedes"},
            {"source": "narrative_disclosure_evidence", "target": "risk_assessment", "relation": "precedes"},
            {"source": "calculation_analysis", "target": "risk_assessment", "relation": "requires"},
            {"source": "calculation_analysis", "target": "compliance_review", "relation": "requires"},
            {"source": "risk_assessment", "target": "audit_trail", "relation": "reviews"},
            {"source": "compliance_review", "target": "audit_trail", "relation": "reviews"},
            {"source": "audit_trail", "target": "final_report", "relation": "precedes"},
            {"source": "compliance_review", "target": "calculation_analysis", "relation": "feedback"},
        ]
    return {
        "task_family": profile["task_family"],
        "financial_focus": "dataset-driven financial analysis, risk assessment and compliance review",
        "risk_level": "high",
        "evidence_mode": "table" if profile["dataset"] == "FinQA" else "filing",
        "tasks": [{"id": task, "objective": objectives[task]} for task in stages],
        "edges": edges,
        "constraints": [{"id": control, "target": "final_report", "predicate": "required"} for control in profile["required_controls"]],
    }


def _sanitize(profile: dict[str, Any], row: dict[str, Any], raw: dict[str, Any], source: str) -> dict[str, Any]:
    raw_tasks = raw.get("tasks") if isinstance(raw.get("tasks"), list) else []
    tasks: list[dict[str, str]] = []
    seen: set[str] = set()
    reserved_tasks = {"final_report", *profile["required_controls"]}
    for item in raw_tasks[:8]:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("id", "")).strip().casefold()
        if not TASK_ID.match(task_id) or task_id in seen or task_id in reserved_tasks:
            continue
        tasks.append({"id": task_id, "objective": str(item.get("objective") or task_id.replace("_", " ")).strip()[:160]})
        seen.add(task_id)
    if not tasks:
        fallback = _deterministic_analysis(profile, row)
        tasks = fallback["tasks"]
        raw = {**fallback, **{key: value for key, value in raw.items() if key not in {"tasks", "edges", "constraints"}}}

    # Keep governance and terminal tasks stable even when the model returns a
    # shallow or linear plan. Q10's claim is a governed financial application,
    # not a raw QA chain.
    final_task = next((item for item in tasks if item["id"] == "final_report"), None)
    tasks = [item for item in tasks if item["id"] not in {"risk_assessment", "compliance_review", "audit_trail", "final_report"}]
    tasks.append({"id": "risk_assessment", "objective": "Assess material risk, sensitivity and unsupported inference risk"})
    tasks.append({"id": "compliance_review", "objective": "Check regulatory, disclosure and non-advice constraints"})
    tasks.append({"id": "audit_trail", "objective": "Record evidence lineage, assumptions and control decisions"})
    tasks.append(final_task or {"id": "final_report", "objective": "Return the dataset-specific financial answer contract"})
    task_ids = {item["id"] for item in tasks}

    edges: list[dict[str, str]] = []
    for item in raw.get("edges", []) if isinstance(raw.get("edges"), list) else []:
        if not isinstance(item, dict):
            continue
        source_id, target_id = str(item.get("source", "")).casefold(), str(item.get("target", "")).casefold()
        relation = str(item.get("relation", "precedes")).casefold()
        if source_id in task_ids and target_id in task_ids and source_id != target_id and relation in ALLOWED_RELATIONS:
            edge = {"source": source_id, "target": target_id, "relation": relation}
            if edge not in edges:
                edges.append(edge)
    if not edges:
        edges = [{"source": left["id"], "target": right["id"], "relation": "precedes"} for left, right in zip(tasks, tasks[1:])]
    substantive = [item["id"] for item in tasks if item["id"] not in {"risk_assessment", "compliance_review", "audit_trail", "final_report"}]
    if len(substantive) >= 2:
        _ensure_path(edges, substantive[0], "risk_assessment")
        _ensure_path(edges, substantive[1], "compliance_review")
    elif substantive:
        _ensure_path(edges, substantive[0], "risk_assessment")
        _ensure_path(edges, substantive[0], "compliance_review")
    _ensure_path(edges, "risk_assessment", "audit_trail", relation="reviews")
    _ensure_path(edges, "compliance_review", "audit_trail", relation="reviews")
    _ensure_path(edges, "audit_trail", "final_report")
    if profile["dataset"] == "FinanceBench":
        _ensure_path(edges, "compliance_review", "risk_assessment", relation="feedback")
    else:
        _ensure_path(edges, "compliance_review", "calculation_analysis", relation="feedback")
    constraints = [{"id": control, "target": "final_report", "predicate": "required"} for control in profile["required_controls"]]
    return {
        "dataset": profile["dataset"],
        "profile": profile,
        "task_family": str(raw.get("task_family") or profile["task_family"]),
        "financial_focus": str(raw.get("financial_focus") or "dataset-driven financial application construction")[:160],
        "risk_level": str(raw.get("risk_level") or "high").casefold() if str(raw.get("risk_level") or "high").casefold() in {"low", "medium", "high"} else "high",
        "evidence_mode": str(raw.get("evidence_mode") or ("table" if profile["dataset"] == "FinQA" else "filing")),
        "tasks": tasks,
        "edges": edges,
        "constraints": constraints,
        "analysis_source": source,
        "model": raw.get("model"),
        "model_calls": raw.get("model_calls", 0),
        "raw_response_sha256": raw.get("raw_response_sha256"),
        "gold_used": False,
    }


def _ensure_path(edges: list[dict[str, str]], source: str, target: str, relation: str = "precedes") -> None:
    if not any(edge["source"] == source and edge["target"] == target for edge in edges):
        edges.append({"source": source, "target": target, "relation": relation})
