"""Dataset-level priors for the Q10 financial application case study."""
from __future__ import annotations

from typing import Any


PROFILES: dict[str, dict[str, Any]] = {
    "FinanceBench": {
        "task_family": "financial_application_case",
        "domain": "finance",
        "input_features": [
            "filing_evidence",
            "metric_reasoning",
            "risk_factors",
            "regulatory_constraints",
        ],
        "baseline_tasks": [
            "filing_evidence",
            "market_risk_evidence",
            "financial_analysis",
            "risk_assessment",
            "compliance_review",
            "audit_trail",
            "final_report",
        ],
        "required_controls": [
            "risk_control",
            "regulatory_compliance",
            "auditability",
        ],
    },
    "FinQA": {
        "task_family": "financial_application_case",
        "domain": "finance",
        "input_features": [
            "financial_table",
            "narrative_disclosure",
            "numeric_program",
            "unit_verification",
            "auditability",
        ],
        "baseline_tasks": [
            "report_table_evidence",
            "narrative_disclosure_evidence",
            "calculation_analysis",
            "risk_assessment",
            "compliance_review",
            "audit_trail",
            "final_report",
        ],
        "required_controls": [
            "risk_control",
            "regulatory_compliance",
            "auditability",
        ],
    },
}


def get_financial_profile(dataset: str) -> dict[str, Any]:
    canonical = {
        "financebench": "FinanceBench",
        "finqa": "FinQA",
    }.get(dataset.casefold(), dataset)
    if canonical not in PROFILES:
        raise ValueError(f"unsupported Q10 financial dataset {dataset!r}")
    return {**PROFILES[canonical], "dataset": canonical}
