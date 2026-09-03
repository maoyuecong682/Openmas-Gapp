"""Dataset-column evaluation for the revised Q2 ablation protocol.

The primary result is one metric per dataset/track, paired against the full
Graph Harness row.  Structural diagnostics remain available, but are not
averaged into a single saturated construction score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class Q2DatasetTrack:
    dataset_id: str
    label: str
    metric_name: str
    description: str
    selector: Callable[[dict[str, Any]], bool]
    scorer: Callable[[dict[str, Any]], float]


def _source(row: dict[str, Any]) -> str:
    return str(row.get("source_dataset") or row.get("source") or "")


def _execution_success(row: dict[str, Any]) -> float:
    """Primary answer success; runtime/trace remain separate diagnostics."""
    answer = row.get("answer_accuracy")
    if answer is None:
        return 0.0
    return float(answer)


def _trace_proxy(row: dict[str, Any]) -> float:
    return float(row.get("answer_accuracy", row.get("patch_score", 0.0)) or 0.0)


def _construction_metric(row: dict[str, Any], name: str) -> float:
    return float(row.get(name, 0.0))


def _all(_: dict[str, Any]) -> bool:
    return True


DATASET_TRACKS: tuple[Q2DatasetTrack, ...] = (
    Q2DatasetTrack("FinQA", "FinQA", "task_success", "Financial QA success gated by runtime contracts",
                   lambda r: "FinQA" in _source(r), _execution_success),
    Q2DatasetTrack("PubMedQA", "PubMedQA", "task_success", "Biomedical QA success gated by runtime contracts",
                   lambda r: "PubMedQA" in _source(r), _execution_success),
    Q2DatasetTrack("SWE-bench", "SWE-bench", "trace_patch_proxy", "Patch/trace proxy gated by executable runtime",
                   lambda r: "SWE" in _source(r), _trace_proxy),
    Q2DatasetTrack("RequirementCases", "Requirement Alignment", "strict_requirement_alignment",
                   "Case contracts for task-capability-constraint binding", _all,
                   lambda r: _construction_metric(r, "strict_requirement_alignment")),
    Q2DatasetTrack("WorkflowCases", "Workflow Structure", "strict_path_success",
                   "Exact relation and path contracts", _all,
                   lambda r: _construction_metric(r, "strict_path_success")),
    Q2DatasetTrack("BlueprintCases", "Blueprint Fidelity", "strict_blueprint_fidelity",
                   "Blueprint node, edge and binding preservation", _all,
                   lambda r: _construction_metric(r, "strict_blueprint_fidelity")),
    Q2DatasetTrack("GovernanceCases", "Governance", "strict_governance_success",
                   "Constraint-heavy application governance contracts",
                   lambda r: r.get("family") == "constraint_heavy",
                   lambda r: _construction_metric(r, "strict_governance_success")),
)


def strict_construction_scores(row: dict[str, Any]) -> dict[str, float]:
    """Compute non-saturated structural diagnostics from serialized outputs.

    Existing Q2 result rows contain the construction metrics and execution
    summaries.  These scores intentionally require multiple dimensions instead
    of checking only whether an ID appeared somewhere.
    """
    task = float(row.get("requirement_task_f1", 0.0))
    cap = float(row.get("capability_requirement_f1", 0.0))
    constraint_detection = float(row.get("constraint_recall", 0.0))
    constraint_orchestration = float(
        row.get("constraint_orchestration_recall",
                row.get("constraint_satisfaction", 0.0))
    )
    relation = float(row.get("orchestration_relation_recall", 0.0))
    realization = float(row.get("realization_fidelity", 0.0))
    executable = float(row.get("executable_validity", 0.0))
    forbidden = float(row.get("forbidden_component_rate", 0.0))
    # Strict alignment includes all declarations and penalizes forbidden use.
    alignment = (task + cap + constraint_detection + (1.0 - forbidden)) / 4.0
    # A path must be represented and executable; relation declaration alone is
    # insufficient for the strict score.
    path = relation * executable
    # Blueprint fidelity is gated by executable validity and realization.
    bp = realization * executable
    governance = constraint_orchestration * executable * (1.0 - forbidden)
    return {"strict_requirement_alignment": alignment,
            "strict_path_success": path,
            "strict_blueprint_fidelity": bp,
            "strict_governance_success": governance}


def add_strict_scores(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row.update(strict_construction_scores(row))
    return row


def primary_table(construction_rows: Iterable[dict[str, Any]], execution_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one primary metric per dataset track and variant."""
    rows = [add_strict_scores(x) for x in construction_rows]
    executions = list(execution_rows)
    # Dataset tracks use execution rows for task datasets and construction rows
    # for the application-construction tracks.
    output: list[dict[str, Any]] = []
    for track in DATASET_TRACKS:
        source = executions if track.dataset_id in {"FinQA", "PubMedQA", "SWE-bench"} else rows
        grouped: dict[str, list[float]] = {}
        for row in source:
            if track.selector(row):
                variant = row.get("variant") or row.get("method")
                grouped.setdefault(str(variant), []).append(track.scorer(row))
        for variant, values in sorted(grouped.items()):
            output.append({"dataset": track.dataset_id, "label": track.label,
                           "metric": track.metric_name, "variant": variant,
                           "mean": round(sum(values) / len(values), 6), "n": len(values)})
    return output


def paired_deltas(table: Iterable[dict[str, Any]], full_variant: str = "full_graph_harness") -> list[dict[str, Any]]:
    rows = list(table)
    full = {(x["dataset"], x["metric"]): x["mean"] for x in rows if x["variant"] == full_variant}
    output = []
    for row in rows:
        key = (row["dataset"], row["metric"])
        output.append({**row, "full_mean": full.get(key),
                       "delta_vs_full": round(row["mean"] - full[key], 6) if key in full else None})
    return output
