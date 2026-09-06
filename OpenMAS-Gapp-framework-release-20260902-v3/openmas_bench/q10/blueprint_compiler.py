"""Compile a Q10 financial analysis into the shared ConstructionCase schema."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..dataset_adapters import DATASET_ADAPTERS
from ..dataset_cases import build_dataset_case
from ..dynamic_graph import execution_layers, topological_order
from .row_analyzer import analyze_financial_row


def build_q10_case(dataset: str, row: dict[str, Any], llm=None, index: int = 0, seed: int = 11):
    adapter = _adapter(dataset)
    analysis = analyze_financial_row(adapter.dataset_id, row, llm=llm, seed=seed)
    if str(analysis.get("analysis_source", "")).casefold() == "deterministic" or str(analysis.get("model", "")).casefold().startswith("deterministic"):
        raise ValueError("Q10 analysis must be DeepSeek-backed; deterministic analysis is no longer accepted")
    stages = tuple((item["id"], item["objective"]) for item in analysis["tasks"])
    constraints = tuple((item["id"], "financial_governance", item["target"], item["predicate"]) for item in analysis["constraints"])
    template = replace(
        adapter.template,
        family="multi_branch",
        stages=stages,
        constraints=constraints,
        text=(
            "Construct a dataset-aware financial MAS application that separates "
            "filing/table evidence analysis, risk assessment, compliance review, "
            "auditability, and final reporting."
        ),
    )
    dynamic_adapter = replace(adapter, template=template, source_file=f"q10_datasets/normalized/{dataset.casefold()}.jsonl")
    case = build_dataset_case(dynamic_adapter, row, index)
    _replace_dependencies(case, analysis)
    case.harness.metadata["render_layout"] = "order_first_layers"
    case.harness.metadata["render_order"] = [node.id for node in case.harness.nodes]
    case.harness.metadata["topological_order"] = topological_order(case.harness)
    case.harness.metadata["parallel_groups"] = execution_layers(case.harness)
    case.reference_blueprint.metadata["render_layout"] = "order_first_layers"
    case.reference_blueprint.metadata["render_order"] = [node.id for node in case.reference_blueprint.nodes]
    case.metadata["q10_analysis"] = analysis
    case.metadata["graph_layout"] = "order_first_layers"
    case.metadata["graph_order"] = [node.id for node in case.harness.nodes]
    case.metadata["construction_budget"] = {
        "max_components": 16,
        "max_edges": 32,
        "max_planning_steps": 80,
        "max_model_calls": 4,
    }
    case.metadata["task_profile"] = {
        **adapter.task_profile,
        "task_family": analysis["task_family"],
        "dataset": adapter.dataset_id,
        "q10_dynamic": True,
        "risk_level": analysis["risk_level"],
        "evidence_mode": analysis["evidence_mode"],
        "requires_multi_branch": True,
        "requires_evidence_merge": True,
        "requires_constraint_gate": True,
    }
    case.validate()
    return case, analysis


def _adapter(dataset: str):
    key = dataset.casefold()
    aliases = {"finance-bench": "financebench", "finance bench": "financebench"}
    key = aliases.get(key, key)
    adapter = DATASET_ADAPTERS.get(key)
    if adapter is None:
        raise ValueError(f"unknown Q10 dataset {dataset!r}")
    return adapter


def _replace_dependencies(case, analysis: dict[str, Any]) -> None:
    from ..schema import BlueprintEdge, HarnessEdge, TaskDependency, RelationContract

    deps = [TaskDependency(edge["source"], edge["target"], edge["relation"]) for edge in analysis["edges"]]
    case.reference_requirement_model.task_dependencies = deps
    case.harness.edges = [edge for edge in case.harness.edges if edge.relation not in {"precedes", "reviews"}]
    case.harness.edges.extend(HarnessEdge(edge.source, edge.target, "reviews" if edge.relation == "reviews" else "precedes") for edge in deps)
    case.reference_blueprint.edges = [edge for edge in case.reference_blueprint.edges if edge.relation not in {"precedes", "reviews", "feedback"}]
    case.reference_blueprint.edges.extend(BlueprintEdge(edge.source, edge.target, edge.relation) for edge in deps)
    case.contracts.acceptable_relations = [RelationContract(edge.source, edge.target, [edge.relation, "precedes"] if edge.relation == "feedback" else [edge.relation]) for edge in deps]
