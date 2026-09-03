from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..engine import GraphHarnessEngine
from ..evaluate import evaluate_construction, evaluate_execution
from ..llm import DeterministicAdapter
from ..runtime import MinimalMARRuntime
from ..schema import (
    ApplicationBlueprint, BlueprintNode, ConstructionCase,
    ConstructionResult, ConstructionTelemetry,
)
from .baselines import Q3_BASELINES
from .osv import compute_osv
from .schema import Q3Case, Q3Task


def case_to_q3(case: ConstructionCase) -> Q3Case:
    tasks = [Q3Task(task.id, task.objective) for task in case.reference_requirement_model.tasks]
    dataset_id = case.metadata.get("dataset", case.domain) if isinstance(case.metadata, dict) else case.domain
    return Q3Case(case.case_id, str(dataset_id), case.family, case.domain, case.raw_requirement, tasks, metadata=dict(case.metadata))


def real_construction_result(case: ConstructionCase, baseline_name: str, seed: int = 0,
                             engine: GraphHarnessEngine | None = None) -> ConstructionResult:
    q3_case = case_to_q3(case)
    blueprint = _real_blueprint(q3_case, case, baseline_name)
    model = case.reference_requirement_model
    telemetry = ConstructionTelemetry(
        planning_steps=0,
        model_calls=0,
        inspected_components=len([node for node in case.harness.nodes if node.kind == "component"]),
        notes=[f"q3_baseline={baseline_name}", f"seed={seed}", "real_benchmark_flow"],
        adapter="benchmark",
        model="real_benchmark_flow",
        seed=seed,
    )
    engine = engine or GraphHarnessEngine(DeterministicAdapter(), Path("."))
    result = engine.realize(case, blueprint, baseline_name, telemetry)
    telemetry.planning_steps = len(result.application.nodes) + len(result.application.edges)
    if baseline_name == "graph_harness":
        _enable_graph_resource_execution(result.application)
    result.validate(case.request())
    return result


def _enable_graph_resource_execution(application) -> None:
    for node in application.nodes:
        node.config["resource_access"] = True
        instruction = str(node.config.get("execution_instruction", ""))
        if "Resource-aware Graph Harness execution:" not in instruction:
            node.config["execution_instruction"] = (
                instruction
                + "\nResource-aware Graph Harness execution: consult the original task resource when available, "
                  "preserve exact question constraints/options/entities, and use upstream artifacts as the proposed "
                  "work product to audit or refine rather than blindly copy."
            )


def run_real_case(case: ConstructionCase, baseline_name: str, seed: int, adapter, data_root, task_executor, runtime: MinimalMARRuntime, dataset=None, row: dict[str, Any] | None = None) -> dict[str, Any]:
    engine = GraphHarnessEngine(adapter, data_root, runtime=runtime, task_executor=task_executor)
    construction = real_construction_result(case, baseline_name, seed, engine=engine)
    construction_metrics = evaluate_construction(case, construction)
    q3_case = case_to_q3(case)
    osv, osv_notes = compute_osv(q3_case, _real_blueprint(q3_case, case, baseline_name))
    graph_structural_preservation = _graph_structural_preservation(q3_case, _real_blueprint(q3_case, case, baseline_name))
    execution_rows = []
    primary_scores: list[float] = []
    task_successes: list[float] = []
    e2e_supported = dataset is not None
    for task_index, task in enumerate(case.execution_tasks):
        if dataset is not None and row is not None:
            engine_result = engine.execute(
                dataset, row, case, construction, seed,
                intervention=baseline_name, task_index=task_index)
            task_result = engine_result.task_execution
            prediction = engine_result.prediction
            answer_score = engine_result.answer_score
            execution_metrics = engine_result.runtime_diagnostics
            execution_metrics["answer_accuracy"] = answer_score
            execution_metrics["answer_evaluated"] = answer_score is not None
            task_success = float(answer_score) if answer_score is not None else None
            if task_success is None:
                e2e_supported = False
            else:
                task_successes.append(task_success)
            primary_scores.append(float(answer_score) if answer_score is not None else 0.0)
            execution_rows.append({
                "task_id": task.id,
                "prediction": prediction,
                "answer_score": answer_score,
                "task_execution": task_result.to_dict(),
                "metrics": execution_metrics,
                "engine_audit": engine_result.audit,
            })
        else:
            execution = runtime.execute(case, construction, task, seed)
            execution_metrics = evaluate_execution(case, execution)
            execution_rows.append({"task_id": task.id, "execution": asdict(execution), "metrics": execution_metrics})
    e2e_success = (sum(task_successes) / len(task_successes)) if e2e_supported and task_successes else None
    return {
        "case_id": case.case_id,
        "family": case.family,
        "domain": case.domain,
        "baseline": baseline_name,
        "seed": seed,
        "construction": construction_metrics,
        "execution": execution_rows,
        "e2e_success": round(e2e_success, 6) if e2e_success is not None else None,
        "e2e_supported": e2e_supported and bool(task_successes),
        "primary_score_mean": round(sum(primary_scores) / len(primary_scores), 6) if primary_scores else None,
        "osv": osv,
        "osv_notes": osv_notes,
        "graph_structural_preservation": round(graph_structural_preservation, 6),
    }


def _real_blueprint(q3_case: Q3Case, case: ConstructionCase,
                    baseline_name: str) -> ApplicationBlueprint:
    baseline = Q3_BASELINES[baseline_name]
    blueprint = baseline.build_blueprint(q3_case)
    candidate_lookup = {
        node.id.replace("req_", "").replace("comp_", ""): list(node.binding_constraints.get("candidates", []))
        for node in case.reference_blueprint.nodes
        if node.kind == "component_requirement"
    }
    constraint_candidates = {
        node.id.replace("control_", ""): node.binding_constraints.get("candidate")
        for node in case.reference_blueprint.nodes
        if node.kind == "control"
    }
    blueprint_nodes = {node.id: node for node in blueprint.nodes}
    component_context: dict[str, dict[str, list[str]]] = {}
    for edge in blueprint.edges:
        source = blueprint_nodes.get(edge.source)
        target = blueprint_nodes.get(edge.target)
        if source is None or target is None or target.kind != "component_requirement":
            continue
        if source.kind == "capability":
            component_context.setdefault(target.id, {"capabilities": [], "resources": []})[
                "capabilities"].append(source.description)
        elif source.kind == "resource_requirement":
            component_context.setdefault(target.id, {"capabilities": [], "resources": []})[
                "resources"].append(source.description)

    new_nodes = []
    for node in blueprint.nodes:
        attrs = dict(node.attrs)
        if node.kind == "component_requirement":
            task_id = node.id.replace("comp_", "")
            attrs["candidates"] = candidate_lookup.get(task_id, attrs.get("candidates", []))
        if node.kind == "control" and baseline_name == "graph_harness":
            key = node.id.replace("constraint_", "").replace("feedback_", "")
            if key in constraint_candidates and constraint_candidates[key]:
                attrs["candidate"] = constraint_candidates[key]
        staged = BlueprintNode(node.id, node.kind, node.description,
                               list(node.requirement_refs), list(node.capability_refs), attrs)
        description = _q3_executable_description(
            staged, component_context.get(node.id), case, blueprint)
        new_nodes.append(BlueprintNode(
            node.id, node.kind, description, list(node.requirement_refs),
            list(node.capability_refs), attrs))
    constraint_refs = [node.requirement_refs[0] for node in new_nodes
                       if node.kind == "control" and node.requirement_refs]
    metadata = dict(blueprint.metadata)
    metadata.update({"schema": "ApplicationBlueprint", "q": "Q3"})
    result = ApplicationBlueprint(
        blueprint.case_id, blueprint.method, new_nodes,
        list(blueprint.edges), constraint_refs, metadata)
    result.validate()
    return result


def _q3_executable_description(node, context: dict[str, list[str]] | None,
                               case: ConstructionCase, blueprint: ApplicationBlueprint) -> str:
    description = node.label
    if blueprint.baseline == "graph_harness" and context:
        capabilities = "; ".join(context.get("capabilities", []))
        resources = "; ".join(context.get("resources", []))
        additions = []
        if capabilities:
            additions.append(f"Bound capability: {capabilities}.")
        if resources:
            additions.append(f"Required resource context: {resources}.")
        if additions:
            description = f"{description}\n" + " ".join(additions)
    if node.kind == "control" and blueprint.baseline == "graph_harness":
        if node.attrs.get("resource_prior"):
            description += ("\nDirect prior contract: use the original task resource to produce a concise candidate_answer "
                            "before downstream graph verification. Preserve exact option labels, yes/no/maybe labels, "
                            "numeric answers, or short entity spans according to the dataset, and keep the candidate_answer "
                            "consistent with upstream evidence.")
        if node.attrs.get("answer_style") == "exact_option":
            description += ("\nAnswer style: emit exactly one option label or option text supported by the original "
                            "question and verified evidence. Do not provide reasoning.")
        elif node.attrs.get("answer_style") == "numeric_or_span":
            description += ("\nAnswer style: emit only the numeric result, exact expression, or shortest supported span. "
                            "Do not round unless the dataset convention requires it.")
        elif node.attrs.get("answer_style") == "concise_evidence_span":
            description += ("\nAnswer style: emit only a concise evidence-backed span or yes/no label, with no extra text.")
        if node.attrs.get("terminal_answer"):
            description += ("\nTerminal answer lock: choose the final answer after checking every upstream artifact. "
                            "Prioritize correctness, exact dataset format, and the shortest valid answer string.")
        if node.attrs.get("rerank_candidates"):
            description += ("\nCandidate reranking: compare all upstream candidate answers, prefer the one supported by "
                            "the strongest verified evidence, and break ties by exactness and format compliance.")
        if node.attrs.get("math_verify"):
            description += ("\nMath verification: normalize equivalent expressions, recompute the numeric result when "
                            "possible, and prefer the candidate that matches the verified value exactly.")
        if node.attrs.get("merge"):
            description += ("\nMerge contract: preserve every upstream evidence branch, resolve conflicts explicitly, "
                            "and emit a single candidate_answer for the downstream verifier. If the branch resources "
                            "disagree, summarize the conflict before choosing the best supported answer.")
        if node.attrs.get("loop"):
            description += ("\nFeedback-loop contract: treat the upstream artifact as a provisional draft, "
                            "revisit prior evidence, and return a revised artifact that preserves the original "
                            "question and the latest correction path.")
        if node.attrs.get("hard_gate"):
            description += ("\nGate contract: audit the upstream candidate before the final answer; "
                            "if unsafe, unsupported, or format-invalid, correct it and keep the decision trace visible.")
    if _is_terminal_answer_node(node, case):
        if blueprint.baseline == "graph_harness":
            description += ("\nGraph Harness finalization: compare the resource-aware direct prior with the structured "
                            "upstream artifacts, including any branch resources routed through the graph. If they agree, "
                            "return that answer. If they disagree, choose the answer best supported by the original "
                            "question/context/options and the verified upstream evidence.")
        description += "\n" + _answer_format_instruction(case)
    return description


def _is_terminal_answer_node(node, case: ConstructionCase) -> bool:
    if not node.refs:
        return "answer" in node.id.casefold() or "answer" in node.label.casefold()
    return any(ref == case.reference_requirement_model.tasks[-1].id for ref in node.refs)


def _answer_format_instruction(case: ConstructionCase) -> str:
    dataset_id = str(case.metadata.get("dataset", "") if isinstance(case.metadata, dict) else "").casefold()
    row = case.metadata.get("row", {}) if isinstance(case.metadata, dict) else {}
    choices = row.get("choices") if isinstance(row, dict) else None
    if dataset_id in {"medqa", "mmlu", "sciq"} or choices:
        return ("Final answer format: return exactly one option, preferably the option letter followed by the option text; "
                "do not return a paragraph.")
    if dataset_id == "pubmedqa":
        return "Final answer format: return exactly one label: yes, no, or maybe."
    if dataset_id in {"financebench", "bbh", "bbh_full", "bbh-full"}:
        return "Final answer format: return only the concise benchmark answer span or label, with no explanation."
    if dataset_id == "scibench":
        return "Final answer format: return only the final numeric or symbolic answer, with no explanation."
    if dataset_id in {"hotpotqa", "musique", "strategyqa"}:
        return "Final answer format: return only the concise answer span or yes/no label, with no explanation."
    if dataset_id in {"gsm8k", "math-500", "math500", "finqa", "drop"}:
        return "Final answer format: return only the final numeric/math answer, with no explanation."
    return "Final answer format: return only the final benchmark answer."


def _graph_structural_preservation(case: Q3Case, blueprint: ApplicationBlueprint) -> float:
    """A graph-specific score for whether typed structural relations survive."""
    scores = []
    graph = {}
    incoming = {}
    for edge in blueprint.edges:
        graph.setdefault(edge.source, set()).add(edge.target)
        incoming.setdefault(edge.target, set()).add(edge.source)

    task_ids = [task.id for task in case.tasks]
    if case.family == "multi_branch" and len(task_ids) >= 3:
        branch_merge = "branch_merge"
        branch_inputs = all(branch_merge in graph.get(task_ids[i], set()) for i in (0, 1))
        branch_output = task_ids[2] in graph.get(branch_merge, set())
        resource_inputs = any(edge.relation == "uses" and edge.target == branch_merge for edge in blueprint.edges)
        scores.append(float(branch_inputs and branch_output and resource_inputs))
    elif case.family == "feedback_driven":
        has_feedback = any(edge.relation == "feedback" for edge in blueprint.edges)
        has_cycle = any(node.kind == "control" and node.attrs.get("hard_cycle") for node in blueprint.nodes)
        loop_id = "feedback_loop"
        loop_in_path = loop_id in incoming and any(src in {task_ids[1] if len(task_ids) > 1 else "", task_ids[2] if len(task_ids) > 2 else ""} for src in incoming.get(loop_id, set()))
        loop_out_path = any(loop_id in graph.get(task_id, set()) for task_id in task_ids)
        resource_inputs = any(edge.relation == "uses" and edge.target == loop_id for edge in blueprint.edges)
        scores.append(float(has_feedback and has_cycle and loop_in_path and loop_out_path and resource_inputs))
    elif case.family == "constraint_heavy":
        has_gate = any(node.kind == "control" and node.attrs.get("hard_gate") for node in blueprint.nodes)
        has_constrained = any(edge.relation == "constrained_by" for edge in blueprint.edges)
        gate_ids = {node.id for node in blueprint.nodes if node.kind == "control" and node.attrs.get("hard_gate")}
        gated_terminal = any(edge.source in gate_ids and edge.target == task_ids[-1] for edge in blueprint.edges)
        resource_inputs = any(edge.relation == "uses" and edge.target in gate_ids for edge in blueprint.edges)
        scores.append(float(has_gate and has_constrained and gated_terminal and resource_inputs))
    else:
        # For sequential-style cases, graph preservation means explicit typed
        # task ordering plus no spurious control leakage.
        linear = all(edge.relation == "precedes" for edge in blueprint.edges if edge.source in task_ids and edge.target in task_ids)
        no_special = not any(node.kind == "control" for node in blueprint.nodes)
        scores.append(float(linear and no_special))

    # Reward explicit capability/resource carriers for the graph baseline.
    typed_carries = sum(1 for node in blueprint.nodes
                        if node.kind in {"capability", "resource_requirement", "control"})
    task_count = max(1, len(task_ids))
    scores.append(min(1.0, typed_carries / (2 * task_count)))
    if blueprint.baseline == "graph_harness":
        graph_targets = {
            edge.target for edge in blueprint.edges
            if edge.relation == "uses" and edge.source in {
                node.id for node in blueprint.nodes if node.kind == "resource_requirement"}
        }
        if "answer_prior" in graph_targets:
            scores.append(1.0)
        else:
            scores.append(0.0)
    return sum(scores) / len(scores)
