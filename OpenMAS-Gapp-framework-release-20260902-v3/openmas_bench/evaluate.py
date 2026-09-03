from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from .schema import ApplicationExecutionResult, ConstructionCase, ConstructionResult, DomainPackage, MASSpec, TraceEvent


def evaluate_construction(case: ConstructionCase, result: ConstructionResult) -> dict[str, Any]:
    """Q1-Q4 structural metrics against contracts, not a unique gold topology."""
    result.validate(case.request())
    predicted = result.requirement_model
    gold_tasks = set(case.contracts.required_tasks)
    predicted_tasks = {x.id for x in predicted.tasks}
    gold_caps = set(case.contracts.required_capabilities)
    predicted_caps = {x.id for x in predicted.capability_requirements}
    gold_constraints = set(case.contracts.required_constraints)
    predicted_constraints = {x.id for x in predicted.constraints}
    task_f1 = _set_f1(gold_tasks, predicted_tasks)
    capability_f1 = _set_f1(gold_caps, predicted_caps)
    constraint_recall = (len(gold_constraints & predicted_constraints) / len(gold_constraints)
                         if gold_constraints else float(not predicted_constraints))
    if result.blueprint.metadata.get("blueprint_present", True):
        injected_constraints = set(result.blueprint.constraint_refs)
    else:
        injected_constraints = {
            ref for node in result.application.nodes
            for ref in node.config.get("requirement_refs", [])
        }
    constraint_orchestration_recall = (
        len(gold_constraints & injected_constraints) / len(gold_constraints)
        if gold_constraints else float(not injected_constraints)
    )

    bp_edges = {(x.source, x.target, x.relation) for x in result.blueprint.edges}
    relation_pass = 0
    for contract in case.contracts.acceptable_relations:
        relation_pass += any((contract.source, contract.target, relation) in bp_edges for relation in contract.relations)
    relation_recall = relation_pass / max(1, len(case.contracts.acceptable_relations))
    required_bp = {f"req_component_{x.task_id}" for x in case.reference_requirement_model.capability_requirements}
    required_bp.update(f"control_{x}" for x in case.contracts.required_constraints)
    predicted_bp = {x.id for x in result.blueprint.nodes if x.kind in {"component_requirement", "control"}}
    blueprint_coverage = len(required_bp & predicted_bp) / max(1, len(required_bp))

    realized = {x.realizes_blueprint_node for x in result.application.nodes}
    realizable = {x.id for x in result.blueprint.nodes
                  if x.kind not in {"task", "resource_requirement"}}
    # Q2 explicitly marks the no-Blueprint ablation.  Its direct plan is not
    # an application-level IR, so it must not receive IR fidelity credit just
    # because the serialization carrier happens to contain matching IDs.
    blueprint_present = result.blueprint.metadata.get("blueprint_present", True)
    preserving = result.application.metadata.get("blueprint_preserving", True)
    realization_fidelity = ((len(realized & realizable) / max(1, len(realizable)))
                            if blueprint_present and preserving else 0.0)
    executable = 1.0 if result.application.nodes and result.application.entrypoints else 0.0
    forbidden = set(case.contracts.forbidden_components)
    selected_implementations = {x.implementation_ref for x in result.application.nodes}
    forbidden_component_rate = len(forbidden & selected_implementations) / max(1, len(forbidden))
    architecture_validity = (relation_recall + realization_fidelity + executable + (1 - forbidden_component_rate)) / 4
    requirement_coverage = (task_f1 + constraint_recall) / 2
    overall = (requirement_coverage + capability_f1 + architecture_validity + constraint_orchestration_recall) / 4
    return {
        "case_id": case.case_id,
        "family": case.family,
        "method": result.method,
        "requirement_task_f1": round(task_f1, 6),
        "capability_requirement_f1": round(capability_f1, 6),
        "constraint_recall": round(constraint_recall, 6),
        "constraint_orchestration_recall": round(constraint_orchestration_recall, 6),
        "orchestration_relation_recall": round(relation_recall, 6),
        "blueprint_coverage": round(blueprint_coverage, 6),
        "realization_fidelity": round(realization_fidelity, 6),
        "executable_validity": executable,
        "requirement_coverage": round(requirement_coverage, 6),
        "capability_completeness": round(capability_f1, 6),
        "architecture_validity": round(architecture_validity, 6),
        "constraint_satisfaction": round(constraint_orchestration_recall, 6),
        "forbidden_component_rate": round(forbidden_component_rate, 6),
        "construction_quality": round(overall, 6),
        "planning_steps": result.telemetry.planning_steps,
        "model_calls": result.telemetry.model_calls,
        "inspected_components": result.telemetry.inspected_components,
        "adapter": result.telemetry.adapter,
        "model": result.telemetry.model,
        "seed": result.telemetry.seed,
        "input_tokens": result.telemetry.input_tokens,
        "output_tokens": result.telemetry.output_tokens,
        "latency_ms": round(result.telemetry.latency_ms, 6),
        "retry_count": result.telemetry.retry_count,
        "json_repaired": float(result.telemetry.json_repaired),
        "fallback": float(result.telemetry.fallback),
    }


def evaluate_execution(case: ConstructionCase, execution: ApplicationExecutionResult) -> dict[str, Any]:
    events = execution.events
    executed_caps = {cap for event in events for cap in event.capability_refs}
    required_caps = {cap for task in case.execution_tasks if task.id == execution.execution_task_id for cap in task.required_capabilities}
    enforced = {ref for event in events for ref in event.constraint_refs}
    used = {event.payload.get("implementation_ref") for event in events}
    capability_rate = len(required_caps & executed_caps) / max(1, len(required_caps))
    constraint_rate = len(set(case.contracts.required_constraints) & enforced) / max(1, len(case.contracts.required_constraints)) if case.contracts.required_constraints else 1.0
    forbidden_rate = len(set(case.contracts.forbidden_components) & used) / max(1, len(case.contracts.forbidden_components))
    trace_pass = 0
    for contract in case.contracts.trace_contracts:
        if contract.kind == "capability_executed":
            trace_pass += contract.target in executed_caps
        elif contract.kind == "constraint_enforced":
            trace_pass += contract.target in enforced
        elif contract.kind == "component_forbidden":
            trace_pass += contract.target not in used
    trace_rate = trace_pass / max(1, len(case.contracts.trace_contracts))
    task = next((x for x in case.execution_tasks if x.id == execution.execution_task_id), None)
    return {
        "case_id": case.case_id, "split": case.split, "family": case.family,
        "method": execution.method, "execution_task_id": execution.execution_task_id,
        "seed": execution.seed, "runtime_valid": float(execution.runtime_valid),
        "capability_execution_rate": round(capability_rate, 6),
        "runtime_constraint_satisfaction": round(constraint_rate, 6),
        "forbidden_invocation_rate": round(forbidden_rate, 6),
        "trace_contract_rate": round(trace_rate, 6),
        "answer_accuracy": (float(execution.metadata.get("answer_correct")) if execution.metadata.get("answer_correct") is not None else None),
        "predicted_answer": execution.predicted_answer,
        "gold_answer": task.answer if task else None,
        "source_dataset": task.source.get("dataset") if task else None,
        "execution_performance": round((float(execution.runtime_valid) + capability_rate + constraint_rate + trace_rate + (1 - forbidden_rate)) / 5, 6),
        "event_count": len(events), "answer_evaluated": bool(execution.metadata.get("answer_evaluated")),
        "answer_input_tokens": execution.metadata.get("answer_input_tokens", 0),
        "answer_output_tokens": execution.metadata.get("answer_output_tokens", 0),
    }


def _set_f1(gold: set[str], predicted: set[str]) -> float:
    true_positive = len(gold & predicted)
    precision = true_positive / max(1, len(predicted))
    recall = true_positive / max(1, len(gold))
    return 2 * precision * recall / max(1e-12, precision + recall)


def evaluate_spec(package: DomainPackage, spec: MASSpec) -> dict[str, Any]:
    selected = set(spec.selected_capabilities)
    required = {x.target for x in package.contracts if x.kind == "capability_required"}
    forbidden = {x.target for x in package.contracts if x.kind == "capability_forbidden"}
    tp = len(required & selected)
    precision = tp / len(selected) if selected else 0.0
    recall = tp / len(required) if required else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    orders = [x.target.split("<", 1) for x in package.contracts if x.kind == "order_required" and "<" in x.target]
    order_pass = sum(_reachable(spec, a, b) for a, b in orders)
    return {
        "package_id": package.package_id,
        "baseline": spec.baseline,
        "required_capabilities": len(required),
        "selected_capabilities": len(selected),
        "capability_precision": round(precision, 6),
        "capability_recall": round(recall, 6),
        "capability_f1": round(f1, 6),
        "forbidden_violation_rate": round(len(forbidden & selected) / max(1, len(forbidden)), 6),
        "order_contract_rate": round(order_pass / max(1, len(orders)), 6),
        "node_count": len(spec.nodes),
        "edge_count": len(spec.edges),
    }


def evaluate_trace(package: DomainPackage, events: list[TraceEvent]) -> dict[str, Any]:
    event_contracts = {ref for event in events for ref in event.contract_refs}
    runtime = {x.id for x in package.contracts if x.kind.startswith("runtime_")}
    return {
        "runtime_contract_rate": len(runtime & event_contracts) / max(1, len(runtime)),
        "event_count": len(events),
        "actors": len({x.actor for x in events}),
    }


def _reachable(spec: MASSpec, source_cap: str, target_cap: str) -> bool:
    cap_nodes = defaultdict(set)
    for node in spec.nodes:
        for cap in node.capabilities:
            cap_nodes[cap].add(node.id)
    graph = defaultdict(list)
    for edge in spec.edges:
        graph[edge.source].append(edge.target)
    targets = cap_nodes[target_cap]
    # A single node containing both capabilities does not demonstrate an
    # explicit workflow dependency. The order contract requires a non-empty
    # execution path between distinct nodes.
    queue = deque()
    for source in cap_nodes[source_cap]:
        for nxt in graph[source]:
            queue.append(nxt)
    seen = set(queue)
    while queue:
        node = queue.popleft()
        if node in targets:
            return True
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False
