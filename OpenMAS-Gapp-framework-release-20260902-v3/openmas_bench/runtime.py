from __future__ import annotations

from collections import defaultdict, deque

from .schema import (
    ApplicationExecutionResult, ConstructionCase, ConstructionExecutionTask, ConstructionResult,
    RuntimeTraceEvent,
)
from .task_adapter import DomainTaskAdapter, answer_matches


class MinimalMARRuntime:
    """Blueprint-preserving capability sandbox for Q1 construction experiments.

    It verifies binding, scheduling, and runtime contracts. It does not solve the
    underlying QA task and therefore never copies a gold answer into predictions.
    """

    def execute(self, case: ConstructionCase, construction: ConstructionResult,
                task: ConstructionExecutionTask, seed: int, task_adapter: DomainTaskAdapter | None = None) -> ApplicationExecutionResult:
        app = construction.application
        order, cycle_broken = _schedule(app.nodes, app.edges)
        by_id = {x.id: x for x in app.nodes}
        blueprint = {x.id: x for x in construction.blueprint.nodes}
        events = []
        for sequence, node_id in enumerate(order):
            node = by_id[node_id]
            bp = blueprint[node.realizes_blueprint_node]
            constraint_refs = list(bp.requirement_refs) if bp.kind == "control" else []
            events.append(RuntimeTraceEvent(
                f"evt_{sequence:04d}", case.case_id, task.id, node.id, "control_enforced" if node.kind == "control" else "component_executed",
                "success", sequence, bp.id, list(node.capabilities), constraint_refs,
                [task.id], [f"artifact_{sequence:04d}"],
                {"implementation_ref": node.implementation_ref, "runtime": "minimal_capability_sandbox"},
            ))
        executed_caps = {cap for event in events for cap in event.capability_refs}
        required_caps = set(task.required_capabilities)
        forbidden = set(case.contracts.forbidden_components)
        used = {event.payload.get("implementation_ref") for event in events}
        required_constraints = set(case.contracts.required_constraints)
        enforced = {ref for event in events for ref in event.constraint_refs}
        binding_valid, binding_errors = _independent_binding_check(case, construction)
        runtime_valid = (binding_valid and required_caps.issubset(executed_caps)
                         and not forbidden.intersection(used)
                         and required_constraints.issubset(enforced))
        task_answer = task_adapter.answer(task, seed) if task_adapter else None
        predicted = task_answer.answer if task_answer else None
        answer_ok = answer_matches(task, predicted)
        return ApplicationExecutionResult(
            case.case_id, construction.method, task.id, seed, events, predicted,
            bool(runtime_valid and answer_ok) if answer_ok is not None else False, runtime_valid,
            {"answer_evaluated": answer_ok is not None, "answer_correct": answer_ok, "cycle_broken_for_schedule": cycle_broken, "gold_answer_used": False, "answer_adapter": task_answer.adapter if task_answer else "disabled", "answer_model": task_answer.model if task_answer else "none", "answer_input_tokens": task_answer.input_tokens if task_answer else 0, "answer_output_tokens": task_answer.output_tokens if task_answer else 0, "binding_valid": binding_valid, "binding_errors": binding_errors},
        )


def _independent_binding_check(case: ConstructionCase, construction: ConstructionResult) -> tuple[bool, list[str]]:
    """Validate executable bindings from contracts + ecosystem, independently
    of the blueprint's candidate lists. This prevents a serialized blueprint
    from receiving credit merely because it contains matching IDs.
    """
    allowed_by_cap: dict[str, set[str]] = defaultdict(set)
    for edge in case.harness.edges:
        if edge.relation != "realizes":
            continue
        cap = next((n for n in case.harness.nodes if n.id == edge.source and n.kind == "capability"), None)
        component = next((n for n in case.harness.nodes if n.id == edge.target and n.kind == "component"), None)
        if cap and component:
            allowed_by_cap[cap.id].add(component.id)
    errors: list[str] = []
    covered: set[str] = set()
    for node in construction.application.nodes:
        if node.kind == "control":
            continue
        for cap in node.capabilities:
            covered.add(cap)
            if node.implementation_ref not in allowed_by_cap.get(cap, set()):
                errors.append(f"{node.id}:{cap}->{node.implementation_ref}")
    missing = set(case.contracts.required_capabilities).difference(covered)
    errors.extend(f"missing:{cap}" for cap in sorted(missing))
    return not errors, errors


def _schedule(nodes, edges):
    node_ids = {x.id for x in nodes}
    graph = defaultdict(list)
    indegree = {x: 0 for x in node_ids}
    for edge in edges:
        if edge.relation == "feedback":
            continue
        graph[edge.source].append(edge.target)
        indegree[edge.target] += 1
    queue = deque(sorted(x for x, degree in indegree.items() if degree == 0))
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in graph[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    remaining = sorted(node_ids.difference(order))
    return order + remaining, bool(remaining)
