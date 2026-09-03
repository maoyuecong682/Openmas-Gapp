from __future__ import annotations

from collections import defaultdict, deque

from ..schema import ApplicationBlueprint
from .schema import Q3Case


def compute_osv(case: Q3Case, blueprint: ApplicationBlueprint) -> tuple[float, list[str]]:
    checks = [_check_allowed_kinds(blueprint), _check_allowed_relations(blueprint)]
    if case.family == "sequential":
        checks.append(_check_sequential(case, blueprint))
    elif case.family == "multi_branch":
        checks.append(_check_multi_branch(case, blueprint))
    elif case.family == "feedback_driven":
        checks.append(_check_feedback(case, blueprint))
    elif case.family == "constraint_heavy":
        checks.append(_check_constraint(case, blueprint))
    else:
        checks.append((False, [f"unknown family {case.family}"]))
    success = all(flag for flag, _ in checks)
    notes = [note for _, items in checks for note in items]
    return (1.0 if success else 0.0), notes


def _check_sequential(case: Q3Case, blueprint: ApplicationBlueprint) -> tuple[bool, list[str]]:
    order = _reachable_order(blueprint)
    expected = [task.id for task in case.tasks]
    if expected and order[:len(expected)] != expected:
        return False, ["sequential order mismatch"]
    if any(edge.relation in {"feedback", "depends", "constrained_by"} for edge in blueprint.edges):
        return False, ["sequential blueprint contains non-linear relations"]
    return True, []


def _check_multi_branch(case: Q3Case, blueprint: ApplicationBlueprint) -> tuple[bool, list[str]]:
    graph = _graph(blueprint)
    if len(case.tasks) < 3:
        return False, ["insufficient branches"]
    branch_sources = {case.tasks[0].id, case.tasks[1].id}
    convergence = case.tasks[2].id
    if not all(convergence in _reachable_set(graph, source) for source in branch_sources):
        return False, ["branch convergence missing"]
    if not any(node.kind == "control" and node.attrs.get("merge") for node in blueprint.nodes):
        return False, ["merge control missing"]
    if not all(any(edge.source == source and edge.target == "branch_merge" for edge in blueprint.edges)
               for source in branch_sources):
        return False, ["branch merge input missing"]
    if not any(edge.source == "branch_merge" and edge.target == convergence for edge in blueprint.edges):
        return False, ["branch merge output missing"]
    return True, []


def _check_feedback(case: Q3Case, blueprint: ApplicationBlueprint) -> tuple[bool, list[str]]:
    graph = _graph(blueprint)
    has_cycle = _has_cycle(graph)
    if not has_cycle and not any(edge.relation == "feedback" for edge in blueprint.edges):
        return False, ["feedback loop missing"]
    if not any(node.kind == "control" and node.attrs.get("hard_cycle") for node in blueprint.nodes):
        return False, ["hard cycle control missing"]
    terminal = case.tasks[-1].id
    if terminal not in {edge.target for edge in blueprint.edges}:
        return False, ["termination missing"]
    return True, []


def _check_constraint(case: Q3Case, blueprint: ApplicationBlueprint) -> tuple[bool, list[str]]:
    if not any(node.kind == "control" and node.attrs.get("hard_gate") for node in blueprint.nodes):
        return False, ["constraint control missing"]
    if not any(edge.relation == "constrained_by" for edge in blueprint.edges):
        return False, ["constraint edge missing"]
    gate_ids = {node.id for node in blueprint.nodes if node.kind == "control" and node.attrs.get("hard_gate")}
    terminal = case.tasks[-1].id
    terminal_directly_constrained = terminal in {edge.target for edge in blueprint.edges if edge.relation == "constrained_by"}
    gated_terminal = any(edge.source in gate_ids and edge.target == terminal for edge in blueprint.edges)
    constrained_gate = any(edge.target in gate_ids and edge.relation == "constrained_by" for edge in blueprint.edges)
    if not terminal_directly_constrained and not (gated_terminal and constrained_gate):
        return False, ["constraint not attached to terminal path"]
    return True, []


def _check_allowed_kinds(blueprint: ApplicationBlueprint) -> tuple[bool, list[str]]:
    allowed = set(blueprint.metadata.get("allowed_kinds", []))
    if not allowed:
        return True, []
    kinds = {node.kind for node in blueprint.nodes}
    extra = kinds.difference(allowed)
    if extra:
        return False, [f"disallowed node kinds: {sorted(extra)}"]
    return True, []


def _check_allowed_relations(blueprint: ApplicationBlueprint) -> tuple[bool, list[str]]:
    allowed = set(blueprint.metadata.get("allowed_relations", []))
    if not allowed:
        return True, []
    extra = {edge.relation for edge in blueprint.edges}.difference(allowed)
    if extra:
        return False, [f"disallowed relations: {sorted(extra)}"]
    return True, []


def _graph(blueprint: ApplicationBlueprint) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = defaultdict(list)
    for edge in blueprint.edges:
        graph[edge.source].append(edge.target)
    return graph


def _reachable_order(blueprint: ApplicationBlueprint) -> list[str]:
    graph = _graph(blueprint)
    indegree = defaultdict(int)
    nodes = [node.id for node in blueprint.nodes]
    for node in nodes:
        indegree[node] = indegree[node]
    for edge in blueprint.edges:
        indegree[edge.target] += 1
    queue = deque([node for node in nodes if indegree[node] == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in graph.get(node, []):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return order


def _reachable_set(graph: dict[str, list[str]], start: str, allow_self: bool = True) -> set[str]:
    queue = deque([start])
    seen = {start}
    reach = set()
    while queue:
        node = queue.popleft()
        for nxt in graph.get(node, []):
            if nxt == start and allow_self:
                return {start}
            if nxt not in seen:
                seen.add(nxt)
                reach.add(nxt)
                queue.append(nxt)
    return reach


def _has_cycle(graph: dict[str, list[str]]) -> bool:
    seen: set[str] = set()
    stack: set[str] = set()

    def visit(node: str) -> bool:
        if node in stack:
            return True
        if node in seen:
            return False
        seen.add(node)
        stack.add(node)
        for nxt in graph.get(node, []):
            if visit(nxt):
                return True
        stack.remove(node)
        return False

    return any(visit(node) for node in graph)
