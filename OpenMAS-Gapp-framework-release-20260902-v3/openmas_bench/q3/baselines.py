from __future__ import annotations

from dataclasses import dataclass

from ..schema import ApplicationBlueprint, BlueprintEdge, BlueprintNode
from .schema import Q3Case


Q3BlueprintEdge = BlueprintEdge


def Q3BlueprintNode(node_id, kind, label, refs=None, attrs=None) -> BlueprintNode:
    canonical_kind = "resource_requirement" if kind == "resource" else kind
    return BlueprintNode(node_id, canonical_kind, label, list(refs or []), [], dict(attrs or {}))


def Q3Blueprint(case_id, baseline, nodes, edges, metadata=None) -> ApplicationBlueprint:
    normalized = dict(metadata or {})
    if "allowed_kinds" in normalized:
        normalized["allowed_kinds"] = [
            "resource_requirement" if kind == "resource" else kind
            for kind in normalized["allowed_kinds"]
        ]
    return ApplicationBlueprint(case_id, baseline, list(nodes), list(edges), [], normalized)


@dataclass(frozen=True)
class Q3Baseline:
    name: str
    label: str

    def build_blueprint(self, case: Q3Case) -> ApplicationBlueprint:
        if self.name == "flat_component_selection":
            return _flat_blueprint(case, self.name)
        if self.name == "sequence_based_orchestration":
            return _sequence_blueprint(case, self.name)
        if self.name == "tree_based_planning":
            return _tree_blueprint(case, self.name)
        if self.name == "workflow_based_template":
            return _workflow_blueprint(case, self.name)
        if self.name == "agent_graph_orchestration":
            return _agent_graph_blueprint(case, self.name)
        if self.name == "graph_harness":
            return _graph_harness_blueprint(case, self.name)
        raise KeyError(self.name)


Q3_BASELINES: dict[str, Q3Baseline] = {
    "flat_component_selection": Q3Baseline("flat_component_selection", "Flat Component Selection"),
    "sequence_based_orchestration": Q3Baseline("sequence_based_orchestration", "Sequence-based Orchestration"),
    "tree_based_planning": Q3Baseline("tree_based_planning", "Tree-based Planning"),
    "workflow_based_template": Q3Baseline("workflow_based_template", "Workflow-based Template"),
    "agent_graph_orchestration": Q3Baseline("agent_graph_orchestration", "Agent Graph Orchestration"),
    "graph_harness": Q3Baseline("graph_harness", "Graph Harness (Ours)"),
}


def get_q3_baseline(name: str) -> Q3Baseline:
    try:
        return Q3_BASELINES[name]
    except KeyError as exc:
        raise KeyError(f"unknown Q3 baseline {name}; choose from {sorted(Q3_BASELINES)}") from exc


def _task_nodes(case: Q3Case) -> list[Q3BlueprintNode]:
    return [Q3BlueprintNode(task.id, "task", task.label, [task.id]) for task in case.tasks]


def _task_edges(case: Q3Case, relation: str = "precedes") -> list[Q3BlueprintEdge]:
    return [Q3BlueprintEdge(left.id, right.id, relation) for left, right in zip(case.tasks, case.tasks[1:])]


def _graph_task_edges(case: Q3Case) -> list[Q3BlueprintEdge]:
    """Task-level structure available only to the graph representation.

    The sequence/tree/workflow baselines intentionally serialize task stages.
    Graph Harness should preserve the dependency shape implied by the scenario:
    parallel evidence branches converge before synthesis/merge, feedback cases
    keep a reverse review edge, and constraint-heavy cases remain a forward
    chain with an explicit gate inserted later.
    """
    if case.family == "multi_branch" and len(case.tasks) >= 3:
        edges = [
            Q3BlueprintEdge(case.tasks[0].id, case.tasks[2].id, "precedes"),
            Q3BlueprintEdge(case.tasks[1].id, case.tasks[2].id, "precedes"),
        ]
        edges.extend(Q3BlueprintEdge(left.id, right.id, "precedes")
                     for left, right in zip(case.tasks[2:], case.tasks[3:]))
        return edges
    return _task_edges(case)


def _flat_blueprint(case: Q3Case, baseline: str) -> Q3Blueprint:
    nodes = [Q3BlueprintNode(f"comp_{task.id}", "component_requirement", task.label, [task.id]) for task in case.tasks]
    return Q3Blueprint(case.case_id, baseline, nodes, [], {"representation": "flat", "allowed_kinds": ["component_requirement"], "allowed_relations": []})


def _sequence_blueprint(case: Q3Case, baseline: str) -> Q3Blueprint:
    nodes = _task_nodes(case) + [Q3BlueprintNode(f"comp_{task.id}", "component_requirement", task.label, [task.id]) for task in case.tasks]
    edges = _task_edges(case)
    edges.extend(Q3BlueprintEdge(task.id, f"comp_{task.id}", "requires") for task in case.tasks)
    return Q3Blueprint(case.case_id, baseline, nodes, edges, {"representation": "sequence", "allowed_kinds": ["task", "component_requirement"], "allowed_relations": ["precedes", "requires"]})


def _tree_blueprint(case: Q3Case, baseline: str) -> Q3Blueprint:
    root = Q3BlueprintNode("root_task", "task", case.requirement, [t.id for t in case.tasks])
    nodes = [root] + [Q3BlueprintNode(f"branch_{task.id}", "task", task.label, [task.id], {"parent": root.id}) for task in case.tasks]
    nodes.extend(Q3BlueprintNode(f"comp_{task.id}", "component_requirement", task.label, [task.id]) for task in case.tasks)
    edges = [Q3BlueprintEdge(root.id, f"branch_{task.id}", "precedes") for task in case.tasks]
    edges.extend(Q3BlueprintEdge(f"branch_{task.id}", f"comp_{task.id}", "requires") for task in case.tasks)
    return Q3Blueprint(case.case_id, baseline, nodes, edges, {"representation": "tree", "allowed_kinds": ["task", "component_requirement"], "allowed_relations": ["precedes", "requires"]})


def _workflow_blueprint(case: Q3Case, baseline: str) -> Q3Blueprint:
    nodes = _task_nodes(case)
    nodes.extend(Q3BlueprintNode(f"wf_{task.id}", "control", f"workflow step for {task.label}", [task.id], {"template": "fixed"}) for task in case.tasks)
    edges = _task_edges(case)
    edges.extend(Q3BlueprintEdge(task.id, f"wf_{task.id}", "requires") for task in case.tasks)
    return Q3Blueprint(case.case_id, baseline, nodes, edges, {"representation": "workflow", "allowed_kinds": ["task", "control"], "allowed_relations": ["precedes", "requires"]})


def _agent_graph_blueprint(case: Q3Case, baseline: str) -> Q3Blueprint:
    nodes = [Q3BlueprintNode(f"agent_{task.id}", "control", f"agent for {task.label}", [task.id], {"agent": True}) for task in case.tasks]
    edges = [Q3BlueprintEdge(left.id, right.id, "reviews" if case.family == "feedback_driven" and idx == 0 else "precedes")
             for idx, (left, right) in enumerate(zip(nodes, nodes[1:]))]
    if case.family == "multi_branch" and len(nodes) >= 3:
        edges.append(Q3BlueprintEdge(nodes[0].id, nodes[-1].id, "reviews"))
    return Q3Blueprint(case.case_id, baseline, nodes, edges, {"representation": "agent_graph", "allowed_kinds": ["control"], "allowed_relations": ["precedes", "reviews"]})


def _graph_harness_blueprint(case: Q3Case, baseline: str) -> Q3Blueprint:
    nodes = _task_nodes(case)
    nodes.extend(Q3BlueprintNode(f"cap_{task.id}", "capability", task.label, [task.id]) for task in case.tasks)
    nodes.extend(Q3BlueprintNode(
        f"res_{task.id}", "resource", f"resource for {task.label}", [task.id], {"resource": True}
    ) for task in case.tasks)
    nodes.extend(Q3BlueprintNode(
        f"comp_{task.id}", "component_requirement", task.label, [task.id],
        {"candidates": [f"agent_{task.id}"], "capability_ref": f"cap_{task.id}", "resource_ref": f"res_{task.id}"})
        for task in case.tasks)
    resource_ids = [f"res_{task.id}" for task in case.tasks]
    if case.family == "constraint_heavy":
        nodes.append(Q3BlueprintNode(
            "constraint_human_approval", "control", "human approval", ["human_approval"],
            {"constraint": "human_approval", "hard_gate": True, "gate_role": "approval_gate"}))
    if case.family == "feedback_driven":
        nodes.append(Q3BlueprintNode(
            "feedback_loop", "control", "feedback loop", ["feedback"],
            {"loop": True, "hard_cycle": True, "gate_role": "feedback_controller"}))
    if case.family == "multi_branch":
        nodes.append(Q3BlueprintNode(
            "branch_merge", "control", "merge branch outputs before synthesis",
            ["merge"], {"merge": True, "hard_gate": True, "gate_role": "merge_gate"}))
    nodes.append(Q3BlueprintNode(
        "answer_prior", "control", "resource-aware direct answer prior",
        ["answer_prior"], {"resource_prior": True, "direct_candidate": True, "gate_role": "answer_prior"}))
    nodes.append(Q3BlueprintNode(
        "answer_lock", "control", "terminal answer lock",
        ["answer_lock"], {
            "terminal_answer": True,
            "answer_style": _graph_answer_style(case),
            "rerank_candidates": True,
            "math_verify": str(case.dataset_id).casefold() in {"math-500", "math500"},
            "gate_role": "terminal_answer",
            "hard_gate": True,
        }))
    edges = [Q3BlueprintEdge(task.id, f"cap_{task.id}", "requires") for task in case.tasks]
    edges.extend(Q3BlueprintEdge(f"cap_{task.id}", f"comp_{task.id}", "uses") for task in case.tasks)
    edges.extend(Q3BlueprintEdge(f"res_{task.id}", f"comp_{task.id}", "uses") for task in case.tasks)
    edges.extend(_graph_task_edges(case))
    # Route the typed resource carriers into the graph controls so the
    # executable realization can materialize branch-specific context at the
    # actual merge / feedback / approval nodes instead of only at the leaf
    # component requirements.
    for resource_id in resource_ids:
        edges.append(Q3BlueprintEdge(resource_id, "answer_prior", "uses"))
        edges.append(Q3BlueprintEdge(resource_id, "answer_lock", "uses"))
    edges.append(Q3BlueprintEdge("answer_prior", "answer_lock", "precedes"))
    if case.family == "multi_branch" and len(case.tasks) >= 3:
        # The merge control is a pre-synthesis gate: both evidence branches
        # must arrive before the synthesizer/merger can produce an answer
        # candidate.  This prevents the executable order from placing the
        # merge node after the final answer.
        edges.append(Q3BlueprintEdge(case.tasks[0].id, "branch_merge", "precedes"))
        edges.append(Q3BlueprintEdge(case.tasks[1].id, "branch_merge", "precedes"))
        edges.append(Q3BlueprintEdge("branch_merge", case.tasks[2].id, "precedes"))
        edges.append(Q3BlueprintEdge("branch_merge", "answer_lock", "precedes"))
        for resource_id in resource_ids:
            edges.append(Q3BlueprintEdge(resource_id, "branch_merge", "uses"))
    if case.family == "feedback_driven" and len(case.tasks) >= 3:
        edges.append(Q3BlueprintEdge(case.tasks[1].id, "feedback_loop", "precedes"))
        edges.append(Q3BlueprintEdge("feedback_loop", case.tasks[2].id, "precedes"))
        edges.append(Q3BlueprintEdge("feedback_loop", "answer_lock", "precedes"))
        for resource_id in resource_ids:
            edges.append(Q3BlueprintEdge(resource_id, "feedback_loop", "uses"))
        edges.append(Q3BlueprintEdge(case.tasks[-2].id, case.tasks[1].id, "feedback"))
    if case.family == "constraint_heavy":
        # Gate the terminal answer with the immediate predecessor's reviewed
        # artifact.  The control then feeds the answer node.
        predecessor = case.tasks[-2].id if len(case.tasks) >= 2 else case.tasks[-1].id
        edges.append(Q3BlueprintEdge(predecessor, "constraint_human_approval", "constrained_by"))
        edges.append(Q3BlueprintEdge("answer_prior", "constraint_human_approval", "precedes"))
        edges.append(Q3BlueprintEdge("constraint_human_approval", "answer_lock", "precedes"))
        edges.append(Q3BlueprintEdge("constraint_human_approval", case.tasks[-1].id, "precedes"))
        for resource_id in resource_ids:
            edges.append(Q3BlueprintEdge(resource_id, "constraint_human_approval", "uses"))
    elif case.tasks:
        edges.append(Q3BlueprintEdge("answer_prior", case.tasks[-1].id, "precedes"))
        edges.append(Q3BlueprintEdge(case.tasks[-1].id, "answer_lock", "precedes"))
    return Q3Blueprint(case.case_id, baseline, nodes, edges, {"representation": "graph_harness", "allowed_kinds": ["task", "capability", "resource", "component_requirement", "control"], "allowed_relations": ["requires", "uses", "precedes", "depends", "constrained_by", "feedback", "reviews"]})


def _graph_answer_style(case: Q3Case) -> str:
    row = case.metadata.get("row") if isinstance(case.metadata, dict) else {}
    choices = row.get("choices") if isinstance(row, dict) else None
    dataset = str(case.dataset_id).casefold()
    if dataset in {"medqa", "mmlu", "mmlu-pro", "arc", "logiqa", "sciq"} or choices:
        return "exact_option"
    if dataset in {"finqa", "gsm8k", "math-500", "math500", "drop"}:
        return "numeric_or_span"
    if dataset in {"hotpotqa", "musique", "strategyqa", "pubmedqa"}:
        return "concise_evidence_span"
    return "minimal_answer"
