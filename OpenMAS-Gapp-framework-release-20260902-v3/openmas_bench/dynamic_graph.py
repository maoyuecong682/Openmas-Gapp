from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any

from .dynamic_planner import PlannerGraphSpec
from .schema import HarnessEdge, HarnessGraph, HarnessNode


@dataclass
class DynamicHarnessBundle:
    task: str
    planner: PlannerGraphSpec
    harness: HarnessGraph

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "planner": self.planner.to_dict(),
            "harness": asdict(self.harness),
        }


def _normalize_id(value: str, fallback: str) -> str:
    text = re.sub(r"[^a-z0-9_]+", "_", str(value).strip().casefold())
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = fallback
    if not text[0].isalpha():
        text = f"a_{text}"
    return text[:64]


class DynamicGraphBuilder:
    """Turn planner decisions into a concrete HarnessGraph.

    The builder does not decide which agents exist. It only validates and
    instantiates the planner's chosen structure.
    """

    def build(self, task: str, plan: PlannerGraphSpec) -> DynamicHarnessBundle:
        plan.validate()
        agent_map: dict[str, str] = {}
        seen: set[str] = set()
        nodes: list[HarnessNode] = []
        for index, agent in enumerate(plan.agents):
            normalized = _normalize_id(agent.id, f"agent_{index + 1}")
            if normalized in seen:
                suffix = 2
                candidate = f"{normalized}_{suffix}"
                while candidate in seen:
                    suffix += 1
                    candidate = f"{normalized}_{suffix}"
                normalized = candidate
            seen.add(normalized)
            agent_map[agent.id] = normalized
            nodes.append(
                HarnessNode(
                    normalized,
                    "component",
                    agent.objective,
                    capabilities=list(agent.capabilities),
                    tags=[agent.role, agent.id],
                    metadata={
                        "runtime_kind": "agent",
                        "role": agent.role,
                        "objective": agent.objective,
                        "tools": list(agent.tools),
                        "planner_agent_id": agent.id,
                    },
                )
            )
        edges: list[HarnessEdge] = []
        for edge in plan.edges:
            source = agent_map.get(edge.source, _normalize_id(edge.source, edge.source))
            target = agent_map.get(edge.target, _normalize_id(edge.target, edge.target))
            if source not in seen or target not in seen:
                raise ValueError(f"planner edge references unknown agent: {edge.source!r} -> {edge.target!r}")
            if source == target:
                raise ValueError(f"planner edge may not be a self-loop: {source!r}")
            edges.append(HarnessEdge(source, target, "precedes"))
        harness = HarnessGraph(
            nodes,
            edges,
            metadata={
                "task": task,
                "planner": "deepseek_graph_planner",
                "agent_count": len(nodes),
                "edge_count": len(edges),
            },
        )
        harness.validate()
        _validate_acyclic(harness)
        harness.metadata["topological_order"] = topological_order(harness)
        harness.metadata["parallel_groups"] = execution_layers(harness)
        return DynamicHarnessBundle(task=task, planner=plan, harness=harness)


def topological_order(harness: HarnessGraph) -> list[str]:
    node_ids = {node.id for node in harness.nodes}
    declaration_order = {node.id: index for index, node in enumerate(harness.nodes)}
    graph: dict[str, list[str]] = defaultdict(list)
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in harness.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            continue
        graph[edge.source].append(edge.target)
        indegree[edge.target] += 1
    queue = deque(sorted((node_id for node_id, degree in indegree.items() if degree == 0), key=declaration_order.get))
    order: list[str] = []
    while queue:
        current = queue.popleft()
        order.append(current)
        for target in graph[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(order) != len(node_ids):
        raise ValueError("harness graph contains a cycle")
    return order


def execution_layers(harness: HarnessGraph) -> list[list[str]]:
    node_ids = {node.id for node in harness.nodes}
    declaration_order = {node.id: index for index, node in enumerate(harness.nodes)}
    graph: dict[str, list[str]] = defaultdict(list)
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in harness.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            continue
        graph[edge.source].append(edge.target)
        indegree[edge.target] += 1
    ready = sorted((node_id for node_id, degree in indegree.items() if degree == 0), key=declaration_order.get)
    layers: list[list[str]] = []
    seen: set[str] = set()
    while ready:
        layer = ready
        layers.append(layer)
        next_ready: list[str] = []
        for current in layer:
            seen.add(current)
            for target in graph[current]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    next_ready.append(target)
        ready = sorted(next_ready, key=declaration_order.get)
    if len(seen) != len(node_ids):
        raise ValueError("harness graph contains a cycle")
    return layers


def _validate_acyclic(harness: HarnessGraph) -> None:
    node_ids = {node.id for node in harness.nodes}
    indegree = {node_id: 0 for node_id in node_ids}
    graph: dict[str, list[str]] = defaultdict(list)
    for edge in harness.edges:
        graph[edge.source].append(edge.target)
        indegree[edge.target] += 1
    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    seen: list[str] = []
    while queue:
        current = queue.popleft()
        seen.append(current)
        for target in graph[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(seen) != len(node_ids):
        raise ValueError("planner graph must be acyclic")
