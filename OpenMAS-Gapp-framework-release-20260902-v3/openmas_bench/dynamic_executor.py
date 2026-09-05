from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .dynamic_graph import topological_order
from .llm import LLMAdapter
from .schema import HarnessGraph


@dataclass
class DynamicNodeRun:
    node_id: str
    role: str
    objective: str
    predecessors: list[str] = field(default_factory=list)
    artifact: str = ""
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class DynamicExecutionResult:
    task: str
    order: list[str]
    node_runs: list[DynamicNodeRun]
    final_node_id: str
    final_output: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "order": list(self.order),
            "node_runs": [asdict(run) for run in self.node_runs],
            "final_node_id": self.final_node_id,
            "final_output": self.final_output,
            "metadata": dict(self.metadata),
        }


class DynamicGraphExecutor:
    """Execute a planner-built HarnessGraph without deciding its topology."""

    def __init__(self, adapter: LLMAdapter):
        self.adapter = adapter

    def execute(self, task: str, harness: HarnessGraph, seed: int = 11, context: dict[str, Any] | None = None) -> DynamicExecutionResult:
        harness.validate()
        order = topological_order(harness)
        node_by_id = {node.id: node for node in harness.nodes}
        predecessor_map: dict[str, list[str]] = {node.id: [] for node in harness.nodes}
        outgoing: dict[str, list[str]] = {node.id: [] for node in harness.nodes}
        for edge in harness.edges:
            predecessor_map.setdefault(edge.target, []).append(edge.source)
            outgoing.setdefault(edge.source, []).append(edge.target)
        outputs: dict[str, str] = {}
        node_runs: list[DynamicNodeRun] = []
        for index, node_id in enumerate(order):
            node = node_by_id[node_id]
            predecessors = predecessor_map.get(node_id, [])
            node_context = {
                "task": task,
                "graph_metadata": harness.metadata,
                "context": context or {},
                "predecessor_outputs": {pid: outputs[pid] for pid in predecessors if pid in outputs},
                "node": {
                    "id": node.id,
                    "kind": node.kind,
                    "description": node.description,
                    "capabilities": list(node.capabilities),
                    "tools": list(node.metadata.get("tools", [])),
                    "role": node.metadata.get("role", ""),
                },
            }
            system = (
                "You are one agent in a dynamic multi-agent graph. "
                "Follow your assigned role and objective. "
                "Do not redesign the graph. "
                "Return valid JSON with one field: artifact."
            )
            user = json.dumps(node_context, ensure_ascii=False)
            response = self.adapter.generate_json(system, user, seed + index, {"artifact"})
            artifact = _extract_artifact(response.value)
            outputs[node_id] = artifact
            node_runs.append(
                DynamicNodeRun(
                    node_id=node_id,
                    role=str(node.metadata.get("role", "")),
                    objective=str(node.metadata.get("objective", node.description)),
                    predecessors=list(predecessors),
                    artifact=artifact,
                    provider=response.provider,
                    model=response.model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
            )
        sink_nodes = [node_id for node_id in order if not outgoing.get(node_id)]
        final_node_id = sink_nodes[-1] if sink_nodes else order[-1]
        final_output = outputs.get(final_node_id, "")
        return DynamicExecutionResult(
            task=task,
            order=order,
            node_runs=node_runs,
            final_node_id=final_node_id,
            final_output=final_output,
            metadata={
                "task": task,
                "node_count": len(harness.nodes),
                "edge_count": len(harness.edges),
            },
        )


def _extract_artifact(value: dict[str, Any]) -> str:
    for key in ("artifact", "answer", "text", "output"):
        candidate = value.get(key)
        if candidate is not None:
            return str(candidate)
    return json.dumps(value, ensure_ascii=False)

