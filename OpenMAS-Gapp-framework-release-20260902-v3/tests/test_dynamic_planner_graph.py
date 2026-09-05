from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

from openmas_bench.dynamic_graph import DynamicGraphBuilder, topological_order
from openmas_bench.dynamic_planner import (
    DeepSeekGraphPlanner,
    PlannerAgentSpec,
    PlannerEdgeSpec,
    PlannerGraphSpec,
)


class _FakePlannerClient:
    def __init__(self, payloads: list[dict[str, object]]):
        self.payloads = list(payloads)
        self.calls: list[tuple[str, str, int, set[str] | None]] = []

    def generate_json(self, system_prompt: str, user_prompt: str, seed: int, required_fields=None):
        self.calls.append((system_prompt, user_prompt, seed, required_fields))
        payload = self.payloads[min(len(self.calls) - 1, len(self.payloads) - 1)]
        return SimpleNamespace(
            value=payload,
            provider="fake",
            model="fake-model",
            raw_text="{}",
        )


def test_planner_to_dict_is_strict():
    spec = PlannerGraphSpec(
        agents=[PlannerAgentSpec("task_splitter", "splitter", "split work", ["analysis"], ["notes"])],
        edges=[],
    )

    assert spec.to_dict() == {
        "agents": [
            {
                "id": "task_splitter",
                "role": "splitter",
                "objective": "split work",
                "capabilities": ["analysis"],
                "tools": ["notes"],
            }
        ],
        "edges": [],
    }


def test_dynamic_graph_builder_produces_distinct_topologies():
    builder = DynamicGraphBuilder()

    plan_a = PlannerGraphSpec(
        agents=[
            PlannerAgentSpec("researcher", "researcher", "collect evidence", ["search"], ["web"]),
            PlannerAgentSpec("writer", "writer", "draft answer", ["synthesis"], ["editor"]),
        ],
        edges=[PlannerEdgeSpec("researcher", "writer")],
    )
    plan_b = PlannerGraphSpec(
        agents=[
            PlannerAgentSpec("planner", "planner", "decompose the task", ["analysis"], ["notes"]),
            PlannerAgentSpec("retriever", "retriever", "gather sources", ["search"], ["web"]),
            PlannerAgentSpec("validator", "validator", "check consistency", ["verification"], ["rubric"]),
            PlannerAgentSpec("finalizer", "finalizer", "merge and deliver", ["synthesis"], ["editor"]),
        ],
        edges=[
            PlannerEdgeSpec("planner", "retriever"),
            PlannerEdgeSpec("planner", "validator"),
            PlannerEdgeSpec("retriever", "finalizer"),
            PlannerEdgeSpec("validator", "finalizer"),
        ],
    )

    bundle_a = builder.build("compare single branch", plan_a)
    bundle_b = builder.build("compare multi branch", plan_b)

    assert len(bundle_a.harness.nodes) == 2
    assert len(bundle_a.harness.edges) == 1
    assert len(bundle_b.harness.nodes) == 4
    assert len(bundle_b.harness.edges) == 4
    assert topological_order(bundle_a.harness) == ["researcher", "writer"]
    assert topological_order(bundle_b.harness)[0] == "planner"
    assert bundle_b.harness.metadata["parallel_groups"] == [["planner"], ["retriever", "validator"], ["finalizer"]]
    assert asdict(bundle_a.harness) != asdict(bundle_b.harness)


def test_deepseek_graph_planner_uses_planner_client():
    payloads = [
        {
            "agents": [
                {
                    "id": "researcher",
                    "role": "researcher",
                    "objective": "collect evidence",
                    "capabilities": ["search"],
                    "tools": ["web"],
                },
                {
                    "id": "writer",
                    "role": "writer",
                    "objective": "draft answer",
                    "capabilities": ["synthesis"],
                    "tools": ["editor"],
                },
            ],
            "edges": [{"source": "researcher", "target": "writer"}],
        }
    ]
    planner = DeepSeekGraphPlanner(client=_FakePlannerClient(payloads))

    plan = planner.plan("Explain why the graph has a dependency edge.", seed=7)

    assert plan.to_dict()["agents"][0]["id"] == "researcher"
    assert plan.to_dict()["edges"] == [{"source": "researcher", "target": "writer"}]
