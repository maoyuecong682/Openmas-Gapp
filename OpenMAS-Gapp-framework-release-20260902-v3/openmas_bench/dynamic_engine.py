from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dynamic_executor import DynamicExecutionResult, DynamicGraphExecutor
from .dynamic_graph import DynamicGraphBuilder, DynamicHarnessBundle
from .dynamic_planner import DeepSeekGraphPlanner, PlannerGraphSpec


@dataclass
class DynamicRunBundle:
    task: str
    plan: PlannerGraphSpec
    bundle: DynamicHarnessBundle
    execution: DynamicExecutionResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "plan": self.plan.to_dict(),
            "bundle": self.bundle.to_dict(),
            "execution": None if self.execution is None else self.execution.to_dict(),
        }


class DynamicGraphEngine:
    """Planner -> graph builder -> executor pipeline."""

    def __init__(
        self,
        planner: DeepSeekGraphPlanner | None = None,
        builder: DynamicGraphBuilder | None = None,
        executor: DynamicGraphExecutor | None = None,
    ):
        self.planner = planner or DeepSeekGraphPlanner.from_env()
        self.builder = builder or DynamicGraphBuilder()
        self.executor = executor

    def plan(self, task: str, seed: int = 11, context: dict[str, Any] | None = None) -> DynamicRunBundle:
        plan = self.planner.plan(task, seed=seed, context=context)
        bundle = self.builder.build(task, plan)
        return DynamicRunBundle(task=task, plan=plan, bundle=bundle)

    def run(
        self,
        task: str,
        seed: int = 11,
        context: dict[str, Any] | None = None,
        execute: bool = False,
    ) -> DynamicRunBundle:
        bundle = self.plan(task, seed=seed, context=context)
        if execute:
            if self.executor is None:
                raise RuntimeError("executor is not configured")
            bundle.execution = self.executor.execute(task, bundle.bundle.harness, seed=seed, context=context)
        return bundle

