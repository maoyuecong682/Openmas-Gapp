"""Public façade for the OpenMAS-Gapp construction and execution pipeline."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .ablation import Q2_VARIANTS, get_ablation_method
from .application_executor import ApplicationTaskExecutor, ApplicationTaskResult
from .construction import get_construction_method, realize_blueprint
from .evaluate import evaluate_construction, evaluate_execution
from .runtime import MinimalMARRuntime
from .sandbox import run_python_tests, run_swebench_tests
from .schema import (ApplicationBlueprint, ConstructionCase, ConstructionResult,
                     ConstructionTelemetry)


@dataclass(frozen=True)
class EngineUsage:
    construction_input_tokens: int = 0
    construction_output_tokens: int = 0
    construction_calls: int = 0
    execution_input_tokens: int = 0
    execution_output_tokens: int = 0
    execution_calls: int = 0
    retries: int = 0
    repairs: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class EngineRunResult:
    dataset_id: str
    case_id: str
    source_id: Any
    intervention: str
    seed: int
    metric_name: str
    primary_score: float
    answer_score: float | None
    construction: ConstructionResult
    task_execution: ApplicationTaskResult
    construction_metrics: dict[str, Any]
    runtime_diagnostics: dict[str, Any]
    sandbox: dict[str, Any] | None
    usage: EngineUsage
    audit: dict[str, Any] = field(default_factory=dict)

    @property
    def prediction(self) -> Any:
        return self.task_execution.prediction

    def to_experiment_record(self) -> dict[str, Any]:
        prediction = self.prediction
        return {
            "status": "completed",
            "dataset": self.dataset_id,
            "case_id": self.case_id,
            "source_id": self.source_id,
            "variant": self.intervention,
            "seed": self.seed,
            "primary_metric": self.metric_name,
            "primary_score": self.primary_score,
            "prediction": prediction,
            "prediction_sha256": hashlib.sha256(str(prediction).encode("utf-8")).hexdigest(),
            "answer_evaluated": self.answer_score is not None,
            "application_digest": self.task_execution.application_digest,
            "application": asdict(self.construction.application),
            "blueprint_metadata": dict(self.construction.blueprint.metadata),
            "application_metadata": dict(self.construction.application.metadata),
            "construction_metrics": self.construction_metrics,
            "runtime_diagnostics": self.runtime_diagnostics,
            "task_execution": self.task_execution.to_dict(),
            "sandbox": self.sandbox,
            "engine_audit": self.audit,
        }


class GraphHarnessEngine:
    """Stable API for R + G_H -> M_R -> Blueprint -> Application -> Trace."""

    def __init__(self, adapter, data_root: str | Path,
                 runtime: MinimalMARRuntime | None = None,
                 task_executor: ApplicationTaskExecutor | None = None):
        self.adapter = adapter
        self.data_root = Path(data_root)
        self.runtime = runtime or MinimalMARRuntime()
        self.task_executor = task_executor or ApplicationTaskExecutor(adapter, self.data_root)

    def construct(self, case: ConstructionCase, seed: int = 0,
                  intervention: str = "full_graph_harness",
                  method: str | None = None) -> ConstructionResult:
        if method is not None and intervention != "full_graph_harness":
            raise ValueError("choose either a construction method or a Q2 intervention")
        constructor = (get_construction_method(method, adapter=self.adapter, seed=seed)
                       if method is not None else
                       get_ablation_method(intervention, adapter=self.adapter, seed=seed))
        return constructor.construct(case.request())

    def execute(self, dataset, row: dict[str, Any], case: ConstructionCase,
                construction: ConstructionResult, seed: int = 0,
                intervention: str | None = None,
                task_index: int = 0) -> EngineRunResult:
        if task_index < 0 or task_index >= len(case.execution_tasks):
            raise IndexError(f"execution task index out of range: {task_index}")
        execution_task = case.execution_tasks[task_index]
        construction_metrics = evaluate_construction(case, construction)
        task_result = self.task_executor.execute(dataset, row, case, construction, seed)
        diagnostic_execution = self.runtime.execute(
            case, construction, execution_task, seed)
        execution_metrics = evaluate_execution(case, diagnostic_execution)
        answer_score = dataset.execution.score(
            task_result.prediction, execution_task.answer)
        sandbox = None
        if dataset.execution.metric_name == "unit_test_pass":
            sandbox = run_python_tests(dataset.dataset_id, row, task_result.prediction)
            answer_score = float(sandbox["passed"])
        elif dataset.execution.metric_name == "swebench_resolved":
            sandbox = run_swebench_tests(row, task_result.prediction, timeout=180)
            answer_score = float(sandbox["passed"])
        primary = float(answer_score) if answer_score is not None else 0.0
        usage = EngineUsage(
            construction_input_tokens=construction.telemetry.input_tokens,
            construction_output_tokens=construction.telemetry.output_tokens,
            construction_calls=construction.telemetry.model_calls,
            execution_input_tokens=task_result.input_tokens,
            execution_output_tokens=task_result.output_tokens,
            execution_calls=task_result.calls,
            retries=construction.telemetry.retry_count + task_result.retries,
            repairs=int(construction.telemetry.json_repaired) + task_result.repairs,
        )
        selected = construction.blueprint.metadata.get("selected_candidate")
        audit = {
            "pipeline": "R+G_H->M_R->B->A->trace",
            "intervention": intervention or construction.application.metadata.get("variant") or construction.method,
            "candidate_count": construction.blueprint.metadata.get("candidate_count", 1),
            "candidate_scores": construction.blueprint.metadata.get("candidate_scores", {}),
            "selected_candidate": selected,
            "task_profile": construction.blueprint.metadata.get("task_profile", {}),
            "gold_used": False,
            "resource_audit": (task_result.metadata or {}).get("resource_audit", {}),
            "tool_audit": (task_result.metadata or {}).get("finqa_tool_audit", {}),
        }
        return EngineRunResult(
            dataset.dataset_id, case.case_id, row.get("id"),
            intervention or construction.application.metadata.get("variant") or construction.method,
            seed, dataset.execution.metric_name, primary, answer_score,
            construction, task_result, construction_metrics, execution_metrics,
            sandbox, usage, audit,
        )

    def realize(self, case: ConstructionCase, blueprint: ApplicationBlueprint,
                method: str, telemetry: ConstructionTelemetry | None = None) -> ConstructionResult:
        """Compile an externally proposed public Blueprint through the shared MEG."""
        application = realize_blueprint(case.request(), blueprint, method)
        telemetry = telemetry or ConstructionTelemetry(
            planning_steps=len(application.nodes) + len(application.edges),
            notes=["engine_realize_public_blueprint=true"],
        )
        result = ConstructionResult(
            case.case_id, method, case.reference_requirement_model,
            blueprint, application, telemetry)
        result.validate(case.request())
        return result

    def run_case(self, dataset, row: dict[str, Any], case: ConstructionCase,
                 seed: int = 0, intervention: str = "full_graph_harness") -> EngineRunResult:
        if intervention not in Q2_VARIANTS:
            raise ValueError(f"unknown intervention {intervention!r}")
        construction = self.construct(case, seed=seed, intervention=intervention)
        return self.execute(dataset, row, case, construction, seed, intervention)


__all__ = ["EngineRunResult", "EngineUsage", "GraphHarnessEngine"]
