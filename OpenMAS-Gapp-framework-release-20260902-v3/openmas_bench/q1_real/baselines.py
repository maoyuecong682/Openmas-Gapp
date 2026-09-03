from __future__ import annotations

from dataclasses import dataclass

from ..construction import realize_blueprint
from ..engine import GraphHarnessEngine
from ..schema import (
    ApplicationBlueprint,
    ApplicationRequirementModel,
    BlueprintEdge,
    BlueprintNode,
    CapabilityRequirement,
    ConstructionCase,
    ConstructionResult,
    ConstructionTelemetry,
    ExecutableMASApplication,
    ExecutableNode,
    Goal,
    RequirementTask,
)


@dataclass(frozen=True)
class Q1RealBaseline:
    name: str
    label: str
    layer: str


Q1_REAL_BASELINES: dict[str, Q1RealBaseline] = {
    "direct_llm_answering": Q1RealBaseline("direct_llm_answering", "Direct LLM Answering", "no_harness"),
    "prompt_only_planning": Q1RealBaseline("prompt_only_planning", "Prompt-only MAS Planning", "implicit_planning"),
    "component_based_assembly": Q1RealBaseline("component_based_assembly", "Component-based Assembly", "component_layer"),
    "plan_based_construction": Q1RealBaseline("plan_based_construction", "Plan-based Construction", "plan_layer"),
    "workflow_based_construction": Q1RealBaseline("workflow_based_construction", "Workflow-based Construction", "workflow_layer"),
    "graph_harness": Q1RealBaseline("graph_harness", "Graph Harness", "harness_layer"),
}


LEGACY_CONSTRUCTION_METHODS = {
    "component_based_assembly",
    "plan_based_construction",
    "workflow_based_construction",
    "graph_harness",
}


def build_q1_real_construction(case: ConstructionCase, baseline_name: str, adapter, seed: int,
                               engine: GraphHarnessEngine | None = None) -> ConstructionResult:
    """Build same-level Q1 baselines for real QA items.

    Q1-real asks whether a Harness layer is necessary.  Therefore the baselines
    are construction paradigms at the same abstraction level as Harness, not
    alternative internal graph representations.  Q3 owns the latter question.
    """
    if baseline_name in LEGACY_CONSTRUCTION_METHODS:
        if engine is None:
            engine = GraphHarnessEngine(adapter, ".")
        result = engine.construct(case, seed=seed, method=baseline_name)
        if baseline_name == "graph_harness":
            _enable_harness_resource_execution(result.application)
        return result
    if baseline_name == "direct_llm_answering":
        return _direct_llm_answering(case, seed)
    if baseline_name == "prompt_only_planning":
        return _prompt_only_planning(case, seed, engine)
    raise KeyError(f"unknown Q1-real baseline {baseline_name}; choose from {sorted(Q1_REAL_BASELINES)}")


def _direct_llm_answering(case: ConstructionCase, seed: int) -> ConstructionResult:
    task = RequirementTask("direct_answer", case.raw_requirement)
    model = ApplicationRequirementModel(
        Goal("goal", case.raw_requirement, ["answer produced"]),
        [task],
        [],
        [CapabilityRequirement("cap_direct_answer", task.id, "Direct benchmark answering without an explicit Harness layer")],
        [],
        {"representation": "direct_llm", "harness_layer": False},
    )
    blueprint = ApplicationBlueprint(
        case.case_id,
        "direct_llm_answering",
        [BlueprintNode(
            "direct_answer_node",
            "component_requirement",
            _direct_answer_instruction(case),
            [task.id],
            ["cap_direct_answer"],
            {"candidates": []},
        )],
        [],
        [],
        {"representation": "direct_llm_answering", "blueprint_present": False, "harness_layer": False},
    )
    application = ExecutableMASApplication(
        case.case_id,
        "direct_llm_answering",
        [ExecutableNode(
            "inst_direct_answer",
            "agent",
            "generalist_direct_llm",
            "direct_answer_node",
            ["cap_direct_answer"],
            {
                "execution_instruction": _direct_answer_instruction(case)
                + "\nDo not emit a plan, decomposition, or intermediate reasoning artifact.",
                "artifact_contract": "Return only the final benchmark prediction. No plan artifact.",
                "resource_access": True,
            },
        )],
        [],
        ["inst_direct_answer"],
        {"blueprint_preserving": False, "harness_layer": False},
    )
    telemetry = ConstructionTelemetry(
        planning_steps=1,
        model_calls=0,
        inspected_components=0,
        notes=[f"q1_real_baseline=direct_llm_answering", f"seed={seed}", "no_explicit_harness_layer"],
        adapter="benchmark",
        model="direct_llm_answering",
        seed=seed,
    )
    result = ConstructionResult(case.case_id, "direct_llm_answering", model, blueprint, application, telemetry)
    result.validate(case.request())
    return result


def _prompt_only_planning(case: ConstructionCase, seed: int,
                          engine: GraphHarnessEngine | None = None) -> ConstructionResult:
    tasks = [RequirementTask("plan", "Infer a natural-language solution plan"),
             RequirementTask("solve", "Execute the plan and produce the answer")]
    model = ApplicationRequirementModel(
        Goal("goal", case.raw_requirement, ["answer produced"]),
        tasks,
        [],
        [CapabilityRequirement("cap_prompt_plan", "plan", "Prompt-only planning"),
         CapabilityRequirement("cap_prompt_solve", "solve", "Prompt-only solving")],
        [],
        {"representation": "prompt_only", "harness_layer": False},
    )
    nodes = [
        BlueprintNode(
            "prompt_plan",
            "component_requirement",
            "Create an explicit natural-language plan from the benchmark item. The plan should list the key reasoning steps, but do not use a typed Harness or component graph.",
            ["plan"],
            ["cap_prompt_plan"],
            {"candidates": []},
        ),
        BlueprintNode(
            "prompt_solve",
            "component_requirement",
            _direct_answer_instruction(case)
            + "\nFollow the upstream plan explicitly, but no Harness graph, capability graph, or constraint graph is available.",
            ["solve"],
            ["cap_prompt_solve"],
            {"candidates": []},
        ),
    ]
    blueprint = ApplicationBlueprint(
        case.case_id,
        "prompt_only_planning",
        nodes,
        [BlueprintEdge("prompt_plan", "prompt_solve", "precedes")],
        [],
        {"representation": "prompt_only_planning", "blueprint_present": False, "harness_layer": False},
    )
    telemetry = ConstructionTelemetry(
        planning_steps=2,
        model_calls=0,
        inspected_components=0,
        notes=[f"q1_real_baseline=prompt_only_planning", f"seed={seed}", "implicit_prompt_plan_without_harness"],
        adapter="benchmark",
        model="prompt_only_planning",
        seed=seed,
    )
    if engine is None:
        application = realize_blueprint(case.request(), blueprint, "prompt_only_planning")
        result = ConstructionResult(case.case_id, "prompt_only_planning", model, blueprint, application, telemetry)
    else:
        result = engine.realize(case, blueprint, "prompt_only_planning", telemetry)
        result.requirement_model = model
    result.application.metadata["blueprint_preserving"] = False
    result.validate(case.request())
    return result


def _enable_harness_resource_execution(application: ExecutableMASApplication) -> None:
    """Let Harness-constructed apps exploit explicit resource/task context.

    This is the Q1 counterpart of the Q3 resource-aware bridge: if the method
    actually constructs a Harness-layer application, its realized nodes may use
    original dataset resources to preserve exact options/entities and audit
    upstream artifacts. Non-Harness baselines do not receive this affordance.
    """
    for node in application.nodes:
        node.config["resource_access"] = True
        instruction = str(node.config.get("execution_instruction", ""))
        if "Harness-layer resource access:" not in instruction:
            node.config["execution_instruction"] = (
                instruction
                + "\nHarness-layer resource access: use the original question/context/options as explicit resources, "
                  "while preserving the structured upstream work and satisfying all constraints."
            )


def _direct_answer_instruction(case: ConstructionCase) -> str:
    dataset_id = str(case.metadata.get("dataset", "") if isinstance(case.metadata, dict) else "").casefold()
    row = case.metadata.get("row", {}) if isinstance(case.metadata, dict) else {}
    if row.get("choices"):
        return "Answer the benchmark item directly. Return exactly one option label or the matching option text."
    if dataset_id == "pubmedqa":
        return "Answer the biomedical question directly. Return exactly one lowercase label: yes, no, or maybe."
    if dataset_id in {"strategyqa"}:
        return "Answer the question directly. Return exactly yes or no."
    if dataset_id in {"gsm8k", "math-500", "math500", "finqa", "drop"}:
        return "Solve the problem directly. Return only the final numeric or mathematical answer."
    return "Answer the benchmark item directly. Return only the concise final answer."
