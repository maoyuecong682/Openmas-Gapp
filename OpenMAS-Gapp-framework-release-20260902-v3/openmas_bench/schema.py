from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass
class Goal:
    id: str
    description: str
    success_criteria: list[str] = field(default_factory=list)


@dataclass
class RequirementTask:
    id: str
    objective: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


@dataclass
class TaskDependency:
    source: str
    target: str
    relation: str = "precedes"
    condition: str | None = None


@dataclass
class CapabilityRequirement:
    id: str
    task_id: str
    description: str
    tags: list[str] = field(default_factory=list)


@dataclass
class RequirementConstraint:
    id: str
    kind: str
    target: str
    predicate: str
    severity: str = "required"


@dataclass
class ApplicationRequirementModel:
    """ARG output. This layer must not contain implementation components."""

    goal: Goal
    tasks: list[RequirementTask]
    task_dependencies: list[TaskDependency]
    capability_requirements: list[CapabilityRequirement]
    constraints: list[RequirementConstraint]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _unique("requirement task", (x.id for x in self.tasks))
        _unique("capability requirement", (x.id for x in self.capability_requirements))
        _unique("requirement constraint", (x.id for x in self.constraints))
        task_ids = {x.id for x in self.tasks}
        for edge in self.task_dependencies:
            if edge.source not in task_ids or edge.target not in task_ids:
                raise ValueError(f"invalid task dependency {edge.source} -> {edge.target}")
        for capability in self.capability_requirements:
            if capability.task_id not in task_ids:
                raise ValueError(f"capability {capability.id} targets unknown task {capability.task_id}")
        forbidden = {"agent", "tool", "workflow", "component_id", "implementation_ref"}
        leaked = forbidden.intersection(self.metadata)
        if leaked:
            raise ValueError(f"requirement model leaks implementation fields: {sorted(leaked)}")


@dataclass
class HarnessNode:
    id: str
    kind: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    risk: str = "low"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessEdge:
    source: str
    target: str
    relation: str
    condition: str | None = None


@dataclass
class HarnessGraph:
    nodes: list[HarnessNode]
    edges: list[HarnessEdge]
    version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        allowed_nodes = {"task_pattern", "capability", "component", "resource", "constraint", "control"}
        allowed_edges = {"requires", "realizes", "uses", "depends", "constrained_by", "precedes", "reviews"}
        _unique("harness node", (x.id for x in self.nodes))
        node_ids = {x.id for x in self.nodes}
        for node in self.nodes:
            if node.kind not in allowed_nodes:
                raise ValueError(f"invalid harness node kind {node.kind}")
        for edge in self.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError(f"invalid harness edge {edge.source} -> {edge.target}")
            if edge.relation not in allowed_edges:
                raise ValueError(f"invalid harness relation {edge.relation}")


@dataclass
class BlueprintNode:
    id: str
    kind: str
    description: str
    requirement_refs: list[str] = field(default_factory=list)
    capability_refs: list[str] = field(default_factory=list)
    binding_constraints: dict[str, Any] = field(default_factory=dict)

    # Read-only semantic aliases used by representation evaluators. They keep
    # one canonical Blueprint schema without forcing experiment-local node DTOs.
    @property
    def label(self) -> str:
        return self.description

    @property
    def refs(self) -> list[str]:
        return self.requirement_refs

    @property
    def attrs(self) -> dict[str, Any]:
        return self.binding_constraints


@dataclass
class BlueprintEdge:
    source: str
    target: str
    relation: str
    condition: str | None = None


@dataclass
class ApplicationBlueprint:
    """Application-level IR. Components are abstract requirements, never instances."""

    case_id: str
    method: str
    nodes: list[BlueprintNode]
    edges: list[BlueprintEdge]
    constraint_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def baseline(self) -> str:
        """Compatibility alias; method is the canonical field name."""
        return self.method

    def validate(self) -> None:
        allowed_nodes = {"task", "capability", "component_requirement", "resource_requirement", "control"}
        allowed_edges = {"requires", "uses", "precedes", "constrained_by", "reviews", "feedback"}
        _unique("blueprint node", (x.id for x in self.nodes))
        node_ids = {x.id for x in self.nodes}
        for node in self.nodes:
            if node.kind not in allowed_nodes:
                raise ValueError(f"invalid blueprint node kind {node.kind}")
            forbidden = {"agent_id", "tool_id", "implementation_ref", "runtime_config"}
            leaked = forbidden.intersection(node.binding_constraints)
            if leaked:
                raise ValueError(f"blueprint node {node.id} contains concrete bindings: {sorted(leaked)}")
        for edge in self.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError(f"invalid blueprint edge {edge.source} -> {edge.target}")
            if edge.relation not in allowed_edges:
                raise ValueError(f"invalid blueprint relation {edge.relation}")


@dataclass
class ExecutableNode:
    id: str
    kind: str
    implementation_ref: str
    realizes_blueprint_node: str
    capabilities: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutableEdge:
    source: str
    target: str
    relation: str
    realizes_blueprint_edge: str | None = None
    condition: str | None = None


@dataclass
class ExecutableMASApplication:
    case_id: str
    method: str
    nodes: list[ExecutableNode]
    edges: list[ExecutableEdge]
    entrypoints: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, blueprint: ApplicationBlueprint | None = None) -> None:
        allowed_nodes = {"agent", "tool", "memory", "control"}
        allowed_edges = {"communication", "invocation", "execution", "review", "feedback"}
        _unique("executable node", (x.id for x in self.nodes))
        node_ids = {x.id for x in self.nodes}
        for node in self.nodes:
            if node.kind not in allowed_nodes:
                raise ValueError(f"invalid executable node kind {node.kind}")
        for edge in self.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError(f"invalid executable edge {edge.source} -> {edge.target}")
            if edge.relation not in allowed_edges:
                raise ValueError(f"invalid executable relation {edge.relation}")
        if not self.nodes or not self.entrypoints or not set(self.entrypoints).issubset(node_ids):
            raise ValueError("executable application requires valid nodes and entrypoints")
        if blueprint is not None:
            bp_ids = {x.id for x in blueprint.nodes}
            dangling = {x.realizes_blueprint_node for x in self.nodes}.difference(bp_ids)
            if dangling:
                raise ValueError(f"realization references unknown blueprint nodes: {sorted(dangling)}")


@dataclass
class ConstructionBudget:
    max_components: int = 12
    max_edges: int = 24
    max_planning_steps: int = 32
    max_model_calls: int = 4


@dataclass
class ConstructionRequest:
    case_id: str
    raw_requirement: str
    harness: HarnessGraph
    budget: ConstructionBudget = field(default_factory=ConstructionBudget)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.case_id or not self.raw_requirement.strip():
            raise ValueError("case_id and raw_requirement are required")
        self.harness.validate()


@dataclass
class ConstructionTelemetry:
    planning_steps: int = 0
    model_calls: int = 0
    inspected_components: int = 0
    notes: list[str] = field(default_factory=list)
    adapter: str = "deterministic"
    model: str = "deterministic-q1-proxy"
    seed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    retry_count: int = 0
    json_repaired: bool = False
    fallback: bool = False


@dataclass
class ConstructionResult:
    case_id: str
    method: str
    requirement_model: ApplicationRequirementModel
    blueprint: ApplicationBlueprint
    application: ExecutableMASApplication
    telemetry: ConstructionTelemetry = field(default_factory=ConstructionTelemetry)

    def validate(self, request: ConstructionRequest | None = None) -> None:
        self.requirement_model.validate()
        self.blueprint.validate()
        self.application.validate(self.blueprint)
        if not (self.case_id == self.blueprint.case_id == self.application.case_id):
            raise ValueError("construction result case ids do not match")
        if not (self.method == self.blueprint.method == self.application.method):
            raise ValueError("construction result method names do not match")
        if request is not None:
            if len(self.application.nodes) > request.budget.max_components:
                raise ValueError("component budget exceeded")
            if len(self.application.edges) > request.budget.max_edges:
                raise ValueError("edge budget exceeded")
            if self.telemetry.planning_steps > request.budget.max_planning_steps:
                raise ValueError("planning-step budget exceeded")
            if self.telemetry.model_calls > request.budget.max_model_calls:
                raise ValueError("model-call budget exceeded")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RelationContract:
    source: str
    target: str
    relations: list[str]
    required: bool = True


@dataclass
class TraceContract:
    id: str
    kind: str
    target: str
    predicate: str
    severity: str = "required"


@dataclass
class ConstructionContracts:
    required_tasks: list[str]
    required_capabilities: list[str]
    acceptable_relations: list[RelationContract]
    forbidden_components: list[str]
    required_constraints: list[str]
    trace_contracts: list[TraceContract]

    def validate(self) -> None:
        _unique("trace contract", (x.id for x in self.trace_contracts))
        for relation in self.acceptable_relations:
            if not relation.relations:
                raise ValueError(f"relation contract {relation.source}->{relation.target} has no accepted relation")


@dataclass
class ConstructionExecutionTask:
    id: str
    prompt: str
    answer: Any
    context: Any = None
    source: dict[str, Any] = field(default_factory=dict)
    required_capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeTraceEvent:
    event_id: str
    case_id: str
    execution_task_id: str
    actor: str
    action: str
    status: str
    sequence: int
    blueprint_node_ref: str
    capability_refs: list[str] = field(default_factory=list)
    constraint_refs: list[str] = field(default_factory=list)
    input_refs: list[str] = field(default_factory=list)
    output_refs: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ApplicationExecutionResult:
    case_id: str
    method: str
    execution_task_id: str
    seed: int
    events: list[RuntimeTraceEvent]
    predicted_answer: Any
    task_success: bool
    runtime_valid: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConstructionCase:
    case_id: str
    family: str
    domain: str
    raw_requirement: str
    harness: HarnessGraph
    reference_requirement_model: ApplicationRequirementModel
    reference_blueprint: ApplicationBlueprint
    contracts: ConstructionContracts
    execution_tasks: list[ConstructionExecutionTask] = field(default_factory=list)
    split: str = "dev"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.family not in {"sequential", "multi_branch", "feedback_driven", "constraint_heavy"}:
            raise ValueError(f"unknown construction family {self.family}")
        self.harness.validate()
        self.reference_requirement_model.validate()
        self.reference_blueprint.validate()
        self.contracts.validate()
        _unique("construction execution task", (x.id for x in self.execution_tasks))
        if self.reference_blueprint.case_id != self.case_id:
            raise ValueError("reference blueprint case id mismatch")
        if self.split not in {"dev", "validation", "test"}:
            raise ValueError(f"invalid split {self.split}")
        if len(self.execution_tasks) < 2:
            raise ValueError("each formal construction case requires at least two execution tasks")

    def request(self, budget: ConstructionBudget | None = None) -> ConstructionRequest:
        metadata = {"family": self.family, "domain": self.domain}
        # Pass only answer-independent construction priors. The full row is
        # intentionally excluded from ConstructionRequest to prevent oracle
        # selection and accidental gold-answer leakage.
        if isinstance(self.metadata.get("task_profile"), dict):
            metadata["task_profile"] = dict(self.metadata["task_profile"])
        return ConstructionRequest(self.case_id, self.raw_requirement, self.harness,
                                   budget or ConstructionBudget(), metadata)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConstructionCase":
        model = _requirement_model_from_dict(value["reference_requirement_model"])
        blueprint = _blueprint_from_dict(value["reference_blueprint"])
        harness = _harness_from_dict(value["harness"])
        contract_value = value["contracts"]
        contracts = ConstructionContracts(
            contract_value["required_tasks"], contract_value["required_capabilities"],
            [RelationContract(**x) for x in contract_value["acceptable_relations"]],
            contract_value["forbidden_components"], contract_value["required_constraints"],
            [TraceContract(**x) for x in contract_value["trace_contracts"]],
        )
        case = cls(
            value["case_id"], value["family"], value["domain"], value["raw_requirement"],
            harness, model, blueprint, contracts,
            [ConstructionExecutionTask(**x) for x in value.get("execution_tasks", [])],
            value.get("split", "dev"), value.get("metadata", {}),
        )
        case.validate()
        return case


# Legacy pilot schemas remain readable while the Q1-Q4 suite migrates.
@dataclass
class Capability:
    id: str
    kind: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    risk: str = "low"


@dataclass
class Contract:
    id: str
    kind: str
    target: str
    predicate: str
    description: str
    severity: str = "required"


@dataclass
class ExecutionTask:
    id: str
    prompt: str
    answer: Any
    context: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Mutation:
    id: str
    operation: str
    path: str
    value: Any
    expected_effects: list[str] = field(default_factory=list)


@dataclass
class DomainPackage:
    package_id: str
    domain: str
    application: str
    source: dict[str, Any]
    requirement: dict[str, Any]
    capabilities: list[Capability]
    contracts: list[Contract]
    execution_tasks: list[ExecutionTask]
    mutations: list[Mutation] = field(default_factory=list)
    split: str = "dev"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        required = {"goal", "process", "resources", "governance", "output"}
        missing = required.difference(self.requirement)
        if missing:
            raise ValueError(f"requirement missing fields: {sorted(missing)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DomainPackage":
        package = cls(value["package_id"], value["domain"], value["application"], value.get("source", {}), value["requirement"], [Capability(**x) for x in value.get("capabilities", [])], [Contract(**x) for x in value.get("contracts", [])], [ExecutionTask(**x) for x in value.get("execution_tasks", [])], [Mutation(**x) for x in value.get("mutations", [])], value.get("split", "dev"), value.get("metadata", {}))
        package.validate()
        return package


@dataclass
class MASNode:
    id: str
    kind: str
    role: str
    capabilities: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class MASEdge:
    source: str
    target: str
    kind: str = "execution_order"
    condition: str | None = None


@dataclass
class MASSpec:
    package_id: str
    baseline: str
    nodes: list[MASNode]
    edges: list[MASEdge]
    selected_capabilities: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _unique("MAS node", (x.id for x in self.nodes))
        node_ids = {x.id for x in self.nodes}
        if not self.nodes:
            raise ValueError("MASSpec must contain at least one node")
        for edge in self.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError(f"invalid edge {edge.source} -> {edge.target}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceEvent:
    event_id: str
    package_id: str
    task_id: str
    actor: str
    action: str
    timestamp: str
    input_refs: list[str] = field(default_factory=list)
    output_refs: list[str] = field(default_factory=list)
    capability_refs: list[str] = field(default_factory=list)
    contract_refs: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


def _harness_from_dict(value: dict[str, Any]) -> HarnessGraph:
    return HarnessGraph([HarnessNode(**x) for x in value["nodes"]], [HarnessEdge(**x) for x in value["edges"]], value.get("version", "1.0"), value.get("metadata", {}))


def _requirement_model_from_dict(value: dict[str, Any]) -> ApplicationRequirementModel:
    return ApplicationRequirementModel(Goal(**value["goal"]), [RequirementTask(**x) for x in value["tasks"]], [TaskDependency(**x) for x in value["task_dependencies"]], [CapabilityRequirement(**x) for x in value["capability_requirements"]], [RequirementConstraint(**x) for x in value["constraints"]], value.get("metadata", {}))


def _blueprint_from_dict(value: dict[str, Any]) -> ApplicationBlueprint:
    return ApplicationBlueprint(value["case_id"], value["method"], [BlueprintNode(**x) for x in value["nodes"]], [BlueprintEdge(**x) for x in value["edges"]], value.get("constraint_refs", []), value.get("metadata", {}))


def construction_result_from_dict(value: dict[str, Any]) -> ConstructionResult:
    blueprint = _blueprint_from_dict(value["blueprint"])
    app = value["application"]
    result = ConstructionResult(value["case_id"], value["method"], _requirement_model_from_dict(value["requirement_model"]), blueprint, ExecutableMASApplication(app["case_id"], app["method"], [ExecutableNode(**x) for x in app["nodes"]], [ExecutableEdge(**x) for x in app["edges"]], app["entrypoints"], app.get("metadata", {})), ConstructionTelemetry(**value.get("telemetry", {})))
    result.validate()
    return result


def _unique(label: str, values: Iterable[str]) -> None:
    values = list(values)
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label} ids")
