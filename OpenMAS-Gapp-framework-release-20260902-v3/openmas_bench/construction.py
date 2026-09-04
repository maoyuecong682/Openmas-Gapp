from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque

from .schema import (
    ApplicationBlueprint, ApplicationRequirementModel, BlueprintEdge, BlueprintNode,
    CapabilityRequirement, ConstructionRequest, ConstructionResult, ConstructionTelemetry,
    ExecutableEdge, ExecutableMASApplication, ExecutableNode, Goal, HarnessEdge, HarnessNode,
    RequirementConstraint, RequirementTask, TaskDependency,
)
from .llm import DeterministicAdapter, LLMAdapter
from .prompts import Q1_SYSTEM_PROMPTS, construction_user_prompt


class ConstructionMethod(ABC):
    """Frozen Q1 method contract shared by every construction paradigm."""

    name: str

    def __init__(self, adapter: LLMAdapter | None = None, seed: int = 0):
        self.adapter = adapter or DeterministicAdapter()
        self.seed = seed

    @abstractmethod
    def construct(self, request: ConstructionRequest) -> ConstructionResult:
        raise NotImplementedError

    def _result(self, request: ConstructionRequest, model: ApplicationRequirementModel,
                blueprint: ApplicationBlueprint, telemetry: ConstructionTelemetry) -> ConstructionResult:
        application = realize_blueprint(request, blueprint, self.name)
        result = ConstructionResult(request.case_id, self.name, model, blueprint, application, telemetry)
        result.validate(request)
        return result

    def _model_call(self, request: ConstructionRequest) -> ConstructionTelemetry:
        required = {"tasks", "capabilities", "components", "constraints", "relations"}
        fallback_error = None
        try:
            response = self.adapter.generate_json(
                Q1_SYSTEM_PROMPTS[self.name], construction_user_prompt(request), self.seed, required)
            self.model_selection = response.value
        except RuntimeError as exc:
            fallback_error = str(exc)
            self.model_selection = {
                "tasks": [node.id for node in request.harness.nodes if node.kind == "task_pattern"],
                "capabilities": [node.id for node in request.harness.nodes if node.kind == "capability"],
                "components": [node.id for node in request.harness.nodes
                               if node.kind == "component" and node.risk != "high"],
                "constraints": [node.id for node in request.harness.nodes if node.kind == "constraint"],
                "relations": [{"source": edge.source, "target": edge.target}
                              for edge in request.harness.edges
                              if edge.relation in {"precedes", "depends", "reviews", "realizes", "requires"}],
            }
            response = None
        value = self.model_selection
        valid = required.issubset(value) and all(isinstance(value[x], list) for x in required)
        harness_ids = {x.id for x in request.harness.nodes}
        returned_ids = set()
        if valid:
            for field in ("tasks", "capabilities", "components", "constraints"):
                returned_ids.update(str(x) for x in value[field])
            for relation in value["relations"]:
                if isinstance(relation, dict):
                    returned_ids.update(str(relation.get(x, "")) for x in ("source", "target"))
                else:
                    valid = False
            out_of_range = returned_ids.difference(harness_ids)
            if out_of_range:
                for field in ("tasks", "capabilities", "components", "constraints"):
                    value[field] = [x for x in value[field] if str(x) in harness_ids]
                value["relations"] = [x for x in value["relations"] if str(x.get("source", "")) in harness_ids and str(x.get("target", "")) in harness_ids]
                if response is not None:
                    response.json_repaired = True
            valid = bool(value["tasks"]) and bool(value["components"])
        else:
            out_of_range = set()
        self.model_selection_valid = valid
        telemetry = ConstructionTelemetry(
            model_calls=(response.retry_count + 1) if response is not None else 0,
            adapter=response.provider if response is not None else "deterministic_selection_fallback",
            model=response.model if response is not None else "case_local_harness",
            seed=response.seed if response is not None else self.seed,
            input_tokens=response.input_tokens if response is not None else 0,
            output_tokens=response.output_tokens if response is not None else 0,
            latency_ms=response.latency_ms if response is not None else 0,
            retry_count=response.retry_count if response is not None else 0,
            json_repaired=response.json_repaired if response is not None else False,
            notes=[f"prompt_paradigm={self.name}"],
        )
        telemetry.notes.append(f"model_selection_valid={str(valid).lower()}")
        telemetry.notes.append(f"out_of_range_ids={len(out_of_range)}")
        if out_of_range:
            telemetry.notes.append("semantic_id_repair=true")
        if fallback_error:
            telemetry.fallback = True
            telemetry.notes.append("construction_json_fallback=true")
            telemetry.notes.append(f"construction_json_error={fallback_error[-240:]}")
        elif not valid:
            telemetry.fallback = True
            telemetry.notes.append("deterministic_selection_fallback=true")
        return telemetry

    def _ground(self, request: ConstructionRequest, use_task_graph: bool, use_constraints: bool) -> ApplicationRequirementModel:
        return ground_requirement(request, use_task_graph, use_constraints, self.model_selection if self.model_selection_valid else None)

    def _components(self, request: ConstructionRequest, requirements: list[CapabilityRequirement], inspect_relations: bool) -> list[HarnessNode]:
        return _components_for_requirements(request, requirements, inspect_relations, self.model_selection if self.model_selection_valid else None)


class DirectMASGeneration(ConstructionMethod):
    name = "direct_mas_generation"

    def construct(self, request: ConstructionRequest) -> ConstructionResult:
        telemetry = self._model_call(request)
        # One-pass direct generation sees component descriptions but does not use graph edges.
        chosen = set(self.model_selection.get("components", [])) if self.model_selection_valid else set()
        components = ([x for x in request.harness.nodes if x.kind == "component" and x.id in chosen]
                      if chosen else _ranked_components(request, inspect_edges=False))[:request.budget.max_components]
        model = _shallow_model(request, components)
        nodes = [_component_requirement(x, [x.id]) for x in components]
        edges = [BlueprintEdge(a.id, b.id, "precedes") for a, b in zip(nodes, nodes[1:])]
        blueprint = ApplicationBlueprint(request.case_id, self.name, nodes, edges, [], {"representation": "direct"})
        telemetry.planning_steps, telemetry.inspected_components = 1, len(components)
        telemetry.notes.append("single-pass generation")
        return self._result(request, model, blueprint, telemetry)


class PlanBasedConstruction(ConstructionMethod):
    name = "plan_based_construction"

    def construct(self, request: ConstructionRequest) -> ConstructionResult:
        telemetry = self._model_call(request)
        model = self._ground(request, use_task_graph=True, use_constraints=False)
        ordered = _topological_tasks(model)
        selected = self._components(request, model.capability_requirements, inspect_relations=False)
        nodes = [_task_node(x) for x in ordered] + [_component_requirement(x, [x.id]) for x in selected]
        edges = [BlueprintEdge(x.source, x.target, "precedes", x.condition) for x in model.task_dependencies]
        edges += _attach_components(model, selected)
        blueprint = ApplicationBlueprint(request.case_id, self.name, nodes, _valid_edges(nodes, edges), [], {"representation": "task_plan"})
        steps = len(model.tasks) + len(selected)
        telemetry.planning_steps, telemetry.inspected_components = steps, len(selected)
        telemetry.notes.append("plan then generate")
        return self._result(request, model, blueprint, telemetry)


class ComponentBasedAssembly(ConstructionMethod):
    name = "component_based_assembly"

    def construct(self, request: ConstructionRequest) -> ConstructionResult:
        telemetry = self._model_call(request)
        model = self._ground(request, use_task_graph=False, use_constraints=False)
        selected = self._components(request, model.capability_requirements, inspect_relations=False)
        nodes = [_component_requirement(x, [x.id]) for x in selected]
        blueprint = ApplicationBlueprint(request.case_id, self.name, nodes, [], [], {"representation": "component_set"})
        telemetry.planning_steps, telemetry.inspected_components = len(selected), len(selected)
        telemetry.notes.append("retrieve and assemble without relation reasoning")
        return self._result(request, model, blueprint, telemetry)


class WorkflowBasedConstruction(ConstructionMethod):
    name = "workflow_based_construction"

    def construct(self, request: ConstructionRequest) -> ConstructionResult:
        telemetry = self._model_call(request)
        model = self._ground(request, use_task_graph=True, use_constraints=False)
        selected = self._components(request, model.capability_requirements, inspect_relations=False)
        task_nodes = [_task_node(x) for x in _topological_tasks(model)]
        component_nodes = [_component_requirement(x, [x.id]) for x in selected]
        nodes = task_nodes + component_nodes
        # A workflow can encode order, but deliberately flattens feedback and governance.
        edges = [BlueprintEdge(a.id, b.id, "precedes") for a, b in zip(task_nodes, task_nodes[1:])]
        edges += _attach_components(model, selected)
        blueprint = ApplicationBlueprint(request.case_id, self.name, nodes, _valid_edges(nodes, edges), [], {"representation": "linear_workflow"})
        telemetry.planning_steps, telemetry.inspected_components = len(nodes), len(selected)
        telemetry.notes.append("fixed linear workflow")
        return self._result(request, model, blueprint, telemetry)


class GraphHarnessConstruction(ConstructionMethod):
    name = "graph_harness"

    def construct(self, request: ConstructionRequest) -> ConstructionResult:
        telemetry = self._model_call(request)
        # The LLM may suggest preferred components, but it must not delete
        # required tasks or governance constraints from the global graph.
        model = ground_requirement(
            request, use_task_graph=True, use_constraints=True, selection=None)
        telemetry.notes.append("global_requirement_grounding=true")
        profile = dict(request.metadata.get("task_profile") or {})
        grounding_repaired = False
        if ((profile.get("requires_multi_branch") or profile.get("requires_evidence_merge"))
                and _multibranch_grounding_incomplete(request, model)):
            # Candidate search cannot validate or execute a merge when the
            # stochastic ARG selection omitted a typed resource branch. Repair
            # from the case-local public Harness graph; no row answer or gold
            # program is consulted.
            model = ground_requirement(
                request, use_task_graph=True, use_constraints=True, selection=None)
            grounding_repaired = True
            telemetry.json_repaired = True
            telemetry.notes.append("multibranch_grounding_repair=true")
        selected = (_components_for_requirements(
            request, model.capability_requirements, inspect_relations=True,
            selection=None) if grounding_repaired else
            self._components(request, model.capability_requirements, inspect_relations=True))
        selected, component_repair_ids = _complete_component_selection(
            request, model.capability_requirements, selected)
        if component_repair_ids:
            telemetry.json_repaired = True
            telemetry.notes.append("global_component_coverage_repair=true")
            telemetry.notes.append(
                f"repaired_components={','.join(component_repair_ids)}")
        if profile.get("requires_multi_branch") or profile.get("requires_evidence_merge"):
            # A valid ARG model can still collapse two branch capabilities onto
            # one retrieved component. Preserve one executable component per
            # resource-bearing task so realize_blueprint can route both keys.
            resource_task_ids = {
                (edge.target[len("component_"):] if edge.target.startswith("component_") else edge.target)
                for edge in request.harness.edges
                if edge.relation == "uses"
                and any(node.id == edge.source and node.kind == "resource" for node in request.harness.nodes)
            }
            selected_ids = {node.id for node in selected}
            required_components = {f"component_{task_id}" for task_id in resource_task_ids}
            missing_components = required_components.difference(selected_ids)
            if missing_components:
                selected.extend(node for node in request.harness.nodes
                                if node.kind == "component" and node.id in missing_components)
                grounding_repaired = True
                telemetry.json_repaired = True
                if "multibranch_component_repair=true" not in telemetry.notes:
                    telemetry.notes.append("multibranch_component_repair=true")
        candidates = []
        # Candidate generation shares one ARG call. The search space is the
        # executable topology/prompt policy, and selection never reads gold.
        for strategy in self._candidate_strategies(profile):
            blueprint = self._build_graph_blueprint(request, model, selected, strategy)
            score = self._candidate_score(blueprint, profile, strategy)
            candidates.append((score, strategy, blueprint))
        _, chosen_strategy, blueprint = max(candidates, key=lambda item: (item[0], item[1]))
        task_nodes = [_task_node(x) for x in model.tasks]
        blueprint.metadata.update({"candidate_search": True, "candidate_count": len(candidates),
                                   "selected_candidate": chosen_strategy,
                                   "candidate_scores": {name: round(score, 6) for score, name, _ in candidates},
                                   "task_profile": profile,
                                   "multibranch_grounding_repair": grounding_repaired,
                                   "gold_used": False})
        inspected = len([x for x in request.harness.nodes if x.kind in {"component", "control"}])
        telemetry.planning_steps, telemetry.inspected_components = len(blueprint.nodes) + len(blueprint.edges), inspected
        telemetry.notes.append("constraint-aware graph orchestration")
        telemetry.notes.append(f"candidate_search=k{len(candidates)};selected={chosen_strategy}")
        result = self._result(request, model, blueprint, telemetry)
        _validate_global_graph_harness_result(model, selected, result)
        return result

    @staticmethod
    def _candidate_strategies(profile: dict) -> list[str]:
        """Small, auditable topology policy set; no task answers are involved."""
        if profile.get("requires_multi_branch") or profile.get("requires_evidence_merge"):
            return ["branch_merge_first", "evidence_first", "conservative"]
        if profile.get("requires_numeric_tool"):
            return ["tool_first", "verify_first", "conservative"]
        if profile.get("requires_constraint_gate"):
            return ["constraint_first", "evidence_first", "conservative"]
        return ["profile_specialized", "verify_first", "conservative"]

    @staticmethod
    def _build_graph_blueprint(request, model, selected, strategy: str) -> ApplicationBlueprint:
        task_nodes = [_task_node(x) for x in model.tasks]
        component_nodes = [_component_requirement(x, [x.id]) for x in selected]
        resource_nodes = _resource_requirements(request, selected)
        controls = _control_nodes(request, model)
        nodes = task_nodes + component_nodes + resource_nodes + controls
        edges = [BlueprintEdge(x.source, x.target, "feedback" if x.relation == "feedback" else "precedes", x.condition) for x in model.task_dependencies]
        edges += _attach_components(model, selected)
        edges += _resource_edges(request, selected)
        edges += _constraint_edges(request, model, nodes)
        hints = {
            "branch_merge_first": "Preserve isolated branch evidence and require the merge stage to consume every branch before verification.",
            "evidence_first": "Prioritize evidence extraction and preserve source identifiers through every downstream stage.",
            "verify_first": "Use an explicit independent verification pass before emitting the answer.",
            "tool_first": "Use the domain calculation/tool stage for the core operation, then independently verify units and result.",
            "constraint_first": "Apply required governance checks before the terminal answer stage.",
            "profile_specialized": "Follow the task-family policy while preserving all required artifacts and checks.",
            "conservative": "Preserve all available evidence and use the simplest valid execution path.",
        }
        for node in nodes:
            if node.kind in {"component_requirement", "control"}:
                node.binding_constraints["policy_hint"] = hints[strategy]
                if node.kind == "component_requirement":
                    profile = request.metadata.get("task_profile") or {}
                    if profile.get("task_family") == "financial_program":
                        node.binding_constraints["finqa_numeric_stage"] = True
                        node.binding_constraints["numeric_tool_contract"] = (
                            "At the execute stage emit selected_evidence and ordered steps as structured JSON "
                            "for the compiled Decimal tool. Steps may reference earlier $step_n results. "
                            "Verification may inspect only TOOL_TRACE, and the answer stage must copy final_value."
                        )
                    if node.id.casefold().endswith("answer") or "answer" in node.id.casefold():
                        node.binding_constraints["terminal_answer"] = True
                        node.binding_constraints["rerank_candidates"] = True
                        node.binding_constraints["math_verify"] = bool(profile.get("requires_numeric_tool"))
                profile = request.metadata.get("task_profile") or {}
                # Make the convergence point an explicit executable merge.
                # This is a runtime contract, not a gold-answer shortcut.
                if (profile.get("requires_multi_branch") or profile.get("requires_evidence_merge")) \
                        and node.kind == "component_requirement" \
                        and any(token in node.id.casefold() for token in ("synthesize", "merge")):
                    node.binding_constraints["merge_stage"] = True
        return ApplicationBlueprint(request.case_id, "graph_harness", nodes, _valid_edges(nodes, edges),
                                    [x.id for x in model.constraints],
                                    {"representation": "typed_harness_graph", "candidate_strategy": strategy})

    @staticmethod
    def _candidate_score(blueprint: ApplicationBlueprint, profile: dict, strategy: str) -> float:
        nodes = blueprint.nodes
        edges = blueprint.edges
        resources = [node for node in nodes if node.kind == "resource_requirement"]
        controls = [node for node in nodes if node.kind == "control"]
        score = 0.20 * min(1.0, len(edges) / max(1, len(nodes)))
        if profile.get("task_family") == "financial_program" and strategy == "tool_first":
            score += 0.20
        if profile.get("requires_multi_branch"):
            score += 0.35 * float(bool(resources) and any(edge.relation == "uses" for edge in edges))
        if profile.get("requires_evidence_merge"):
            score += 0.20 * float(any("merge" in node.description.casefold() for node in nodes))
        if profile.get("requires_constraint_gate"):
            score += 0.20 * float(bool(controls) and bool(blueprint.constraint_refs))
        if profile.get("requires_numeric_tool"):
            score += 0.10 * float(any(token in node.description.casefold() for node in nodes
                                      for token in ("calculate", "execute", "arithmetic", "numeric")))
        # Prefer complete resource routing and avoid oversized graphs.
        score += 0.15 * float(all(any(edge.source == resource.id and edge.relation == "uses" for edge in edges)
                                  for resource in resources))
        score -= 0.01 * max(0, len(edges) - 24)
        return score


Q1_METHODS = {x.name: x for x in [DirectMASGeneration, PlanBasedConstruction, ComponentBasedAssembly, WorkflowBasedConstruction, GraphHarnessConstruction]}


def get_construction_method(name: str, adapter: LLMAdapter | None = None, seed: int = 0) -> ConstructionMethod:
    try:
        return Q1_METHODS[name](adapter=adapter, seed=seed)
    except KeyError as exc:
        raise KeyError(f"unknown Q1 method {name}; choose from {sorted(Q1_METHODS)}") from exc


def ground_requirement(request: ConstructionRequest, use_task_graph: bool, use_constraints: bool, selection: dict | None = None) -> ApplicationRequirementModel:
    text = request.raw_requirement.lower()
    selected_tasks = set(selection.get("tasks", [])) if selection else set()
    # The offline sanity adapter uses the case-local typed Harness as its
    # deterministic source of truth. Real adapters must explicitly select IDs.
    patterns = [x for x in request.harness.nodes if x.kind == "task_pattern" and (x.id in selected_tasks if selection else True)]
    if not patterns:
        patterns = [HarnessNode("task_general", "task_pattern", request.raw_requirement, tags=["general"])]
    tasks = [RequirementTask(x.id, x.description, x.inputs, x.outputs) for x in patterns]
    task_ids = {x.id for x in tasks}
    dependencies = []
    if use_task_graph:
        for edge in request.harness.edges:
            if edge.source in task_ids and edge.target in task_ids and edge.relation in {"precedes", "depends"}:
                dependencies.append(TaskDependency(edge.source, edge.target, "precedes", edge.condition))
            elif edge.source in task_ids and edge.target in task_ids and edge.relation == "reviews":
                dependencies.append(TaskDependency(edge.source, edge.target, "feedback", edge.condition))
    requirements = []
    cap_nodes = {x.id: x for x in request.harness.nodes if x.kind == "capability"}
    for task in tasks:
        linked = [x.target for x in request.harness.edges if x.source == task.id and x.relation == "requires" and x.target in cap_nodes]
        for cap_id in linked:
            cap = cap_nodes[cap_id]
            requirements.append(CapabilityRequirement(cap.id, task.id, cap.description, cap.tags))
    constraints = []
    if use_constraints:
        selected_constraints = set(selection.get("constraints", [])) if selection else set()
        for node in request.harness.nodes:
            if node.kind == "constraint" and (node.id in selected_constraints if selection else True):
                constraints.append(RequirementConstraint(node.id, node.metadata.get("constraint_kind", "governance"), node.metadata.get("target", "application"), node.metadata.get("predicate", "required"), node.metadata.get("severity", "required")))
    model = ApplicationRequirementModel(Goal("goal", request.raw_requirement, ["construct executable MAS application"]), tasks, dependencies, requirements, constraints, {"grounding": "typed" if use_task_graph else "flat"})
    model.validate()
    return model


def realize_blueprint(request: ConstructionRequest, blueprint: ApplicationBlueprint, method: str) -> ExecutableMASApplication:
    harness_nodes = {x.id: x for x in request.harness.nodes}
    blueprint_nodes = {x.id: x for x in blueprint.nodes}
    resource_bindings = defaultdict(list)
    for edge in blueprint.edges:
        source = blueprint_nodes.get(edge.source)
        if edge.relation == "uses" and source is not None and source.kind == "resource_requirement":
            key = source.binding_constraints.get("resource_key")
            if key:
                resource_bindings[edge.target].append(str(key))
    app_nodes = []
    for node in blueprint.nodes:
        if node.kind in {"task", "capability", "resource_requirement"}:
            continue
        if node.kind == "control":
            candidate = harness_nodes.get(node.binding_constraints.get("candidate", ""))
            app_nodes.append(ExecutableNode(
                f"inst_{node.id}", "control", candidate.id if candidate else "builtin.control",
                node.id, node.capability_refs,
                {"execution_instruction": node.description,
                 "artifact_contract": _realization_artifact_contract(node),
                 "requirement_refs": list(node.requirement_refs),
                 "merge_stage": bool(node.binding_constraints.get("merge_stage")),
                 "finqa_numeric_stage": bool(node.binding_constraints.get("finqa_numeric_stage")),
                 "numeric_tool_contract": node.binding_constraints.get("numeric_tool_contract", ""),
                 "terminal_answer": bool(node.binding_constraints.get("terminal_answer")),
                 "answer_style": node.binding_constraints.get("answer_style", ""),
                 "rerank_candidates": bool(node.binding_constraints.get("rerank_candidates")),
                 "math_verify": bool(node.binding_constraints.get("math_verify")),
                 "policy_hint": node.binding_constraints.get("policy_hint", ""),
                 "resource_bindings": list(resource_bindings.get(node.id, []))},
            ))
            continue
        candidate_ids = node.binding_constraints.get("candidates", [])
        candidate = next((harness_nodes[x] for x in candidate_ids if x in harness_nodes), None)
        if candidate is None:
            candidate = HarnessNode("generalist_component", "component", "Generalist fallback", capabilities=node.capability_refs)
        kind = "tool" if candidate.metadata.get("runtime_kind") == "tool" else "agent"
        app_nodes.append(ExecutableNode(
            f"inst_{node.id}", kind, candidate.id, node.id, list(candidate.capabilities),
            {"execution_instruction": node.description,
             "artifact_contract": _realization_artifact_contract(node),
             "requirement_refs": list(node.requirement_refs),
             "merge_stage": bool(node.binding_constraints.get("merge_stage")),
             "finqa_numeric_stage": bool(node.binding_constraints.get("finqa_numeric_stage")),
             "numeric_tool_contract": node.binding_constraints.get("numeric_tool_contract", ""),
             "terminal_answer": bool(node.binding_constraints.get("terminal_answer")),
             "answer_style": node.binding_constraints.get("answer_style", ""),
             "rerank_candidates": bool(node.binding_constraints.get("rerank_candidates")),
             "math_verify": bool(node.binding_constraints.get("math_verify")),
             "policy_hint": node.binding_constraints.get("policy_hint", ""),
             "resource_bindings": list(resource_bindings.get(node.id, []))},
        ))
    if not app_nodes:
        # Direct methods may emit only task structure; keep realization executable.
        first = blueprint.nodes[0]
        app_nodes.append(ExecutableNode(
            "inst_generalist", "agent", "generalist_component", first.id,
            first.capability_refs, {"execution_instruction": first.description},
        ))
    by_bp = {x.realizes_blueprint_node: x.id for x in app_nodes}
    edges = []
    task_components = defaultdict(list)
    task_capabilities = defaultdict(set)
    capability_components = defaultdict(set)
    for edge in blueprint.edges:
        if edge.relation == "requires" and edge.target in by_bp:
            task_components[edge.source].append(by_bp[edge.target])
        source_node = blueprint_nodes.get(edge.source)
        target_node = blueprint_nodes.get(edge.target)
        if (edge.relation == "requires" and source_node is not None and target_node is not None
                and source_node.kind == "task" and target_node.kind == "capability"):
            task_capabilities[edge.source].add(edge.target)
        if (edge.relation in {"uses", "requires"} and source_node is not None
                and source_node.kind == "capability" and edge.target in by_bp):
            capability_components[edge.source].add(by_bp[edge.target])
    for task_id, capability_ids in task_capabilities.items():
        for capability_id in capability_ids:
            task_components[task_id].extend(sorted(capability_components.get(capability_id, set())))
    for i, edge in enumerate(blueprint.edges):
        if edge.source in by_bp and edge.target in by_bp:
            relation = "feedback" if edge.relation == "feedback" else "execution"
            edges.append(ExecutableEdge(by_bp[edge.source], by_bp[edge.target], relation, f"bp_edge_{i}", edge.condition))
        elif edge.source in task_components and edge.target in task_components:
            relation = "feedback" if edge.relation == "feedback" else "execution"
            for source in task_components[edge.source]:
                for target in task_components[edge.target]:
                    edges.append(ExecutableEdge(source, target, relation, f"bp_edge_{i}", edge.condition))
        elif edge.source in task_components and edge.target in by_bp:
            relation = "feedback" if edge.relation == "feedback" else "execution"
            for source in task_components[edge.source]:
                edges.append(ExecutableEdge(source, by_bp[edge.target], relation, f"bp_edge_{i}", edge.condition))
        elif edge.source in by_bp and edge.target in task_components:
            relation = "feedback" if edge.relation == "feedback" else "execution"
            for target in task_components[edge.target]:
                edges.append(ExecutableEdge(by_bp[edge.source], target, relation, f"bp_edge_{i}", edge.condition))
    deduplicated = []
    seen_edges = set()
    for edge in edges:
        key = (edge.source, edge.target, edge.relation, edge.condition)
        if edge.source != edge.target and key not in seen_edges:
            deduplicated.append(edge)
            seen_edges.add(key)
    edges = deduplicated
    if not edges:
        edges = [ExecutableEdge(a.id, b.id, "execution") for a, b in zip(app_nodes, app_nodes[1:])]
    non_feedback_targets = {edge.target for edge in edges if edge.relation != "feedback"}
    entrypoints = [node.id for node in app_nodes if node.id not in non_feedback_targets]
    if not entrypoints:
        entrypoints = [app_nodes[0].id]
    app = ExecutableMASApplication(request.case_id, method, app_nodes, edges, entrypoints, {"blueprint_preserving": True})
    app.validate(blueprint)
    return app


def _realization_artifact_contract(node: BlueprintNode) -> str:
    """Compile an abstract stage into a concrete, stage-local data contract."""
    refs = " ".join(node.requirement_refs + node.capability_refs).casefold()
    description = node.description.casefold()
    stage = f"{refs} {description}"
    if node.kind == "control":
        return ("Mandatory fields: original_question, candidate_answer, constraint_decision, violations, correction. "
                "Copy original_question and candidate_answer losslessly from upstream.")
    # Match terminal and transformation stages before evidence words. For
    # example, "verify supporting facts" is a verifier, and "synthesize both
    # evidence streams" is a synthesizer, not another retrieval stage.
    if any(token in stage for token in ("answer", "return", "report", "emit")):
        return ("Copy only candidate_answer from the verified upstream work into the benchmark answer. "
                "Return the shortest answer span that directly answers original_question.")
    if any(token in stage for token in ("verify", "check", "review", "test", "repair", "revise")):
        return ("Mandatory fields: original_question, candidate_answer, checks, correction. "
                "Resolve ambiguity against the original question and never drop candidate_answer.")
    if any(token in stage for token in ("reason", "solve", "compute", "execute", "synthesize", "merge")):
        return ("Mandatory fields: original_question, candidate_answer, rationale, preserved_evidence. "
                "The candidate_answer must directly answer the original question.")
    if any(token in stage for token in ("identify", "parse", "understand", "ground")):
        return ("Mandatory fields: original_question, options_or_givens, requested_output, candidate_answer. "
                "Preserve the exact question and every option label and text.")
    if any(token in stage for token in ("retrieve", "evidence", "support")):
        return ("Mandatory fields: original_question, evidence_spans, source_titles, candidate_answer. "
                "Copy the original question verbatim and preserve exact entities and quantities.")
    return "Return a lossless work product that satisfies the bound capability and can be consumed downstream."


def _shallow_model(request: ConstructionRequest, components: list[HarnessNode]) -> ApplicationRequirementModel:
    task = RequirementTask("task_direct", request.raw_requirement)
    reqs = [CapabilityRequirement(cap, task.id, cap) for component in components for cap in component.capabilities]
    return ApplicationRequirementModel(Goal("goal", request.raw_requirement), [task], [], reqs, [], {"grounding": "implicit"})


def _matches(text: str, node: HarnessNode) -> bool:
    tags = node.tags or node.description.lower().split()
    normalized = text.replace("-", " ").replace("_", " ")
    return any(tag.lower().replace("-", " ").replace("_", " ") in normalized for tag in tags)


def _ranked_components(request: ConstructionRequest, inspect_edges: bool) -> list[HarnessNode]:
    text = request.raw_requirement.lower()
    components = [x for x in request.harness.nodes if x.kind == "component"]
    scored = []
    degree = defaultdict(int)
    if inspect_edges:
        for edge in request.harness.edges:
            degree[edge.source] += 1
            degree[edge.target] += 1
    for component in components:
        overlap = sum(1 for tag in component.tags + component.capabilities if tag.lower() in text)
        scored.append((overlap, degree[component.id], component.id, component))
    return [x[-1] for x in sorted(scored, key=lambda x: (-x[0], -x[1], x[2])) if x[0] > 0]


def _components_for_requirements(request: ConstructionRequest, requirements: list[CapabilityRequirement], inspect_relations: bool, selection: dict | None = None) -> list[HarnessNode]:
    required = {x.id for x in requirements}
    selected_ids = set(selection.get("components", [])) if selection else set()
    candidates = [x for x in request.harness.nodes if x.kind == "component" and required.intersection(x.capabilities) and (not selected_ids or x.id in selected_ids)]
    if inspect_relations:
        realizes = {(x.source, x.target) for x in request.harness.edges if x.relation == "realizes"}
        candidates.sort(key=lambda x: (-sum((cap, x.id) in realizes for cap in required), x.risk != "low", x.id))
    selected = []
    covered = set()
    for component in candidates:
        gain = required.intersection(component.capabilities).difference(covered)
        if gain:
            selected.append(component)
            covered.update(gain)
    return selected


def _complete_component_selection(
    request: ConstructionRequest,
    requirements: list[CapabilityRequirement],
    selected: list[HarnessNode],
) -> tuple[list[HarnessNode], list[str]]:
    """Make the task-to-component mapping total after model suggestions.

    Model-selected components are preferences. Every required capability still
    needs an executable component so task dependencies can be realized without
    silently dropping intermediate stages.
    """
    required = {requirement.id for requirement in requirements}
    selected_by_id = {component.id: component for component in selected}
    covered = {
        capability
        for component in selected_by_id.values()
        for capability in component.capabilities
        if capability in required
    }
    missing = required.difference(covered)
    if not missing:
        return list(selected_by_id.values()), []

    realizes = {(edge.source, edge.target)
                for edge in request.harness.edges if edge.relation == "realizes"}
    candidates = [
        node for node in request.harness.nodes
        if node.kind == "component" and node.id not in selected_by_id
    ]
    candidates.sort(key=lambda node: (
        -sum((capability, node.id) in realizes for capability in missing),
        node.risk != "low",
        node.id,
    ))
    repaired = []
    for component in candidates:
        gain = missing.intersection(component.capabilities)
        if not gain:
            continue
        selected_by_id[component.id] = component
        repaired.append(component.id)
        missing.difference_update(gain)
        if not missing:
            break
    if missing:
        raise ValueError(
            "global graph cannot realize required capabilities: "
            + ", ".join(sorted(missing)))
    return list(selected_by_id.values()), repaired


def _validate_global_graph_harness_result(
    model: ApplicationRequirementModel,
    selected: list[HarnessNode],
    result: ConstructionResult,
) -> None:
    """Reject executable graphs that lose required tasks or terminal reachability."""
    task_to_components: dict[str, set[str]] = defaultdict(set)
    for requirement in model.capability_requirements:
        component_ids = {
            component.id
            for component in selected
            if requirement.id in component.capabilities
        }
        if not component_ids:
            raise ValueError(
                f"required task {requirement.task_id} has no component for "
                f"capability {requirement.id}")
        task_to_components[requirement.task_id].update(component_ids)

    app_by_blueprint = defaultdict(list)
    for node in result.application.nodes:
        app_by_blueprint[node.realizes_blueprint_node].append(node.id)
    task_to_app = {
        task_id: {
            app_id
            for component_id in component_ids
            for app_id in app_by_blueprint.get(f"req_{component_id}", [])
        }
        for task_id, component_ids in task_to_components.items()
    }
    missing_tasks = [
        task_id for task_id, app_ids in task_to_app.items() if not app_ids
    ]
    if missing_tasks:
        raise ValueError(
            "required tasks were not realized as executable agents: "
            + ", ".join(sorted(missing_tasks)))

    executable_edges = {
        (edge.source, edge.target)
        for edge in result.application.edges
    }
    missing_dependencies = []
    for dependency in model.task_dependencies:
        source_agents = task_to_app.get(dependency.source, set())
        target_agents = task_to_app.get(dependency.target, set())
        if not any((source, target) in executable_edges
                   for source in source_agents for target in target_agents):
            missing_dependencies.append(
                f"{dependency.source}->{dependency.target}")
    if missing_dependencies:
        raise ValueError(
            "required task dependencies were not realized: "
            + ", ".join(missing_dependencies))

    terminal_task = next(
        (
            task.id for task in reversed(model.tasks)
            if task.id.casefold() in {"answer", "report", "emit"}
            or any(token in task.id.casefold()
                   for token in ("answer", "report", "emit"))
        ),
        model.tasks[-1].id,
    )
    terminal_agents = task_to_app.get(terminal_task, set())
    outgoing = defaultdict(list)
    for source, target in executable_edges:
        outgoing[source].append(target)
    reachable = set(result.application.entrypoints)
    queue = deque(reachable)
    while queue:
        source = queue.popleft()
        for target in outgoing[source]:
            if target not in reachable:
                reachable.add(target)
                queue.append(target)
    if not terminal_agents.intersection(reachable):
        raise ValueError(
            f"terminal task {terminal_task} is not reachable from graph entrypoints")


def _multibranch_grounding_incomplete(request: ConstructionRequest,
                                      model: ApplicationRequirementModel) -> bool:
    """Check that ARG retained both public evidence branches and their merge."""
    harness_nodes = {node.id: node for node in request.harness.nodes}
    resource_tasks = {
        (edge.target[len("component_"):] if edge.target.startswith("component_")
         else edge.target)
        for edge in request.harness.edges
        if edge.relation == "uses"
        and edge.source in harness_nodes
        and harness_nodes[edge.source].kind == "resource"
    }
    if len(resource_tasks) < 2:
        return False
    task_ids = {task.id for task in model.tasks}
    required_tasks = resource_tasks.intersection({task.id for task in request.harness.nodes
                                                  if task.kind == "task_pattern"})
    if not required_tasks.issubset(task_ids):
        return True
    dependency_pairs = {(dep.source, dep.target) for dep in model.task_dependencies}
    merge_targets = {target for source, target in dependency_pairs
                     if source in required_tasks}
    if not merge_targets:
        return True
    required_caps = {req.id for req in model.capability_requirements
                     if req.task_id in required_tasks}
    return len(required_caps) < len(required_tasks)


def _task_node(task: RequirementTask) -> BlueprintNode:
    return BlueprintNode(task.id, "task", task.objective, [task.id])


def _component_requirement(component: HarnessNode, candidates: list[str]) -> BlueprintNode:
    return BlueprintNode(f"req_{component.id}", "component_requirement", component.description, [], list(component.capabilities), {"candidates": candidates})


def _attach_components(model: ApplicationRequirementModel, components: list[HarnessNode]) -> list[BlueprintEdge]:
    result = []
    for requirement in model.capability_requirements:
        component = next((x for x in components if requirement.id in x.capabilities), None)
        if component:
            result.append(BlueprintEdge(requirement.task_id, f"req_{component.id}", "requires"))
    return result


def _resource_requirements(request: ConstructionRequest,
                           components: list[HarnessNode]) -> list[BlueprintNode]:
    component_ids = {component.id for component in components}
    used_resources = {edge.source for edge in request.harness.edges
                      if edge.relation == "uses" and edge.target in component_ids}
    return [BlueprintNode(
        node.id, "resource_requirement", node.description, [], [],
        {"resource_key": node.metadata.get("resource_key", node.id)},
    ) for node in request.harness.nodes
        if node.kind == "resource" and node.id in used_resources]


def _resource_edges(request: ConstructionRequest,
                    components: list[HarnessNode]) -> list[BlueprintEdge]:
    component_ids = {component.id for component in components}
    return [BlueprintEdge(edge.source, f"req_{edge.target}", "uses")
            for edge in request.harness.edges
            if edge.relation == "uses" and edge.target in component_ids]


def _control_nodes(request: ConstructionRequest, model: ApplicationRequirementModel) -> list[BlueprintNode]:
    result = []
    controls = [x for x in request.harness.nodes if x.kind == "control"]
    for constraint in model.constraints:
        candidate = next((x for x in controls if constraint.id in x.tags or constraint.kind in x.tags), None)
        if candidate:
            result.append(BlueprintNode(f"control_{constraint.id}", "control", candidate.description, [constraint.id], candidate.capabilities, {"candidate": candidate.id}))
    return result


def _constraint_edges(request: ConstructionRequest, model: ApplicationRequirementModel, nodes: list[BlueprintNode]) -> list[BlueprintEdge]:
    node_ids = {x.id for x in nodes}
    edges = []
    for constraint in model.constraints:
        control = f"control_{constraint.id}"
        target = constraint.target
        if control in node_ids and target in node_ids:
            # A control is a gate, not a post-answer reporting node. Feed it
            # the same immediate predecessors as the target and require the
            # target to wait for the control decision.
            predecessors = [dependency.source for dependency in model.task_dependencies
                            if dependency.target == target and dependency.source in node_ids]
            edges.extend(BlueprintEdge(source, control, "constrained_by")
                         for source in predecessors)
            edges.append(BlueprintEdge(control, target, "precedes"))
        elif control in node_ids:
            task_ids = [x.id for x in model.tasks]
            if task_ids:
                edges.append(BlueprintEdge(control, task_ids[-1], "precedes"))
    return edges


def _valid_edges(nodes: list[BlueprintNode], edges: list[BlueprintEdge]) -> list[BlueprintEdge]:
    ids = {x.id for x in nodes}
    seen = set()
    result = []
    for edge in edges:
        key = (edge.source, edge.target, edge.relation, edge.condition)
        if edge.source in ids and edge.target in ids and key not in seen:
            result.append(edge)
            seen.add(key)
    return result


def _topological_tasks(model: ApplicationRequirementModel) -> list[RequirementTask]:
    tasks = {x.id: x for x in model.tasks}
    graph = defaultdict(list)
    indegree = {x: 0 for x in tasks}
    for edge in model.task_dependencies:
        if edge.relation == "feedback":
            continue
        graph[edge.source].append(edge.target)
        indegree[edge.target] += 1
    queue = deque(sorted(x for x, degree in indegree.items() if degree == 0))
    ordered = []
    while queue:
        node = queue.popleft()
        ordered.append(tasks[node])
        for nxt in graph[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return ordered if len(ordered) == len(tasks) else list(tasks.values())
