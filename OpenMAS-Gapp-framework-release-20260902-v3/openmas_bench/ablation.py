"""Q2 component-wise ablations for the Graph Harness pipeline.

Each variant shares the Q1 adapter, ecosystem, budget and runtime.  The
variant changes exactly one construction module and records that change in
metadata so that causal comparisons remain auditable.
"""
from __future__ import annotations

import json

from .construction import (
    ConstructionMethod, GraphHarnessConstruction, _attach_components,
    _component_requirement, _ranked_components, _shallow_model,
    ground_requirement,
    _task_node, _topological_tasks, _valid_edges,
    _control_nodes, _constraint_edges, _resource_edges, _resource_requirements,
)
from .schema import (
    ApplicationBlueprint, BlueprintEdge, BlueprintNode, ConstructionRequest,
    ConstructionResult, ConstructionTelemetry, ExecutableEdge, ExecutableMASApplication,
    ExecutableNode, HarnessNode,
)
from .prompts import Q1_SYSTEM_PROMPTS, construction_user_prompt


class _Q2Base(ConstructionMethod):
    variant = "unknown"
    removed_module = "none"

    def _model_call(self, request):
        # Q2 uses one shared construction prompt/adapter budget; only the
        # post-selection pipeline module changes.
        original = self.name
        self.name = "graph_harness"
        try:
            return super()._model_call(request)
        finally:
            self.name = original

    def _tag(self, result: ConstructionResult) -> ConstructionResult:
        result.blueprint.metadata.update({"q": "Q2", "variant": self.variant,
                                          "removed_module": self.removed_module})
        result.application.metadata.update({"q": "Q2", "variant": self.variant,
                                            "removed_module": self.removed_module})
        result.requirement_model.metadata.update({"q": "Q2", "variant": self.variant})
        return result

    @staticmethod
    def _stage_count(request: ConstructionRequest) -> int:
        # A fixed execution budget makes the ablation intervention the only
        # causal difference. Include governance stages so MedQA-like cases do
        # not give Full an extra control node for free.
        return max(1, sum(node.kind == "task_pattern" for node in request.harness.nodes)
                   + sum(node.kind == "control" for node in request.harness.nodes))

    def _pad_components(self, request: ConstructionRequest, selected: list[HarnessNode], reserved_stages: int = 0) -> list[HarnessNode]:
        target = max(1, self._stage_count(request) - reserved_stages)
        pool = [node for node in request.harness.nodes
                if node.kind == "component" and node not in selected
                and node.risk != "high"
                and not {"cap_shortcut", "cap_untrusted"}.intersection(node.capabilities)]
        result = list(selected)
        result.extend(pool[:max(0, target - len(result))])
        return result[:target]

    @staticmethod
    def _neutral_stages(count: int) -> list[BlueprintNode]:
        return [BlueprintNode(
            f"neutral_budget_stage_{index}", "component_requirement",
            "Pass the upstream work to the next stage without adding requirements, constraints, or relations.",
            [], [], {"candidates": [], "budget_equalizer": True},
        ) for index in range(count)]


class FullGraphHarnessAblation(GraphHarnessConstruction):
    # Reuse the frozen Q1 prompt key; the result is labelled with the Q2 variant.
    name = "graph_harness"
    variant = "full_graph_harness"
    removed_module = "none"

    def construct(self, request: ConstructionRequest) -> ConstructionResult:
        result = super().construct(request)
        result.blueprint.metadata.update({"q": "Q2", "variant": self.variant, "removed_module": self.removed_module})
        result.application.metadata.update({"q": "Q2", "variant": self.variant, "removed_module": self.removed_module})
        return result


class WithoutRequirementGrounding(_Q2Base):
    name = "without_requirement_grounding"
    variant = "w/o_requirement_grounding"
    removed_module = "requirement_grounding"

    def construct(self, request):
        telemetry = self._raw_component_call(request)
        # Directly consume the catalog order and raw component descriptions;
        # do not derive a structured requirement/task model. Catalog order
        # still preserves the shared output protocol (the answer stage last).
        chosen = set(self.model_selection.get("components", [])) if self.model_selection_valid else set()
        components = [node for node in request.harness.nodes
                      if node.kind == "component" and (not chosen or node.id in chosen)
                      and node.risk != "high"
                      and not {"cap_shortcut", "cap_untrusted"}.intersection(node.capabilities)]
        components = self._pad_components(request, components[:request.budget.max_components])
        model = _shallow_model(request, components)
        # Keep one executable stage per task-pattern/control slot while the
        # requirement model itself remains intentionally shallow.
        component_nodes = [_component_requirement(x, [x.id]) for x in components]
        neutral = self._neutral_stages(max(0, self._stage_count(request) - len(component_nodes)))
        nodes = component_nodes[:-1] + neutral + component_nodes[-1:] if component_nodes else neutral
        blueprint = ApplicationBlueprint(request.case_id, self.name, nodes,
                                         [BlueprintEdge(a.id, b.id, "precedes") for a, b in zip(nodes, nodes[1:])],
                                         [], {"representation": "raw_requirement_direct", "blueprint_present": True})
        telemetry.planning_steps, telemetry.inspected_components = 1, len(components)
        telemetry.notes.append("Q2: raw utterance direct component assembly; no ARG prompt")
        return self._tag(self._result(request, model, blueprint, telemetry))

    def _raw_component_call(self, request: ConstructionRequest) -> ConstructionTelemetry:
        """Select components without emitting an ARG-shaped model response."""
        catalog = [{"id": node.id, "description": node.description,
                    "capabilities": node.capabilities, "tags": node.tags}
                   for node in request.harness.nodes if node.kind == "component"
                   and node.risk != "high"]
        system = ("Select reusable components directly from the raw user utterance. "
                  "Do not decompose tasks, infer constraints, construct a requirement "
                  "model, or inspect graph relations. Return JSON with only components.")
        user = json.dumps({"raw_user_utterance": request.raw_requirement,
                           "component_catalog": catalog,
                           "max_components": request.budget.max_components,
                           "output_schema": {"components": ["component_id"]}},
                          ensure_ascii=False)
        fallback_error = None
        try:
            response = self.adapter.generate_json(system, user, self.seed, {"components"})
            values = response.value.get("components")
            valid_ids = {node["id"] for node in catalog}
            chosen = [str(value) for value in values] if isinstance(values, list) else []
            chosen = [value for value in chosen if value in valid_ids]
            if not chosen:
                raise RuntimeError("direct component selection returned no valid component IDs")
        except RuntimeError as exc:
            fallback_error = str(exc)
            chosen = [node["id"] for node in catalog[:request.budget.max_components]]
            response = None
        self.model_selection = {"components": chosen}
        self.model_selection_valid = bool(chosen)
        telemetry = ConstructionTelemetry(
            model_calls=(response.retry_count + 1) if response is not None else 0,
            adapter=response.provider if response is not None else "direct_selection_fallback",
            model=response.model if response is not None else "case_local_catalog",
            seed=response.seed if response is not None else self.seed,
            input_tokens=response.input_tokens if response is not None else 0,
            output_tokens=response.output_tokens if response is not None else 0,
            latency_ms=response.latency_ms if response is not None else 0.0,
            retry_count=response.retry_count if response is not None else 0,
            json_repaired=response.json_repaired if response is not None else False,
            fallback=fallback_error is not None,
            notes=["prompt_paradigm=raw_component_selection",
                   "arg_output_fields=0", "graph_relations_visible=false"],
        )
        if fallback_error:
            telemetry.notes.append(f"direct_selection_error={fallback_error[-240:]}")
        return telemetry


class WithoutGraphOrchestration(_Q2Base):
    name = "without_graph_orchestration"
    variant = "w/o_graph_orchestration"
    removed_module = "graph_orchestration"

    def construct(self, request):
        telemetry = self._model_call(request)
        model = self._ground(request, use_task_graph=True, use_constraints=True)
        controls = _control_nodes(request, model)
        selected = self._pad_components(
            request, self._components(request, model.capability_requirements, inspect_relations=False), len(controls))
        tasks = [_task_node(x) for x in model.tasks]
        comps = [_component_requirement(x, [x.id]) for x in selected]
        nodes = tasks + comps + controls
        # Flat composition intentionally discards typed dependency relations.
        linear_nodes = list(comps)
        # Preserve governance as a stage while discarding its typed graph
        # placement. Put the control immediately before the final answer stage.
        if controls and linear_nodes:
            linear_nodes[-1:-1] = controls
        else:
            linear_nodes.extend(controls)
        edges = [BlueprintEdge(a.id, b.id, "precedes") for a, b in zip(linear_nodes, linear_nodes[1:])]
        blueprint = ApplicationBlueprint(request.case_id, self.name, nodes, _valid_edges(nodes, edges),
                                         [x.id for x in model.constraints],
                                         {"representation": "flat_component_retrieval"})
        telemetry.planning_steps, telemetry.inspected_components = len(nodes), len(selected)
        telemetry.notes.append("Q2: component retrieval without graph relations")
        return self._tag(self._result(request, model, blueprint, telemetry))


class WithoutBlueprint(_Q2Base):
    name = "without_blueprint"
    variant = "w/o_blueprint"
    removed_module = "application_blueprint"

    def construct(self, request):
        telemetry = self._model_call(request)
        model = self._ground(request, use_task_graph=True, use_constraints=True)
        controls = _control_nodes(request, model)
        selected = self._pad_components(request, self._components(request, model.capability_requirements, inspect_relations=True), len(controls))
        app, blueprint = _build_direct_mas(request, model, selected, controls, self.name)
        result = ConstructionResult(request.case_id, self.name, model, blueprint, app, telemetry)
        result.validate(request)
        telemetry.planning_steps = len(app.nodes) + len(app.edges)
        telemetry.inspected_components = len(selected)
        telemetry.notes.append("Q2: orchestration mapped directly to executable MAS")
        return self._tag(result)


class WithoutConstraintAwareOrchestration(_Q2Base):
    name = "without_constraint_aware_orchestration"
    variant = "w/o_constraint_aware_orchestration"
    removed_module = "constraint_aware_orchestration"

    def construct(self, request):
        telemetry = self._model_call(request)
        # ARG still detects constraints. The ablation is applied only at CCG
        # orchestration: no governance node or constraint edge is injected.
        model = self._ground(request, use_task_graph=True, use_constraints=True)
        # Remove governance controls, but keep the same executable stage
        # budget as Full by padding with ordinary components.
        selected = self._pad_components(request, self._components(request, model.capability_requirements, inspect_relations=True))
        tasks = [_task_node(x) for x in model.tasks]
        comps = [_component_requirement(x, [x.id]) for x in selected]
        resources = _resource_requirements(request, selected)
        neutral = self._neutral_stages(max(0, self._stage_count(request) - len(comps)))
        nodes = tasks + comps + resources + neutral
        edges = [BlueprintEdge(x.source, x.target, "precedes", x.condition) for x in model.task_dependencies]
        edges += _attach_components(model, selected)
        edges += _resource_edges(request, selected)
        if neutral and comps:
            # Equalize runtime budget at the removed control's location without
            # reproducing its predicate or governance behavior.
            component_by_capability = {
                capability: component_node
                for component, component_node in zip(selected, comps)
                for capability in component.capabilities
            }
            task_component = {
                requirement.task_id: component_by_capability.get(requirement.id)
                for requirement in model.capability_requirements
            }
            answer_task = model.tasks[-1].id
            answer_component = task_component.get(answer_task)
            answer_predecessors = [dependency.source for dependency in model.task_dependencies
                                   if dependency.target == answer_task]
            predecessor_component = next(
                (task_component.get(task_id) for task_id in answer_predecessors
                 if task_component.get(task_id) is not None), None)
            if predecessor_component is not None and answer_component is not None:
                edges.append(BlueprintEdge(predecessor_component.id, neutral[0].id, "precedes"))
                edges.append(BlueprintEdge(neutral[-1].id, answer_component.id, "precedes"))
            edges.extend(BlueprintEdge(a.id, b.id, "precedes")
                         for a, b in zip(neutral, neutral[1:]))
        blueprint = ApplicationBlueprint(request.case_id, self.name, nodes, _valid_edges(nodes, edges), [],
                                         {"representation": "graph_without_constraint_injection",
                                          "constraints_detected_but_not_injected": [x.id for x in model.constraints]})
        telemetry.planning_steps, telemetry.inspected_components = len(nodes) + len(edges), len(selected)
        telemetry.notes.append("Q2: graph orchestration with governance disabled")
        return self._tag(self._result(request, model, blueprint, telemetry))


class WithoutRealization(_Q2Base):
    name = "without_realization"
    variant = "w/o_realization"
    removed_module = "blueprint_preserving_realization"

    def construct(self, request):
        telemetry = self._model_call(request)
        model = self._ground(request, use_task_graph=True, use_constraints=True)
        controls = _control_nodes(request, model)
        selected = self._pad_components(request, self._components(request, model.capability_requirements, inspect_relations=True), len(controls))
        tasks = [_task_node(x) for x in model.tasks]
        comps = [_component_requirement(x, [x.id]) for x in selected]
        resources = _resource_requirements(request, selected)
        nodes = tasks + comps + resources + controls
        edges = [BlueprintEdge(x.source, x.target, "feedback" if x.relation == "feedback" else "precedes", x.condition) for x in model.task_dependencies]
        edges += _attach_components(model, selected)
        edges += _resource_edges(request, selected)
        edges += _constraint_edges(request, model, nodes)
        blueprint = ApplicationBlueprint(request.case_id, self.name, nodes, _valid_edges(nodes, edges),
                                         [x.id for x in model.constraints], {"representation": "typed_harness_graph"})
        # Do not call realize_blueprint here. This is a separate one-pass,
        # prompt-style generator: it sees Blueprint stage text, but performs no
        # catalog binding, typed interface compilation, or execution-policy
        # realization. The schema reference is only a serialization bridge and
        # is explicitly excluded from fidelity scoring.
        app = _prompt_generate_application(blueprint, self.name)
        result = ConstructionResult(request.case_id, self.name, model, blueprint, app, telemetry)
        result.validate(request)
        telemetry.planning_steps, telemetry.inspected_components = len(nodes) + len(edges), len(selected)
        telemetry.notes.append("Q2: ordinary Prompt generation from Blueprint; no MAR compiler")
        return self._tag(result)


def _prompt_generate_application(blueprint: ApplicationBlueprint,
                                 method: str) -> ExecutableMASApplication:
    """Generate runnable agents from Blueprint text without MAR semantics."""
    abstract_nodes = [node for node in blueprint.nodes
                      if node.kind not in {"task", "resource_requirement"}]
    app_nodes = [ExecutableNode(
        f"prompt_agent_{index}", "agent", "generic_blueprint_interpreter", node.id,
        [],
        {"execution_instruction": node.description,
         "construction_mode": "prompt_generation",
         "generic_prompt_generation": True},
    ) for index, node in enumerate(abstract_nodes)]
    if not app_nodes:
        first = blueprint.nodes[0]
        app_nodes = [ExecutableNode(
            "prompt_agent_0", "agent", "generic_blueprint_interpreter", first.id, [],
            {"execution_instruction": first.description,
             "construction_mode": "prompt_generation",
             "generic_prompt_generation": True},
        )]
    # A generic code prompt commonly emits declaration-order hand-offs. It
    # does not receive or reproduce typed Blueprint edges.
    app_edges = [ExecutableEdge(left.id, right.id, "execution")
                 for left, right in zip(app_nodes, app_nodes[1:])]
    return ExecutableMASApplication(
        blueprint.case_id, method, app_nodes, app_edges, [app_nodes[0].id],
        {"blueprint_present": True, "blueprint_preserving": False,
         "uses_blueprint_realization": False,
         "construction_mode": "prompt_generation"},
    )


def _build_direct_mas(request, model, selected, controls, method):
    """Build a runnable MAS directly, without a Blueprint realization pass.

    ConstructionResult currently requires a Blueprint-shaped serialization
    field. The returned carrier contains only opaque executable-stage IDs and
    is never exposed to the executor; it is not an application IR.
    """
    task_by_cap = {cap.id: cap.task_id for cap in model.capability_requirements}
    objective_by_task = {task.id: task.objective for task in model.tasks}
    app_nodes = []
    task_to_node = {}
    for index, component in enumerate(selected):
        node_id = f"direct_stage_{index}"
        task_ids = [task_by_cap[cap] for cap in component.capabilities if cap in task_by_cap]
        instruction = "; ".join(objective_by_task[task] for task in task_ids)
        app_nodes.append(ExecutableNode(
            node_id, "tool" if component.metadata.get("runtime_kind") == "tool" else "agent",
            component.id, node_id, list(component.capabilities),
            {"execution_instruction": instruction or "Directly contribute to the requested task",
             "construction_mode": "direct_mas",
             "artifact_contract": "Preserve CANDIDATE_ANSWER and all upstream evidence for the next direct stage."},
        ))
        for task_id in task_ids:
            task_to_node.setdefault(task_id, node_id)
    constraint_to_control = {}
    for control in controls:
        candidate = control.binding_constraints.get("candidate", "builtin.control")
        node_id = f"direct_stage_{len(app_nodes)}"
        app_nodes.append(ExecutableNode(
            node_id, "control", candidate, node_id, list(control.capability_refs),
            {"execution_instruction": control.description,
             "requirement_refs": list(control.requirement_refs),
             "construction_mode": "direct_mas",
             "artifact_contract": "Preserve CANDIDATE_ANSWER and all upstream evidence for the next direct stage."},
        ))
        for requirement in control.requirement_refs:
            constraint_to_control[requirement] = node_id

    edges = []
    for dependency in model.task_dependencies:
        source, target = task_to_node.get(dependency.source), task_to_node.get(dependency.target)
        if source and target and source != target:
            edges.append(ExecutableEdge(source, target,
                                        "feedback" if dependency.relation == "feedback" else "execution"))
    for constraint in model.constraints:
        target = task_to_node.get(constraint.target) or task_to_node.get(model.tasks[-1].id)
        control = constraint_to_control.get(constraint.id)
        predecessors = [task_to_node.get(dependency.source)
                        for dependency in model.task_dependencies
                        if dependency.target == constraint.target]
        for source in predecessors:
            if source and control and source != control:
                edges.append(ExecutableEdge(source, control, "review"))
        if control and target and control != target:
            edges.append(ExecutableEdge(control, target, "execution"))
    if not edges:
        edges = [ExecutableEdge(a.id, b.id, "execution") for a, b in zip(app_nodes, app_nodes[1:])]
    targets = {edge.target for edge in edges if edge.relation != "feedback"}
    entrypoints = [node.id for node in app_nodes if node.id not in targets] or [app_nodes[0].id]

    carrier_nodes = [BlueprintNode(node.id,
                                   "control" if node.kind == "control" else "component_requirement",
                                   "opaque direct MAS stage",
                                   list(node.config.get("requirement_refs", [])),
                                   list(node.capabilities), {})
                     for node in app_nodes]
    carrier_edges = [BlueprintEdge(edge.source, edge.target,
                                   "feedback" if edge.relation == "feedback" else "precedes")
                     for edge in edges]
    carrier = ApplicationBlueprint(
        request.case_id, method, carrier_nodes, carrier_edges, [],
        {"representation": "opaque_schema_carrier", "blueprint_present": False,
         "carrier_only": True},
    )
    app = ExecutableMASApplication(
        request.case_id, method, app_nodes, edges, entrypoints,
        {"blueprint_present": False, "blueprint_preserving": False,
         "construction_mode": "direct_mas", "uses_blueprint_realization": False},
    )
    return app, carrier


Q2_VARIANTS = {x.variant: x for x in [FullGraphHarnessAblation, WithoutRequirementGrounding,
    WithoutGraphOrchestration, WithoutBlueprint, WithoutConstraintAwareOrchestration, WithoutRealization]}


def get_ablation_method(variant: str, adapter=None, seed: int = 0) -> ConstructionMethod:
    try:
        return Q2_VARIANTS[variant](adapter=adapter, seed=seed)
    except KeyError as exc:
        raise KeyError(f"unknown Q2 variant {variant}; choose from {sorted(Q2_VARIANTS)}") from exc
