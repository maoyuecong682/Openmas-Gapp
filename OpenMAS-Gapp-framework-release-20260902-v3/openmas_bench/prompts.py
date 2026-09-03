from __future__ import annotations

import json

from .schema import ConstructionRequest


Q1_SYSTEM_PROMPTS = {
    "direct_mas_generation": "Directly generate an executable MAS application from the requirement. Do not create a separate task plan, requirement model, component graph, or blueprint.",
    "plan_based_construction": "First decompose the requirement into an ordered task plan, then generate an executable MAS. Use the plan only; do not perform typed component-relation or constraint-graph reasoning.",
    "component_based_assembly": "Retrieve reusable components that match the requirement and assemble them into an executable MAS. Treat the retrieved result as a component set; do not introduce a separate relation-orchestration stage.",
    "workflow_based_construction": "Construct the MAS through a workflow representation. Organize tasks as a conventional sequential workflow and bind available components to workflow stages.",
    "graph_harness": "Ground the requirement into goals, task dependencies, capability requirements, and constraints. Orchestrate typed Harness Graph relations into an Application Blueprint, then realize the Blueprint without redesigning it.",
}

COMMON_OUTPUT_CONTRACT = """
Return one JSON object with exactly these fields:
{
  "tasks": [task_pattern_id],
  "capabilities": [capability_id],
  "components": [component_or_control_id],
  "constraints": [constraint_id],
  "relations": [{"source": node_id, "target": node_id, "relation": "precedes|feedback|constrained_by"}]
}
Use only IDs present in the supplied component ecosystem. Select only items needed by the requirement.
"""


def construction_user_prompt(request: ConstructionRequest) -> str:
    ecosystem = [{"id": x.id, "kind": x.kind, "description": x.description, "capabilities": x.capabilities, "tags": x.tags} for x in request.harness.nodes]
    return json.dumps({
        "requirement": request.raw_requirement,
        "component_ecosystem": ecosystem,
        "budget": {
            "max_components": request.budget.max_components,
            "max_edges": request.budget.max_edges,
            "max_planning_steps": request.budget.max_planning_steps,
            "max_model_calls": request.budget.max_model_calls,
        },
        "required_output": "ConstructionResult JSON",
        "common_model_output_contract": COMMON_OUTPUT_CONTRACT,
    }, ensure_ascii=False)
