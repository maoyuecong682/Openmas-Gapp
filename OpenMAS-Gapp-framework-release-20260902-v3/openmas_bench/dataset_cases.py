"""Build small, reproducible ConstructionCase objects from dataset adapters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dataset_adapters import DatasetAdapter
from .schema import (ApplicationBlueprint, ApplicationRequirementModel, BlueprintEdge,
    BlueprintNode, CapabilityRequirement, ConstructionCase, ConstructionContracts,
    ConstructionExecutionTask, Goal, HarnessEdge, HarnessGraph, HarnessNode,
    RelationContract, RequirementConstraint, RequirementTask, TaskDependency,
    TraceContract)


def build_dataset_case(adapter: DatasetAdapter, row: dict[str, Any], index: int) -> ConstructionCase:
    template = adapter.template
    case_id = f"ds_{adapter.dataset_id.lower().replace('-', '_')}_{index:04d}"
    tasks = [RequirementTask(task_id, desc, ["dataset_item"], [f"{task_id}_output"]) for task_id, desc in template.stages]
    deps = []
    if template.family == "multi_branch" and len(template.stages) >= 3:
        # Two independent evidence branches converge at the third stage.
        # Serializing branch A before branch B makes the Graph ablation
        # observationally identical to Full on multi-hop datasets.
        deps.extend([
            TaskDependency(template.stages[0][0], template.stages[2][0], "precedes"),
            TaskDependency(template.stages[1][0], template.stages[2][0], "precedes"),
        ])
        dependency_stages = template.stages[2:]
    else:
        dependency_stages = template.stages
    for left, right in zip(dependency_stages, dependency_stages[1:]):
        relation = "feedback" if template.family == "feedback_driven" and left[0] in {"test", "validate", "check"} else "precedes"
        deps.append(TaskDependency(left[0], right[0], relation))
    caps = [CapabilityRequirement(f"cap_{task.id}", task.id, task.objective, [template.domain, task.id]) for task in tasks]
    constraints = [RequirementConstraint(cid, kind, target, predicate) for cid, kind, target, predicate in template.constraints]
    raw_requirement = _raw_user_requirement(adapter, row)
    model = ApplicationRequirementModel(Goal("goal", raw_requirement, ["answer produced", "contracts satisfied"]), tasks, deps, caps, constraints, {"dataset": adapter.dataset_id})

    nodes = []; edges = []
    for task in tasks:
        cap = f"cap_{task.id}"; component = f"component_{task.id}"
        nodes.extend([HarnessNode(task.id, "task_pattern", task.objective, tags=[task.id, template.domain]), HarnessNode(cap, "capability", task.objective, tags=[task.id, template.domain]), HarnessNode(component, "component", f"Dataset component for {task.objective.lower()}", [cap], tags=[task.id, template.domain], metadata={"runtime_kind": "agent"})])
        edges.extend([HarnessEdge(task.id, cap, "requires"), HarnessEdge(cap, component, "realizes")])
    for dep in deps:
        edges.append(HarnessEdge(dep.source, dep.target, "reviews" if dep.relation == "feedback" else dep.relation))
    resource_specs = _branch_resource_specs(adapter, row)
    for task_id, resource_id, resource_key in resource_specs:
        nodes.append(HarnessNode(
            resource_id, "resource", f"Isolated evidence resource for {task_id}",
            tags=[task_id, "branch_resource"], metadata={"resource_key": resource_key},
        ))
        edges.append(HarnessEdge(resource_id, f"component_{task_id}", "uses"))
    for cid, kind, target, predicate in template.constraints:
        nodes.extend([HarnessNode(cid, "constraint", f"{kind} {predicate}", tags=[cid, kind], metadata={"constraint_kind": kind, "target": target, "predicate": predicate, "severity": "required"}), HarnessNode(f"component_{cid}", "control", f"Control enforcing {cid}", [f"cap_{cid}"], tags=[cid, kind])])
        edges.extend([HarnessEdge(target, cid, "constrained_by"), HarnessEdge(cid, f"component_{cid}", "realizes")])
    forbidden = [f"forbidden_{case_id}_unsafe", f"forbidden_{case_id}_untrusted"]
    nodes.extend([HarnessNode(forbidden[0], "component", "Unsafe shortcut component", ["cap_shortcut"], risk="high"), HarnessNode(forbidden[1], "component", "Untrusted source component", ["cap_untrusted"], risk="high")])
    harness = HarnessGraph(nodes, edges, metadata={"dataset": adapter.dataset_id, "case_id": case_id})

    bp_nodes = [BlueprintNode(t.id, "task", t.objective, [t.id]) for t in tasks]
    bp_nodes += [BlueprintNode(f"req_component_{c.task_id}", "component_requirement", c.description, [c.task_id], [c.id], {"candidates": [f"component_{c.task_id}"]}) for c in caps]
    bp_nodes += [BlueprintNode(f"control_{c.id}", "control", f"Enforce {c.id}", [c.id], [f"cap_{c.id}"], {"candidate": f"component_{c.id}"}) for c in constraints]
    bp_nodes += [BlueprintNode(resource_id, "resource_requirement",
                               f"Isolated evidence resource for {task_id}", [], [],
                               {"resource_key": resource_key})
                 for task_id, resource_id, resource_key in resource_specs]
    bp_edges = [BlueprintEdge(d.source, d.target, d.relation) for d in deps]
    bp_edges += [BlueprintEdge(c.task_id, f"req_component_{c.task_id}", "requires") for c in caps]
    bp_edges += [BlueprintEdge(c.target, f"control_{c.id}", "constrained_by") for c in constraints]
    bp_edges += [BlueprintEdge(resource_id, f"req_component_{task_id}", "uses")
                 for task_id, resource_id, _ in resource_specs]
    blueprint = ApplicationBlueprint(case_id, "reference", bp_nodes, bp_edges, [c.id for c in constraints], {"gold": True, "dataset": adapter.dataset_id})
    relation_contracts = [RelationContract(d.source, d.target, [d.relation, "precedes" if d.relation == "feedback" else d.relation]) for d in deps]
    relation_contracts += [RelationContract(resource_id, f"req_component_{task_id}", ["uses"])
                           for task_id, resource_id, _ in resource_specs]
    trace_contracts = [TraceContract(f"trace_cap_{c.id}", "capability_executed", c.id, "at_least_once") for c in caps]
    trace_contracts += [TraceContract(f"trace_constraint_{c.id}", "constraint_enforced", c.id, c.predicate) for c in constraints]
    trace_contracts += [TraceContract(f"trace_no_forbidden_{i}", "component_forbidden", x, "never") for i, x in enumerate(forbidden)]
    contracts = ConstructionContracts([t.id for t in tasks], [c.id for c in caps], relation_contracts, forbidden, [c.id for c in constraints], trace_contracts)
    gold_answer = _canonical_choice_gold(row.get("answer"), row.get("choices"))
    if adapter.dataset_id == "SciBench":
        gold_answer = {
            "value": row.get("answer"),
            "unit": str((row.get("raw") or {}).get("unit") or "").strip(),
        }
    # DROP's normalized rows may be produced by older manifests; recover the
    # first annotated answer span defensively.
    if adapter.dataset_id == "DROP" and gold_answer is None:
        spans = (row.get("raw") or {}).get("answers_spans", {}).get("spans", [])
        gold_answer = spans[0] if spans else None
    execution = ConstructionExecutionTask(f"{case_id}_exec_1", raw_requirement, gold_answer, row.get("context"), {"dataset": adapter.dataset_id, "row_id": row.get("id")}, [c.id for c in caps], {"adapter": adapter.dataset_id})
    # Keep two execution tasks per construction case as required by the frozen
    # schema. The second is a deterministic replay with a distinct task id.
    replay = ConstructionExecutionTask(f"{case_id}_exec_2", raw_requirement, gold_answer, row.get("context"), {"dataset": adapter.dataset_id, "row_id": row.get("id"), "replay": True}, [c.id for c in caps], {"adapter": adapter.dataset_id, "replay": True})
    case = ConstructionCase(case_id, template.family, template.domain, raw_requirement, harness, model, blueprint, contracts, [execution, replay], "validation", {"dataset": adapter.dataset_id, "task_profile": adapter.task_profile, "row": row})
    case.validate()
    return case


def _raw_user_requirement(adapter: DatasetAdapter, row: dict[str, Any]) -> str:
    """Return the user-visible request without leaking the reference pipeline.

    RequirementTemplate.text describes the gold construction protocol and must
    not be supplied as the raw utterance in a Requirement Grounding ablation.
    """
    question = str(row.get("question") or row.get("prompt") or "").strip()
    final_output = adapter.template.stages[-1][1] if adapter.template.stages else "Return the answer"
    return f"{question}\nRequired output: {final_output}."


def _branch_resource_specs(adapter: DatasetAdapter,
                           row: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Declare branch routing keys without embedding evidence or gold answers."""
    if adapter.template.family != "multi_branch":
        return []
    task_ids = [task_id for task_id, _ in adapter.template.stages[:2]]
    return [(task_id, f"resource_{task_id}", f"branch_{index}")
            for index, task_id in enumerate(task_ids)]


def _canonical_choice_gold(answer: Any, choices: Any) -> Any:
    """Carry both the option label and text for every choice-shaped dataset."""
    if isinstance(choices, list):
        if isinstance(answer, int) and 0 <= answer < len(choices):
            return f"choice:{chr(97 + answer)}|{choices[answer]}"
        # ARC/MMLU-Pro use answer letters (A-J) with a list of option text.
        label = str(answer).strip().casefold()
        if len(label) == 1 and label in "abcdefghij":
            index = ord(label) - ord("a")
            if index < len(choices):
                return f"choice:{label}|{choices[index]}"
        for index, choice in enumerate(choices):
            if str(choice).strip().casefold() == str(answer).strip().casefold():
                return f"choice:{chr(97 + index)}|{choice}"
    if isinstance(choices, dict):
        answer_text = str(answer).strip()
        for label, choice in choices.items():
            if (str(label).strip().casefold() == answer_text.casefold()
                    or str(choice).strip().casefold() == answer_text.casefold()):
                return f"choice:{str(label).strip().casefold()}|{choice}"
    return answer


def load_normalized_rows(root: Path, adapter: DatasetAdapter, limit: int = 5) -> list[dict[str, Any]]:
    path = root / adapter.source_file.replace("q2_datasets/", "")
    # source_file points at normalized files under q2_datasets.
    path = root / adapter.source_file
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if _is_qualified_row(adapter, row):
            rows.append(row)
        if len(rows) >= limit:
            break
    if len(rows) < limit:
        raise ValueError(f"{adapter.dataset_id} has only {len(rows)} qualified rows; requested {limit}")
    return rows


def _is_qualified_row(adapter: DatasetAdapter, row: dict[str, Any]) -> bool:
    """Pre-register validity criteria before any model execution occurs."""
    question, answer, context = row.get("question"), row.get("answer"), row.get("context")
    if not str(question or "").strip():
        return False
    # Code benchmarks are scored by sandbox execution and intentionally have
    # no textual gold answer in the normalized row.
    if adapter.execution.metric_name not in {"unit_test_pass", "swebench_resolved"} \
            and (answer is None or (isinstance(answer, str) and not answer.strip())):
        return False
    if adapter.dataset_id == "FinQA":
        raw = row.get("raw") or {}
        return (bool(str((raw.get("metadata") or {}).get("program", "")).strip())
                and len(str(context or "")) <= 5000)
    if adapter.dataset_id == "FinanceBench":
        evidence = (row.get("raw") or {}).get("evidence") or []
        nonempty = [item for item in evidence if isinstance(item, dict)
                    and str(item.get("evidence_text") or "").strip()]
        return len(nonempty) >= 2
    if adapter.dataset_id == "MuSiQue":
        raw = row.get("raw") or {}
        hops = raw.get("question_decomposition") or []
        return (raw.get("answerable") is True and 2 <= len(hops) <= 4
                and all(isinstance(hop, dict) and isinstance(hop.get("paragraph_support_idx"), int)
                        for hop in hops)
                and len(str(context or "")) <= 16000)
    return True
