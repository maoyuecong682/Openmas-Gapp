from __future__ import annotations

import json
import sys
import argparse
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openmas_bench.io import write_json
from openmas_bench.schema import (
    ApplicationBlueprint, ApplicationRequirementModel, BlueprintEdge, BlueprintNode,
    CapabilityRequirement, ConstructionCase, ConstructionContracts, ConstructionExecutionTask,
    Goal, HarnessEdge, HarnessGraph, HarnessNode, RelationContract, RequirementConstraint,
    RequirementTask, TaskDependency, TraceContract,
)


def d(case_id, split, family, domain, requirement, tasks, edges, constraints=()):
    return case_id, split, family, domain, requirement, tasks, edges, constraints


DEFINITIONS = [
    d("seq_finance", "dev", "sequential", "finance", "Build a financial filing assistant that must retrieve an approved filing, extract figures, calculate the requested metric, verify the calculation, and report a cited answer.",
      [("retrieve", "Retrieve approved filing"), ("extract", "Extract relevant figures"), ("calculate", "Calculate requested metric"), ("verify", "Verify the calculation"), ("report", "Report a cited answer")], [("retrieve", "extract", "precedes"), ("extract", "calculate", "precedes"), ("calculate", "verify", "precedes"), ("verify", "report", "precedes")]),
    d("seq_software", "dev", "sequential", "software", "Build a repository repair assistant that must localize the defect, modify the code, run tests, review the patch, and return a validated diff.",
      [("localize", "Localize the repository defect"), ("modify", "Modify relevant code"), ("test", "Run tests"), ("review", "Review the patch"), ("report", "Return a validated diff")], [("localize", "modify", "precedes"), ("modify", "test", "precedes"), ("test", "review", "precedes"), ("review", "report", "precedes")]),
    d("branch_evidence", "dev", "multi_branch", "biomedical", "Build a biomedical evidence assistant that must search two approved evidence streams in parallel, extract both, synthesize findings, verify citations, and report uncertainty.",
      [("search_primary", "Search primary biomedical evidence"), ("search_guideline", "Search guideline evidence"), ("extract_primary", "Extract primary evidence"), ("extract_guideline", "Extract guideline evidence"), ("synthesize", "Synthesize evidence streams"), ("verify", "Verify citations"), ("report", "Report findings and uncertainty")], [("search_primary", "extract_primary", "precedes"), ("search_guideline", "extract_guideline", "precedes"), ("extract_primary", "synthesize", "precedes"), ("extract_guideline", "synthesize", "precedes"), ("synthesize", "verify", "precedes"), ("verify", "report", "precedes")]),
    d("branch_due_diligence", "dev", "multi_branch", "finance", "Build a due-diligence assistant that must analyze statements and risk disclosures in parallel, reconcile analyses, run compliance review, and produce a sourced memo.",
      [("statement_analysis", "Analyze financial statements"), ("risk_analysis", "Analyze risk disclosures"), ("reconcile", "Reconcile both analyses"), ("compliance", "Run compliance review"), ("report", "Produce a sourced memo")], [("statement_analysis", "reconcile", "precedes"), ("risk_analysis", "reconcile", "precedes"), ("reconcile", "compliance", "precedes"), ("compliance", "report", "precedes")]),
    d("feedback_material", "dev", "feedback_driven", "science", "Build a scientific screening assistant that must retrieve candidates, simulate properties, validate plausibility, revise failed candidates using feedback, and report only validated candidates.",
      [("retrieve", "Retrieve scientific candidates"), ("simulate", "Simulate candidate properties"), ("validate", "Validate plausibility"), ("revise", "Revise failed candidates"), ("report", "Report validated candidates")], [("retrieve", "simulate", "precedes"), ("simulate", "validate", "precedes"), ("validate", "revise", "feedback"), ("revise", "simulate", "feedback"), ("validate", "report", "precedes")]),
    d("feedback_code", "dev", "feedback_driven", "software", "Build a test-driven repair assistant that must generate a patch, execute tests, diagnose failures, revise from test feedback until validation passes, and return the final diff.",
      [("patch", "Generate a candidate patch"), ("test", "Execute tests"), ("diagnose", "Diagnose failures"), ("revise", "Revise the patch"), ("report", "Return validated diff")], [("patch", "test", "precedes"), ("test", "diagnose", "feedback"), ("diagnose", "revise", "precedes"), ("revise", "test", "feedback"), ("test", "report", "precedes")]),
    d("constraint_medical", "dev", "constraint_heavy", "biomedical", "Build a clinical evidence support assistant that must retrieve approved evidence, assess applicability, check safety, and produce a cited recommendation; every high-risk recommendation requires human approval and no unapproved source may be used.",
      [("retrieve", "Retrieve approved clinical evidence"), ("assess", "Assess applicability"), ("safety", "Check safety risks"), ("recommend", "Produce cited recommendation")], [("retrieve", "assess", "precedes"), ("assess", "safety", "precedes"), ("safety", "recommend", "precedes")], [("human_approval", "human_approval", "recommend", "required"), ("approved_source", "resource_policy", "retrieve", "approved_only")]),
    d("constraint_legal", "dev", "constraint_heavy", "legal", "Build a legal research assistant that must analyze facts, retrieve authoritative law, organize evidence, draft cited advice, and obtain professional review; confidential data must remain isolated.",
      [("facts", "Analyze case facts"), ("retrieve", "Retrieve authoritative law"), ("evidence", "Organize evidence"), ("draft", "Draft cited advice"), ("review", "Obtain professional review")], [("facts", "evidence", "precedes"), ("retrieve", "evidence", "precedes"), ("evidence", "draft", "precedes"), ("draft", "review", "precedes")], [("professional_review", "human_approval", "draft", "required"), ("data_isolation", "privacy", "facts", "isolated")]),
    # Validation cases are held separate from development cases. They cover new task and constraint compositions.
    d("val_seq_filing_audit", "validation", "sequential", "finance", "Construct a filing-audit application that retrieves a report, extracts evidence, computes a ratio, independently verifies it, and publishes a cited audit note.",
      [("retrieve", "Retrieve filing report"), ("extract", "Extract audit evidence"), ("calculate", "Compute financial ratio"), ("verify", "Independently verify ratio"), ("publish", "Publish cited audit note")], [("retrieve", "extract", "precedes"), ("extract", "calculate", "precedes"), ("calculate", "verify", "precedes"), ("verify", "publish", "precedes")]),
    d("val_seq_evidence", "validation", "sequential", "biomedical", "Construct an evidence appraisal application that retrieves abstracts, screens relevance, extracts claims, verifies support, and writes a qualified answer.",
      [("retrieve", "Retrieve biomedical abstracts"), ("screen", "Screen relevance"), ("extract", "Extract claims"), ("verify", "Verify evidential support"), ("report", "Write qualified answer")], [("retrieve", "screen", "precedes"), ("screen", "extract", "precedes"), ("extract", "verify", "precedes"), ("verify", "report", "precedes")]),
    d("val_seq_patch", "validation", "sequential", "software", "Construct a patch application that reproduces an issue, localizes its cause, edits the repository, executes regression tests, and emits the verified patch.",
      [("reproduce", "Reproduce repository issue"), ("localize", "Localize root cause"), ("edit", "Edit repository"), ("test", "Execute regression tests"), ("report", "Emit verified patch")], [("reproduce", "localize", "precedes"), ("localize", "edit", "precedes"), ("edit", "test", "precedes"), ("test", "report", "precedes")]),
    d("val_branch_finance", "validation", "multi_branch", "finance", "Construct an investment-analysis application that analyzes tabular performance and narrative risks in parallel, reconciles findings, verifies calculations, and reports a sourced conclusion.",
      [("table_analysis", "Analyze tabular performance"), ("narrative_analysis", "Analyze narrative risks"), ("reconcile", "Reconcile findings"), ("verify", "Verify calculations"), ("report", "Report sourced conclusion")], [("table_analysis", "reconcile", "precedes"), ("narrative_analysis", "reconcile", "precedes"), ("reconcile", "verify", "precedes"), ("verify", "report", "precedes")]),
    d("val_branch_biomedical", "validation", "multi_branch", "biomedical", "Construct a biomedical answer application that separately extracts methods and results, checks consistency across both branches, synthesizes evidence, and reports confidence.",
      [("methods", "Extract study methods"), ("results", "Extract study results"), ("consistency", "Check branch consistency"), ("synthesize", "Synthesize evidence"), ("report", "Report confidence")], [("methods", "consistency", "precedes"), ("results", "consistency", "precedes"), ("consistency", "synthesize", "precedes"), ("synthesize", "report", "precedes")]),
    d("val_branch_software", "validation", "multi_branch", "software", "Construct a repository diagnosis application that inspects implementation and tests in parallel, merges evidence, proposes a patch, runs validation, and reports the result.",
      [("inspect_code", "Inspect implementation"), ("inspect_tests", "Inspect tests"), ("merge", "Merge diagnosis evidence"), ("patch", "Propose patch"), ("validate", "Run validation"), ("report", "Report result")], [("inspect_code", "merge", "precedes"), ("inspect_tests", "merge", "precedes"), ("merge", "patch", "precedes"), ("patch", "validate", "precedes"), ("validate", "report", "precedes")]),
    d("val_feedback_finance", "validation", "feedback_driven", "finance", "Construct a financial reasoning application that computes an answer, audits the derivation, diagnoses failed checks, revises the computation, and publishes only an audited result.",
      [("compute", "Compute financial answer"), ("audit", "Audit derivation"), ("diagnose", "Diagnose failed checks"), ("revise", "Revise computation"), ("publish", "Publish audited result")], [("compute", "audit", "precedes"), ("audit", "diagnose", "feedback"), ("diagnose", "revise", "precedes"), ("revise", "audit", "feedback"), ("audit", "publish", "precedes")]),
    d("val_feedback_biomedical", "validation", "feedback_driven", "biomedical", "Construct an evidence synthesis application that drafts a conclusion, checks citation support, diagnoses unsupported claims, revises the synthesis, and releases only a verified answer.",
      [("draft", "Draft evidence conclusion"), ("check", "Check citation support"), ("diagnose", "Diagnose unsupported claims"), ("revise", "Revise synthesis"), ("release", "Release verified answer")], [("draft", "check", "precedes"), ("check", "diagnose", "feedback"), ("diagnose", "revise", "precedes"), ("revise", "check", "feedback"), ("check", "release", "precedes")]),
    d("val_feedback_software", "validation", "feedback_driven", "software", "Construct a repair loop that patches code, runs focused tests, analyzes failures, revises the patch, reruns validation, and emits only a passing diff.",
      [("patch", "Patch code"), ("test", "Run focused tests"), ("analyze", "Analyze failures"), ("revise", "Revise patch"), ("emit", "Emit passing diff")], [("patch", "test", "precedes"), ("test", "analyze", "feedback"), ("analyze", "revise", "precedes"), ("revise", "test", "feedback"), ("test", "emit", "precedes")]),
    d("val_constraint_finance", "validation", "constraint_heavy", "finance", "Construct a financial advisory application that retrieves approved filings, calculates and verifies metrics, and drafts a recommendation; a compliance reviewer must approve publication and speculative sources are forbidden.",
      [("retrieve", "Retrieve approved filings"), ("calculate", "Calculate metrics"), ("verify", "Verify metrics"), ("draft", "Draft recommendation"), ("publish", "Publish approved recommendation")], [("retrieve", "calculate", "precedes"), ("calculate", "verify", "precedes"), ("verify", "draft", "precedes"), ("draft", "publish", "precedes")], [("compliance_approval", "human_approval", "publish", "required"), ("approved_source", "resource_policy", "retrieve", "approved_only")]),
    d("val_constraint_biomedical", "validation", "constraint_heavy", "biomedical", "Construct a biomedical evidence application that uses approved literature, checks safety and uncertainty, and drafts a response; human review is mandatory before release and unsupported claims are forbidden.",
      [("retrieve", "Retrieve approved literature"), ("assess", "Assess evidence"), ("safety", "Check safety and uncertainty"), ("draft", "Draft response"), ("release", "Release reviewed response")], [("retrieve", "assess", "precedes"), ("assess", "safety", "precedes"), ("safety", "draft", "precedes"), ("draft", "release", "precedes")], [("human_review", "human_approval", "release", "required"), ("supported_claims", "evidence_policy", "draft", "supported_only")]),
    d("val_constraint_software", "validation", "constraint_heavy", "software", "Construct a repository repair application that modifies only allowed files, runs required regression tests, obtains reviewer approval, and emits a patch; destructive shell operations are forbidden.",
      [("localize", "Localize defect"), ("modify", "Modify allowed files"), ("test", "Run required regression tests"), ("review", "Obtain reviewer approval"), ("emit", "Emit approved patch")], [("localize", "modify", "precedes"), ("modify", "test", "precedes"), ("test", "review", "precedes"), ("review", "emit", "precedes")], [("review_approval", "human_approval", "emit", "required"), ("safe_operations", "operation_policy", "modify", "non_destructive")]),
]


SOURCE_FILES = {"finance": "finance_tasks.jsonl", "biomedical": "biomedical_tasks.jsonl", "software": "software_tasks.jsonl", "science": "biomedical_tasks.jsonl", "legal": "software_tasks.jsonl"}


@lru_cache(maxsize=1)
def load_sources():
    result = {}
    for domain, name in SOURCE_FILES.items():
        records = [json.loads(line) for line in (ROOT / "cleaned" / name).read_text(encoding="utf-8").splitlines() if line.strip()]
        result[domain] = records
    return result


def build_case(definition, sources, source_offset):
    case_id, split, family, domain, requirement, task_defs, dependency_defs, constraint_defs = definition
    tasks = [RequirementTask(task_id, description, ["application_input"], [f"{task_id}_output"]) for task_id, description in task_defs]
    dependencies = [TaskDependency(source, target, relation) for source, target, relation in dependency_defs]
    capabilities = [CapabilityRequirement(f"cap_{task_id}", task_id, description, [task_id, domain]) for task_id, description in task_defs]
    constraints = [RequirementConstraint(cid, kind, target, predicate) for cid, kind, target, predicate in constraint_defs]
    model = ApplicationRequirementModel(Goal("goal", requirement, ["required outputs produced", "contracts satisfied"]), tasks, dependencies, capabilities, constraints, {"gold": True})

    nodes, edges = [], []
    for task_id, description in task_defs:
        cap_id, component_id = f"cap_{task_id}", f"component_{task_id}"
        nodes += [HarnessNode(task_id, "task_pattern", description, tags=[task_id, *description.lower().split()[:2]]), HarnessNode(cap_id, "capability", description, tags=[task_id, domain]), HarnessNode(component_id, "component", f"Reusable component for {description.lower()}", [cap_id], tags=[task_id, domain], metadata={"runtime_kind": "agent"})]
        edges += [HarnessEdge(task_id, cap_id, "requires"), HarnessEdge(cap_id, component_id, "realizes")]
    for source, target, relation in dependency_defs:
        edges.append(HarnessEdge(source, target, "reviews" if relation == "feedback" else relation))
    for cid, kind, target, predicate in constraint_defs:
        nodes += [HarnessNode(cid, "constraint", f"{kind} {predicate}", tags=[cid, kind, predicate, "human" if kind == "human_approval" else kind], metadata={"constraint_kind": kind, "target": target, "predicate": predicate, "severity": "required"}), HarnessNode(f"component_{cid}", "control", f"Control enforcing {cid}", [f"cap_{cid}"], tags=[cid, kind])]
        edges += [HarnessEdge(target, cid, "constrained_by"), HarnessEdge(cid, f"component_{cid}", "realizes")]
    forbidden = [f"forbidden_{case_id}_unsafe", f"forbidden_{case_id}_untrusted"]
    nodes += [HarnessNode(forbidden[0], "component", "Unsafe shortcut component", ["cap_shortcut"], tags=["shortcut", domain], risk="high"), HarnessNode(forbidden[1], "component", "Untrusted source component", ["cap_untrusted"], tags=["untrusted"], risk="high")]
    harness = HarnessGraph(nodes, edges, metadata={"case_id": case_id, "family": family, "shared_across_q1_methods": True})

    bp_nodes = [BlueprintNode(x.id, "task", x.objective, [x.id]) for x in tasks]
    bp_nodes += [BlueprintNode(f"req_component_{x.task_id}", "component_requirement", x.description, [x.id], [x.id], {"candidates": [f"component_{x.task_id}"]}) for x in capabilities]
    bp_nodes += [BlueprintNode(f"control_{x.id}", "control", f"Enforce {x.id}", [x.id], [f"cap_{x.id}"], {"candidate": f"component_{x.id}"}) for x in constraints]
    bp_edges = [BlueprintEdge(x.source, x.target, x.relation) for x in dependencies] + [BlueprintEdge(x.task_id, f"req_component_{x.task_id}", "requires") for x in capabilities] + [BlueprintEdge(x.target, f"control_{x.id}", "constrained_by") for x in constraints]
    blueprint = ApplicationBlueprint(case_id, "reference", bp_nodes, bp_edges, [x.id for x in constraints], {"gold": True, "non_unique": True})

    relation_contracts = [RelationContract(x.source, x.target, [x.relation, "precedes" if x.relation == "feedback" else x.relation]) for x in dependencies]
    trace_contracts = [TraceContract(f"trace_cap_{x.id}", "capability_executed", x.id, "at_least_once") for x in capabilities]
    trace_contracts += [TraceContract(f"trace_constraint_{x.id}", "constraint_enforced", x.id, x.predicate) for x in constraints]
    trace_contracts += [TraceContract(f"trace_no_forbidden_{index}", "component_forbidden", component, "never") for index, component in enumerate(forbidden)]
    contracts = ConstructionContracts([x.id for x in tasks], [x.id for x in capabilities], relation_contracts, forbidden, [x.id for x in constraints], trace_contracts)

    records = sources[domain]
    execution_tasks = []
    for index in range(3):
        row = records[(source_offset + index) % len(records)]
        execution_tasks.append(ConstructionExecutionTask(f"{case_id}_exec_{index+1}", row["prompt"], row.get("answer"), row.get("context"), {"dataset": row.get("metadata", {}).get("source", SOURCE_FILES[domain]), "task_id": row["task_id"], "proxy_domain": domain in {"science", "legal"}}, [x.id for x in capabilities], row.get("metadata", {})))
    case = ConstructionCase(case_id, family, domain, requirement, harness, model, blueprint, contracts, execution_tasks, split, {"annotation": "expert-authored Q1 pilot", "version": "2.0"})
    case.validate()
    return case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "q1_formal_cases"))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sources, offsets, cases = load_sources(), {x: 0 for x in SOURCE_FILES}, []
    for definition in DEFINITIONS:
        domain = definition[3]
        case = build_case(definition, sources, offsets[domain])
        offsets[domain] += 3
        cases.append(case)
        write_json(output / f"{case.case_id}.json", case.to_dict())
    # Remove stale generated cases that are not part of the frozen 20-case suite.
    valid = {f"{x.case_id}.json" for x in cases} | {"index.json"}
    for path in output.glob("*.json"):
        if path.name not in valid:
            path.unlink()
    write_json(output / "index.json", {"schema_version": "2.0", "count": len(cases), "splits": {"dev": 8, "validation": 12, "test": 0}, "cases": [{"case_id": x.case_id, "split": x.split, "family": x.family, "domain": x.domain, "execution_tasks": len(x.execution_tasks), "path": f"{x.case_id}.json"} for x in cases]})
    print(f"wrote {len(cases)} cases with {sum(len(x.execution_tasks) for x in cases)} execution tasks")


@lru_cache(maxsize=1)
def build_suite():
    """Build the frozen suite in memory when the host blocks generated files."""
    sources, offsets, cases = load_sources(), {x: 0 for x in SOURCE_FILES}, []
    for definition in DEFINITIONS:
        domain = definition[3]
        cases.append(build_case(definition, sources, offsets[domain]))
        offsets[domain] += 3
    return cases


if __name__ == "__main__":
    main()
