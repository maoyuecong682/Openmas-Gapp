from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openmas_bench.ablation import Q2_VARIANTS, get_ablation_method
from openmas_bench.evaluate import evaluate_construction, evaluate_execution
from openmas_bench.llm import DeterministicAdapter, LLMConfig
from openmas_bench.runtime import MinimalMARRuntime
from openmas_bench.dataset_adapters import DATASET_ADAPTERS
from openmas_bench.dataset_cases import build_dataset_case, load_normalized_rows
from scripts.prepare_construction_cases import build_suite


def _mean(rows, key):
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(statistics.mean(values), 6) if values else None


def main():
    cases = build_suite()
    adapter = DeterministicAdapter(LLMConfig(provider="deterministic", model="deterministic-q2-proxy"))
    runtime = MinimalMARRuntime()
    construction_rows, execution_rows, signatures = [], [], []
    for case in cases:
        for variant in Q2_VARIANTS:
            result = get_ablation_method(variant, adapter=adapter, seed=11).construct(case.request())
            c = evaluate_construction(case, result)
            c.update({"variant": variant, "split": case.split,
                      "blueprint_present": bool(result.blueprint.metadata.get("blueprint_present", True)),
                      "blueprint_preserving": bool(result.application.metadata.get("blueprint_preserving", True))})
            construction_rows.append(c)
            executions = [runtime.execute(case, result, task, 11) for task in case.execution_tasks]
            for x in executions:
                row = evaluate_execution(case, x)
                row["variant"] = variant
                execution_rows.append(row)
            signatures.append({"case_id": case.case_id, "variant": variant,
                               "removed_module": result.blueprint.metadata.get("removed_module"),
                               "requirement_model_grounding": result.requirement_model.metadata.get("grounding"),
                               "representation": result.blueprint.metadata.get("representation"),
                               "constraint_refs": len(result.blueprint.constraint_refs),
                               "arg_constraints": len(result.requirement_model.constraints),
                               "typed_bp_relations": sum(edge.relation in {"feedback", "reviews", "constrained_by"}
                                                         for edge in result.blueprint.edges),
                               "bp_nodes": len(result.blueprint.nodes),
                               "app_nodes": len(result.application.nodes),
                               "app_mode": result.application.metadata.get("construction_mode"),
                               "uses_blueprint_realization": result.application.metadata.get("uses_blueprint_realization", True),
                               "compiled_edge_refs": sum(edge.realizes_blueprint_edge is not None for edge in result.application.edges),
                               "bound_implementations": sum(node.implementation_ref not in {"prompt_generated_agent", "generic_blueprint_interpreter"}
                                                            for node in result.application.nodes),
                               # Direct MAS may carry a plain hand-off
                               # contract for prompt continuity; that is not
                               # a compiler-emitted typed artifact contract.
                               "compiled_contracts": sum(
                                   bool(node.config.get("artifact_contract"))
                                   and result.application.metadata.get("construction_mode") != "direct_mas"
                                   for node in result.application.nodes),
                               "telemetry_notes": list(result.telemetry.notes)})
    groups = defaultdict(list)
    for row in construction_rows: groups[row["variant"]].append(row)
    summary = []
    for variant, rows in groups.items():
        ex = [x for x in execution_rows if x["variant"] == variant]
        summary.append({"variant": variant, "n_cases": len(rows),
                        "construction_quality_mean": _mean(rows, "construction_quality"),
                        "requirement_satisfaction_mean": _mean(rows, "requirement_coverage"),
                        "capability_organization_mean": _mean(rows, "orchestration_relation_recall"),
                        "blueprint_fidelity_mean": _mean(rows, "realization_fidelity"),
                        "constraint_satisfaction_mean": _mean(rows, "constraint_satisfaction"),
                        "execution_performance_mean": _mean(ex, "execution_performance"),
                        "runtime_validity_mean": _mean(ex, "runtime_valid"),
                        "trace_contract_rate_mean": _mean(ex, "trace_contract_rate"),
                        "fallback_rate": _mean(rows, "fallback")})
    resource_checks = []
    data_root = ROOT.parents[1]
    for adapter_name in ("hotpotqa", "musique"):
        dataset = DATASET_ADAPTERS[adapter_name]
        row = load_normalized_rows(data_root, dataset, 1)[0]
        case = build_dataset_case(dataset, row, 0)
        full = get_ablation_method("full_graph_harness", adapter=adapter, seed=11).construct(case.request())
        flat = get_ablation_method("w/o_graph_orchestration", adapter=adapter, seed=11).construct(case.request())
        full_bindings = sorted(key for node in full.application.nodes
                               for key in node.config.get("resource_bindings", []))
        flat_bindings = sorted(key for node in flat.application.nodes
                               for key in node.config.get("resource_bindings", []))
        resource_checks.append({"dataset": dataset.dataset_id,
                                "full_bindings": full_bindings,
                                "flat_bindings": flat_bindings})
    payload = {"protocol": "Q2 component-wise ablation deterministic sanity",
               "case_count": len(cases), "variant_count": len(Q2_VARIANTS),
               "construction_runs": len(construction_rows), "execution_runs": len(execution_rows),
               "summary": sorted(summary, key=lambda x: x["variant"]),
               "signatures": signatures,
               "checks": {"all_construct_valid": len(construction_rows) == len(cases) * len(Q2_VARIANTS),
                          "full_preserves_case_constraints": all((x["constraint_refs"] > 0) == (next(c for c in cases if c.case_id == x["case_id"]).contracts.required_constraints != []) for x in signatures if x["variant"] == "full_graph_harness"),
                          "requirement_ablation_has_no_arg_output": all(
                              x["requirement_model_grounding"] == "implicit"
                              and "arg_output_fields=0" in x["telemetry_notes"]
                              and "graph_relations_visible=false" in x["telemetry_notes"]
                              for x in signatures if x["variant"] == "w/o_requirement_grounding"),
                          "constraint_ablation_removes_constraints": all(x["constraint_refs"] == 0 for x in signatures if x["variant"] == "w/o_constraint_aware_orchestration"),
                          "constraint_ablation_preserves_arg_detection": all(
                              x["arg_constraints"] == len(next(c for c in cases if c.case_id == x["case_id"]).contracts.required_constraints)
                              for x in signatures if x["variant"] == "w/o_constraint_aware_orchestration"),
                          "graph_ablation_removes_typed_relations": all(
                              x["typed_bp_relations"] == 0
                              for x in signatures if x["variant"] == "w/o_graph_orchestration"),
                          "graph_resource_isolation_active": all(
                              x["full_bindings"] == ["branch_0", "branch_1"]
                              and x["flat_bindings"] == [] for x in resource_checks),
                          "blueprint_ablation_marked": all(not x["blueprint_present"] for x in construction_rows if x["variant"] == "w/o_blueprint"),
                          "blueprint_ablation_has_no_compiled_contracts": all(
                              x["compiled_contracts"] == 0
                              for x in signatures if x["variant"] == "w/o_blueprint"),
                          "realization_ablation_marked": all(not x["blueprint_preserving"] for x in construction_rows if x["variant"] == "w/o_realization"),
                          "realization_ablation_bypasses_compiler": all(
                              x["app_mode"] == "prompt_generation"
                              and not x["uses_blueprint_realization"]
                              and x["compiled_edge_refs"] == 0
                              and x["bound_implementations"] == 0
                              and x["compiled_contracts"] == 0
                              for x in signatures if x["variant"] == "w/o_realization")},
               "resource_checks": resource_checks}
    # Keep generated reports in the writable project root; some benchmark
    # subdirectories are read-only in the managed workspace.
    out = ROOT.parents[1] / "q2_ablation_deterministic.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(json.dumps(payload["checks"], indent=2))


if __name__ == "__main__":
    main()
