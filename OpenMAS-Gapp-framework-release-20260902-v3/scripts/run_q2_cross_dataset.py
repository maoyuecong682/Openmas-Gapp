from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openmas_bench.ablation import Q2_VARIANTS
from openmas_bench.dataset_adapters import all_adapters
from openmas_bench.dataset_cases import build_dataset_case, load_normalized_rows
from openmas_bench.engine import GraphHarnessEngine
from openmas_bench.llm import DeterministicAdapter, LLMConfig, OpenAICompatibleAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-dataset", type=int, default=5)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--output", default=str(ROOT.parents[1] / "q2_cross_dataset_pilot_v6.json"))
    parser.add_argument("--provider", choices=["deterministic", "openai_compatible"], default="deterministic")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-key-env", default="Q1_LLM_API_KEY")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--variants", default="")
    parser.add_argument("--datasets", default="", help="comma-separated dataset IDs; empty means all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=["pilot", "formal"], default="pilot")
    parser.add_argument("--workers", type=int, default=1,
                        help="concurrent run workers; API-bound experiments usually use 2-6")
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    variants = [value for value in args.variants.split(",") if value] or list(Q2_VARIANTS)
    unknown = set(variants).difference(Q2_VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    dataset_filter = {value.strip().lower() for value in args.datasets.split(",") if value.strip()}
    datasets = [dataset for dataset in all_adapters()
                if not dataset_filter or dataset.dataset_id.lower() in dataset_filter]
    if not datasets:
        raise ValueError("dataset filter selected no datasets")

    config = LLMConfig(
        provider=args.provider,
        model="deterministic-q2-protocol" if args.provider == "deterministic" else args.model,
        base_url=args.base_url if args.provider != "deterministic" else None,
        api_key=os.environ.get(args.api_key_env) if args.provider != "deterministic" else None,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    if args.provider == "deterministic":
        adapter = DeterministicAdapter(config)
    else:
        if not config.api_key:
            raise RuntimeError(f"missing API key environment variable {args.api_key_env}")
        adapter = OpenAICompatibleAdapter(config)

    output = Path(args.output)
    checkpoint = output.with_suffix(output.suffix + ".checkpoint")
    rows, usage = _load_checkpoint(checkpoint) if args.resume else ([], _empty_usage())
    completed = {(row["dataset"], row["case_id"], row["variant"], row["seed"])
                 for row in rows if row.get("status", "completed") == "completed"}
    engine = GraphHarnessEngine(adapter, ROOT.parents[1])

    if args.workers > 1:
        _run_parallel(args, datasets, variants, seeds, engine,
                      rows, usage, completed, checkpoint)
        payload = _build_payload(args, datasets, variants, seeds, rows, usage)
        _write_json_atomic(output, payload)
        if checkpoint.exists() and not payload["failed_runs"]:
            checkpoint.unlink()
        print(json.dumps(payload["primary_table"], indent=2, ensure_ascii=False))
        print(json.dumps(payload["pilot_diagnostics"], indent=2, ensure_ascii=False))
        print(f"Wrote {output}")
        return

    for dataset in datasets:
        source_rows = load_normalized_rows(ROOT.parents[1], dataset, args.rows_per_dataset)
        for index, raw in enumerate(source_rows):
            case = build_dataset_case(dataset, raw, index)
            for variant in variants:
                for seed in seeds:
                    run_key = (dataset.dataset_id, case.case_id, variant, seed)
                    if run_key in completed:
                        continue
                    rows[:] = [row for row in rows if not (
                        row.get("status", "completed") == "failed" and
                        (row.get("dataset"), row.get("case_id"), row.get("variant"), row.get("seed")) == run_key
                    )]
                    construction_completed = False
                    try:
                        construction = engine.construct(case, seed=seed, intervention=variant)
                        construction_completed = True
                        result = engine.execute(dataset, raw, case, construction, seed, variant)
                        rows.append(result.to_experiment_record())
                        for name, value in result.usage.to_dict().items():
                            usage[name] += value
                        print(f"completed {dataset.dataset_id} row={index} variant={variant} seed={seed} score={result.primary_score:.4f}", flush=True)
                    except Exception as exc:
                        rows.append({
                            "status": "failed", "dataset": dataset.dataset_id, "case_id": case.case_id,
                            "source_id": raw.get("id"), "variant": variant, "seed": seed,
                            "primary_metric": dataset.execution.metric_name, "primary_score": None,
                            "prediction": None, "prediction_sha256": None, "answer_evaluated": False,
                            "application_digest": None, "construction_completed": construction_completed,
                            "error_type": type(exc).__name__, "error": str(exc)[-2000:], "retryable": True,
                        })
                        print(f"failed {dataset.dataset_id} row={index} variant={variant} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
                    _write_checkpoint(checkpoint, rows, usage)

    payload = _build_payload(args, datasets, variants, seeds, rows, usage)
    _write_json_atomic(output, payload)
    if checkpoint.exists() and not payload["failed_runs"]:
        checkpoint.unlink()
    print(json.dumps(payload["primary_table"], indent=2, ensure_ascii=False))
    print(json.dumps(payload["pilot_diagnostics"], indent=2, ensure_ascii=False))
    print(f"Wrote {output}")


def _run_one(dataset, raw, case, variant, seed, engine):
    """Execute one independent case. Only the caller mutates checkpoint state."""
    construction_completed = False
    try:
        construction = engine.construct(case, seed=seed, intervention=variant)
        construction_completed = True
        result = engine.execute(dataset, raw, case, construction, seed, variant)
        return result.to_experiment_record(), result.usage.to_dict()
    except Exception as exc:
        return {
            "status": "failed", "dataset": dataset.dataset_id, "case_id": case.case_id,
            "source_id": raw.get("id"), "variant": variant, "seed": seed,
            "primary_metric": dataset.execution.metric_name, "primary_score": None,
            "prediction": None, "prediction_sha256": None, "answer_evaluated": False,
            "application_digest": None, "construction_completed": construction_completed,
            "error_type": type(exc).__name__, "error": str(exc)[-2000:], "retryable": True,
        }, _empty_usage()


def _run_parallel(args, datasets, variants, seeds, engine,
                  rows, usage, completed, checkpoint):
    jobs = []
    for dataset in datasets:
        source_rows = load_normalized_rows(ROOT.parents[1], dataset, args.rows_per_dataset)
        for index, raw in enumerate(source_rows):
            case = build_dataset_case(dataset, raw, index)
            for variant in variants:
                for seed in seeds:
                    key = (dataset.dataset_id, case.case_id, variant, seed)
                    if key not in completed:
                        jobs.append((dataset, raw, case, variant, seed, key))
    print(f"parallel workers={args.workers}; pending runs={len(jobs)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, dataset, raw, case, variant, seed,
                               engine): (dataset, variant, seed, key)
                   for dataset, raw, case, variant, seed, key in jobs}
        for future in as_completed(futures):
            dataset, variant, seed, key = futures[future]
            row, delta = future.result()
            rows[:] = [old for old in rows if not (
                old.get("status") == "failed" and
                (old.get("dataset"), old.get("case_id"), old.get("variant"), old.get("seed")) == key
            )]
            rows.append(row)
            for name, value in delta.items():
                usage[name] = usage.get(name, 0) + value
            if row["status"] == "completed":
                completed.add(key)
                print(f"completed {dataset.dataset_id} variant={variant} seed={seed} score={row['primary_score']:.4f}", flush=True)
            else:
                print(f"failed {dataset.dataset_id} variant={variant} seed={seed}: {row['error_type']}: {row['error']}", flush=True)
            _write_checkpoint(checkpoint, rows, usage)


def _build_payload(args, datasets, variants, seeds, rows, usage) -> dict[str, Any]:
    expected = len(datasets) * args.rows_per_dataset * len(variants) * len(seeds)
    failed = [row for row in rows if row.get("status", "completed") == "failed"]
    return {
        "protocol": "Q2 application-executed paired cross-dataset ablation v7",
        "causal_protocol_version": "q2-multistep-finqa-tool-v5",
        "stage": args.stage,
        "provider": args.provider,
        "model": args.model if args.provider != "deterministic" else "deterministic-q2-protocol",
        "temperature": args.temperature,
        "datasets": [dataset.dataset_id for dataset in datasets],
        "metrics": {dataset.dataset_id: dataset.execution.metric_name for dataset in datasets},
        "variants": variants,
        "rows_per_dataset": args.rows_per_dataset,
        "seeds": seeds,
        "expected_runs": expected,
        "completed_runs": sum(row.get("status", "completed") == "completed" for row in rows),
        "failed_runs": len(failed),
        "failure_rate": round(len(failed) / expected, 6) if expected else 0.0,
        "complete": len(rows) == expected and not failed,
        "causal_contract": {
            "prediction_source": "constructed executable MAS application",
            "raw_requirement_excludes_reference_pipeline": True,
            "requirement_ablation_uses_arg_prompt": False,
            "blueprint_ablation_uses_compiled_artifact_contracts": False,
            "typed_graph_branch_resources_isolated": True,
            "flat_graph_ablation_receives_branch_mapping": False,
            "resource_routing_uses_answers": False,
            "variant_label_visible_to_task_executor": False,
            "gold_answer_visible_to_task_executor": False,
            "primary_metric_runtime_gated": False,
            "answers_shared_across_variants": False,
            "swebench_proxy_allowed": False,
            "full_candidate_search": True,
            "full_candidate_count": 3,
            "candidate_selection_uses_gold": False,
            "candidate_selection_signal": "answer-independent structural/resource/constraint policy score",
            "finqa_structured_decimal_tool": True,
            "finqa_multistep_tool_trace": True,
            "finqa_max_tool_steps": 8,
            "finqa_selected_evidence_checked_against_public_task": True,
            "finqa_tool_gold_used": False,
            "finqa_tool_requires_blueprint_and_realization": True,
            "finqa_verifier_can_replace_tool_result": False,
            "finqa_answer_can_replace_tool_result": False,
        },
        "usage": usage,
        "primary_table": _primary_table(rows, variants),
        "structural_table": _structural_table(rows),
        "pilot_diagnostics": _pilot_diagnostics(rows, datasets, variants, expected),
        "pairing_diagnostics": _pairing_diagnostics(rows, datasets, variants, args.rows_per_dataset, seeds),
        "finqa_tool_diagnostics": _finqa_tool_diagnostics(rows, variants),
        "runs": rows,
    }


def _complete_paired_rows(rows: list[dict[str, Any]], variants: list[str]) -> list[dict[str, Any]]:
    completed = [row for row in rows if row.get("status", "completed") == "completed"]
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        groups[(row["dataset"], row["case_id"], row["seed"])].append(row)
    return [row for group in groups.values()
            if {row["variant"] for row in group} == set(variants)
            for row in group]


def _primary_table(rows: list[dict[str, Any]], variants: list[str] | None = None) -> list[dict[str, Any]]:
    variants = variants or list(Q2_VARIANTS)
    rows = _complete_paired_rows(rows, variants)
    rows = [row for row in rows if row.get("status", "completed") == "completed"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    full_by_pair = {}
    for row in rows:
        groups[(row["dataset"], row["variant"])].append(row)
        if row["variant"] == "full_graph_harness":
            full_by_pair[(row["dataset"], row["case_id"], row["seed"])] = row["primary_score"]
    table = []
    for (dataset, variant), group in sorted(groups.items()):
        values = [float(row["primary_score"]) for row in group]
        deltas = [float(row["primary_score"]) - full_by_pair[(dataset, row["case_id"], row["seed"])]
                  for row in group if (dataset, row["case_id"], row["seed"]) in full_by_pair]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        delta_mean = statistics.mean(deltas) if deltas else None
        delta_std = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        half_width = 1.96 * delta_std / math.sqrt(len(deltas)) if deltas else None
        table.append({
            "dataset": dataset,
            "variant": variant,
            "metric": group[0]["primary_metric"],
            "mean": round(mean, 6),
            "std": round(std, 6),
            "n": len(values),
            "paired_delta_vs_full": round(delta_mean, 6) if delta_mean is not None else None,
            "paired_delta_ci95": ([round(delta_mean - half_width, 6), round(delta_mean + half_width, 6)]
                                  if half_width is not None else None),
            "wins_ties_losses_vs_full": {
                "wins": sum(delta > 0 for delta in deltas),
                "ties": sum(delta == 0 for delta in deltas),
                "losses": sum(delta < 0 for delta in deltas),
            },
        })
    return table


def _pairing_diagnostics(rows, datasets, variants, rows_per_dataset, seeds):
    expected_per_variant = rows_per_dataset * len(seeds)
    result = []
    for dataset in datasets:
        dataset_id = dataset.dataset_id
        subset = [row for row in rows if row.get("dataset") == dataset_id and row.get("status", "completed") == "completed"]
        groups: dict[tuple[str, int], set[str]] = defaultdict(set)
        for row in subset:
            groups[(row["case_id"], row["seed"])].add(row["variant"])
        complete = sum(set(variants) == values for values in groups.values())
        failure = {}
        for variant in variants:
            completed = sum(row.get("variant") == variant and row.get("status", "completed") == "completed" for row in rows if row.get("dataset") == dataset_id)
            failed = sum(row.get("variant") == variant and row.get("status") == "failed" for row in rows if row.get("dataset") == dataset_id)
            failure[variant] = {"expected": expected_per_variant, "completed": completed,
                                "failed": failed, "failure_rate": round(failed / expected_per_variant, 6) if expected_per_variant else 0.0}
        result.append({"dataset": dataset_id, "expected_pairs": rows_per_dataset * len(seeds),
                       "complete_pairs": complete, "variant_status": failure})
    return result


def _structural_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in rows if row.get("status", "completed") == "completed"]
    metrics = ["requirement_coverage", "orchestration_relation_recall", "realization_fidelity",
               "constraint_recall", "constraint_orchestration_recall",
               "constraint_satisfaction", "construction_quality"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["variant"])].append(row["construction_metrics"])
    return [{"dataset": dataset, "variant": variant, "n": len(group),
             **{metric: round(statistics.mean(float(row[metric]) for row in group), 6) for metric in metrics}}
             for (dataset, variant), group in sorted(groups.items())]


def _finqa_tool_diagnostics(rows: list[dict[str, Any]], variants: list[str]) -> list[dict[str, Any]]:
    completed = [row for row in rows
                 if row.get("dataset") == "FinQA" and row.get("status", "completed") == "completed"]
    diagnostics = []
    for variant in variants:
        group = [row for row in completed if row.get("variant") == variant]
        if not group:
            continue
        audits = [row.get("task_execution", {}).get("metadata", {}).get("finqa_tool_audit", {})
                  for row in group]
        diagnostics.append({
            "variant": variant,
            "n": len(group),
            "eligible_rate": round(sum(bool(audit.get("eligible")) for audit in audits) / len(audits), 6),
            "invocation_rate": round(sum(bool(audit.get("invoked")) for audit in audits) / len(audits), 6),
            "tool_ok_rate": round(sum(audit.get("tool_status") == "ok" for audit in audits) / len(audits), 6),
            "gold_used_any": any(bool(audit.get("gold_used")) for audit in audits),
        })
    return diagnostics


def _pilot_diagnostics(rows, datasets, variants, expected) -> dict[str, Any]:
    all_rows = list(rows)
    rows = [row for row in all_rows if row.get("status", "completed") == "completed"]
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paired: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
        paired[(row["dataset"], row["case_id"], row["seed"])].append(row)
    checks = []
    for dataset in datasets:
        group = by_dataset[dataset.dataset_id]
        means = []
        for variant in variants:
            values = [row["primary_score"] for row in group if row["variant"] == variant]
            if values:
                means.append(round(statistics.mean(values), 10))
        checks.append({
            "dataset": dataset.dataset_id,
            "all_primary_scores_zero": bool(group) and all(row["primary_score"] == 0 for row in group),
            "distinct_variant_means": len(set(means)),
            "all_variant_means_identical": len(set(means)) <= 1,
            "variant_health": {
                variant: {
                    "completed": sum(row.get("variant") == variant for row in group),
                    "failed": sum(row.get("variant") == variant and row.get("status") == "failed" for row in all_rows if row.get("dataset") == dataset.dataset_id),
                    "failure_rate": round(sum(row.get("variant") == variant and row.get("status") == "failed" for row in all_rows if row.get("dataset") == dataset.dataset_id) / max(1, expected // max(1, len(datasets)) // max(1, len(variants))), 6),
                    "fallback_rate": round(statistics.mean(float(row.get("construction_metrics", {}).get("fallback", 0.0)) for row in group if row.get("variant") == variant), 6) if any(row.get("variant") == variant for row in group) else None,
                    "nonzero_rate": round(sum(float(row.get("primary_score") or 0.0) > 0 for row in group if row.get("variant") == variant) / max(1, sum(row.get("variant") == variant for row in group)), 6),
                } for variant in variants
            },
        })
    complete_pairs = [group for group in paired.values() if len(group) == len(variants)]
    return {
        "expected_runs_complete": len(rows) == expected,
        "paired_groups": len(paired),
        "complete_paired_groups": len(complete_pairs),
        "groups_with_identical_primary_score": sum(len({row["primary_score"] for row in group}) == 1 for group in complete_pairs),
        "groups_with_identical_prediction": sum(len({row["prediction_sha256"] for row in group}) == 1 for group in complete_pairs),
        "datasets": checks,
        "interpretation": "Identical predictions are reported, not treated as protocol failures. Causal claims require paired effects and confidence intervals on the expanded run.",
    }


def _empty_usage() -> dict[str, int]:
    return {"construction_input_tokens": 0, "construction_output_tokens": 0, "construction_calls": 0,
            "execution_input_tokens": 0, "execution_output_tokens": 0, "execution_calls": 0,
            "retries": 0, "repairs": 0}


def _load_checkpoint(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not path.exists():
        return [], _empty_usage()
    value = json.loads(path.read_text(encoding="utf-8"))
    return list(value.get("runs", [])), {**_empty_usage(), **value.get("usage", {})}


def _write_checkpoint(path: Path, rows: list[dict[str, Any]], usage: dict[str, int]) -> None:
    _write_json_atomic(path, {"runs": rows, "usage": usage})


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
