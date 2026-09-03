from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openmas_bench.dataset_adapters import DATASET_ADAPTERS
from openmas_bench.dataset_cases import build_dataset_case
from openmas_bench.evaluate import evaluate_construction
from openmas_bench.engine import GraphHarnessEngine
from openmas_bench.io import write_json
from openmas_bench.llm import AdapterResponse, DeterministicAdapter, LLMConfig, OpenAICompatibleAdapter
from openmas_bench.q1_real import Q1_REAL_BASELINES, build_q1_real_construction, harness_necessity_score

DATASET_PRESETS: dict[str, list[str]] = {
    # Main Q1 table: harder, structure-sensitive tasks that can expose the
    # value of an intermediate Harness layer.
    "structural": [
        "hotpotqa",
        "musique",
        "medqa",
        "pubmedqa",
        "finqa",
        "drop",
        "math500",
    ],
    # Supplementary breadth check: easier or more saturated tasks are kept here
    # so they can still be run when the user wants a broader appendix-style run.
    "broad": [
        "gsm8k",
        "mmlu",
        "strategyqa",
        "hotpotqa",
        "musique",
        "medqa",
        "pubmedqa",
        "finqa",
        "drop",
        "math500",
    ],
    # Lightweight sanity mode.
    "sanity": [
        "gsm8k",
        "strategyqa",
        "hotpotqa",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="run_q1_real_qa")
    parser.add_argument("--datasets", default="",
                        help="comma-separated dataset IDs; overrides --dataset-preset when provided")
    parser.add_argument("--dataset-preset", choices=sorted(DATASET_PRESETS), default="structural",
                        help="named dataset bundle used when --datasets is omitted")
    parser.add_argument("--baselines", default=",".join(Q1_REAL_BASELINES),
                        help="comma-separated Q1-real baselines")
    parser.add_argument("--rows-per-dataset", type=int, default=3)
    parser.add_argument("--row-offset", type=int, default=0,
                        help="skip first N normalized rows before selection; ignored by --row-indices")
    parser.add_argument("--row-indices", default="",
                        help="comma-separated original normalized row indices to run for every selected dataset")
    parser.add_argument("--sample", choices=["first", "random"], default="random")
    parser.add_argument("--sample-seed", type=int, default=20260827)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--provider", choices=["deterministic", "openai_compatible", "aliyun_bailian"], default="aliyun_bailian")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    datasets = _select_datasets(args.datasets or ",".join(DATASET_PRESETS[args.dataset_preset]))
    baselines = _select_baselines(args.baselines)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    output = Path(args.output)
    checkpoint = output.with_suffix(output.suffix + ".checkpoint")
    results = _load_runs(checkpoint) if args.resume else []
    completed = {(r.get("dataset_id"), r.get("row_index"), r.get("baseline"), r.get("seed"))
                 for r in results if r.get("status", "completed") == "completed"}

    jobs = []
    selected_by_dataset = {}
    for dataset in datasets:
        selected_rows = _select_rows(ROOT.parents[1], dataset, args)
        selected_by_dataset[dataset.dataset_id] = [index for index, _ in selected_rows]
        for index, raw in selected_rows:
            case = build_dataset_case(dataset, raw, index)
            for baseline in baselines:
                for seed in seeds:
                    key = (dataset.dataset_id, index, baseline, seed)
                    if key not in completed:
                        jobs.append((dataset, raw, case, index, baseline, seed))

    if args.workers <= 1:
        for job in jobs:
            record = _run_job(args, job)
            results.append(record)
            _print_record(record)
            _write_payload(checkpoint, args, datasets, baselines, seeds, selected_by_dataset, results)
    else:
        print(f"running {len(jobs)} remaining Q1-real QA jobs with workers={args.workers}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_job, args, job) for job in jobs]
            for future in as_completed(futures):
                record = future.result()
                results.append(record)
                _print_record(record)
                _write_payload(checkpoint, args, datasets, baselines, seeds, selected_by_dataset, results)

    _write_payload(output, args, datasets, baselines, seeds, selected_by_dataset, results)
    print(f"completed runs={len(results)} output={output}", flush=True)


def _run_job(args, job) -> dict[str, Any]:
    dataset, raw, case, index, baseline, seed = job
    adapter = Q1RealRobustAdapter(_build_adapter(args))
    engine = GraphHarnessEngine(adapter, ROOT)
    try:
        construction = build_q1_real_construction(case, baseline, adapter, seed, engine=engine)
        construction_metrics = evaluate_construction(case, construction)
        task_results = []
        scores = []
        for task_index, task in enumerate(case.execution_tasks):
            engine_result = engine.execute(
                dataset, raw, case, construction, seed,
                intervention=baseline, task_index=task_index)
            task_result = engine_result.task_execution
            answer_score = engine_result.answer_score
            if answer_score is not None:
                scores.append(float(answer_score))
            task_results.append({
                "task_id": task.id,
                "prediction": task_result.prediction,
                "answer_score": answer_score,
                "task_execution": task_result.to_dict(),
                "engine_audit": engine_result.audit,
            })
        e2e = round(statistics.mean(scores), 6) if scores else None
        record = {
            "status": "completed",
            "dataset_id": dataset.dataset_id,
            "row_index": index,
            "source_id": raw.get("id"),
            "case_id": case.case_id,
            "family": case.family,
            "domain": case.domain,
            "baseline": baseline,
            "baseline_label": Q1_REAL_BASELINES[baseline].label,
            "baseline_layer": Q1_REAL_BASELINES[baseline].layer,
            "seed": seed,
            "primary_metric": dataset.execution.metric_name,
            "primary_score": e2e,
            "e2e_success": e2e,
            "e2e_supported": bool(scores),
            "construction": construction_metrics,
            "execution": task_results,
            "app_node_count": len(construction.application.nodes),
            "app_edge_count": len(construction.application.edges),
            "entrypoint_count": len(construction.application.entrypoints),
        }
        record["harness_necessity_score"] = harness_necessity_score(record)
        return record
    except Exception as exc:
        return {
            "status": "failed",
            "dataset_id": dataset.dataset_id,
            "row_index": index,
            "source_id": raw.get("id"),
            "case_id": case.case_id,
            "family": case.family,
            "domain": case.domain,
            "baseline": baseline,
            "baseline_label": Q1_REAL_BASELINES[baseline].label,
            "baseline_layer": Q1_REAL_BASELINES[baseline].layer,
            "seed": seed,
            "primary_metric": dataset.execution.metric_name,
            "primary_score": None,
            "e2e_success": None,
            "e2e_supported": False,
            "harness_necessity_score": None,
            "error_type": type(exc).__name__,
            "error": str(exc)[-2000:],
        }


def _select_datasets(value: str):
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    datasets = []
    for name in requested:
        matched = next((adapter for key, adapter in DATASET_ADAPTERS.items()
                        if key.lower() == name or adapter.dataset_id.lower() == name), None)
        if matched is None:
            raise KeyError(f"unknown dataset {name}; choose from {sorted(DATASET_ADAPTERS)}")
        datasets.append(matched)
    return datasets


def _select_baselines(value: str) -> list[str]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in requested if name not in Q1_REAL_BASELINES]
    if unknown:
        raise KeyError(f"unknown Q1-real baselines {unknown}; choose from {sorted(Q1_REAL_BASELINES)}")
    return requested


def _select_rows(root: Path, dataset, args) -> list[tuple[int, dict[str, Any]]]:
    all_rows = _load_all_normalized_rows(root, dataset)
    explicit = _parse_indices(args.row_indices)
    if explicit:
        selected = []
        for index in explicit:
            if index < 0 or index >= len(all_rows):
                raise ValueError(f"{dataset.dataset_id} row index {index} out of range 0..{len(all_rows) - 1}")
            selected.append((index, all_rows[index]))
        return selected
    candidates = list(enumerate(all_rows))[args.row_offset:]
    if not candidates:
        raise ValueError(f"{dataset.dataset_id} has no rows after offset {args.row_offset}")
    limit = min(args.rows_per_dataset, len(candidates))
    if args.sample == "random":
        rng = random.Random(_dataset_sample_seed(args.sample_seed, dataset.dataset_id))
        return sorted(rng.sample(candidates, limit), key=lambda item: item[0])
    return candidates[:limit]


def _load_all_normalized_rows(root: Path, dataset) -> list[dict[str, Any]]:
    path = root / dataset.source_file
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if _is_qualified_row(dataset, row):
                rows.append(row)
    if not rows:
        raise ValueError(f"{dataset.dataset_id} has no qualified rows in {path}")
    return rows


def _is_qualified_row(dataset, row: dict[str, Any]) -> bool:
    question = str(row.get("question") or row.get("prompt") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not question or not answer:
        return False
    context = str(row.get("context") or "")
    if dataset.dataset_id in {"HotpotQA", "MuSiQue"}:
        return bool(context.strip()) and len(context) <= 16000
    return True


def _parse_indices(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _dataset_sample_seed(sample_seed: int, dataset_id: str) -> int:
    digest = hashlib.sha256(f"{sample_seed}:{dataset_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _build_adapter(args):
    if args.provider == "deterministic":
        return DeterministicAdapter(LLMConfig(provider="deterministic", model="deterministic-q1-real", max_output_tokens=args.max_output_tokens))
    key = os.environ.get(args.api_key_env)
    if not key:
        raise RuntimeError(f"missing API key environment variable {args.api_key_env}")
    provider = "openai_compatible" if args.provider == "aliyun_bailian" else args.provider
    return OpenAICompatibleAdapter(LLMConfig(
        provider=provider,
        model=args.model,
        base_url=args.base_url,
        api_key=key,
        temperature=0.0,
        max_output_tokens=args.max_output_tokens,
    ))


class Q1RealRobustAdapter:
    def __init__(self, inner):
        self.inner = inner
        self.config = inner.config

    def generate_json(self, system_prompt: str, user_prompt: str, seed: int,
                      required_fields: set[str] | None = None) -> AdapterResponse:
        try:
            return self.inner.generate_json(system_prompt, user_prompt, seed, required_fields)
        except RuntimeError:
            if not required_fields:
                raise
            response = self.inner.generate_text(
                system_prompt + " Return plain text content only. Do not use JSON.",
                user_prompt,
                seed,
            )
            field = _preferred_field(required_fields)
            value = response.value.get("text", response.raw_text)
            return AdapterResponse(
                {field: value}, response.provider, response.model, response.seed,
                response.input_tokens, response.output_tokens, response.latency_ms,
                response.raw_text, response.retry_count + 1, True, response.finish_reason,
            )

    def generate_text(self, system_prompt: str, user_prompt: str, seed: int) -> AdapterResponse:
        return self.inner.generate_text(system_prompt, user_prompt, seed)


def _preferred_field(required_fields: set[str]) -> str:
    for field in ("answer", "artifact", "text", "output"):
        if field in required_fields:
            return field
    return sorted(required_fields)[0]


def _load_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(json.load(handle).get("runs", []))


def _write_payload(path: Path, args, datasets, baselines, seeds, selected_by_dataset, results) -> None:
    expected = sum(len(v) for v in selected_by_dataset.values()) * len(baselines) * len(seeds)
    completed = sum(row.get("status", "completed") == "completed" for row in results)
    failed = sum(row.get("status") == "failed" for row in results)
    payload = {
        "protocol": "Q1 real QA Harness-layer necessity",
        "purpose": "Compare Harness-layer construction against same-level non-Harness construction paradigms on real QA benchmark flows.",
        "datasets": [dataset.dataset_id for dataset in datasets],
        "selected_rows": selected_by_dataset,
        "dataset_preset": args.dataset_preset if not args.datasets.strip() else None,
        "rows_per_dataset": args.rows_per_dataset,
        "row_offset": args.row_offset,
        "row_indices": _parse_indices(args.row_indices),
        "sample": args.sample,
        "sample_seed": args.sample_seed,
        "seeds": seeds,
        "baselines": baselines,
        "baseline_labels": {name: Q1_REAL_BASELINES[name].label for name in baselines},
        "expected_runs": expected,
        "completed_runs": completed,
        "failed_runs": failed,
        "progress": round((completed + failed) / expected, 6) if expected else 0.0,
        "runs": results,
        "summary": _summarize(results),
    }
    write_json(path, payload)


def _summarize(results) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in results:
        if row.get("status", "completed") == "completed":
            grouped[(row["dataset_id"], row["baseline"])].append(row)
    output = []
    for (dataset, baseline), rows in sorted(grouped.items()):
        output.append({
            "dataset": dataset,
            "baseline": baseline,
            "baseline_label": rows[0].get("baseline_label", baseline),
            "baseline_layer": rows[0].get("baseline_layer"),
            "n": len(rows),
            "primary_score_mean": _mean([row.get("primary_score") for row in rows]),
            "e2e_success_mean": _mean([row.get("e2e_success") for row in rows]),
            "harness_necessity_score_mean": _mean([row.get("harness_necessity_score") for row in rows]),
            "construction_quality_mean": _mean([_nested(row, "construction", "construction_quality") for row in rows]),
            "architecture_validity_mean": _mean([_nested(row, "construction", "architecture_validity") for row in rows]),
            "constraint_satisfaction_mean": _mean([_nested(row, "construction", "constraint_satisfaction") for row in rows]),
            "constraint_violation_rate_mean": _one_minus_mean([_nested(row, "construction", "constraint_satisfaction") for row in rows]),
            "orchestration_relation_recall_mean": _mean([_nested(row, "construction", "orchestration_relation_recall") for row in rows]),
            "blueprint_coverage_mean": _mean([_nested(row, "construction", "blueprint_coverage") for row in rows]),
            "realization_fidelity_mean": _mean([_nested(row, "construction", "realization_fidelity") for row in rows]),
            "avg_exec_steps": _mean([_first_execution_steps(row) for row in rows]),
        })
    return output


def _nested(row: dict[str, Any], section: str, metric: str):
    return (row.get(section) or {}).get(metric)


def _first_execution_steps(row: dict[str, Any]):
    executions = row.get("execution") or []
    if not executions:
        return None
    return len((executions[0].get("task_execution") or {}).get("node_executions") or [])


def _mean(values) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return round(statistics.mean(clean), 6) if clean else None


def _one_minus_mean(values) -> float | None:
    value = _mean(values)
    return round(1 - value, 6) if value is not None else None


def _print_record(record: dict[str, Any]) -> None:
    prefix = "completed" if record.get("status") == "completed" else "failed"
    message = f"{prefix} {record['dataset_id']} row={record['row_index']} baseline={record['baseline']} seed={record['seed']}"
    if record.get("status") == "failed":
        message += f": {record.get('error_type')}: {record.get('error')}"
    print(message, flush=True)


if __name__ == "__main__":
    main()
