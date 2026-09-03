from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import random
import sys
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openmas_bench.application_executor import ApplicationTaskExecutor
from openmas_bench.dataset_adapters import DATASET_ADAPTERS
from openmas_bench.dataset_cases import build_dataset_case
from openmas_bench.io import write_json
from openmas_bench.llm import AdapterResponse, DeterministicAdapter, LLMConfig, OpenAICompatibleAdapter
from openmas_bench.q3.real import real_construction_result, run_real_case
from openmas_bench.runtime import MinimalMARRuntime


Q3_REAL_BASELINES = [
    "flat_component_selection",
    "sequence_based_orchestration",
    "tree_based_planning",
    "workflow_based_template",
    "agent_graph_orchestration",
    "graph_harness",
]

Q3_DATASET_PRESETS: dict[str, list[str]] = {
    # Main paper table: genuinely structure-sensitive tasks.
    "structural_core": [
        "musique",
        "financebench",
        "pubmedqa",
        "medqa",
        "mmlu_pro",
    ],
    "structural_core_legacy": [
        "hotpotqa",
        "musique",
        "finqa",
        "drop",
        "mmlu_pro",
    ],
    # Wider but still defensible: keep MedQA as a supplement, not a main claim.
    "structural_extended": [
        "musique",
        "financebench",
        "pubmedqa",
        "medqa",
        "mmlu_pro",
        "arc",
    ],
    # Replacement table for the current pilot: removes MATH-500 and uses
    # broader, less saturated reasoning datasets.
    "structural_core_no_math500": [
        "pubmedqa",
        "musique",
        "financebench",
        "medqa",
        "mmlu_pro",
    ],
    # Trial bundle for exploring additional replacement candidates.
    "structural_candidates": [
        "arc",
        "financebench",
        "medqa",
        "scibench",
        "pubmedqa",
        "mmlu_pro",
        "musique",
    ],
    # Supplementary appendix-style bundle.
    "supplementary": [
        "strategyqa",
        "pubmedqa",
        "medqa",
        "scibench",
    ],
}

Q3_ROW_PRESETS: dict[str, dict[str, list[int]]] = {
    # Harder rows chosen from the normalized datasets after a quick structural scan.
    "structural_hard": {
        "hotpotqa": [22, 20, 28],
        "musique": [15, 17, 42, 49, 51, 58, 78, 93, 96, 99],
        "finqa": [18, 10, 24],
        "drop": [93, 92, 58],
        "medqa": [0, 2, 10, 11, 14, 18, 22, 28],
        "mmlu_pro": [0, 1, 2, 4, 9, 10, 14, 16, 17, 18],
    },
    # Stricter hard rows used when the pilot table needs to avoid
    # oversaturated examples and trivial numeric-year HotpotQA cases.
    "structural_hard_strict": {
        "hotpotqa": [1, 2, 4],
        "musique": [15, 17, 42, 49, 51, 58, 78, 93, 96, 99],
        "finqa": [8, 28, 29],
        "drop": [93, 92, 58],
        "medqa": [0, 2, 10, 11, 14, 18, 22, 28],
        "mmlu_pro": [0, 1, 2, 4, 9, 10, 14, 16, 17, 18],
    },
    "structural_core_no_math500": {
        "hotpotqa": [1, 2, 4],
        "musique": [15, 17, 42, 49, 51, 58, 78, 93, 96, 99],
        "finqa": [8, 28, 29],
        "drop": [93, 92, 58],
        "medqa": [0, 2, 10, 11, 14, 18, 22, 28],
        "mmlu_pro": [0, 1, 2, 4, 9, 10, 14, 16, 17, 18],
    },
    "structural_candidates": {
        "arc": [84, 86, 93],
        "medqa": [0, 2, 10, 11, 14, 18, 22, 28],
        "mmlu_pro": [0, 1, 2, 4, 9, 10, 14, 16, 17, 18],
        "hotpotqa": [1, 2, 4],
        "musique": [15, 17, 42, 49, 51, 58, 78, 93, 96, 99],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="run_q3_real")
    parser.add_argument("--datasets", default="", help="comma-separated dataset IDs; overrides --dataset-preset when set")
    parser.add_argument("--dataset-preset", choices=sorted(Q3_DATASET_PRESETS), default="structural_core",
                        help="named dataset bundle used when --datasets is omitted")
    parser.add_argument("--rows-per-dataset", type=int, default=2)
    parser.add_argument("--row-offset", type=int, default=0,
                        help="skip the first N normalized rows before selecting rows; ignored when --row-indices is set")
    parser.add_argument("--row-indices", default="",
                        help="comma-separated original normalized row indices to run for every selected dataset")
    parser.add_argument("--row-preset", choices=sorted(Q3_ROW_PRESETS), default="",
                        help="dataset-specific hard-row preset; ignored when --row-indices is set")
    parser.add_argument("--sample", choices=["first", "random"], default="first",
                        help="row selection strategy after applying --row-offset; default keeps previous first-N behavior")
    parser.add_argument("--sample-seed", type=int, default=2026,
                        help="seed for --sample random; mixed with dataset id for reproducible per-dataset sampling")
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--provider", choices=["deterministic", "openai_compatible", "aliyun_bailian"], default="aliyun_bailian")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="number of parallel API workers")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    dataset_filter = {x.strip().lower() for x in (args.datasets or ",".join(Q3_DATASET_PRESETS[args.dataset_preset])).split(",") if x.strip()}
    datasets = [adapter for key, adapter in DATASET_ADAPTERS.items()
                if not dataset_filter or key.lower() in dataset_filter or adapter.dataset_id.lower() in dataset_filter]
    if not datasets:
        raise RuntimeError("dataset filter selected no supported real-run datasets")

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
            for baseline in Q3_REAL_BASELINES:
                for seed in seeds:
                    key = (dataset.dataset_id, index, baseline, seed)
                    if key in completed:
                        continue
                    jobs.append((dataset, raw, case, index, baseline, seed))
    if args.workers <= 1:
        for job in jobs:
            record = _run_job(args, job)
            results.append(record)
            _print_record(record)
            _write_payload(checkpoint, args, datasets, seeds, results, selected_by_dataset)
    else:
        print(f"running {len(jobs)} remaining Q3 jobs with workers={args.workers}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_job, args, job) for job in jobs]
            for future in as_completed(futures):
                record = future.result()
                results.append(record)
                _print_record(record)
                _write_payload(checkpoint, args, datasets, seeds, results, selected_by_dataset)

    _write_payload(output, args, datasets, seeds, results, selected_by_dataset)


def _select_rows(root: Path, dataset, args) -> list[tuple[int, dict[str, Any]]]:
    """Select normalized dataset rows without touching the shared Q2 loader.

    The old runner always used the first N rows.  Q3 needs reproducible
    sampling because early rows in datasets such as GSM8K can be too easy and
    make orchestration differences invisible.
    """
    all_rows = _load_all_normalized_rows(root, dataset)
    if args.row_preset:
        preset = Q3_ROW_PRESETS[args.row_preset].get(_normalize_dataset_key(dataset.dataset_id))
        if preset:
            selected = []
            for index in preset:
                if index < 0 or index >= len(all_rows):
                    raise ValueError(f"{dataset.dataset_id} row index {index} out of range 0..{len(all_rows) - 1}")
                row = all_rows[index]
                if _is_structurally_qualified_row(dataset, row):
                    selected.append((index, row))
            if selected:
                if args.rows_per_dataset > 0:
                    return selected[: min(args.rows_per_dataset, len(selected))]
                return selected
    explicit_indices = _parse_row_indices(args.row_indices)
    if explicit_indices:
        selected = []
        for index in explicit_indices:
            if index < 0 or index >= len(all_rows):
                raise ValueError(f"{dataset.dataset_id} row index {index} out of range 0..{len(all_rows) - 1}")
            selected.append((index, all_rows[index]))
        return selected

    if args.row_offset < 0:
        raise ValueError("--row-offset must be >= 0")
    candidates = list(enumerate(all_rows))[args.row_offset:]
    candidates = [(index, row) for index, row in candidates if _is_structurally_qualified_row(dataset, row)]
    if not candidates:
        raise ValueError(f"{dataset.dataset_id} has no structurally qualified rows after --row-offset {args.row_offset}")

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
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{dataset.dataset_id} normalized file has no rows: {path}")
    return rows


def _is_structurally_qualified_row(dataset, row: dict[str, Any]) -> bool:
    """Prefer rows that can actually surface orchestration structure."""
    dataset_id = str(dataset.dataset_id).casefold()
    raw = row.get("raw") or {}
    if dataset_id == "hotpotqa":
        answer = str(row.get("answer") or "").strip()
        # Exclude low-entropy numeric year answers from the main table; they
        # tend to saturate all sequential baselines and are better used only
        # in sanity checks.
        if re.fullmatch(r"\d{4}", answer):
            return False
        return (
            str(raw.get("type", "")).casefold() in {"bridge", "comparison"}
            and str(raw.get("level", "")).casefold() == "hard"
            and answer.casefold() not in {"yes", "no"}
            and len(str(row.get("question") or "").split()) >= 12
        )
    if dataset_id == "musique":
        hops = raw.get("question_decomposition") or []
        answer = str(row.get("answer") or "").strip()
        return raw.get("answerable") is True and len(hops) >= 3 and len(answer.split()) >= 1
    if dataset_id == "finqa":
        program = str(((raw.get("metadata") or {}).get("program")) or "").strip()
        answer = str(row.get("answer") or "").strip()
        return bool(answer) and len(program.split()) >= 6 and len(str(row.get("question") or "").split()) >= 8
    if dataset_id == "drop":
        passage = str(raw.get("passage") or "")
        spans = ((raw.get("answers_spans") or {}).get("spans")) or []
        return len(passage) >= 1800 and len(spans) >= 3 and len(str(row.get("question") or "").split()) >= 8
    if dataset_id == "math500":
        return int(raw.get("level") or 0) >= 5 and len(str(raw.get("problem") or "").split()) >= 20
    if dataset_id == "arc":
        choices = raw.get("choices") or {}
        options = choices.get("text") if isinstance(choices, dict) else []
        return len(str(row.get("question") or "").split()) >= 15 and len(options or []) >= 4
    if dataset_id == "mmlu_pro":
        options = raw.get("options") or []
        return len(str(row.get("question") or "").split()) >= 20 and len(options) >= 9
    if dataset_id == "medqa":
        data = raw.get("data") or {}
        choices = (
            raw.get("choices")
            or raw.get("options")
            or data.get("Options")
            or {}
        )
        if isinstance(choices, dict):
            choice_count = len(choices)
        elif isinstance(choices, list):
            choice_count = len(choices)
        else:
            choice_count = 0
        return len(str(row.get("question") or "").split()) >= 60 and choice_count >= 4
    if dataset_id == "financebench":
        evidence = raw.get("evidence") or []
        qtype = str(raw.get("question_type") or "").casefold()
        reasoning = str(raw.get("question_reasoning") or "").casefold()
        return (
            isinstance(evidence, list) and len(evidence) >= 2
            and qtype in {"metrics-generated", "novel-generated"}
            and ("reason" in reasoning or "numerical" in reasoning or "logical" in reasoning)
            and len(str(row.get("question") or "").split()) >= 12
        )
    if dataset_id == "pubmedqa":
        context = str(row.get("context") or "")
        answer = str(row.get("answer") or "").strip().casefold()
        return (
            len(context) >= 1000
            and answer in {"yes", "no", "maybe"}
            and len(str(row.get("question") or "").split()) >= 10
        )
    if dataset_id == "bbh-full":
        task = str(raw.get("task") or "").casefold()
        preferred = {
            "logical_deduction",
            "tracking_shuffled_objects",
            "object_tracking",
            "date_understanding",
            "temporal_sequence",
            "reasoning_about_colored_objects",
            "causal_judgement",
            "disambiguation_qa",
            "geometric_shapes",
            "hyperbaton",
            "ruin_names",
            "salient_translation_error_detection",
            "snarks",
            "sports_understanding",
        }
        inp = str(raw.get("input") or row.get("question") or "")
        return task in preferred and len(inp.split()) >= 20
    if dataset_id == "scibench":
        problem = str(raw.get("problem_text") or row.get("question") or "")
        answer = str(row.get("answer") or "").strip()
        source = str(raw.get("source") or "").casefold()
        return bool(answer) and len(problem.split()) >= 18 and source in {"atkins", "openstax", "chemistry", "physics", "biology"}
    if dataset_id == "bbh":
        inp = str(raw.get("input") or row.get("question") or "")
        return len(inp.split()) >= 8 and (inp.count("(") + inp.count("and") + inp.count("or")) >= 3
    # Supplementary datasets are kept broad; they are no longer the main claim.
    return True


def _parse_row_indices(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _dataset_sample_seed(sample_seed: int, dataset_id: str) -> int:
    digest = hashlib.sha256(f"{sample_seed}:{dataset_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _normalize_dataset_key(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def _build_adapter(args):
    if args.provider == "deterministic":
        return DeterministicAdapter(LLMConfig(provider="deterministic", model="deterministic-q3-real", max_output_tokens=args.max_output_tokens))
    key = os.environ.get(args.api_key_env)
    if not key:
        raise RuntimeError(f"missing API key environment variable {args.api_key_env}")
    return OpenAICompatibleAdapter(LLMConfig(provider=args.provider, model=args.model, base_url=args.base_url, api_key=key, temperature=0.0, max_output_tokens=args.max_output_tokens))


def _run_job(args, job):
    dataset, raw, case, index, baseline, seed = job
    adapter = Q3RobustAdapter(_build_adapter(args))
    runtime = MinimalMARRuntime()
    task_executor = ApplicationTaskExecutor(adapter, ROOT)
    try:
        record = run_real_case(case, baseline, seed, adapter, ROOT, task_executor, runtime, dataset=dataset, row=raw)
        record.update({
            "status": "completed",
            "dataset_id": dataset.dataset_id,
            "row_index": index,
            "source_id": raw.get("id"),
            "primary_metric": dataset.execution.metric_name,
            "primary_score": record.get("primary_score_mean"),
        })
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
            "seed": seed,
            "primary_metric": dataset.execution.metric_name,
            "primary_score": None,
            "e2e_success": None,
            "e2e_supported": False,
            "osv": None,
            "error_type": type(exc).__name__,
            "error": str(exc)[-2000:],
        }


def _print_record(record: dict[str, Any]) -> None:
    prefix = "completed" if record.get("status") == "completed" else "failed"
    message = f"{prefix} {record['dataset_id']} row={record['row_index']} baseline={record['baseline']} seed={record['seed']}"
    if record.get("status") == "failed":
        message += f": {record.get('error_type')}: {record.get('error')}"
    print(message, flush=True)


class Q3RobustAdapter:
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
    for field in ("artifact", "answer", "text", "output"):
        if field in required_fields:
            return field
    return sorted(required_fields)[0]


def _load_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload.get("runs", []))


def _write_payload(path: Path, args, datasets, seeds, results: list[dict[str, Any]],
                   selected_by_dataset: dict[str, list[int]]) -> None:
    selected_count = sum(len(indices) for indices in selected_by_dataset.values())
    expected = selected_count * len(seeds) * len(Q3_REAL_BASELINES)
    completed = sum(r.get("status", "completed") == "completed" for r in results)
    failed = sum(r.get("status") == "failed" for r in results)
    payload = {
        "protocol": "Q3 real benchmark run",
        "datasets": [d.dataset_id for d in datasets],
        "dataset_preset": args.dataset_preset if not args.datasets.strip() else None,
        "rows_per_dataset": args.rows_per_dataset,
        "row_offset": args.row_offset,
        "row_indices": _parse_row_indices(args.row_indices),
        "row_preset": args.row_preset if args.row_preset else None,
        "sample": args.sample,
        "sample_seed": args.sample_seed,
        "selected_rows": selected_by_dataset,
        "seeds": seeds,
        "baselines": Q3_REAL_BASELINES,
        "expected_runs": expected,
        "completed_runs": completed,
        "failed_runs": failed,
        "progress": round((completed + failed) / expected, 6) if expected else 0.0,
        "runs": results,
        "summary": _summarize(results),
    }
    write_json(path, payload)


def _summarize(results):
    grouped = defaultdict(list)
    for row in results:
        if row.get("status", "completed") != "completed":
            continue
        grouped[(row["dataset_id"], row["baseline"])].append(row)
    summary = []
    for (dataset, baseline), group in sorted(grouped.items()):
        primary_values = [float(x["primary_score"]) for x in group if x.get("primary_score") is not None]
        e2e_values = [float(x["e2e_success"]) for x in group if x.get("e2e_success") is not None]
        architecture_values = _nested_metric_values(group, "construction", "architecture_validity")
        relation_values = _nested_metric_values(group, "construction", "orchestration_relation_recall")
        blueprint_values = _nested_metric_values(group, "construction", "blueprint_coverage")
        realization_values = _nested_metric_values(group, "construction", "realization_fidelity")
        constraint_values = _nested_metric_values(group, "construction", "constraint_satisfaction")
        step_values = []
        for row in group:
            executions = row.get("execution") or []
            if executions:
                step_values.append(float(len(executions[0].get("task_execution", {}).get("node_executions", []))))
        summary.append({
            "dataset": dataset,
            "baseline": baseline,
            "n": len(group),
            "primary_score_mean": round(sum(primary_values) / len(primary_values), 6) if primary_values else None,
            "primary_score_std": _stdev_or_none(primary_values),
            "primary_score_var": _variance_or_none(primary_values),
            "primary_score_ci95_low": _ci95(primary_values, lower=True),
            "primary_score_ci95_high": _ci95(primary_values, lower=False),
            "e2e_success_mean": round(sum(e2e_values) / len(e2e_values), 6) if e2e_values else None,
            "e2e_success_std": _stdev_or_none(e2e_values),
            "e2e_success_var": _variance_or_none(e2e_values),
            "e2e_success_ci95_low": _ci95(e2e_values, lower=True),
            "e2e_success_ci95_high": _ci95(e2e_values, lower=False),
            "e2e_supported_n": len(e2e_values),
            "osv_mean": round(sum(float(x["osv"]) for x in group) / len(group), 6),
            "graph_structural_preservation_mean": _mean_or_none([float(x["graph_structural_preservation"]) for x in group if x.get("graph_structural_preservation") is not None]),
            "graph_structural_preservation_std": _stdev_or_none([float(x["graph_structural_preservation"]) for x in group if x.get("graph_structural_preservation") is not None]),
            "architecture_validity_mean": _mean_or_none(architecture_values),
            "orchestration_relation_recall_mean": _mean_or_none(relation_values),
            "blueprint_coverage_mean": _mean_or_none(blueprint_values),
            "realization_fidelity_mean": _mean_or_none(realization_values),
            "constraint_satisfaction_mean": _mean_or_none(constraint_values),
            "constraint_violation_rate_mean": (round(1 - _mean_or_none(constraint_values), 6)
                                               if constraint_values else None),
            "avg_exec_steps": _mean_or_none(step_values),
        })
    return summary


def _nested_metric_values(rows, section: str, metric: str) -> list[float]:
    values = []
    for row in rows:
        section_value = row.get(section) or {}
        value = section_value.get(metric)
        if value is not None:
            values.append(float(value))
    return values


def _mean_or_none(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _stdev_or_none(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return round(statistics.stdev(values), 6)


def _variance_or_none(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return round(statistics.variance(values), 6)


def _ci95(values: list[float], *, lower: bool) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return round(mean, 6)
    margin = 1.96 * statistics.stdev(values) / (len(values) ** 0.5)
    value = mean - margin if lower else mean + margin
    return round(value, 6)


if __name__ == "__main__":
    main()
