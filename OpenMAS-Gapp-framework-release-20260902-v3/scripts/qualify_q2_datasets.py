"""Preflight qualification for Q2 cross-dataset ablation.

This check is deliberately API-free.  It validates the data/evaluator
contract before spending LLM calls, and never uses model scores to decide
whether a dataset is included.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from openmas_bench.ablation import Q2_VARIANTS, get_ablation_method
from openmas_bench.dataset_adapters import all_adapters
from openmas_bench.dataset_cases import build_dataset_case, load_normalized_rows
from openmas_bench.llm import DeterministicAdapter, LLMConfig
from openmas_bench.runtime import MinimalMARRuntime


MAIN_TABLE_CANDIDATES = {
    "GSM8K", "MATH-500", "MMLU", "HotpotQA", "MedQA", "PubMedQA", "FinQA"
}
EXTENSION_CANDIDATES = {"MuSiQue", "StrategyQA", "SciQ"}


def _dataset_preconditions(dataset_id: str, rows: list[dict]) -> tuple[list[str], list[str]]:
    """Return (warnings, blockers) for evaluator-specific requirements."""
    warnings: list[str] = []
    blockers: list[str] = []
    if dataset_id in {"HumanEval", "MBPP"}:
        if not any((row.get("raw") or {}).get("test") or
                   (row.get("raw") or {}).get("tests") or
                   (row.get("raw") or {}).get("test_list") for row in rows):
            blockers.append("no executable tests in normalized rows")
        else:
            warnings.append("code track: report unit-test pass separately from QA accuracy")
    elif dataset_id == "SWE-bench":
        if not any((REPO / "q2_datasets" / "swebench_repos").glob("*/base-*.tar.gz")):
            blockers.append("no frozen repository base snapshots")
        warnings.append("patch track: include only instances with validated git-apply and test environments")
    elif dataset_id == "DROP":
        if not all(row.get("answer") or row.get("answers_spans") or (row.get("raw") or {}).get("answers_spans") for row in rows):
            blockers.append("missing span/answer annotations")
        warnings.append("run a separate span/numeric evaluator smoke before formal inclusion")
    elif dataset_id in {"FinQA", "PubMedQA"}:
        warnings.append("requires source-level split and provenance record before publication")
    return warnings, blockers


def qualify(dataset_ids: set[str], rows_per_dataset: int) -> dict:
    adapter = DeterministicAdapter(LLMConfig(provider="deterministic", model="q2-qualification"))
    runtime = MinimalMARRuntime()
    reports = []
    for ds in all_adapters():
        if dataset_ids and ds.dataset_id.lower() not in dataset_ids:
            continue
        report = {"dataset": ds.dataset_id, "source_file": ds.source_file,
                  "main_table_candidate": ds.dataset_id in MAIN_TABLE_CANDIDATES,
                  "status": "blocked", "rows_checked": 0, "checks": {},
                  "warnings": [], "blockers": []}
        source = REPO / ds.source_file
        report["checks"]["source_exists"] = source.exists()
        if not source.exists():
            report["blockers"].append(f"missing source file: {source}")
            reports.append(report)
            continue
        try:
            rows = load_normalized_rows(REPO, ds, rows_per_dataset)
            report["rows_checked"] = len(rows)
            report["checks"]["rows_nonempty"] = bool(rows)
            # Code and patch tracks intentionally have no scalar answer. Their
            # gold contract is executable tests or a reference patch.
            if ds.dataset_id in {"HumanEval", "MBPP"}:
                gold_ok = all((row.get("raw") or {}).get("test") or
                               (row.get("raw") or {}).get("test_list") for row in rows)
            elif ds.dataset_id == "SWE-bench":
                gold_ok = all((row.get("raw") or {}).get("patch") for row in rows)
            else:
                gold_ok = all(row.get("answer") is not None for row in rows)
            report["checks"]["gold_nonempty"] = gold_ok
            if not rows:
                report["blockers"].append("no normalized rows")
            if not report["checks"]["gold_nonempty"]:
                report["blockers"].append("one or more rows have no normalized gold answer")
            warnings, blockers = _dataset_preconditions(ds.dataset_id, rows)
            report["warnings"].extend(warnings)
            report["blockers"].extend(blockers)
            if ds.dataset_id == "SWE-bench":
                report["blockers"].append("per-instance repository/test environment audit is required before primary-table inclusion")

            stage_counts = {}
            for row_index, raw in enumerate(rows):
                case = build_dataset_case(ds, raw, row_index)
                counts = {}
                for variant in Q2_VARIANTS:
                    result = get_ablation_method(variant, adapter=adapter, seed=0).construct(case.request())
                    result.validate(case.request())
                    counts[variant] = len(result.application.nodes)
                stage_counts[str(row_index)] = counts
                # Runtime smoke checks the constructed application without a
                # domain answer, so it cannot inflate the primary score.
                runtime.execute(case, get_ablation_method("full_graph_harness", adapter=adapter).construct(case.request()), case.execution_tasks[0], 0)
            report["stage_counts"] = stage_counts
            report["checks"]["variant_stage_parity"] = all(len(set(c.values())) == 1 for c in stage_counts.values())
            if not report["checks"]["variant_stage_parity"]:
                report["blockers"].append("variant executable stage counts differ")
        except Exception as exc:
            report["blockers"].append(f"qualification exception: {type(exc).__name__}: {exc}")
        if not report["blockers"]:
            if ds.dataset_id in {"FinQA", "PubMedQA"} | EXTENSION_CANDIDATES:
                report["status"] = "conditional_candidate"
            else:
                report["status"] = "candidate" if report["main_table_candidate"] else "separate_track"
        reports.append(report)
    return {"protocol": "Q2 dataset qualification v1", "rows_per_dataset": rows_per_dataset,
            "policy": {"main_table": sorted(MAIN_TABLE_CANDIDATES),
                       "exclude_without_repair": ["HumanEval", "SWE-bench"],
                       "DROP": "qualification smoke required", "code_metrics": "separate track"},
            "datasets": reports}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-dataset", type=int, default=3)
    parser.add_argument("--datasets", default="")
    parser.add_argument("--output", default=str(REPO / "q2_dataset_qualification.json"))
    args = parser.parse_args()
    selected = {x.strip().lower() for x in args.datasets.split(",") if x.strip()}
    payload = qualify(selected, args.rows_per_dataset)
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({x["dataset"]: {"status": x["status"], "blockers": x["blockers"]} for x in payload["datasets"]}, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
