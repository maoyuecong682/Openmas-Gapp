"""Run the full DeepSeek-backed Q10 financial evaluation over all normalized rows."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [row for row in rows if row.get("status") == "completed"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in completed:
        groups.setdefault(row["dataset"], []).append(row)
    table = []
    for dataset, group in sorted(groups.items()):
        scores = [float(row["primary_score"]) for row in group if row.get("primary_score") is not None]
        table.append({
            "dataset": dataset,
            "metric": group[0]["primary_metric"],
            "n": len(scores),
            "mean_primary_score": round(statistics.mean(scores), 6) if scores else None,
            "std_primary_score": round(statistics.stdev(scores), 6) if len(scores) > 1 else 0.0 if scores else None,
            "completed": len(group),
            "failed": sum(row.get("status") == "failed" for row in rows if row.get("dataset") == dataset),
        })
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="financebench,finqa")
    parser.add_argument(
        "--provider",
        choices=("deterministic", "openai_compatible"),
        default="openai_compatible",
        help="deterministic = fixed-seed DeepSeek-backed run; openai_compatible = explicit DeepSeek-compatible run",
    )
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--base-url", default=os.environ.get("Q10_LLM_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--model", default=os.environ.get("Q10_LLM_MODEL", "deepseek-chat"))
    parser.add_argument("--api-key-env", default="OPENMAS_LLM_API_KEY")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    output_root = args.output_root or args.data_root / "outputs" / "q10_financial" / "full_evaluations"
    run_root = output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    code_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(code_root))
    from openmas_bench.dataset_adapters import all_adapters
    from openmas_bench.q10 import build_q10_case
    from openmas_bench.engine import GraphHarnessEngine
    from openmas_bench.io import write_json
    from openmas_bench.llm import LLMConfig, OpenAICompatibleAdapter

    adapters = {item.dataset_id.casefold(): item for item in all_adapters()}
    api_key = os.environ.get(args.api_key_env) or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(f"missing API key environment variable {args.api_key_env} or DEEPSEEK_API_KEY")
    llm_provider = "deepseek" if args.provider == "deterministic" else "openai_compatible"
    llm = OpenAICompatibleAdapter(
        LLMConfig(
            provider=llm_provider,
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            timeout_seconds=args.timeout_seconds,
        )
    )

    engine = GraphHarnessEngine(llm, args.data_root)
    dataset_filter = [item.strip() for item in args.datasets.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    for dataset in dataset_filter:
        adapter = adapters.get(dataset.casefold())
        if adapter is None:
            raise ValueError(f"unknown dataset {dataset}")
        dataset_path = args.data_root / "q10_datasets" / "normalized" / f"{dataset.casefold().replace('-', '')}.jsonl"
        source_rows = _load_rows(dataset_path)
        for index, row in enumerate(source_rows):
            case, analysis = build_q10_case(adapter.dataset_id, row, llm=llm, index=index, seed=args.seed)
            output = run_root / f"{dataset.lower()}_row{index}_{args.provider}.json"
            print("running", dataset, "row", index)
            result = engine.run_case(adapter, row, case, seed=args.seed, intervention="full_graph_harness")
            record = result.to_experiment_record()
            record["q10_analysis"] = analysis
            record["q10_harness"] = {
                "nodes": [node.__dict__ for node in case.harness.nodes],
                "edges": [edge.__dict__ for edge in case.harness.edges],
                "metadata": case.harness.metadata,
            }
            record["q10_three_layer_pipeline"] = ["dataset_profile", "row_semantic_analysis", "constrained_harness_compilation"]
            write_json(output, record)
            rows.append(record)

    payload = {
        "schema_version": "q10-full-eval-v1",
        "provider": args.provider,
        "model": args.model,
        "datasets": dataset_filter,
        "seed": args.seed,
        "run_count": len(rows),
        "completed_runs": sum(row.get("status") == "completed" for row in rows),
        "failed_runs": sum(row.get("status") == "failed" for row in rows),
        "summary": _summary(rows),
        "runs": rows,
    }
    write_json(output_root / f"q10_full_{args.provider}.json", payload)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"Wrote {output_root / f'q10_full_{args.provider}.json'}")


if __name__ == "__main__":
    main()
