"""Run one deterministic or explicitly model-backed Q10 financial case per dataset."""
from __future__ import annotations

import argparse
import os
import sys
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="financebench,finqa")
    parser.add_argument("--provider", choices=("deterministic", "openai_compatible"), default="deterministic")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--base-url", default=os.environ.get("Q10_LLM_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--model", default=os.environ.get("Q10_LLM_MODEL", "deepseek-chat"))
    parser.add_argument("--api-key-env", default="OPENMAS_LLM_API_KEY")
    parser.add_argument("--analysis-only", action="store_true", help="Run profile + DeepSeek row analysis + compiler, without MAS execution")
    args = parser.parse_args()
    output_root = args.output_root or args.data_root / "outputs" / "q10_financial" / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    code_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(code_root))
    from openmas_bench.dataset_adapters import all_adapters
    from openmas_bench.q10 import build_q10_case
    from openmas_bench.engine import GraphHarnessEngine
    from openmas_bench.io import write_json
    from openmas_bench.llm import DeterministicAdapter, LLMConfig, OpenAICompatibleAdapter

    adapters = {item.dataset_id.casefold(): item for item in all_adapters()}
    if args.provider == "deterministic":
        llm = DeterministicAdapter(LLMConfig(provider="deterministic", model="deterministic-engine-protocol"))
    else:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key environment variable {args.api_key_env}")
        llm = OpenAICompatibleAdapter(LLMConfig(provider="openai_compatible", model=args.model, base_url=args.base_url, api_key=api_key))
    for dataset in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        adapter = adapters.get(dataset.casefold())
        if adapter is None:
            raise ValueError(f"unknown dataset {dataset}")
        dataset_path = args.data_root / "q10_datasets" / "normalized" / f"{dataset.casefold().replace('-', '')}.jsonl"
        rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        row = rows[args.row_index]
        case, analysis = build_q10_case(adapter.dataset_id, row, llm=llm, index=args.row_index, seed=args.seed)
        output = output_root / f"{dataset.lower()}_row{args.row_index}_{args.provider}.json"
        print("running", dataset)
        if args.analysis_only:
            record = {
                "status": "analysis_completed",
                "dataset": adapter.dataset_id,
                "source_id": row.get("id"),
                "q10_analysis": analysis,
                "q10_harness": {"nodes": [node.__dict__ for node in case.harness.nodes], "edges": [edge.__dict__ for edge in case.harness.edges], "metadata": case.harness.metadata},
                "q10_blueprint": {"nodes": [node.__dict__ for node in case.reference_blueprint.nodes], "edges": [edge.__dict__ for edge in case.reference_blueprint.edges], "metadata": case.reference_blueprint.metadata},
                "q10_three_layer_pipeline": ["dataset_profile", "row_semantic_analysis", "constrained_harness_compilation"],
            }
            write_json(output, record)
            continue
        result = GraphHarnessEngine(llm, args.data_root).run_case(adapter, row, case, seed=args.seed, intervention="full_graph_harness")
        record = result.to_experiment_record()
        record["q10_analysis"] = analysis
        record["q10_harness"] = {"nodes": [node.__dict__ for node in case.harness.nodes], "edges": [edge.__dict__ for edge in case.harness.edges], "metadata": case.harness.metadata}
        record["q10_three_layer_pipeline"] = ["dataset_profile", "row_semantic_analysis", "constrained_harness_compilation"]
        write_json(output, record)


if __name__ == "__main__":
    main()
