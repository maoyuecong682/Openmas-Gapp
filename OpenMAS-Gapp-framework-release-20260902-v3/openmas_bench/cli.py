from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .baselines import BUILDERS, get_builder
from .construction import Q1_METHODS, get_construction_method
from .dataset_adapters import all_adapters
from .dataset_cases import build_dataset_case, load_normalized_rows
from .dynamic_engine import DynamicGraphEngine
from .dynamic_executor import DynamicGraphExecutor
from .dynamic_planner import DeepSeekGraphPlanner
from .engine import GraphHarnessEngine
from .evaluate import evaluate_construction, evaluate_spec
from .io import load_construction_case, load_construction_result, load_package, save_construction_result, save_spec, write_json
from .llm import DeterministicAdapter, LLMConfig, OpenAICompatibleAdapter
from .remote_datasets import DEFAULT_ROWS_ENDPOINT, load_remote_row


def main() -> None:
    parser = argparse.ArgumentParser(prog="openmas-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="construct a MASSpec using one controlled baseline")
    build.add_argument("--baseline", required=True, choices=sorted(BUILDERS))
    build.add_argument("--package", required=True)
    build.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate", help="evaluate a generated MASSpec")
    evaluate.add_argument("--package", required=True)
    evaluate.add_argument("--spec", required=True)
    evaluate.add_argument("--output", required=True)
    construct = sub.add_parser("construct", help="construct M_R, Blueprint and executable MAS for Q1-Q4")
    construct.add_argument("--method", required=True, choices=sorted(Q1_METHODS))
    construct.add_argument("--case", required=True)
    construct.add_argument("--output", required=True)
    evaluate_construction_cmd = sub.add_parser("evaluate-construction", help="evaluate a Q1-Q4 construction result")
    evaluate_construction_cmd.add_argument("--case", required=True)
    evaluate_construction_cmd.add_argument("--result", required=True)
    evaluate_construction_cmd.add_argument("--output", required=True)
    run = sub.add_parser("run", help="construct and execute one dataset task through GraphHarnessEngine")
    run.add_argument("--dataset", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--source", choices=["local", "remote"], default="local",
                     help="local reads normalized JSONL; remote fetches one source row without a local dataset cache")
    run.add_argument("--row-index", type=int, default=0)
    run.add_argument("--remote-endpoint", default=DEFAULT_ROWS_ENDPOINT,
                     help="Hugging Face datasets-server rows endpoint used with --source remote")
    run.add_argument("--remote-timeout", type=int, default=30,
                     help="per-request timeout in seconds for --source remote")
    run.add_argument("--seed", type=int, default=11)
    run.add_argument("--intervention", default="full_graph_harness")
    run.add_argument("--provider", choices=["deterministic", "openai_compatible"], default="deterministic")
    run.add_argument("--base-url", default="https://api.deepseek.com")
    run.add_argument("--model", default="deepseek-chat")
    run.add_argument("--api-key-env", default="OPENMAS_LLM_API_KEY")
    run.add_argument("--output", required=True)
    plan_demo = sub.add_parser("plan-graph-demo", help="plan two tasks with DeepSeek and print their dynamic graphs")
    plan_demo.add_argument("--task-a", required=True)
    plan_demo.add_argument("--task-b", required=True)
    plan_demo.add_argument("--seed", type=int, default=11)
    plan_demo.add_argument("--base-url", default="https://api.deepseek.com")
    plan_demo.add_argument("--model", default="deepseek-chat")
    plan_demo.add_argument("--api-key-env", default="OPENMAS_LLM_API_KEY")
    plan_demo.add_argument("--execute", action="store_true")
    plan_demo.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[2] / "outputs" / "dynamic_harness_graphs" / "runs"),
        help="directory for per-task planner, harness graph, and optional execution JSON files",
    )
    args = parser.parse_args()
    if args.command == "build":
        package = load_package(args.package)
        save_spec(args.output, get_builder(args.baseline).build(package))
    elif args.command == "evaluate":
        from .io import read_json
        from .schema import MASEdge, MASNode, MASSpec
        raw = read_json(args.spec)
        spec = MASSpec(raw["package_id"], raw["baseline"], [MASNode(**x) for x in raw["nodes"]],
                       [MASEdge(**x) for x in raw["edges"]], raw["selected_capabilities"], raw.get("metadata", {}))
        write_json(args.output, evaluate_spec(load_package(args.package), spec))
    elif args.command == "construct":
        case = load_construction_case(args.case)
        save_construction_result(args.output, get_construction_method(args.method).construct(case.request()))
    elif args.command == "evaluate-construction":
        write_json(args.output, evaluate_construction(load_construction_case(args.case), load_construction_result(args.result)))
    elif args.command == "run":
        datasets = {item.dataset_id.casefold(): item for item in all_adapters()}
        dataset = datasets.get(args.dataset.casefold())
        if dataset is None:
            raise ValueError(f"unknown dataset {args.dataset!r}; choose from {sorted(item.dataset_id for item in datasets.values())}")
        config = LLMConfig(
            provider=args.provider,
            model="deterministic-engine-protocol" if args.provider == "deterministic" else args.model,
            base_url=None if args.provider == "deterministic" else args.base_url,
            api_key=None if args.provider == "deterministic" else os.environ.get(args.api_key_env),
        )
        if args.provider != "deterministic" and not config.api_key:
            raise RuntimeError(f"missing API key environment variable {args.api_key_env}")
        adapter = (DeterministicAdapter(config) if args.provider == "deterministic"
                   else OpenAICompatibleAdapter(config))
        data_root = Path(args.data_root).resolve()
        if args.source == "remote":
            row = load_remote_row(
                dataset,
                args.row_index,
                endpoint=args.remote_endpoint,
                timeout_seconds=args.remote_timeout,
            )
        else:
            rows = load_normalized_rows(data_root, dataset, args.row_index + 1)
            row = rows[args.row_index]
        case = build_dataset_case(dataset, row, args.row_index)
        result = GraphHarnessEngine(adapter, data_root).run_case(
            dataset, row, case, seed=args.seed, intervention=args.intervention)
        write_json(args.output, result.to_experiment_record())
    elif args.command == "plan-graph-demo":
        planner = DeepSeekGraphPlanner.from_env(
            base_url=args.base_url,
            model=args.model,
            api_key_envs=(args.api_key_env, "DEEPSEEK_API_KEY"),
        )
        engine = DynamicGraphEngine(planner=planner)
        if args.execute:
            engine.executor = DynamicGraphExecutor(planner.client.adapter)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, task in enumerate((args.task_a, args.task_b)):
            bundle = engine.run(task, seed=args.seed + index, execute=args.execute)
            run_record = bundle.to_dict()
            output_path = output_dir / f"task_{index + 1}_seed_{args.seed + index}_dynamic_harness.json"
            write_json(output_path, run_record)
            print(f"=== Task {index + 1} ===")
            print(task)
            print("Saved output:")
            print(str(output_path.resolve()))
            print("Planner output:")
            print(json.dumps(bundle.plan.to_dict(), ensure_ascii=False, indent=2))
            print("Final agent list:")
            print(json.dumps(bundle.plan.to_dict()["agents"], ensure_ascii=False, indent=2))
            print("Final edges:")
            print(json.dumps(bundle.plan.to_dict()["edges"], ensure_ascii=False, indent=2))
            print("Harness Graph:")
            print(json.dumps(bundle.bundle.to_dict()["harness"], ensure_ascii=False, indent=2))
            if bundle.execution is not None:
                print("Executor result:")
                print(json.dumps(bundle.execution.to_dict(), ensure_ascii=False, indent=2))
            if index == 0:
                print()


if __name__ == "__main__":
    main()
