from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openmas_bench.construction import Q1_METHODS
from openmas_bench.engine import GraphHarnessEngine
from openmas_bench.evaluate import evaluate_construction, evaluate_execution
from openmas_bench.llm import DeterministicAdapter, LLMConfig, OpenAICompatibleAdapter
from openmas_bench.runtime import MinimalMARRuntime
from openmas_bench.task_adapter import DomainTaskAdapter
from scripts.prepare_construction_cases import build_suite


def adapter_from_args(args):
    if args.provider == "deterministic":
        return DeterministicAdapter(LLMConfig(provider="deterministic", model="deterministic-q1-proxy", max_output_tokens=args.max_output_tokens))
    key = os.environ.get(args.api_key_env)
    if not key:
        raise RuntimeError(f"missing API key environment variable {args.api_key_env}")
    return OpenAICompatibleAdapter(LLMConfig(provider=args.provider, model=args.model, base_url=args.base_url, api_key=key, temperature=0.0, max_output_tokens=args.max_output_tokens))


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[x] for x in keys)].append(row)
    result = []
    ignored = set(keys) | {"case_id", "execution_task_id", "adapter", "model", "family", "split", "method"}
    for group, items in sorted(groups.items()):
        record = dict(zip(keys, group))
        numeric = {key for item in items for key, value in item.items() if isinstance(value, (int, float)) and not isinstance(value, bool) and key not in ignored}
        for metric in sorted(numeric):
            values = [float(x[metric]) for x in items if x.get(metric) is not None]
            if values:
                record[f"{metric}_mean"] = round(statistics.mean(values), 6)
                record[f"{metric}_std"] = round(statistics.stdev(values), 6) if len(values) > 1 else 0.0
        record["n"] = len(items)
        result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["deterministic", "openai_compatible"], default="deterministic")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-key-env", default="OPENMAS_LLM_API_KEY")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--limit", type=int, default=0, help="optional case limit for connectivity checks")
    parser.add_argument("--case-id", default="")
    parser.add_argument("--skip-task-answers", action="store_true")
    parser.add_argument("--input-price-per-million", type=float, default=0.28)
    parser.add_argument("--output-price-per-million", type=float, default=0.42)
    parser.add_argument("--output", default=str(ROOT / "q1_formal_results" / "latest.json"))
    args = parser.parse_args()
    adapter = adapter_from_args(args)
    seeds = [int(x) for x in args.seeds.split(",")]
    cases = [x for x in build_suite() if not args.case_id or x.case_id == args.case_id]
    cases = cases[:args.limit or None]
    runtime = MinimalMARRuntime()
    engine = GraphHarnessEngine(adapter, ROOT.parents[1], runtime=runtime)
    task_adapter = DomainTaskAdapter(adapter if args.provider != "deterministic" else None)
    construction_rows, execution_rows, run_rows = [], [], []
    for case in cases:
        for method_name in Q1_METHODS:
            for seed in seeds:
                result = engine.construct(case, seed=seed, method=method_name)
                construction = evaluate_construction(case, result)
                executions = []
                for task in case.execution_tasks:
                    execution = runtime.execute(case, result, task, seed, None if args.skip_task_answers else task_adapter)
                    score = evaluate_execution(case, execution)
                    executions.append(score)
                    execution_rows.append(score)
                fallback = result.telemetry.fallback
                run_rows.append({
                    "case_id": case.case_id, "split": case.split, "family": case.family,
                    "method": method_name, "seed": seed, "adapter": result.telemetry.adapter,
                    "model": result.telemetry.model, "fallback": fallback,
                    "construction": construction, "executions": executions,
                    "trace_event_count": sum(x["event_count"] for x in executions),
                })
                construction.update({"split": case.split, "fallback": float(fallback)})
                construction_rows.append(construction)
                print(f"{len(run_rows):03d}/{len(cases) * len(Q1_METHODS) * len(seeds)} {case.case_id} {method_name} seed={seed} fallback={fallback}", flush=True)
    formal = args.provider != "deterministic" and not any(x["fallback"] for x in run_rows)
    total_input_tokens = sum(x["construction"]["input_tokens"] for x in run_rows)
    total_output_tokens = sum(x["construction"]["output_tokens"] for x in run_rows)
    answer_input_tokens = sum(x.get("answer_input_tokens", 0) for x in execution_rows)
    answer_output_tokens = sum(x.get("answer_output_tokens", 0) for x in execution_rows)
    # Domain answers are cached across methods, but appear in five execution
    # rows. Divide by the method count to recover actual billed adapter usage.
    answer_input_tokens //= len(Q1_METHODS)
    answer_output_tokens //= len(Q1_METHODS)
    estimated_cost = ((total_input_tokens + answer_input_tokens) / 1_000_000 * args.input_price_per_million +
                      (total_output_tokens + answer_output_tokens) / 1_000_000 * args.output_price_per_million)
    payload = {
        "protocol": "Q1 Harness Paradigm Comparison v2.0",
        "formal_result": formal,
        "provider": args.provider, "model": adapter.config.model,
        "fairness": {"same_backbone": True, "same_ecosystem": True, "same_budget": True, "same_execution_environment": True, "temperature": 0.0, "max_output_tokens": args.max_output_tokens},
        "case_count": len(cases), "method_count": len(Q1_METHODS), "seeds": seeds,
        "construction_run_count": len(run_rows), "execution_run_count": len(execution_rows),
        "usage": {"construction_input_tokens": total_input_tokens, "construction_output_tokens": total_output_tokens, "answer_input_tokens": answer_input_tokens, "answer_output_tokens": answer_output_tokens, "configured_input_usd_per_million": args.input_price_per_million, "configured_output_usd_per_million": args.output_price_per_million, "estimated_total_cost_usd": round(estimated_cost, 6)},
        "aggregate_by_method": aggregate(construction_rows, ["method"]),
        "aggregate_by_method_split": aggregate(construction_rows, ["method", "split"]),
        "aggregate_by_method_family": aggregate(construction_rows, ["method", "family"]),
        "execution_by_method": aggregate(execution_rows, ["method"]),
        "runs": run_rows,
        "limitations": [] if formal else ["Deterministic adapter or fallback was used; results are protocol validation rather than formal LLM evidence.", "Minimal MAR evaluates execution structure and Trace contracts; answer accuracy is not yet evaluated."],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"completed constructions={len(run_rows)} executions={len(execution_rows)} formal_result={formal} output={output}")


if __name__ == "__main__":
    main()
