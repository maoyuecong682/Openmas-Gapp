"""Run Q2 component-wise ablation under the frozen Q1 controls."""
from __future__ import annotations

import argparse, json, os, statistics, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openmas_bench.ablation import Q2_VARIANTS, get_ablation_method
from openmas_bench.evaluate import evaluate_construction, evaluate_execution
from openmas_bench.llm import DeterministicAdapter, LLMConfig, OpenAICompatibleAdapter
from openmas_bench.runtime import MinimalMARRuntime
from scripts.prepare_construction_cases import build_suite


def adapter_from_args(args):
    if args.provider == "deterministic":
        return DeterministicAdapter(LLMConfig(provider="deterministic", model="deterministic-q2-proxy", max_output_tokens=args.max_output_tokens))
    key = os.environ.get(args.api_key_env)
    if not key:
        raise RuntimeError(f"missing API key environment variable {args.api_key_env}")
    return OpenAICompatibleAdapter(LLMConfig(provider=args.provider, model=args.model, base_url=args.base_url, api_key=key, temperature=0.0, max_output_tokens=args.max_output_tokens))


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    ignored = set(keys) | {"case_id", "method", "variant", "family", "split", "execution_task_id", "adapter", "model"}
    for group, items in sorted(groups.items()):
        record = dict(zip(keys, group)); record["n"] = len(items)
        numeric = {k for item in items for k, v in item.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ignored}
        for metric in sorted(numeric):
            vals = [float(x[metric]) for x in items if x.get(metric) is not None]
            if vals:
                record[f"{metric}_mean"] = round(statistics.mean(vals), 6)
                record[f"{metric}_std"] = round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0
        output.append(record)
    return output


def paired_effects(rows, metric="construction_quality"):
    """Report paired effects without assuming Full must dominate ablations."""
    indexed = {(row["case_id"], row["seed"], row["variant"]): row for row in rows}
    effects = []
    for variant in sorted(set(row["variant"] for row in rows) - {"full_graph_harness"}):
        deltas = []
        for row in rows:
            if row["variant"] != variant or row.get(metric) is None:
                continue
            full = indexed.get((row["case_id"], row["seed"], "full_graph_harness"))
            if full is not None and full.get(metric) is not None:
                deltas.append(float(full[metric]) - float(row[metric]))
        effects.append({
            "variant": variant, "metric": metric, "n_pairs": len(deltas),
            "full_minus_ablation_mean": round(statistics.mean(deltas), 6) if deltas else None,
            "full_wins": sum(delta > 1e-12 for delta in deltas),
            "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
            "ablation_wins": sum(delta < -1e-12 for delta in deltas),
        })
    return effects


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["deterministic", "openai_compatible"], default="deterministic")
    p.add_argument("--base-url", default="https://api.deepseek.com")
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--api-key-env", default="OPENMAS_LLM_API_KEY")
    p.add_argument("--max-output-tokens", type=int, default=2048)
    p.add_argument("--seeds", default="11,22,33")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", default=str(ROOT.parents[1] / "q2_ablation_results.json"))
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    cases = build_suite()[:args.limit or None]
    adapter = adapter_from_args(args); runtime = MinimalMARRuntime()
    construction, execution, runs = [], [], []
    total = len(cases) * len(Q2_VARIANTS) * len(seeds); index = 0
    for case in cases:
        for variant in Q2_VARIANTS:
            for seed in seeds:
                result = get_ablation_method(variant, adapter=adapter, seed=seed).construct(case.request())
                c = evaluate_construction(case, result)
                c.update({"variant": variant, "split": case.split,
                          "blueprint_present": float(result.blueprint.metadata.get("blueprint_present", True)),
                          "blueprint_preserving": float(result.application.metadata.get("blueprint_preserving", True))})
                construction.append(c)
                ex_rows = []
                for task in case.execution_tasks:
                    score = evaluate_execution(case, runtime.execute(case, result, task, seed))
                    score["variant"] = variant; execution.append(score); ex_rows.append(score)
                runs.append({"case_id": case.case_id, "family": case.family, "split": case.split,
                             "variant": variant, "seed": seed, "construction": c, "executions": ex_rows,
                             "fallback": float(result.telemetry.fallback)})
                index += 1
                print(f"{index:03d}/{total} {case.case_id} {variant} seed={seed} fallback={result.telemetry.fallback}", flush=True)
    input_tokens = sum(x["input_tokens"] for x in construction); output_tokens = sum(x["output_tokens"] for x in construction)
    cost = (input_tokens / 1_000_000 * 0.28) + (output_tokens / 1_000_000 * 0.42)
    payload = {"protocol": "Q2 Pipeline Component-wise Ablation v1.0", "provider": args.provider,
               "model": adapter.config.model, "case_count": len(cases), "variant_count": len(Q2_VARIANTS),
               "variants": sorted(Q2_VARIANTS), "seeds": seeds, "construction_run_count": len(construction),
               "execution_run_count": len(execution), "fairness": {"same_backbone": True, "same_ecosystem": True,
               "same_budget": True, "same_runtime": True, "temperature": 0.0, "max_output_tokens": args.max_output_tokens},
               "usage": {"construction_input_tokens": input_tokens, "construction_output_tokens": output_tokens,
                         "estimated_construction_cost_usd": round(cost, 6)},
               "aggregate_by_variant": aggregate(construction, ["variant"]),
               "aggregate_by_variant_split": aggregate(construction, ["variant", "split"]),
               "aggregate_by_variant_family": aggregate(construction, ["variant", "family"]),
               "aggregate_by_variant_seed": aggregate(construction, ["variant", "seed"]),
               "paired_effects_vs_full": paired_effects(construction),
               "execution_by_variant": aggregate(execution, ["variant"]), "runs": runs,
               "formal_result": args.provider == "openai_compatible" and not any(x["fallback"] for x in runs),
               "limitations": ["Minimal MAR is a structural runtime sandbox; domain answer accuracy is not a Q2 primary metric."]}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
