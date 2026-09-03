from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[x] for x in keys)].append(row)
    output = []
    ignored = set(keys) | {"case_id", "execution_task_id", "adapter", "model", "family", "split", "method", "predicted_answer", "gold_answer", "source_dataset"}
    for group, items in sorted(groups.items()):
        record = dict(zip(keys, group))
        numeric = {key for item in items for key, value in item.items() if isinstance(value, (int, float, bool)) and key not in ignored}
        for metric in sorted(numeric):
            values = [float(x[metric]) for x in items if x.get(metric) is not None]
            if values:
                record[f"{metric}_mean"] = round(statistics.mean(values), 6)
                record[f"{metric}_std"] = round(statistics.stdev(values), 6) if len(values) > 1 else 0.0
        record["n"] = len(items)
        output.append(record)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--replacement", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    replacement = json.loads(Path(args.replacement).read_text(encoding="utf-8"))
    runs = [x for x in base["runs"] if x["case_id"] != args.case_id] + replacement["runs"]
    construction = []
    execution = []
    for run in runs:
        row = dict(run["construction"])
        row.update({"split": run["split"], "family": run["family"], "fallback": float(run["fallback"])})
        construction.append(row)
        for item in run["executions"]:
            execution.append(item)
    base["runs"] = runs
    base["construction_run_count"] = len(runs)
    base["execution_run_count"] = len(execution)
    base["formal_result"] = not any(x["fallback"] for x in runs)
    base["aggregate_by_method"] = aggregate(construction, ["method"])
    base["aggregate_by_method_split"] = aggregate(construction, ["method", "split"])
    base["aggregate_by_method_family"] = aggregate(construction, ["method", "family"])
    base["execution_by_method"] = aggregate(execution, ["method"])
    base["execution_by_source"] = aggregate([x for x in execution if x.get("answer_evaluated")], ["source_dataset"])
    construction_input = sum(x["construction"].get("input_tokens", 0) for x in runs)
    construction_output = sum(x["construction"].get("output_tokens", 0) for x in runs)
    answer_input = sum(x.get("answer_input_tokens", 0) for x in execution) // 5
    answer_output = sum(x.get("answer_output_tokens", 0) for x in execution) // 5
    input_rate = float(base["usage"]["configured_input_usd_per_million"])
    output_rate = float(base["usage"]["configured_output_usd_per_million"])
    final_cost = (construction_input + answer_input) / 1_000_000 * input_rate + (construction_output + answer_output) / 1_000_000 * output_rate
    actual_cost = float(base["usage"].get("estimated_total_cost_usd", 0)) + float(replacement["usage"].get("estimated_total_cost_usd", 0))
    base["usage"] = {"construction_input_tokens": construction_input, "construction_output_tokens": construction_output, "answer_input_tokens": answer_input, "answer_output_tokens": answer_output, "configured_input_usd_per_million": input_rate, "configured_output_usd_per_million": output_rate, "estimated_final_dataset_cost_usd": round(final_cost, 6), "estimated_actual_billed_cost_with_recovery_usd": round(actual_cost, 6)}
    base["recovery"] = {"replaced_case": args.case_id, "reason": "semantic ecosystem-ID repair after bounded-output validation"}
    Path(args.output).write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"merged runs={len(runs)} formal_result={base['formal_result']}")


if __name__ == "__main__":
    main()
