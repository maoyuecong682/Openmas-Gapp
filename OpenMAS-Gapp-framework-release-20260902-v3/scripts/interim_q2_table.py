from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

VARIANTS = ["full_graph_harness", "w/o_requirement_grounding",
            "w/o_graph_orchestration", "w/o_blueprint",
            "w/o_constraint_aware_orchestration", "w/o_realization"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    args = parser.parse_args()
    rows = json.loads(Path(args.result).read_text(encoding="utf-8"))["runs"]
    groups = defaultdict(dict)
    for row in rows:
        groups[(row["dataset"], row["case_id"], row["seed"])][row["variant"]] = row
    pairs = {key: group for key, group in groups.items()
             if set(group) == set(VARIANTS) and
             all(row.get("status", "completed") == "completed" and row.get("primary_score") is not None
                 for row in group.values())}
    datasets = sorted({row["dataset"] for row in rows})
    print("| Dataset | n | Full | w/o Req | w/o Graph | w/o Blueprint | w/o Constraint | w/o Realization |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        selected = [group for key, group in pairs.items() if key[0] == dataset]
        values = [statistics.mean(float(group[variant]["primary_score"]) for group in selected)
                  if selected else None for variant in VARIANTS]
        formatted = [f"{value:.3f}" if value is not None else "-" for value in values]
        print(f"| {dataset} | {len(selected)} | " + " | ".join(formatted) + " |")
    print("\nFailures by dataset/variant:")
    for dataset in datasets:
        failures = {variant: sum(row["dataset"] == dataset and row["variant"] == variant and
                                 row.get("status") == "failed" for row in rows)
                    for variant in VARIANTS}
        print(dataset, failures)
    print("\nrows", len(rows), "completed", sum(r.get("status", "completed") == "completed" for r in rows),
          "failed", sum(r.get("status") == "failed" for r in rows), "paired", Counter(k[0] for k in pairs))


if __name__ == "__main__":
    main()
