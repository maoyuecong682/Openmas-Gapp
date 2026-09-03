"""Audit a Q2 result JSON without rerunning any model or evaluator."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def audit(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    variants = payload.get("variants", [])
    by_dataset = defaultdict(list)
    groups = defaultdict(dict)
    for row in runs:
        by_dataset[row.get("dataset")].append(row)
        groups[(row.get("dataset"), row.get("case_id"), row.get("seed"))][row.get("variant")] = row
    datasets = []
    for dataset, rows in sorted(by_dataset.items()):
        health = {}
        for variant in variants:
            subset = [r for r in rows if r.get("variant") == variant]
            completed = [r for r in subset if r.get("status", "completed") == "completed"]
            scores = [float(r["primary_score"]) for r in completed if r.get("primary_score") is not None]
            fallback = [float(r.get("construction_metrics", {}).get("fallback", 0.0)) for r in completed]
            health[variant] = {
                "expected": len(subset), "completed": len(completed),
                "failed": len(subset) - len(completed),
                "failure_rate": round((len(subset) - len(completed)) / max(1, len(subset)), 6),
                "fallback_rate": round(statistics.mean(fallback), 6) if fallback else None,
                "nonzero_rate": round(sum(x > 0 for x in scores) / max(1, len(scores)), 6),
                "mean": round(statistics.mean(scores), 6) if scores else None,
            }
        paired = [g for k, g in groups.items() if k[0] == dataset and set(g) == set(variants)
                  and all(g[v].get("status", "completed") == "completed" and
                          g[v].get("primary_score") is not None for v in variants)]
        deltas = {}
        for variant in variants:
            if variant == "full_graph_harness":
                continue
            values = [float(g[variant]["primary_score"]) - float(g["full_graph_harness"]["primary_score"]) for g in paired]
            deltas[variant] = {"n": len(values), "mean_delta_vs_full": round(statistics.mean(values), 6) if values else None,
                               "wins": sum(x > 0 for x in values), "ties": sum(x == 0 for x in values), "losses": sum(x < 0 for x in values)}
        datasets.append({"dataset": dataset, "complete_paired_groups": len(paired), "health": health, "paired_effects": deltas})
    return {"source": str(path), "expected_runs": payload.get("expected_runs"),
            "completed_runs": payload.get("completed_runs"), "failed_runs": payload.get("failed_runs"),
            "datasets": datasets}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.result)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
