from __future__ import annotations
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_name(source.stem + "_table.json")
payload = json.loads(source.read_text(encoding="utf-8"))
groups = defaultdict(list)
metrics = {}
for run in payload["runs"]:
    groups[(run["dataset"], run["variant"])].append(float(run["primary_score"]))
    metrics[run["dataset"]] = run["primary_metric"]
full = {d: statistics.mean(v) for (d, variant), v in groups.items() if variant == "full_graph_harness"}
table = []
for (dataset, variant), values in sorted(groups.items()):
    mean = statistics.mean(values)
    table.append({"dataset": dataset, "variant": variant, "metric": metrics[dataset], "mean": round(mean, 6), "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0, "n": len(values), "delta_vs_full": round(mean - full[dataset], 6)})
datasets = payload.get("datasets", sorted(metrics))
variants = payload.get("variants", sorted({x["variant"] for x in table}))
lookup = {(x["dataset"], x["variant"]): x for x in table}
pivot = []
for variant in variants:
    row = {"variant": variant}
    for dataset in datasets:
        cell = lookup.get((dataset, variant))
        row[dataset] = f"{cell['mean']:.4f} ± {cell['std']:.4f}" if cell else "NA"
        row[f"{dataset}__delta_vs_full"] = cell["delta_vs_full"] if cell else None
    pivot.append(row)
payload["protocol"] = "Q2 cross-dataset paired ablation v4"
payload["table"] = table
payload["pivot_table"] = pivot
payload["metrics"] = metrics
target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(target)
