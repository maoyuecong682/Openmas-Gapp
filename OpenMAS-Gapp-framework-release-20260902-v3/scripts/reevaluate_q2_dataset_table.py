from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openmas_bench.q2_dataset_eval import add_strict_scores, paired_deltas, primary_table


def main():
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parents[1] / "q2_deepseek_360.json"
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parents[1] / "q2_dataset_table_deepseek.json"
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    construction = [add_strict_scores(x["construction"]) for x in payload["runs"]]
    execution = []
    for run in payload["runs"]:
        for event in run["executions"]:
            event = dict(event)
            event["variant"] = run["variant"]
            execution.append(event)
    table = paired_deltas(primary_table(construction, execution))
    out = {"protocol": "Q2 dataset-column paired ablation re-evaluation v1.0",
           "source_result": str(source),
           "primary_metric_definition": "one primary metric per dataset/track; all variants paired against Full Graph Harness",
           "table": table,
           "construction_rows": construction}
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for dataset in sorted({x["dataset"] for x in table}):
        print(f"\n[{dataset}]")
        for x in table:
            if x["dataset"] == dataset:
                print(f"{x['variant']}: {x['mean']:.4f} (delta {x['delta_vs_full']})")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
