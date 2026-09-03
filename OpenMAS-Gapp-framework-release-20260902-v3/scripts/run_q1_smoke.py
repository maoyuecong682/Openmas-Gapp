from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openmas_bench.construction import Q1_METHODS
from openmas_bench.engine import GraphHarnessEngine
from openmas_bench.llm import DeterministicAdapter
from openmas_bench.evaluate import evaluate_construction
from openmas_bench.io import load_construction_case, save_construction_result, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(ROOT / "construction_cases"))
    parser.add_argument("--output", default=str(ROOT / "q1_smoke"))
    args = parser.parse_args()
    case_dir, output = Path(args.cases), Path(args.output)
    engine = GraphHarnessEngine(DeterministicAdapter(), ROOT.parents[1])
    records = []
    for path in sorted(case_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        case = load_construction_case(path)
        for method_name in Q1_METHODS:
            result = engine.construct(case, method=method_name)
            save_construction_result(output / "results" / method_name / f"{case.case_id}.json", result)
            records.append(evaluate_construction(case, result))
    summary = defaultdict(lambda: defaultdict(list))
    for row in records:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                summary[row["method"]][key].append(value)
    aggregated = {method: {metric: round(sum(values) / len(values), 6) for metric, values in metrics.items()} for method, metrics in summary.items()}
    write_json(output / "records.json", records)
    write_json(output / "summary.json", {"run_type": "deterministic_interface_smoke_test", "formal_result": False, "case_count": 8, "method_count": 5, "aggregate": aggregated})
    print(f"completed {len(records)} runs; formal_result=False; output={output}")


if __name__ == "__main__":
    main()
