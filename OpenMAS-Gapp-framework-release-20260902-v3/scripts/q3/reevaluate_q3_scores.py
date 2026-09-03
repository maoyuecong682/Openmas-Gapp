from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openmas_bench.dataset_adapters import DATASET_ADAPTERS
from openmas_bench.io import write_json
from scripts.q3.run_q3_real import _summarize


def main() -> None:
    parser = argparse.ArgumentParser(prog="reevaluate_q3_scores")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    runs = list(payload.get("runs", []))
    changed = 0
    unsupported = defaultdict(int)

    adapters_by_id = {adapter.dataset_id: adapter for adapter in DATASET_ADAPTERS.values()}
    for run in runs:
        if run.get("status", "completed") != "completed":
            continue
        dataset_id = run.get("dataset_id")
        adapter = adapters_by_id.get(dataset_id)
        if adapter is None:
            unsupported[dataset_id] += 1
            continue
        execution_rows = run.get("execution") or []
        scores = []
        for execution in execution_rows:
            prediction = execution.get("prediction")
            metrics = execution.get("metrics") or {}
            gold = metrics.get("gold_answer")
            score = adapter.execution.score(prediction, gold)
            old_score = execution.get("answer_score")
            if score != old_score:
                changed += 1
            execution["answer_score"] = score
            metrics["answer_accuracy"] = score
            metrics["answer_evaluated"] = score is not None
            if score is not None:
                scores.append(float(score))
        if scores:
            value = round(sum(scores) / len(scores), 6)
            run["primary_score"] = value
            run["primary_score_mean"] = value
            run["e2e_success"] = value
            run["e2e_supported"] = True
        else:
            run["primary_score"] = None
            run["primary_score_mean"] = None
            run["e2e_success"] = None
            run["e2e_supported"] = False

    completed = sum(row.get("status", "completed") == "completed" for row in runs)
    failed = sum(row.get("status") == "failed" for row in runs)
    expected = int(payload.get("expected_runs") or len(runs))
    payload["runs"] = runs
    payload["completed_runs"] = completed
    payload["failed_runs"] = failed
    payload["progress"] = round((completed + failed) / expected, 6) if expected else 0.0
    payload["summary"] = _summarize(runs)
    payload.setdefault("reevaluation", {})
    payload["reevaluation"].update({
        "source": str(source),
        "score_fields_changed": changed,
        "unsupported_datasets": dict(unsupported),
    })
    write_json(Path(args.output), payload)
    print(f"wrote {args.output}")
    print(f"score_fields_changed={changed}")


if __name__ == "__main__":
    main()
