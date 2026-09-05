"""API-free qualification for Q10 normalized financial rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_row(name: str, row: dict) -> list[str]:
    errors = []
    if not str(row.get("id", "")).strip(): errors.append("missing id")
    if not str(row.get("question", "")).strip(): errors.append("missing question")
    if row.get("answer") is None or not str(row.get("answer")).strip(): errors.append("missing answer")
    if not isinstance(row.get("raw"), dict): errors.append("missing raw object")
    if name == "financebench":
        evidence = (row.get("raw") or {}).get("evidence") or []
        if not isinstance(evidence, list) or not evidence:
            errors.append("missing filing evidence snippets")
        if not str((row.get("raw") or {}).get("question_reasoning", "")).strip():
            errors.append("missing question reasoning label")
    if name == "finqa":
        raw = row.get("raw") or {}
        context = raw.get("context") or {}
        metadata = raw.get("metadata") or {}
        program = metadata.get("program")
        gold_evidence = metadata.get("gold_evidence") or raw.get("gold_evidence")
        if not isinstance(context, dict) or not context.get("table"):
            errors.append("missing financial table context")
        if not str(program or "").strip() and not gold_evidence:
            errors.append("missing numeric program or gold evidence metadata for audit")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[3] / "q10_datasets")
    parser.add_argument("--datasets", default="financebench,finqa")
    parser.add_argument("--rows-per-dataset", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    results = []
    for name in [item.strip().lower() for item in args.datasets.split(",") if item.strip()]:
        path = args.data_root / "normalized" / f"{name}.jsonl"
        item = {"dataset": name, "file": str(path), "status": "blocked", "rows_checked": 0, "errors": []}
        if not path.exists():
            item["errors"] = ["normalized file missing"]
        else:
            for line in path.read_text(encoding="utf-8").splitlines()[: args.rows_per_dataset]:
                if line.strip():
                    item["rows_checked"] += 1
                    item["errors"].extend(check_row(name, json.loads(line)))
            item["status"] = "qualified" if item["rows_checked"] >= args.rows_per_dataset and not item["errors"] else "blocked"
        results.append(item)
    payload = {"schema_version": "q10-financial-qualification-v1", "formal_result": False, "purpose": "pre-model financial contract check", "datasets": results}
    output = args.output or args.data_root / "manifests" / "q10_qualification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
