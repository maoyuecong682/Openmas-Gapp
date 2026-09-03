from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / "q2_datasets"
OUT = DATA / "normalized"

LICENSES = {
    "gsm8k": "MIT",
    "math500": "MIT",
    "mmlu": "MIT",
    "humaneval": "MIT",
    "mbpp": "CC-BY-4.0",
    "hotpotqa": "CC-BY-SA-4.0",
    "drop": "CC-BY-SA-4.0",
    "medqa": "unverified",
    "swebench_verified": "MIT/code; dataset terms to verify",
    "pubmedqa": "MIT; verify source terms",
    "finqa": "CC-BY-4.0",
}


def raw_rows(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    for item in payload.get("rows", []):
        yield item.get("row", item)


def cleaned_rows(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def normalize(name: str, row: dict, index: int) -> dict:
    # HF schemas differ; preserve the original row while exposing a stable
    # evaluation interface. Downstream adapters can use dataset-specific fields.
    nested = row.get("data") if isinstance(row.get("data"), dict) else {}
    question = row.get("question") or row.get("prompt") or row.get("problem") or row.get("instruction") or row.get("title") or nested.get("Question") or ""
    context = row.get("context") or row.get("passage") or row.get("background") or ""
    answer = row.get("answer")
    if answer is None and nested:
        answer = nested.get("Correct Option") or nested.get("Correct Answer")
    if answer is None:
        answer = row.get("answers") or row.get("target") or row.get("solution")
    # DROP stores one or more gold spans under answers_spans.
    if answer is None and isinstance(row.get("answers_spans"), dict):
        spans = row["answers_spans"].get("spans") or []
        answer = spans[0] if spans else None
    choices = row.get("choices") or row.get("options") or row.get("endings") or nested.get("Options")
    task_id = row.get("task_id") or row.get("id") or row.get("idx") or f"{name}_{index:04d}"
    return {"id": str(task_id), "question": question, "context": context,
            "choices": choices, "answer": answer, "source": name,
            "raw": row}


def normalize_cleaned(name: str, row: dict, index: int) -> dict:
    """Normalize the repository's audited 30-row FinQA/PubMedQA pilot."""
    context = row.get("context") or {}
    if name == "pubmedqa":
        context_text = "\n\n".join(context.get("contexts", [])) if isinstance(context, dict) else str(context)
    else:
        parts = context.get("pre_text", []) + context.get("post_text", []) if isinstance(context, dict) else [str(context)]
        table = context.get("table", []) if isinstance(context, dict) else []
        context_text = "\n".join(parts + [" | ".join(map(str, row_)) for row_ in table])
    return {"id": str(row.get("task_id", f"{name}_{index:04d}")),
            "question": row.get("prompt", ""), "context": context_text,
            "choices": None, "answer": row.get("answer"), "source": name,
            "raw": row}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in sorted(DATA.glob("*.json")):
        name = path.stem
        if name == "manifest":
            continue
        rows = [normalize(name, row, i) for i, row in enumerate(raw_rows(path))]
        output = OUT / f"{name}.jsonl"
        output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        manifest.append({"dataset": name, "source_dataset": payload.get("dataset"),
                         "config": payload.get("config"), "split": payload.get("split"),
                         "rows": len(rows), "license": LICENSES.get(name, "unverified"),
                         "normalized_file": str(output),
                         "primary_metric": "trace_patch_proxy" if "swe" in name else "accuracy_or_exact_match"})
    cleaned_sources = {
        "finqa": ROOT / "cleaned" / "finance_tasks.jsonl",
        "pubmedqa": ROOT / "cleaned" / "biomedical_tasks.jsonl",
    }
    for name, path in cleaned_sources.items():
        if not path.exists():
            continue
        rows = [normalize_cleaned(name, row, i) for i, row in enumerate(cleaned_rows(path))]
        output = OUT / f"{name}.jsonl"
        output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        manifest.append({"dataset": name, "source_dataset": name.upper(), "config": "cleaned_pilot",
                         "split": "pilot", "rows": len(rows), "license": LICENSES[name],
                         "normalized_file": str(output), "primary_metric": "accuracy"})
    (DATA / "manifest.json").write_text(json.dumps({"schema_version": "q2-dataset-v1", "datasets": manifest}, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
