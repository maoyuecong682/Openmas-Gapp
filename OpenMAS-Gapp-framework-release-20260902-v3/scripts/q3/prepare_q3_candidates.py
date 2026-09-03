"""Download and normalize Q3 candidate datasets.

This script keeps Q2 untouched and writes normalized rows under
``q2_datasets/normalized`` so Q3 can consume them through the existing
benchmark pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "q2_datasets"
RAW = DATA / "raw_q3_candidates"
OUT = DATA / "normalized"

FINANCEBENCH_SPEC = ("PatronusAI/financebench", "default", "train")
SCIBENCH_SPEC = ("xw27/scibench", "default", "train")
BBH_API = "https://api.github.com/repos/suzgunmirac/BIG-Bench-Hard/contents/bbh"
BBH_TASK_ORDER = (
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "causal_judgement",
    "disambiguation_qa",
    "date_understanding",
    "temporal_sequences",
    "geometric_shapes",
    "hyperbaton",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "formal_fallacies",
    "movie_recommendation",
    "navigate",
    "web_of_lies",
    "word_sorting",
    "dyck_languages",
    "boolean_expressions",
    "multistep_arithmetic_two",
)


def _get_json(url: str, *, params: dict[str, Any] | None = None,
              proxies: dict[str, str] | None = None,
              token: str | None = None) -> Any:
    error = None
    for attempt in range(4):
        try:
            headers = {"Authorization": f"Bearer {token}"} if token else None
            response = requests.get(url, params=params, proxies=proxies, headers=headers, timeout=(20, 180))
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed: {url}: {error}")


def _paged_hf_rows(dataset: str, config: str, split: str, total: int,
                   endpoint: str, proxies: dict[str, str] | None = None,
                   token: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < total:
        chunk = min(100, total - len(rows))
        payload = _get_json(
            endpoint.rstrip("/") + "/rows",
            params={"dataset": dataset, "config": config, "split": split, "offset": offset, "length": chunk},
            proxies=proxies,
            token=token,
        )
        batch = [item.get("row", {}) for item in payload.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < chunk:
            break
    return rows[:total]


def _bbh_rows(total: int) -> list[dict[str, Any]]:
    listing = _get_json(BBH_API)
    available = {
        item["name"][:-5]: item["download_url"]
        for item in listing
        if item.get("type") == "file" and str(item.get("name", "")).endswith(".json")
    }
    task_rows: dict[str, list[dict[str, Any]]] = {}
    for task in BBH_TASK_ORDER:
        url = available.get(task)
        if not url:
            continue
        payload = _get_json(url)
        task_rows[task] = list(payload.get("examples", []))
    rows: list[dict[str, Any]] = []
    index = 0
    while len(rows) < total:
        progressed = False
        for task in BBH_TASK_ORDER:
            examples = task_rows.get(task, [])
            if index < len(examples):
                rows.append({"task": task, "index": index, **examples[index]})
                progressed = True
                if len(rows) >= total:
                    break
        if not progressed:
            break
        index += 1
    return rows[:total]


def _normalize_financebench(row: dict[str, Any], index: int) -> dict[str, Any]:
    evidence = row.get("evidence") or []
    snippets = []
    if isinstance(evidence, list):
        for item_index, item in enumerate(evidence[:4]):
            if isinstance(item, dict):
                text = str(item.get("evidence_text") or "").strip()
                if text:
                    snippets.append(f"[E{item_index + 1}] {text}")
    context_parts = [
        f"company: {row.get('company', '')}",
        f"doc_name: {row.get('doc_name', '')}",
        f"question_type: {row.get('question_type', '')}",
        f"question_reasoning: {row.get('question_reasoning', '')}",
    ]
    if snippets:
        context_parts.append("evidence:\n" + "\n\n".join(snippets))
    return {
        "id": str(row.get("financebench_id", f"financebench_{index:04d}")),
        "question": str(row.get("question") or "").strip(),
        "context": "\n".join(part for part in context_parts if part),
        "choices": None,
        "answer": row.get("answer"),
        "source": "financebench",
        "raw": row,
    }


def _normalize_scibench(row: dict[str, Any], index: int) -> dict[str, Any]:
    context_parts = [
        f"source: {row.get('source', '')}",
        f"unit: {row.get('unit', '')}",
    ]
    if row.get("comment"):
        context_parts.append(f"comment: {row.get('comment')}")
    return {
        "id": str(row.get("problemid", f"scibench_{index:04d}")).strip(),
        "question": str(row.get("problem_text") or "").strip(),
        "context": "\n".join(part for part in context_parts if part),
        "choices": None,
        "answer": row.get("answer_number") or row.get("answer_latex"),
        "source": "scibench",
        "raw": row,
    }


def _normalize_bbh(row: dict[str, Any], index: int) -> dict[str, Any]:
    task = str(row.get("task") or "").strip()
    input_text = str(row.get("input") or row.get("question") or "").strip()
    context = f"task: {task}\ninput: {input_text}".strip()
    answer = row.get("target")
    if answer is None:
        answer = row.get("answer")
    return {
        "id": str(row.get("index", index)),
        "question": input_text,
        "context": context,
        "choices": None,
        "answer": answer,
        "source": "bbh_full",
        "raw": row,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--financebench-length", type=int, default=150)
    parser.add_argument("--scibench-length", type=int, default=250)
    parser.add_argument("--bbh-length", type=int, default=240)
    parser.add_argument("--hf-endpoint", default=os.environ.get("Q2_HF_DATASETS_ENDPOINT", "https://datasets-server.huggingface.co"))
    parser.add_argument("--http-proxy", default=os.environ.get("Q2_HTTP_PROXY"))
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    args = parser.parse_args()

    proxies = {"http": args.http_proxy, "https": args.http_proxy} if args.http_proxy else None
    hf_token = os.environ.get(args.hf_token_env)
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []

    finance_rows = _paged_hf_rows(*FINANCEBENCH_SPEC, args.financebench_length, args.hf_endpoint, proxies, hf_token)
    finance_raw = RAW / "financebench.jsonl"
    finance_norm = OUT / "financebench.jsonl"
    _write_jsonl(finance_raw, finance_rows)
    finance_norm_rows = [_normalize_financebench(row, i) for i, row in enumerate(finance_rows)]
    _write_jsonl(finance_norm, finance_norm_rows)
    manifest.append({
        "dataset": "financebench",
        "source_dataset": FINANCEBENCH_SPEC[0],
        "config": FINANCEBENCH_SPEC[1],
        "split": FINANCEBENCH_SPEC[2],
        "rows": len(finance_norm_rows),
        "normalized_file": str(finance_norm),
        "sha256": hashlib.sha256(finance_norm.read_bytes()).hexdigest(),
    })

    scibench_rows = _paged_hf_rows(*SCIBENCH_SPEC, args.scibench_length, args.hf_endpoint, proxies, hf_token)
    scibench_raw = RAW / "scibench.jsonl"
    scibench_norm = OUT / "scibench.jsonl"
    _write_jsonl(scibench_raw, scibench_rows)
    scibench_norm_rows = [_normalize_scibench(row, i) for i, row in enumerate(scibench_rows)]
    _write_jsonl(scibench_norm, scibench_norm_rows)
    manifest.append({
        "dataset": "scibench",
        "source_dataset": SCIBENCH_SPEC[0],
        "config": SCIBENCH_SPEC[1],
        "split": SCIBENCH_SPEC[2],
        "rows": len(scibench_norm_rows),
        "normalized_file": str(scibench_norm),
        "sha256": hashlib.sha256(scibench_norm.read_bytes()).hexdigest(),
    })

    bbh_rows = _bbh_rows(args.bbh_length)
    bbh_raw = RAW / "bbh_full.jsonl"
    bbh_norm = OUT / "bbh_full.jsonl"
    _write_jsonl(bbh_raw, bbh_rows)
    bbh_norm_rows = [_normalize_bbh(row, i) for i, row in enumerate(bbh_rows)]
    _write_jsonl(bbh_norm, bbh_norm_rows)
    manifest.append({
        "dataset": "bbh_full",
        "source_dataset": "suzgunmirac/BIG-Bench-Hard",
        "config": "official",
        "split": "test",
        "rows": len(bbh_norm_rows),
        "normalized_file": str(bbh_norm),
        "sha256": hashlib.sha256(bbh_norm.read_bytes()).hexdigest(),
    })

    manifest_path = DATA / "q3_candidate_manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": "q3-candidates-v1", "datasets": manifest}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
