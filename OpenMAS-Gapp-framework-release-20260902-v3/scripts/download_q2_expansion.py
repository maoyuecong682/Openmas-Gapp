"""Download and normalize Q2 expansion datasets.

The script uses public dataset-server rows (HF) or the official BBH GitHub
repository, preserves every source row under ``raw``, and emits the stable
Q2 JSONL schema. It is intentionally pilot-sized by default; increase
``--length`` only after checking licenses and storage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import os
from pathlib import Path
from typing import Any

import requests


HF_SPECS = {
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "default", "test", "MIT"),
    "arc": ("allenai/ai2_arc", "ARC-Challenge", "test", "Apache-2.0"),
    "humaneval": ("openai/openai_humaneval", "openai_humaneval", "test", "MIT"),
    "mbpp": ("google-research-datasets/mbpp", "full", "test", "CC-BY-4.0"),
    "logiqa": ("lmguan/logiqa", "default", "test", "Apache-2.0"),
}
BBH_TASKS = ("boolean_expressions", "causal_judgement", "date_understanding",
             "disambiguation_qa", "formal fallacies", "geometric_shapes",
             "hyperbaton", "logical_deduction", "object_tracking",
             "penguins_in_a_table", "reasoning_about_colored_objects",
             "ruin_names", "salient_translation_error_detection", "snarks",
             "sports_understanding", "temporal_sequence", "tracking_shuffled_objects")


def _get_json(url: str, *, params: dict[str, Any] | None = None,
              proxies: dict[str, str] | None = None,
              token: str | None = None) -> Any:
    error = None
    for attempt in range(4):
        try:
            headers = {"Authorization": f"Bearer {token}"} if token else None
            response = requests.get(url, params=params, proxies=proxies,
                                    headers=headers,
                                    timeout=(20, 180))
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(
        f"download failed: {url}: {error}. Check network access, or rerun with "
        "--http-proxy http://127.0.0.1:PORT / --hf-endpoint ENDPOINT."
    )


def _hf_rows(dataset: str, config: str, split: str, length: int,
             endpoint: str, proxies: dict[str, str] | None = None,
             token: str | None = None) -> list[dict[str, Any]]:
    payload = _get_json(endpoint.rstrip("/") + "/rows",
                       params={"dataset": dataset, "config": config,
                               "split": split, "offset": 0, "length": length},
                       proxies=proxies, token=token)
    rows = [item.get("row", {}) for item in payload.get("rows", [])]
    if not rows:
        raise RuntimeError(f"{dataset}/{config}/{split}: endpoint returned no rows")
    return rows


def _direct_jsonl(url: str, length: int, proxies: dict[str, str] | None = None) -> list[dict[str, Any]]:
    response = requests.get(url, proxies=proxies, timeout=(20, 180))
    response.raise_for_status()
    rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    return rows[:length]


def _bbh_rows(length: int) -> list[dict[str, Any]]:
    rows = []
    base = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh"
    for task in BBH_TASKS:
        encoded = task.replace(" ", "_")
        payload = _get_json(f"{base}/{encoded}.json")
        for index, example in enumerate(payload.get("examples", [])):
            rows.append({"task": task, "index": index, **example})
            if len(rows) >= length:
                return rows
    return rows


def _normalize(name: str, row: dict[str, Any], index: int) -> dict[str, Any]:
    question = (row.get("question") or row.get("input") or row.get("prompt")
                or row.get("problem") or row.get("instruction") or row.get("text") or "")
    context = row.get("context") or row.get("passage") or row.get("background") or ""
    choices = row.get("choices") or row.get("options") or row.get("endings")
    answer = row.get("answer")
    if answer is None:
        answer = row.get("answerKey")
    if answer is None:
        answer = row.get("target") or row.get("label") or row.get("solution")
    # MMLU-Pro exposes options as a list and answer as a letter.
    if name == "mmlu_pro" and isinstance(row.get("options"), list):
        choices = row["options"]
    source_id = row.get("id") or row.get("task_id") or row.get("idx")
    if source_id is None:
        source_id = f"{name}_{index:06d}"
    return {"id": str(source_id), "question": str(question), "context": context,
            "choices": choices, "answer": answer, "source": name, "raw": row}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join([*HF_SPECS, "bbh"]))
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--hf-endpoint",
                        default=os.environ.get("Q2_HF_DATASETS_ENDPOINT",
                                               "https://datasets-server.huggingface.co"),
                        help="datasets-server base URL; can be a mirror or local proxy")
    parser.add_argument("--http-proxy", default=os.environ.get("Q2_HTTP_PROXY"),
                        help="optional proxy URL, e.g. http://127.0.0.1:7890")
    parser.add_argument("--hf-token-env", default="HF_TOKEN",
                        help="environment variable containing a Hugging Face access token")
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parents[3] / "q2_datasets")
    args = parser.parse_args()
    proxies = ({"http": args.http_proxy, "https": args.http_proxy}
               if args.http_proxy else None)
    hf_token = os.environ.get(args.hf_token_env)
    raw_root = args.output_root / "raw_expansion"
    normalized_root = args.output_root / "normalized"
    raw_root.mkdir(parents=True, exist_ok=True)
    normalized_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in [x.strip().lower() for x in args.datasets.split(",") if x.strip()]:
        if name == "bbh":
            dataset, config, split, license_name = "suzgunmirac/BIG-Bench-Hard", "official", "test", "MIT"
            rows = _bbh_rows(args.length)
        elif name in HF_SPECS:
            dataset, config, split, license_name = HF_SPECS[name]
            rows = _hf_rows(dataset, config, split, args.length, args.hf_endpoint, proxies, hf_token)
        else:
            raise ValueError(f"unknown expansion dataset {name}; choose from {sorted([*HF_SPECS, 'bbh'])}")
        raw_path = raw_root / f"{name}.jsonl"
        normalized_path = normalized_root / f"{name}.jsonl"
        raw_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        normalized = [_normalize(name, row, index) for index, row in enumerate(rows)]
        normalized_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in normalized) + "\n", encoding="utf-8")
        digest = hashlib.sha256(normalized_path.read_bytes()).hexdigest()
        manifest.append({"dataset": name, "source_dataset": dataset, "config": config,
                         "split": split, "rows": len(normalized), "license": license_name,
                         "normalized_file": str(normalized_path), "sha256": digest})
        print(f"normalized {name}: {len(normalized)} rows sha256={digest}")
    manifest_path = args.output_root / "expansion_manifest.json"
    existing = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
    by_name = {item.get("dataset"): item for item in existing.get("datasets", [])}
    by_name.update({item["dataset"]: item for item in manifest})
    manifest_path.write_text(json.dumps({"schema_version": "q2-dataset-v1-expansion",
                                         "datasets": list(by_name.values())}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
