"""Create the local frozen Q10 financial pilot datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


FINANCEBENCH_ROWS: list[dict[str, Any]] = [
    {
        "id": "q10_financebench_000",
        "question": (
            "Using the supplied filing evidence, determine whether revenue "
            "growth appears supported by disclosed operating drivers and note "
            "the main risk or compliance caveat."
        ),
        "answer": "Revenue growth is supported by disclosed volume and pricing drivers, subject to margin and non-advice caveats.",
        "context": "",
        "choices": [],
        "source": "financebench",
        "raw": {
            "question_type": "metrics-generated",
            "question_reasoning": "Numerical reasoning with disclosure and risk review",
            "evidence": [
                {"evidence_text": "Annual filing excerpt: revenue increased from 10.0 billion to 11.2 billion, with management attributing growth to higher shipment volume and selective price increases."},
                {"evidence_text": "Risk factor excerpt: management states that demand softness, supplier concentration, and foreign exchange volatility could pressure future margins."},
                {"evidence_text": "Compliance excerpt: forward-looking statements are subject to uncertainty and should not be treated as investment advice."},
            ],
        },
    },
    {
        "id": "q10_financebench_001",
        "question": "Assess whether the issuer's liquidity position is adequate based on cash, debt maturity, and disclosed covenant risk.",
        "answer": "Liquidity appears adequate, but debt maturities and covenant sensitivity require risk and compliance review.",
        "context": "",
        "choices": [],
        "source": "financebench",
        "raw": {
            "question_type": "novel-generated",
            "question_reasoning": "Logical and numerical reasoning over filing evidence",
            "evidence": [
                {"evidence_text": "Balance sheet excerpt: cash and equivalents were 2.4 billion at year end."},
                {"evidence_text": "Debt note excerpt: 650 million of senior notes mature within twelve months."},
                {"evidence_text": "Risk factor excerpt: covenant headroom may tighten if adjusted EBITDA declines materially."},
            ],
        },
    },
    {
        "id": "q10_financebench_002",
        "question": "Explain whether free cash flow quality supports the company's capital allocation plan, including any auditability caveat.",
        "answer": "Free cash flow quality supports the plan only if working-capital benefits are recurring and traceable to audited disclosures.",
        "context": "",
        "choices": [],
        "source": "financebench",
        "raw": {
            "question_type": "metrics-generated",
            "question_reasoning": "Financial analysis with audit and risk constraints",
            "evidence": [
                {"evidence_text": "Cash flow excerpt: operating cash flow was 1.8 billion and capital expenditures were 0.6 billion."},
                {"evidence_text": "MD&A excerpt: working-capital timing contributed materially to the current-year cash flow improvement."},
                {"evidence_text": "Controls excerpt: estimates are subject to internal control over financial reporting."},
            ],
        },
    },
]


FINQA_ROWS: list[dict[str, Any]] = [
    {
        "id": "q10_finqa_000",
        "question": "What was free cash flow in 2025, calculated as operating cash flow minus capital expenditures?",
        "answer": "1200",
        "context": "Operating cash flow was 1,800 million. Capital expenditures were 600 million.",
        "choices": [],
        "source": "finqa",
        "raw": {
            "context": {
                "pre_text": ["The company reports cash flows in USD millions."],
                "table": [["metric", "2025"], ["operating cash flow", "1800"], ["capital expenditures", "600"]],
                "post_text": ["Free cash flow is calculated as operating cash flow less capital expenditures."],
            },
            "metadata": {"program": "subtract(1800, 600)"},
        },
    },
    {
        "id": "q10_finqa_001",
        "question": "What is the debt-to-EBITDA ratio using total debt of 4500 and EBITDA of 1500?",
        "answer": "3",
        "context": "Total debt was 4,500 million. EBITDA was 1,500 million.",
        "choices": [],
        "source": "finqa",
        "raw": {
            "context": {
                "pre_text": ["All values are stated in USD millions."],
                "table": [["metric", "value"], ["total debt", "4500"], ["EBITDA", "1500"]],
                "post_text": ["Leverage is total debt divided by EBITDA."],
            },
            "metadata": {"program": "divide(4500, 1500)"},
        },
    },
    {
        "id": "q10_finqa_002",
        "question": "What was revenue growth percentage from 10000 to 11200?",
        "answer": "12",
        "context": "Revenue increased from 10,000 million to 11,200 million.",
        "choices": [],
        "source": "finqa",
        "raw": {
            "context": {
                "pre_text": ["Revenue is disclosed in USD millions."],
                "table": [["metric", "2024", "2025"], ["revenue", "10000", "11200"]],
                "post_text": ["Growth rate is the change divided by the prior year amount."],
            },
            "metadata": {"program": "subtract(11200, 10000), divide(#0, 10000), multiply(#1, 100)"},
        },
    },
]


PUBLIC_SPECS: dict[str, dict[str, str]] = {
    "financebench": {
        "dataset": "PatronusAI/financebench",
        "config": "default",
        "split": "train",
        "source": "FinanceBench open-source split via Hugging Face rows API",
        "license": "CC BY-NC 4.0; verify upstream dataset card before publication",
    },
    "finqa": {
        "dataset": "dreamerdeo/finqa",
        "config": "default",
        "split": "test",
        "source": "FinQA test split via Hugging Face rows API",
        "license": "CC-BY-4.0; verify upstream dataset card before publication",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def fetch_public_rows(
    name: str,
    *,
    rows_per_dataset: int,
    endpoint: str,
    timeout_seconds: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = PUBLIC_SPECS[name]
    query = urllib.parse.urlencode({
        "dataset": spec["dataset"],
        "config": spec["config"],
        "split": spec["split"],
        "offset": 0,
        "length": rows_per_dataset,
    })
    url = f"{endpoint.rstrip('/')}/rows?{query}" if not endpoint.rstrip("/").endswith("/rows") else f"{endpoint}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "openmas-gapp-q10/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        reason = getattr(exc, "reason", None) or str(exc)
        raise RuntimeError(f"could not fetch public {name} rows from {url}: {reason}") from exc
    if not isinstance(payload, dict) or payload.get("error"):
        raise RuntimeError(f"public {name} rows API error from {url}: {payload.get('error', payload)!r}")
    rows = []
    for entry in payload.get("rows") or []:
        source = entry.get("row", entry) if isinstance(entry, dict) else entry
        if not isinstance(source, dict):
            continue
        offset = int(entry.get("row_idx", len(rows))) if isinstance(entry, dict) else len(rows)
        rows.append(normalize_public_row(name, source, offset=offset, spec=spec))
    if len(rows) < rows_per_dataset:
        raise RuntimeError(f"public {name} returned {len(rows)} rows, expected {rows_per_dataset}")
    return rows, {
        "dataset": spec["dataset"],
        "config": spec["config"],
        "split": spec["split"],
        "rows_api": url,
        "num_rows_total": payload.get("num_rows_total"),
    }


def normalize_public_row(name: str, source: dict[str, Any], *, offset: int, spec: dict[str, str]) -> dict[str, Any]:
    if name == "financebench":
        evidence = source.get("evidence") if isinstance(source.get("evidence"), list) else []
        context_parts = [str(item.get("evidence_text", "")).strip() for item in evidence if isinstance(item, dict)]
        source_id = source.get("financebench_id") or source.get("id") or f"financebench_{offset:06d}"
        return {
            "id": str(source_id),
            "question": str(source.get("question") or ""),
            "answer": source.get("answer"),
            "context": "\n\n".join(part for part in context_parts if part),
            "choices": [],
            "source": "financebench",
            "raw": source,
            "remote": {"dataset": spec["dataset"], "config": spec["config"], "split": spec["split"], "split_offset": offset},
        }
    if name == "finqa":
        pre_text = _as_string_list(source.get("pre_text"))
        post_text = _as_string_list(source.get("post_text"))
        table = source.get("table") if isinstance(source.get("table"), list) else []
        table_text = "\n".join(" | ".join(str(cell) for cell in row) for row in table if isinstance(row, list))
        raw = {
            **source,
            "context": {"pre_text": pre_text, "table": table, "post_text": post_text},
            "metadata": {"gold_evidence": _as_string_list(source.get("gold_evidence"))},
        }
        return {
            "id": str(source.get("id") or f"finqa_{offset:06d}"),
            "question": str(source.get("question") or ""),
            "answer": source.get("answer"),
            "context": "\n".join([*pre_text, table_text, *post_text]).strip(),
            "choices": [],
            "source": "finqa",
            "raw": raw,
            "remote": {"dataset": spec["dataset"], "config": spec["config"], "split": spec["split"], "split_offset": offset},
        }
    raise ValueError(f"unsupported public Q10 dataset {name!r}")


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local Q10 financial pilot rows.")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[3] / "q10_datasets")
    parser.add_argument("--source", choices=("local", "public"), default="local")
    parser.add_argument("--datasets", default="financebench,finqa")
    parser.add_argument("--rows-per-dataset", type=int, default=3)
    parser.add_argument("--endpoint", default=os.environ.get("Q10_HF_DATASETS_ENDPOINT", "https://datasets-server.huggingface.co"))
    parser.add_argument("--timeout-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.rows_per_dataset < 1:
        raise ValueError("--rows-per-dataset must be >= 1")
    manifest_root = args.output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    selected = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    for name in selected:
        if args.source == "local":
            local_specs = {
                "financebench": ("FinanceBench-style local financial filing pilot", FINANCEBENCH_ROWS),
                "finqa": ("FinQA-style local financial table pilot", FINQA_ROWS),
            }
            if name not in local_specs:
                raise ValueError(f"unsupported local Q10 dataset {name!r}")
            source, rows = local_specs[name]
            rows = rows[: args.rows_per_dataset]
            remote = None
            split = "local_pilot"
            license_text = "local research fixture; replace with frozen public split for formal reporting"
            source_status = "local_fixture"
        else:
            if name not in PUBLIC_SPECS:
                raise ValueError(f"unsupported public Q10 dataset {name!r}")
            rows, remote = fetch_public_rows(
                name,
                rows_per_dataset=args.rows_per_dataset,
                endpoint=args.endpoint,
                timeout_seconds=args.timeout_seconds,
            )
            public_spec = PUBLIC_SPECS[name]
            source = public_spec["source"]
            split = public_spec["split"]
            license_text = public_spec["license"]
            source_status = "public_frozen_rows_api"
        raw_path = args.output_root / "raw" / "financial" / f"{name}.jsonl"
        normalized_path = args.output_root / "normalized" / f"{name}.jsonl"
        pilot_path = args.output_root / "pilot" / f"{name}_pilot.jsonl"
        write_jsonl(raw_path, [row["raw"] for row in rows])
        write_jsonl(normalized_path, rows)
        write_jsonl(pilot_path, rows)
        item = {
            "dataset": name,
            "source_dataset": source,
            "split": split,
            "rows": len(rows),
            "pilot_rows": len(rows),
            "license": license_text,
            "raw_file": str(raw_path.relative_to(args.output_root)),
            "normalized_file": str(normalized_path.relative_to(args.output_root)),
            "pilot_file": str(pilot_path.relative_to(args.output_root)),
            "sha256_raw": sha256(raw_path),
            "sha256_normalized": sha256(normalized_path),
            "source_status": source_status,
        }
        if remote is not None:
            item["remote"] = remote
        manifest.append(item)
        print(f"prepared {name}: {len(rows)} rows")
    output = manifest_root / "q10_dataset_manifest.json"
    output.write_text(json.dumps({
        "schema_version": "q10-financial-v1",
        "purpose": "Graph Harness financial MAS application case study",
        "datasets": manifest,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
