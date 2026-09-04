"""Minimal-storage access to Q9 datasets through the HF datasets-server API.

This module intentionally does not import ``datasets`` and does not write a
cache.  It requests one source row at a time, normalizes it in memory, and
returns only the selected row to the existing construction pipeline.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .dataset_adapters import DatasetAdapter
from .dataset_cases import _is_qualified_row


DEFAULT_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


class RemoteDatasetError(RuntimeError):
    """Raised when a remote row cannot be obtained without local fallback."""


def load_remote_row(
    adapter: DatasetAdapter,
    row_index: int,
    *,
    timeout_seconds: int = 30,
    endpoint: str = DEFAULT_ROWS_ENDPOINT,
) -> dict[str, Any]:
    """Return the ``row_index``-th qualified row without materializing a file.

    MMLU medical is represented by several configs, so its index is the
    concatenation of those test splits in the declared config order.  The rows
    API reports each config's row count, allowing the selected config and row
    to be addressed without scanning or downloading prior source rows.
    """
    if row_index < 0:
        raise ValueError("row_index must be >= 0")
    spec = adapter.remote_spec
    if not spec:
        raise RemoteDatasetError(f"{adapter.dataset_id} has no remote row source configured")

    configs = tuple(spec.get("configs") or (spec.get("config", "default"),))
    if len(configs) == 1:
        return _load_remote_row_at(
            adapter, spec, configs[0], row_index, endpoint, timeout_seconds)

    remaining = row_index
    for config in configs:
        probe = _fetch_rows(
            endpoint,
            dataset=str(spec["dataset"]),
            config=config,
            split=str(spec["split"]),
            offset=0,
            length=1,
            timeout_seconds=timeout_seconds,
        )
        row_count = _reported_row_count(probe, adapter.dataset_id, config)
        if remaining < row_count:
            if remaining == 0:
                return _normalize_payload_row(adapter, probe, config, 0)
            return _load_remote_row_at(
                adapter, spec, config, remaining, endpoint, timeout_seconds)
        remaining -= row_count

    raise RemoteDatasetError(
        f"{adapter.dataset_id} remote source has fewer than {row_index + 1} rows"
    )


def _load_remote_row_at(
    adapter: DatasetAdapter,
    spec: dict[str, Any],
    config: str,
    offset: int,
    endpoint: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    payload = _fetch_rows(
        endpoint,
        dataset=str(spec["dataset"]),
        config=config,
        split=str(spec["split"]),
        offset=offset,
        length=1,
        timeout_seconds=timeout_seconds,
    )
    return _normalize_payload_row(adapter, payload, config, offset)


def _normalize_payload_row(
    adapter: DatasetAdapter,
    payload: dict[str, Any],
    config: str,
    offset: int,
) -> dict[str, Any]:
    entries = payload.get("rows") or []
    if len(entries) != 1:
        raise RemoteDatasetError(
            f"{adapter.dataset_id} remote source has no row at offset {offset} in config {config}"
        )
    entry = entries[0]
    source_row = entry.get("row", entry) if isinstance(entry, dict) else entry
    if not isinstance(source_row, dict):
        raise RemoteDatasetError(
            f"{adapter.dataset_id} remote source returned a non-object row at offset {offset}"
        )
    row = normalize_remote_row(adapter.dataset_id, source_row, config=config, offset=offset)
    if not _is_qualified_row(adapter, row):
        raise RemoteDatasetError(
            f"{adapter.dataset_id} remote source row {offset} is not a runnable benchmark row"
        )
    return row


def _reported_row_count(payload: dict[str, Any], dataset_id: str, config: str) -> int:
    value = payload.get("num_rows_total")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise RemoteDatasetError(
        f"{dataset_id} remote rows API did not report num_rows_total for config {config}; "
        "cannot map a global MMLU-Medical row index without downloading rows"
    )


def normalize_remote_row(
    dataset_id: str,
    source: dict[str, Any],
    *,
    config: str,
    offset: int,
) -> dict[str, Any]:
    """Map a source-specific HF row to the benchmark's stable row schema."""
    kind = dataset_id.casefold()
    if kind == "medqa":
        medqa_source = source.get("data") if isinstance(source.get("data"), dict) else source
        choices = _choices_from_medqa(medqa_source)
        answer = _first_value(
            medqa_source,
            "answer",
            "answer_idx",
            "answer_index",
            "cop",
            "Correct Answer",
            "Correct Option",
        )
        question = _first_value(medqa_source, "question", "Question")
        context = _first_value(medqa_source, "context", "exp", "explanation", "Explanation") or ""
    elif kind == "medmcqa":
        choices = _choices_from_medmcqa(source)
        answer = _first_value(source, "cop", "answer", "answer_idx", "answer_index")
        question = _first_value(source, "question", "Question")
        context = _first_value(source, "exp", "explanation", "context") or ""
    elif kind == "pubmedqa":
        question = _first_value(source, "QUESTION", "question")
        answer = _first_value(source, "final_decision", "FINAL_DECISION", "label", "answer")
        context = _pubmed_context(source)
        choices = ["yes", "no", "maybe"]
    elif kind == "mmlu-medical":
        question = _first_value(source, "question", "Question")
        answer = _first_value(source, "answer", "Answer")
        context = _first_value(source, "context", "explanation") or ""
        choices = source.get("choices")
        if not isinstance(choices, list):
            choices = list(choices.values()) if isinstance(choices, dict) else []
    else:
        raise RemoteDatasetError(f"unsupported remote Q9 dataset {dataset_id!r}")

    source_id = _first_value(source, "id", "qid", "pubid", "uid")
    if source_id is None or not str(source_id).strip():
        source_id = f"{dataset_id.casefold()}_{config}_{offset:06d}"
    return {
        "id": str(source_id),
        "question": str(question or ""),
        "answer": answer,
        "context": context,
        "choices": choices,
        "source": dataset_id,
        "raw": source,
        "remote": {"dataset": dataset_id, "config": config, "split_offset": offset},
    }


def _fetch_rows(
    endpoint: str,
    *,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    query = urllib.parse.urlencode({
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    })
    url = f"{endpoint}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "openmas-bench-q9/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        reason = getattr(exc, "reason", None) or str(exc)
        raise RemoteDatasetError(
            f"could not fetch one remote row from {url}: {reason}. "
            "Check HTTPS/proxy access to datasets-server.huggingface.co or use --source local."
        ) from exc
    if not isinstance(payload, dict):
        raise RemoteDatasetError(f"remote rows API returned a non-object payload: {url}")
    if payload.get("error"):
        raise RemoteDatasetError(f"remote rows API error for {url}: {payload['error']}")
    return payload


def _first_value(source: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = source.get(key)
        if value is not None:
            return value
    return None


def _choices_from_medqa(source: dict[str, Any]) -> list[Any]:
    options = source.get("options") or source.get("Options")
    if isinstance(options, dict):
        return [value for _, value in sorted(options.items(), key=lambda item: str(item[0]))]
    if isinstance(options, list):
        return options
    return _choice_columns(source)


def _choices_from_medmcqa(source: dict[str, Any]) -> list[Any]:
    options = source.get("options")
    if isinstance(options, dict):
        return [value for _, value in sorted(options.items(), key=lambda item: str(item[0]))]
    if isinstance(options, list):
        return options
    return _choice_columns(source)


def _choice_columns(source: dict[str, Any]) -> list[Any]:
    values = [source.get(key) for key in ("opa", "opb", "opc", "opd")]
    return [value for value in values if value is not None]


def _pubmed_context(source: dict[str, Any]) -> str:
    context = _first_value(source, "CONTEXTS", "contexts", "context")
    if isinstance(context, dict):
        values = context.get("contexts") or context.get("sentences") or []
        if isinstance(values, list):
            return "\n".join(str(value) for value in values if str(value).strip())
        return str(values or "")
    if isinstance(context, list):
        return "\n".join(str(value) for value in context if str(value).strip())
    return str(context or "")
