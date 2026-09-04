from __future__ import annotations

import pytest

from openmas_bench.dataset_adapters import DATASET_ADAPTERS
from openmas_bench.dataset_cases import _is_qualified_row
from openmas_bench.remote_datasets import (
    RemoteDatasetError,
    load_remote_row,
    normalize_remote_row,
)


def test_normalize_medqa_row_keeps_multiple_choice_contract():
    row = normalize_remote_row(
        "MedQA",
        {"question": "Which finding is expected?", "options": {"A": "alpha", "B": "beta"}, "answer_idx": 1},
        config="med_qa_en_4options_source",
        offset=4,
    )
    assert row["choices"] == ["alpha", "beta"]
    assert row["answer"] == 1
    assert row["remote"]["split_offset"] == 4


def test_normalize_medmcqa_row_keeps_option_columns():
    row = normalize_remote_row(
        "MedMCQA",
        {"id": "7", "question": "Which option?", "opa": "A", "opb": "B", "opc": "C", "opd": "D", "cop": 2},
        config="default",
        offset=0,
    )
    assert row["id"] == "7"
    assert row["choices"] == ["A", "B", "C", "D"]
    assert row["answer"] == 2


def test_normalize_pubmedqa_row_preserves_context_and_label():
    row = normalize_remote_row(
        "PubMedQA",
        {"pubid": "123", "QUESTION": "Does treatment help?", "CONTEXTS": ["First abstract.", "Second abstract."], "final_decision": "yes"},
        config="pqa_labeled",
        offset=1,
    )
    assert row["id"] == "123"
    assert row["context"] == "First abstract.\nSecond abstract."
    assert row["answer"] == "yes"
    assert row["choices"] == ["yes", "no", "maybe"]


def test_normalize_mmlu_medical_row_preserves_answer_index():
    row = normalize_remote_row(
        "MMLU-Medical",
        {"question": "Medical question", "choices": ["A", "B", "C", "D"], "answer": 3},
        config="anatomy",
        offset=2,
    )
    assert row["choices"] == ["A", "B", "C", "D"]
    assert row["answer"] == 3


def test_choice_answer_zero_is_a_qualified_row():
    row = {"id": "zero", "question": "Which option?", "answer": 0, "context": "", "choices": ["A", "B"], "raw": {}}
    assert _is_qualified_row(DATASET_ADAPTERS["medqa"], row)


def test_remote_loader_requests_only_the_selected_row(monkeypatch):
    seen_urls = []

    def fake_fetch(endpoint, **kwargs):
        seen_urls.append((endpoint, kwargs))
        return {
            "num_rows_total": 1273,
            "rows": [{"row": {"question": "One question", "options": {"A": "one", "B": "two"}, "answer_idx": 0}}],
        }

    monkeypatch.setattr("openmas_bench.remote_datasets._fetch_rows", fake_fetch)
    row = load_remote_row(DATASET_ADAPTERS["medqa"], 29)
    assert row["question"] == "One question"
    assert len(seen_urls) == 1
    _, request = seen_urls[0]
    assert request == {
        "dataset": "openlifescienceai/medqa",
        "config": "med_qa_en_4options_source",
        "split": "test",
        "offset": 29,
        "length": 1,
        "timeout_seconds": 30,
    }


def test_remote_loader_surfaces_network_problem_without_local_fallback(monkeypatch):
    def fail_fetch(*args, **kwargs):
        raise RemoteDatasetError("network unavailable")

    monkeypatch.setattr("openmas_bench.remote_datasets._fetch_rows", fail_fetch)
    with pytest.raises(RemoteDatasetError, match="network unavailable"):
        load_remote_row(DATASET_ADAPTERS["pubmedqa"], 0)
