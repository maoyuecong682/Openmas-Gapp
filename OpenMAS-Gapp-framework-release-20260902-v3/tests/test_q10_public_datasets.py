from __future__ import annotations

from openmas_bench.application_executor import _branch_resources
from openmas_bench.dataset_adapters import DATASET_ADAPTERS
from openmas_bench.dataset_cases import _is_qualified_row
from scripts.q10.prepare_q10_financial import normalize_public_row
from scripts.q10.qualify_q10_datasets import check_row


def test_normalize_public_financebench_row_keeps_filing_contract():
    row = normalize_public_row(
        "financebench",
        {
            "financebench_id": "financebench_id_03029",
            "question": "What is capex?",
            "answer": "$1577.00",
            "question_reasoning": "Information extraction",
            "justification": "Extracted from the cash flow statement.",
            "evidence": [{"evidence_text": "Purchases of property, plant and equipment (PP&E) (1,577)."}],
        },
        offset=0,
        spec={"dataset": "PatronusAI/financebench", "config": "default", "split": "train"},
    )

    assert row["id"] == "financebench_id_03029"
    assert row["context"]
    assert row["remote"]["split_offset"] == 0
    assert check_row("financebench", row) == []
    assert _is_qualified_row(DATASET_ADAPTERS["financebench"], row)
    assert set(_branch_resources("FinanceBench", row)) == {"branch_0", "branch_1"}


def test_normalize_public_finqa_row_uses_gold_evidence_as_audit_metadata():
    row = normalize_public_row(
        "finqa",
        {
            "id": "ETR/2016/page_23.pdf-2",
            "pre_text": ["amount (in millions)"],
            "table": [["", "amount"], ["2014 net revenue", "$ 5735"], ["2015 net revenue", "$ 5829"]],
            "post_text": ["net change is the difference between 2015 and 2014."],
            "question": "what is the net change?",
            "answer": "94",
            "gold_evidence": ["2014 net revenue is $5735", "2015 net revenue is $5829"],
        },
        offset=0,
        spec={"dataset": "dreamerdeo/finqa", "config": "default", "split": "test"},
    )

    assert row["raw"]["context"]["table"][2][1] == "$ 5829"
    assert row["raw"]["metadata"]["gold_evidence"]
    assert row["remote"]["dataset"] == "dreamerdeo/finqa"
    assert check_row("finqa", row) == []
    assert _is_qualified_row(DATASET_ADAPTERS["finqa"], row)
    assert "2015 net revenue" in _branch_resources("FinQA", row)["branch_0"]
