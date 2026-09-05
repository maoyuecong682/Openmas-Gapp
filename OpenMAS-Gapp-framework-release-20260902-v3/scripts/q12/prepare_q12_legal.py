from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from domain_harness_common import (  # noqa: E402
    build_domain_record,
    dataset_key,
    render_evolution_png,
    render_harness_png,
    write_dataset_space,
    write_json,
)


QUESTION_ID = "Q12"
DOMAIN_SLUG = "q12_legal"
DOMAIN_TITLE = "Legal Application"
CASE_STUDY = "Legal Application Case Study"
QUESTION_DESCRIPTION = (
    "Can Graph Harness construct a domain-specific legal MAS application by "
    "organizing legal retrieval, case analysis, statute interpretation, and "
    "compliance review?"
)


PROFILES = {
    "casehold": {
        "dataset": "CaseHOLD",
        "source_dataset": "casehold/casehold",
        "split": "validation",
        "license": "unverified",
        "download_url": "https://huggingface.co/datasets/casehold/casehold",
        "domain": "legal_application",
        "task_family": "legal_application_case",
        "problem_type": "case_citation_selection",
        "domain_focus": "case citation retrieval, holding identification and legal justification",
        "metric": "Accuracy",
        "tasks": [
            {"id": "opinion_retrieval", "objective": "Retrieve the relevant judicial opinion"},
            {"id": "citation_matching", "objective": "Match the cited holding to the legal issue"},
            {"id": "ratio_analysis", "objective": "Analyze the ratio decidendi and precedent strength"},
            {"id": "holding_selection", "objective": "Select the best legal holding"},
            {"id": "review_gate", "objective": "Apply professional legal review"},
            {"id": "final_legal_report", "objective": "Return the selected holding with traceable support"},
        ],
        "edges": [
            {"source": "opinion_retrieval", "target": "citation_matching", "relation": "precedes"},
            {"source": "citation_matching", "target": "ratio_analysis", "relation": "precedes"},
            {"source": "ratio_analysis", "target": "holding_selection", "relation": "requires"},
            {"source": "holding_selection", "target": "review_gate", "relation": "reviews"},
            {"source": "review_gate", "target": "final_legal_report", "relation": "precedes"},
            {"source": "review_gate", "target": "ratio_analysis", "relation": "feedback"},
        ],
        "resources": [
            {
                "id": "resource_case_opinion",
                "task_id": "opinion_retrieval",
                "resource_key": "case_opinion",
                "description": "Opinion text and citation context",
            },
            {
                "id": "resource_holding_bank",
                "task_id": "citation_matching",
                "resource_key": "holding_bank",
                "description": "Holding candidates and citation anchors",
            },
        ],
        "constraints": [
            {
                "id": "precedent_consistency",
                "kind": "legal_consistency",
                "target": "final_legal_report",
                "predicate": "required",
            },
            {
                "id": "professional_review",
                "kind": "legal_review",
                "target": "final_legal_report",
                "predicate": "required",
            },
        ],
    },
    "legalbench": {
        "dataset": "LegalBench",
        "source_dataset": "nguha/legalbench",
        "split": "validation",
        "license": "mixed / task-level",
        "download_url": "https://huggingface.co/datasets/nguha/legalbench",
        "domain": "legal_application",
        "task_family": "legal_application_case",
        "problem_type": "legal_mixed_reasoning",
        "domain_focus": "legal retrieval, case analysis, statutory interpretation and compliance checking",
        "metric": "Task-specific",
        "tasks": [
            {"id": "legal_text_retrieval", "objective": "Retrieve the legal source text or statute"},
            {"id": "case_fact_analysis", "objective": "Analyze the facts and the legal question"},
            {"id": "rule_interpretation", "objective": "Interpret the governing legal rule"},
            {"id": "compliance_check", "objective": "Check consistency with legal constraints"},
            {"id": "review_gate", "objective": "Apply professional legal review"},
            {"id": "final_legal_report", "objective": "Return the legally supported result"},
        ],
        "edges": [
            {"source": "legal_text_retrieval", "target": "case_fact_analysis", "relation": "precedes"},
            {"source": "case_fact_analysis", "target": "rule_interpretation", "relation": "precedes"},
            {"source": "rule_interpretation", "target": "compliance_check", "relation": "requires"},
            {"source": "compliance_check", "target": "review_gate", "relation": "reviews"},
            {"source": "review_gate", "target": "final_legal_report", "relation": "precedes"},
            {"source": "review_gate", "target": "case_fact_analysis", "relation": "feedback"},
        ],
        "resources": [
            {
                "id": "resource_legal_text",
                "task_id": "legal_text_retrieval",
                "resource_key": "legal_text",
                "description": "Relevant legal text or statute excerpt",
            },
            {
                "id": "resource_evidence_chain",
                "task_id": "case_fact_analysis",
                "resource_key": "evidence_chain",
                "description": "Evidence chain and issue spotting notes",
            },
        ],
        "constraints": [
            {
                "id": "evidence_integrity",
                "kind": "legal_evidence",
                "target": "final_legal_report",
                "predicate": "required",
            },
            {
                "id": "compliance_review",
                "kind": "legal_review",
                "target": "final_legal_report",
                "predicate": "required",
            },
        ],
    },
}


ROWS = {
    "casehold": [
        {
            "id": "q12_casehold_000",
            "question": "Which holding best matches a case about the admissibility of business records prepared in the regular course of operations?",
            "answer": "business records prepared in the regular course of operations are admissible when the foundational requirements are met",
            "context": "The opinion discusses hearsay exceptions and the admissibility of business records prepared in the ordinary course of business.",
            "choices": [
                "Business records are admissible when they were prepared in the regular course of operations and the foundation is satisfied.",
                "All business records are inadmissible because they are hearsay.",
                "Only criminal cases may use business records.",
                "Business records are admissible only if signed by a judge.",
                "The rule concerns patents rather than evidence."
            ],
            "source": "casehold",
            "raw": {
                "citation": "Federal Rule of Evidence 803(6)",
                "issue": "business_records",
            },
        },
        {
            "id": "q12_casehold_001",
            "question": "Which holding best matches a dispute about implied warranty and the sale of merchantable goods?",
            "answer": "merchantable goods may carry an implied warranty of fitness and merchantability under the governing code",
            "context": "The opinion concerns contract law, merchantability and implied warranties in the sale of goods.",
            "choices": [
                "Merchantable goods may carry an implied warranty of fitness and merchantability under the governing code.",
                "Warranty claims never apply to goods.",
                "Only tort law governs merchantable goods.",
                "Merchantability applies only to real property.",
                "The case addresses sentencing only."
            ],
            "source": "casehold",
            "raw": {
                "citation": "UCC implied warranty doctrine",
                "issue": "merchantability",
            },
        },
        {
            "id": "q12_casehold_002",
            "question": "Which holding best matches a case discussing summary judgment when no genuine dispute of material fact exists?",
            "answer": "summary judgment is proper when there is no genuine dispute of material fact",
            "context": "The opinion addresses civil procedure and the summary judgment standard.",
            "choices": [
                "Summary judgment is proper when there is no genuine dispute of material fact.",
                "Summary judgment is proper only after trial.",
                "The standard applies only to criminal sentencing.",
                "Summary judgment is never allowed in federal court.",
                "The case is about bankruptcy discharge."
            ],
            "source": "casehold",
            "raw": {
                "citation": "Rule 56",
                "issue": "summary_judgment",
            },
        },
    ],
    "legalbench": [
        {
            "id": "q12_legalbench_000",
            "question": "Does the contract clause require written notice before termination?",
            "answer": "yes",
            "context": "The clause states that a party must provide written notice at least 30 days before termination.",
            "choices": ["yes", "no"],
            "source": "legalbench",
            "raw": {"task": "contract_interpretation", "label": "yes"},
        },
        {
            "id": "q12_legalbench_001",
            "question": "Is the quoted statute compatible with the described evidence preservation duty?",
            "answer": "yes",
            "context": "The quoted statute requires parties to preserve material evidence after litigation is reasonably anticipated.",
            "choices": ["yes", "no"],
            "source": "legalbench",
            "raw": {"task": "statutory_interpretation", "label": "yes"},
        },
        {
            "id": "q12_legalbench_002",
            "question": "Does the fact pattern create a conflict of interest under the professional conduct rule?",
            "answer": "no",
            "context": "The facts show independent representation, disclosed relationships and no adverse client overlap.",
            "choices": ["yes", "no"],
            "source": "legalbench",
            "raw": {"task": "professional_conduct", "label": "no"},
        },
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Q12 legal-harness pilots and figures.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--datasets", default="casehold,legalbench")
    parser.add_argument("--rows-per-dataset", type=int, default=3)
    parser.add_argument("--render-rows", type=int, default=1)
    args = parser.parse_args()

    selected = [dataset_key(item) for item in args.datasets.split(",") if item.strip()]
    rows_by_dataset = {key: ROWS[key][: args.rows_per_dataset] for key in selected}
    profiles = {key: PROFILES[key] for key in selected}
    dataset_root = args.data_root / "q12_datasets"
    output_root = args.data_root / "outputs" / "q12_legal"
    write_dataset_space(
        dataset_root=dataset_root,
        raw_subdir="legal",
        schema_version="q12_legal_v1",
        purpose="Graph Harness legal MAS application case study",
        rows_by_dataset=rows_by_dataset,
        profiles=profiles,
    )
    for key, rows in rows_by_dataset.items():
        for index, row in enumerate(rows[: args.render_rows]):
            profile = profiles[key]
            record = build_domain_record(
                question_id=QUESTION_ID,
                domain_slug=DOMAIN_SLUG,
                domain_title=DOMAIN_TITLE,
                case_study=CASE_STUDY,
                question_description=QUESTION_DESCRIPTION,
                profile=profile,
                row=row,
                row_index=index,
            )
            record_path = output_root / "runs" / f"{key}_row{index}_harness.json"
            figure_path = output_root / "figures" / f"Q12_{profile['dataset']}_harness.png"
            evolution_path = output_root / "figures" / f"Q12_{profile['dataset']}_evolution.png"
            write_json(record_path, record)
            render_harness_png(record, figure_path)
            render_evolution_png(record, evolution_path)
    print(f"prepared Q12 datasets under {dataset_root}")


if __name__ == "__main__":
    main()
