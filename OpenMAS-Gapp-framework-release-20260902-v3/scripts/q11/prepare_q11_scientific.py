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


QUESTION_ID = "Q11"
DOMAIN_SLUG = "q11_scientific"
DOMAIN_TITLE = "Scientific Discovery"
CASE_STUDY = "Scientific Discovery Application Case Study"
QUESTION_DESCRIPTION = (
    "Can Graph Harness construct a domain-specific scientific research MAS "
    "application by organizing retrieval, reasoning, simulation and validation?"
)


PROFILES = {
    "hotpotqa": {
        "dataset": "HotpotQA",
        "source_dataset": "hotpotqa/hotpotqa",
        "split": "validation",
        "license": "CC BY-SA 4.0",
        "download_url": "https://huggingface.co/datasets/hotpotqa/hotpotqa",
        "domain": "scientific_discovery",
        "task_family": "scientific_application_case",
        "problem_type": "multi_hop_literature_bridge",
        "domain_focus": "scientific literature bridge reasoning and citation validation",
        "metric": "F1 / EM",
        "tasks": [
            {"id": "literature_retrieval_a", "objective": "Retrieve first-hop scientific evidence"},
            {"id": "literature_retrieval_b", "objective": "Retrieve second-hop scientific evidence"},
            {"id": "evidence_synthesis", "objective": "Synthesize evidence across literature hops"},
            {"id": "hypothesis_reasoning", "objective": "Reason over the candidate scientific finding"},
            {"id": "validation_check", "objective": "Validate support and detect unsupported claims"},
            {"id": "final_discovery_report", "objective": "Return the scientific answer with traceable support"},
        ],
        "edges": [
            {"source": "literature_retrieval_a", "target": "evidence_synthesis", "relation": "precedes"},
            {"source": "literature_retrieval_b", "target": "evidence_synthesis", "relation": "precedes"},
            {"source": "evidence_synthesis", "target": "hypothesis_reasoning", "relation": "requires"},
            {"source": "hypothesis_reasoning", "target": "validation_check", "relation": "reviews"},
            {"source": "validation_check", "target": "final_discovery_report", "relation": "precedes"},
            {"source": "validation_check", "target": "evidence_synthesis", "relation": "feedback"},
        ],
        "resources": [
            {
                "id": "resource_first_hop_literature",
                "task_id": "literature_retrieval_a",
                "resource_key": "first_hop_literature",
                "description": "First-hop supporting paragraph or paper evidence",
            },
            {
                "id": "resource_second_hop_literature",
                "task_id": "literature_retrieval_b",
                "resource_key": "second_hop_literature",
                "description": "Second-hop supporting paragraph or paper evidence",
            },
        ],
        "constraints": [
            {
                "id": "citation_traceability",
                "kind": "evidence_trace",
                "target": "final_discovery_report",
                "predicate": "required",
            },
            {
                "id": "claim_support_check",
                "kind": "scientific_validation",
                "target": "final_discovery_report",
                "predicate": "required",
            },
        ],
    },
    "musique": {
        "dataset": "MuSiQue",
        "source_dataset": "StonyBrookNLP/musique",
        "split": "validation",
        "license": "CC BY 4.0",
        "download_url": "https://huggingface.co/datasets/StonyBrookNLP/musique",
        "domain": "scientific_discovery",
        "task_family": "scientific_application_case",
        "problem_type": "controlled_multi_hop_retrieval_reasoning",
        "domain_focus": "controlled decomposition, evidence chain reasoning and validation",
        "metric": "F1 / EM",
        "tasks": [
            {"id": "question_decomposition", "objective": "Decompose the research question into linked subgoals"},
            {"id": "support_chain_retrieval", "objective": "Retrieve the annotated support chain"},
            {"id": "bridge_reasoning", "objective": "Bridge evidence across controlled hops"},
            {"id": "contradiction_filter", "objective": "Reject distractor or unanswerable evidence"},
            {"id": "experiment_design", "objective": "Map the reasoning result to an experimental check"},
            {"id": "validation_check", "objective": "Validate chain consistency and answer support"},
            {"id": "final_discovery_report", "objective": "Return the supported scientific finding"},
        ],
        "edges": [
            {"source": "question_decomposition", "target": "support_chain_retrieval", "relation": "precedes"},
            {"source": "support_chain_retrieval", "target": "bridge_reasoning", "relation": "precedes"},
            {"source": "bridge_reasoning", "target": "contradiction_filter", "relation": "reviews"},
            {"source": "bridge_reasoning", "target": "experiment_design", "relation": "requires"},
            {"source": "contradiction_filter", "target": "validation_check", "relation": "precedes"},
            {"source": "experiment_design", "target": "validation_check", "relation": "precedes"},
            {"source": "validation_check", "target": "final_discovery_report", "relation": "precedes"},
        ],
        "resources": [
            {
                "id": "resource_decomposition",
                "task_id": "question_decomposition",
                "resource_key": "question_decomposition",
                "description": "Controlled decomposition annotations",
            },
            {
                "id": "resource_support_paragraphs",
                "task_id": "support_chain_retrieval",
                "resource_key": "support_paragraphs",
                "description": "Linked support paragraphs and distractors",
            },
        ],
        "constraints": [
            {
                "id": "chain_consistency",
                "kind": "reasoning_chain",
                "target": "final_discovery_report",
                "predicate": "required",
            },
            {
                "id": "verification_gate",
                "kind": "scientific_validation",
                "target": "final_discovery_report",
                "predicate": "required",
            },
        ],
    },
    "drop": {
        "dataset": "DROP",
        "source_dataset": "ucinlp/drop",
        "split": "validation",
        "license": "CC BY-SA 4.0",
        "download_url": "https://huggingface.co/datasets/ucinlp/drop",
        "domain": "scientific_discovery",
        "task_family": "scientific_application_case",
        "problem_type": "reading_numeric_experiment_reasoning",
        "domain_focus": "evidence extraction, discrete computation and reproducibility checks",
        "metric": "F1 / EM",
        "tasks": [
            {"id": "passage_evidence", "objective": "Retrieve relevant passage evidence"},
            {"id": "variable_extraction", "objective": "Extract quantities, dates or entities"},
            {"id": "numeric_operation", "objective": "Perform the required discrete operation"},
            {"id": "simulation_check", "objective": "Replay the computation as a lightweight simulation"},
            {"id": "result_validation", "objective": "Validate units, operation and evidence lineage"},
            {"id": "final_discovery_report", "objective": "Return the validated result"},
        ],
        "edges": [
            {"source": "passage_evidence", "target": "variable_extraction", "relation": "precedes"},
            {"source": "variable_extraction", "target": "numeric_operation", "relation": "precedes"},
            {"source": "numeric_operation", "target": "simulation_check", "relation": "requires"},
            {"source": "simulation_check", "target": "result_validation", "relation": "reviews"},
            {"source": "result_validation", "target": "final_discovery_report", "relation": "precedes"},
            {"source": "result_validation", "target": "numeric_operation", "relation": "feedback"},
        ],
        "resources": [
            {
                "id": "resource_passage",
                "task_id": "passage_evidence",
                "resource_key": "passage_evidence",
                "description": "Source passage used for scientific-style evidence extraction",
            },
            {
                "id": "resource_numeric_program",
                "task_id": "numeric_operation",
                "resource_key": "numeric_program",
                "description": "Discrete operation and calculation trace",
            },
        ],
        "constraints": [
            {
                "id": "unit_consistency",
                "kind": "calculation_audit",
                "target": "final_discovery_report",
                "predicate": "required",
            },
            {
                "id": "reproducibility_review",
                "kind": "scientific_validation",
                "target": "final_discovery_report",
                "predicate": "required",
            },
        ],
    },
}


ROWS = {
    "hotpotqa": [
        {
            "id": "q11_hotpotqa_000",
            "question": "Which research area links Rosalind Franklin's X-ray diffraction work to the model reported by Watson and Crick?",
            "answer": "molecular biology",
            "context": "Franklin produced X-ray diffraction images of DNA. Watson and Crick proposed the double-helix structure of DNA, a central result in molecular biology.",
            "choices": [],
            "source": "hotpotqa",
            "raw": {
                "type": "bridge",
                "supporting_facts": [["Rosalind Franklin", 0], ["DNA", 1]],
                "context": [
                    ["Rosalind Franklin", ["Franklin produced X-ray diffraction images of DNA."]],
                    ["DNA", ["The DNA double helix became a central model in molecular biology."]],
                ],
            },
        },
        {
            "id": "q11_hotpotqa_001",
            "question": "What field studies both the greenhouse effect and the atmospheric measurements used to validate it?",
            "answer": "climate science",
            "context": "The greenhouse effect concerns atmospheric heat retention. Instrumental atmospheric measurements are used to validate climate models.",
            "choices": [],
            "source": "hotpotqa",
            "raw": {"type": "bridge", "supporting_facts": [["Greenhouse effect", 0], ["Climate model", 0]]},
        },
        {
            "id": "q11_hotpotqa_002",
            "question": "What discipline connects protein folding studies with X-ray crystallography evidence?",
            "answer": "structural biology",
            "context": "Protein folding concerns three-dimensional biomolecular structure. X-ray crystallography provides structural evidence for proteins.",
            "choices": [],
            "source": "hotpotqa",
            "raw": {"type": "bridge", "supporting_facts": [["Protein folding", 0], ["X-ray crystallography", 0]]},
        },
    ],
    "musique": [
        {
            "id": "q11_musique_000",
            "question": "The method used to infer a star's chemical composition relies on what type of observed signal?",
            "answer": "spectrum",
            "context": "Spectroscopy separates light into a spectrum. Astronomers infer stellar chemical composition from absorption and emission lines in spectra.",
            "choices": [],
            "source": "musique",
            "raw": {
                "answerable": True,
                "question_decomposition": [
                    {"question": "Which method infers stellar composition?", "paragraph_support_idx": 0},
                    {"question": "What signal does spectroscopy analyze?", "paragraph_support_idx": 1},
                ],
            },
        },
        {
            "id": "q11_musique_001",
            "question": "The experiment that verifies enzyme activity usually tracks the concentration change of what?",
            "answer": "substrate or product",
            "context": "Enzyme assays measure reaction progress. Reaction progress is observed as substrate depletion or product formation over time.",
            "choices": [],
            "source": "musique",
            "raw": {
                "answerable": True,
                "question_decomposition": [
                    {"question": "What does an enzyme assay measure?", "paragraph_support_idx": 0},
                    {"question": "How is reaction progress observed?", "paragraph_support_idx": 1},
                ],
            },
        },
        {
            "id": "q11_musique_002",
            "question": "The proxy used to estimate past temperature from ice cores is commonly measured in what material?",
            "answer": "trapped gases and ice isotopes",
            "context": "Ice cores preserve trapped gases and isotope ratios. Paleoclimate studies use these measurements to infer past temperature.",
            "choices": [],
            "source": "musique",
            "raw": {
                "answerable": True,
                "question_decomposition": [
                    {"question": "What do ice cores preserve?", "paragraph_support_idx": 0},
                    {"question": "What do paleoclimate studies infer?", "paragraph_support_idx": 1},
                ],
            },
        },
    ],
    "drop": [
        {
            "id": "q11_drop_000",
            "question": "If Trial A produced 42 colonies and Trial B produced 57 colonies, how many more colonies were observed in Trial B?",
            "answer": "15",
            "context": "A lab notebook reports Trial A with 42 colonies and Trial B with 57 colonies after incubation.",
            "choices": [],
            "source": "drop",
            "raw": {"answer_type": "number", "operation": "subtract(57, 42)"},
        },
        {
            "id": "q11_drop_001",
            "question": "A sensor recorded 12, 15, and 18 events across three runs. What was the total event count?",
            "answer": "45",
            "context": "The passage lists three experimental runs with 12, 15, and 18 detected events.",
            "choices": [],
            "source": "drop",
            "raw": {"answer_type": "number", "operation": "sum(12, 15, 18)"},
        },
        {
            "id": "q11_drop_002",
            "question": "The baseline measurement was 80 ms and the optimized measurement was 65 ms. What was the reduction in milliseconds?",
            "answer": "15",
            "context": "The experiment compares a baseline of 80 ms with an optimized condition of 65 ms.",
            "choices": [],
            "source": "drop",
            "raw": {"answer_type": "number", "operation": "subtract(80, 65)"},
        },
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Q11 scientific-discovery harness pilots and figures.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--datasets", default="hotpotqa,musique,drop")
    parser.add_argument("--rows-per-dataset", type=int, default=3)
    parser.add_argument("--render-rows", type=int, default=1)
    args = parser.parse_args()

    selected = [dataset_key(item) for item in args.datasets.split(",") if item.strip()]
    rows_by_dataset = {key: ROWS[key][: args.rows_per_dataset] for key in selected}
    profiles = {key: PROFILES[key] for key in selected}
    dataset_root = args.data_root / "q11_datasets"
    output_root = args.data_root / "outputs" / "q11_scientific"
    write_dataset_space(
        dataset_root=dataset_root,
        raw_subdir="scientific",
        schema_version="q11_scientific_v1",
        purpose="Graph Harness scientific discovery MAS application case study",
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
            figure_path = output_root / "figures" / f"Q11_{profile['dataset']}_harness.png"
            evolution_path = output_root / "figures" / f"Q11_{profile['dataset']}_evolution.png"
            write_json(record_path, record)
            render_harness_png(record, figure_path)
            render_evolution_png(record, evolution_path)
    print(f"prepared Q11 datasets under {dataset_root}")


if __name__ == "__main__":
    main()
