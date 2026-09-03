import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from openmas_bench.dataset_adapters import DATASET_ADAPTERS
from openmas_bench.dataset_cases import build_dataset_case
from openmas_bench.llm import DeterministicAdapter
from openmas_bench.application_executor import _branch_resources
from openmas_bench.q1_real import Q1_REAL_BASELINES, build_q1_real_construction, harness_necessity_score
from scripts.q1.run_q1_real_qa import DATASET_PRESETS


def test_q1_real_baselines_are_harness_level_paradigms():
    assert set(Q1_REAL_BASELINES) == {
        "direct_llm_answering",
        "prompt_only_planning",
        "component_based_assembly",
        "plan_based_construction",
        "workflow_based_construction",
        "graph_harness",
    }
    assert Q1_REAL_BASELINES["graph_harness"].layer == "harness_layer"
    assert Q1_REAL_BASELINES["direct_llm_answering"].layer == "no_harness"


def test_q1_real_dataset_presets_prioritize_structure_sensitive_tasks():
    structural = DATASET_PRESETS["structural"]
    assert structural[:4] == ["hotpotqa", "musique", "medqa", "pubmedqa"]
    assert "gsm8k" not in structural
    assert "strategyqa" not in structural


def test_q1_real_constructs_every_baseline_for_qa_case():
    adapter = DATASET_ADAPTERS["gsm8k"]
    row = {"id": "q1-real-regression", "question": "What is 2 plus 2?", "context": "", "answer": "4"}
    case = build_dataset_case(adapter, row, 0)
    llm = DeterministicAdapter()
    for baseline in Q1_REAL_BASELINES:
        result = build_q1_real_construction(case, baseline, llm, seed=11)
        result.validate(case.request())
        assert result.method == baseline
        assert result.application.nodes


def test_q1_real_graph_harness_gets_resource_access_but_direct_baselines_do_not():
    adapter = DATASET_ADAPTERS["hotpotqa"]
    row = {
        "id": "q1-resource-access-regression",
        "question": "Which entity is supported?",
        "context": "The supporting facts point to Ada.",
        "answer": "Ada",
    }
    case = build_dataset_case(adapter, row, 0)
    llm = DeterministicAdapter()
    graph = build_q1_real_construction(case, "graph_harness", llm, seed=11)
    component = build_q1_real_construction(case, "component_based_assembly", llm, seed=11)
    assert all(node.config.get("resource_access") is True for node in graph.application.nodes)
    assert not any(node.config.get("resource_access") for node in component.application.nodes)


def test_q1_real_strategyqa_branch_resources_are_materialized():
    row = {
        "id": "q1-strategyqa-resource-regression",
        "question": "Is there a full Neptunian orbit between the first two burials of women in the Panth\u00e9on?",
        "context": "",
        "answer": "no",
        "raw": {
            "facts": ["Fact A", "Fact B", "Fact C", "Fact D"],
            "decomposition": ["Step 1", "Step 2", "Step 3"],
            "description": "Reason over the supporting facts.",
        },
    }
    resources = _branch_resources("StrategyQA", row)
    assert "branch_0" in resources and "branch_1" in resources
    assert resources["branch_0"]
    assert resources["branch_1"]


def test_q1_real_direct_answering_rejects_plan_artifacts():
    adapter = DATASET_ADAPTERS["gsm8k"]
    row = {"id": "q1-direct-regression", "question": "What is 2 plus 2?", "context": "", "answer": "4"}
    case = build_dataset_case(adapter, row, 0)
    llm = DeterministicAdapter()
    direct = build_q1_real_construction(case, "direct_llm_answering", llm, seed=11)
    instruction = direct.application.nodes[0].config["execution_instruction"]
    assert "Do not emit a plan" in instruction
    assert "No plan artifact" in direct.application.nodes[0].config["artifact_contract"]


def test_harness_necessity_score_combines_e2e_and_structure():
    record = {
        "e2e_success": 0.5,
        "construction": {
            "architecture_validity": 0.8,
            "constraint_satisfaction": 1.0,
        },
    }
    assert harness_necessity_score(record) == 0.65
