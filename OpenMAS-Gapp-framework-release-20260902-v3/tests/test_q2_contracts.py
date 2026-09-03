import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openmas_bench.ablation import get_ablation_method
from openmas_bench.dataset_adapters import (
    DATASET_ADAPTERS, MultipleChoiceAdapter, TextF1Adapter, DropAdapter,
    FinQAAdapter, SciBenchAdapter,
)
from openmas_bench.dataset_cases import build_dataset_case, load_normalized_rows
from openmas_bench.llm import DeterministicAdapter
from openmas_bench.application_executor import ApplicationTaskExecutor, _branch_resources
from openmas_bench.llm import AdapterResponse, LLMAdapter, LLMConfig


def test_choice_accepts_letter_and_text():
    adapter = MultipleChoiceAdapter()
    assert adapter.score("A", "choice:a|oxidants") == 1.0
    assert adapter.score("oxidants", "choice:a|oxidants") == 1.0
    assert adapter.score("B", "choice:a|oxidants") == 0.0
    assert adapter.score("D. Cross-linking of DNA", "choice:d|Cross-linking of DNA") == 1.0
    assert adapter.score("The answer is option D because cisplatin cross-links DNA.",
                         "choice:d|Cross-linking of DNA") == 1.0
    assert adapter.score("B. Hyperstabilization of microtubules",
                         "choice:d|Cross-linking of DNA") == 0.0


def test_token_f1_counts_duplicate_tokens():
    assert TextF1Adapter().score("a a b", "a b b") == 2 / 3
    assert DropAdapter().score("a a b", "a b b") == 2 / 3


def test_hotpot_f1_uses_official_answer_normalization():
    adapter = TextF1Adapter()
    assert adapter.score("Yes, both are American.", "yes") == 0.4
    assert adapter.score("The Animorphs", "Animorphs") == 1.0


def test_finqa_accepts_documented_numeric_rounding_but_not_large_errors():
    adapter = FinQAAdapter()
    assert adapter.score("41932.2", "41932") == 1.0
    assert adapter.score("41930", "41932") == 0.0
    assert adapter.score("100", "") is None


def test_scibench_scores_scientific_notation_tolerance_and_units():
    adapter = SciBenchAdapter()
    gold = {"value": "50.7", "unit": "$\\mathrm{atm}$"}
    assert adapter.score("50.70 atm", gold) == 1.0
    assert adapter.score(r"5.07 x 10^{1} \mathrm{atm}", gold) == 1.0
    assert adapter.score("50.8 atm", gold) == 1.0
    assert adapter.score("50.7 kPa", gold) == 0.0
    assert adapter.score("51.1 atm", gold) == 0.0
    assert _branch_resources("SciBench", {
        "question": "science problem", "raw": {"unit": "atm", "source": "atkins"},
    }) == {}


def test_financebench_routes_independent_evidence_records_only():
    row = {
        "question": "Reconcile the two filing disclosures.",
        "raw": {
            "question_reasoning": "Numerical reasoning",
            "evidence": [
                {"evidence_text": "First filing disclosure: revenue was 10."},
                {"evidence_text": "Second filing disclosure: costs were 4."},
                {"evidence_text": "Third filing disclosure: tax was 1."},
            ],
        },
    }
    resources = _branch_resources("FinanceBench", row)
    assert set(resources) == {"branch_0", "branch_1"}
    assert "EVIDENCE_1" in resources["branch_0"]
    assert "EVIDENCE_3" in resources["branch_0"]
    assert "EVIDENCE_2" in resources["branch_1"]
    assert "FINANCEBENCH_REASONING" not in " ".join(resources.values())


def test_bbh_task_aware_split_preserves_fact_and_option_boundaries():
    row = {
        "question": "ordering task",
        "raw": {
            "task": "logical_deduction_three_objects",
            "input": (
                "There are three objects. The falcon is right of the jay. "
                "The jay is right of the quail.\nOptions:\n(A) jay\n(B) quail\n(C) falcon"
            ),
        },
    }
    resources = _branch_resources("BBH-Full", row)
    assert set(resources) == {"branch_0", "branch_1"}
    assert "The falcon is right of the jay." in resources["branch_0"]
    assert "The jay is right of the quail." in resources["branch_1"]
    assert "Options:\n(A) jay" in resources["branch_1"]
    assert all("logical_deduction_three_objects" in value for value in resources.values())


def test_full_repairs_arg_that_omits_one_bbh_resource_branch():
    case = _case("bbh_full")
    llm = DeterministicAdapter()
    method = get_ablation_method("full_graph_harness", llm, seed=11)

    def omit_first_branch(system_prompt, user_prompt, seed, required_fields=None):
        from openmas_bench.llm import AdapterResponse
        harness = case.harness
        value = {
            "tasks": [n.id for n in harness.nodes if n.kind == "task_pattern" and n.id != "parse"],
            "capabilities": [n.id for n in harness.nodes if n.kind == "capability"],
            "components": [n.id for n in harness.nodes if n.kind == "component"
                           and n.risk != "high" and n.id != "component_parse"],
            "constraints": [n.id for n in harness.nodes if n.kind == "constraint"],
            "relations": [{"source": e.source, "target": e.target} for e in harness.edges],
        }
        return AdapterResponse(value, "test", "test", seed, 1, 1, 0.0)

    llm.generate_json = omit_first_branch
    result = method.construct(case.request())
    bindings = {key for node in result.application.nodes
                for key in node.config.get("resource_bindings", [])}
    assert bindings == {"branch_0", "branch_1"}
    assert result.blueprint.metadata["multibranch_grounding_repair"] is True
    assert result.blueprint.metadata["gold_used"] is False


def _case(dataset_key):
    root = Path(__file__).resolve().parents[3]
    adapter = DATASET_ADAPTERS[dataset_key]
    if not (root / adapter.source_file).exists():
        pytest.skip(f"{adapter.dataset_id} fixture is external to the source distribution")
    row = load_normalized_rows(root, adapter, 1)[0]
    return build_dataset_case(adapter, row, 0)


def test_multibranch_full_converges_but_graph_ablation_is_linear():
    case = _case("hotpotqa")
    full = get_ablation_method("full_graph_harness", DeterministicAdapter()).construct(case.request())
    without = get_ablation_method("w/o_graph_orchestration", DeterministicAdapter()).construct(case.request())
    full_edges = {(edge.source, edge.target) for edge in full.application.edges}
    assert ("inst_req_component_retrieve_a", "inst_req_component_synthesize") in full_edges
    assert ("inst_req_component_retrieve_b", "inst_req_component_synthesize") in full_edges
    assert set(full.application.entrypoints) == {"inst_req_component_retrieve_a", "inst_req_component_retrieve_b"}
    without_edges = {(edge.source, edge.target) for edge in without.application.edges}
    assert ("inst_req_component_retrieve_a", "inst_req_component_retrieve_b") in without_edges
    assert len(without.application.entrypoints) == 1


def test_without_blueprint_is_direct_mas_not_hidden_realization():
    case = _case("gsm8k")
    result = get_ablation_method("w/o_blueprint", DeterministicAdapter()).construct(case.request())
    assert result.blueprint.metadata["carrier_only"] is True
    assert result.application.metadata["uses_blueprint_realization"] is False
    assert result.application.metadata["construction_mode"] == "direct_mas"
    assert all(node.id.startswith("direct_stage_") for node in result.application.nodes)
    assert all(node.description == "opaque direct MAS stage" for node in result.blueprint.nodes)
    assert all("CANDIDATE_ANSWER" in node.config["artifact_contract"]
               for node in result.application.nodes)
    assert all("Blueprint" not in node.config["artifact_contract"]
               for node in result.application.nodes)


def test_medqa_choice_is_canonical_and_constraint_precedes_answer():
    case = _case("medqa")
    assert str(case.execution_tasks[0].answer).startswith("choice:")
    full = get_ablation_method("full_graph_harness", DeterministicAdapter()).construct(case.request())
    order, _ = __import__("openmas_bench.application_executor", fromlist=["_execution_order"])._execution_order(
        full.application.nodes, full.application.edges)
    roles = [next(node for node in full.application.nodes if node.id == node_id)
             .config["execution_instruction"] for node_id in order]
    assert "enforc" in roles[-2].lower()
    assert "return one option" in roles[-1].lower()


def test_graph_ablation_keeps_constraint_stage_but_removes_typed_dependencies():
    case = _case("medqa")
    result = get_ablation_method("w/o_graph_orchestration", DeterministicAdapter()).construct(case.request())
    assert any(node.kind == "control" for node in result.application.nodes)
    assert len(result.application.nodes) == 6


def test_medqa_variants_keep_answer_terminal_and_equal_runtime_budget():
    case = _case("medqa")
    for variant in ("w/o_requirement_grounding", "w/o_constraint_aware_orchestration"):
        result = get_ablation_method(variant, DeterministicAdapter()).construct(case.request())
        order, _ = __import__("openmas_bench.application_executor", fromlist=["_execution_order"])._execution_order(
            result.application.nodes, result.application.edges)
        terminal = next(node for node in result.application.nodes if node.id == order[-1])
        assert "return one option" in terminal.config["execution_instruction"].lower()
        assert len(result.application.nodes) == 6


class _CaptureAdapter(LLMAdapter):
    def __init__(self):
        self.config = LLMConfig()
        self.calls = []

    def generate_json(self, system_prompt, user_prompt, seed, required_fields=None):
        import json
        self.calls.append(json.loads(user_prompt))
        field = next(iter(required_fields or {"artifact"}))
        return AdapterResponse({field: "captured"}, "capture", "capture", seed, 1, 1, 0.0)


def test_full_multibranch_gives_task_to_both_entries_and_merges_artifacts():
    case = _case("hotpotqa")
    dataset = DATASET_ADAPTERS["hotpotqa"]
    construction = get_ablation_method("full_graph_harness", DeterministicAdapter()).construct(case.request())
    capture = _CaptureAdapter()
    root = Path(__file__).resolve().parents[3]
    row = load_normalized_rows(root, dataset, 1)[0]
    ApplicationTaskExecutor(capture, root).execute(dataset, row, case, construction, 0)
    visible = [call for call in capture.calls if call["task"].get("question")]
    assert len(visible) == 2
    merge = next(call for call in capture.calls if call["role"] == "Dataset component for synthesize both evidence streams")
    assert len(merge["upstream_artifacts"]) == 2
