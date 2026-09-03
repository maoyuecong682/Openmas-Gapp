import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from openmas_bench.ablation import Q2_VARIANTS, get_ablation_method
from openmas_bench.application_executor import (
    ApplicationTaskExecutor, _execute_finqa_tool_call, _execution_order,
)
from openmas_bench.dataset_adapters import DATASET_ADAPTERS, NumericAdapter
from openmas_bench.dataset_cases import build_dataset_case
from openmas_bench.llm import AdapterResponse, DeterministicAdapter, LLMAdapter, LLMConfig


class RecordingAdapter(LLMAdapter):
    def __init__(self):
        self.config = LLMConfig(provider="recording", model="recording")
        self.prompts = []

    def generate_json(self, system_prompt, user_prompt, seed, required_fields=None):
        self.prompts.append((system_prompt, user_prompt, seed, required_fields))
        field = next(iter(required_fields or {"artifact"}))
        value = "7" if field == "answer" else "work product"
        return AdapterResponse({field: value}, "recording", "recording", seed, 1, 1, 1.0)


class FailingConstructionAdapter(LLMAdapter):
    def __init__(self):
        self.config = LLMConfig(provider="failing", model="failing")

    def generate_json(self, system_prompt, user_prompt, seed, required_fields=None):
        raise RuntimeError("malformed provider JSON")


class FinQAStructuredAdapter(LLMAdapter):
    def __init__(self):
        self.config = LLMConfig(provider="finqa-test", model="finqa-test")
        self.prompts = []

    def generate_json(self, system_prompt, user_prompt, seed, required_fields=None):
        fields = set(required_fields or {"artifact"})
        self.prompts.append((system_prompt, user_prompt, fields))
        if fields == {"selected_evidence", "steps"}:
            value = {
                "selected_evidence": [
                    {"type": "cell", "location": "2018 / aircraft fuel expense", "raw_value": "9,896"},
                    {"type": "cell", "location": "2018 / operating expense percent", "raw_value": "23.6%"},
                ],
                "steps": [{
                    "operation": "divide",
                    "operands": ["9,896", "23.6%"],
                    "result_unit": "million",
                }],
            }
        else:
            field = next(iter(fields))
            value = {field: "999" if field == "answer" else "upstream work"}
        return AdapterResponse(value, "finqa-test", "finqa-test", seed, 1, 1, 1.0)


def _case():
    row = {"id": "causal-1", "question": "What is 3 plus 4?", "context": "",
           "choices": None, "answer": "GOLD_SECRET", "raw": {}}
    adapter = DATASET_ADAPTERS["gsm8k"]
    return adapter, row, build_dataset_case(adapter, row, 0)


def test_every_q2_variant_emits_a_runnable_application():
    _, _, case = _case()
    results = {variant: get_ablation_method(variant).construct(case.request()) for variant in Q2_VARIANTS}
    for result in results.values():
        result.validate(case.request())
        assert result.application.nodes
        assert result.application.entrypoints
    assert results["w/o_blueprint"].blueprint.metadata["blueprint_present"] is False
    generic = results["w/o_realization"].application
    assert generic.metadata["blueprint_preserving"] is False
    assert all(node.implementation_ref == "generic_blueprint_interpreter" for node in generic.nodes)


def test_executor_is_blind_to_variant_and_gold_and_uses_application():
    dataset, row, case = _case()
    construction = get_ablation_method("w/o_blueprint", adapter=DeterministicAdapter()).construct(case.request())
    recorder = RecordingAdapter()
    result = ApplicationTaskExecutor(recorder, Path(__file__).parents[3]).execute(
        dataset, row, case, construction, 11)
    serialized_prompts = json.dumps([(system, user, seed, sorted(fields or []))
                                     for system, user, seed, fields in recorder.prompts])
    assert "w/o_blueprint" not in serialized_prompts
    assert "GOLD_SECRET" not in serialized_prompts
    assert result.prediction == "7"
    assert len(result.node_executions) == len(construction.application.nodes)
    assert result.application_digest


def test_primary_answer_metric_is_not_runtime_or_trace_gated():
    adapter = NumericAdapter()
    dataset = DATASET_ADAPTERS["gsm8k"]
    assert adapter.score("7", "7") == 1.0
    assert dataset.primary_score("7", "7", runtime_valid=0.0, trace_rate=0.0) == 1.0


def test_swebench_primary_metric_is_real_resolution():
    assert DATASET_ADAPTERS["swebench_verified"].execution.metric_name == "swebench_resolved"


def test_full_realization_projects_task_dependencies_to_components():
    _, _, case = _case()
    result = get_ablation_method("full_graph_harness").construct(case.request())
    order, _ = _execution_order(result.application.nodes, result.application.edges)
    roles = [next(node for node in result.application.nodes if node.id == node_id)
             .config["execution_instruction"] for node_id in order]
    assert "parse" in roles[0].lower()
    assert "answer" in roles[-1].lower()


def test_realization_compiles_stage_contracts_but_generic_interpreter_does_not():
    _, _, case = _case()
    full = get_ablation_method("full_graph_harness").construct(case.request())
    generic = get_ablation_method("w/o_realization").construct(case.request())
    assert all(node.config.get("artifact_contract") for node in full.application.nodes)
    assert all("artifact_contract" not in node.config for node in generic.application.nodes)


def test_hotpot_realization_contracts_distinguish_retrieve_synthesize_and_verify():
    root = Path(__file__).resolve().parents[3]
    dataset = DATASET_ADAPTERS["hotpotqa"]
    if not (root / dataset.source_file).exists():
        pytest.skip("HotpotQA fixture is external to the source distribution")
    row = __import__("openmas_bench.dataset_cases", fromlist=["load_normalized_rows"]).load_normalized_rows(
        root, dataset, 1)[0]
    case = build_dataset_case(dataset, row, 0)
    full = get_ablation_method("full_graph_harness").construct(case.request())
    contracts = {node.config["execution_instruction"]: node.config["artifact_contract"]
                 for node in full.application.nodes}
    synth = next(value for role, value in contracts.items() if "synthesize" in role.lower())
    verify = next(value for role, value in contracts.items() if "verify" in role.lower())
    answer = next(value for role, value in contracts.items() if "return the answer" in role.lower())
    assert "rationale" in synth and "candidate_answer" in synth
    assert "checks" in verify and "candidate_answer" in verify
    assert "shortest answer span" in answer


def test_generic_realization_keeps_abstract_roles_without_component_bindings():
    _, _, case = _case()
    generic = get_ablation_method("w/o_realization").construct(case.request())
    roles = [node.config["execution_instruction"].lower() for node in generic.application.nodes]
    assert any("parse" in role for role in roles)
    assert any("answer" in role for role in roles)
    assert all(node.implementation_ref == "generic_blueprint_interpreter"
               for node in generic.application.nodes)


def test_construction_json_failure_falls_back_without_losing_executable_pair():
    _, _, case = _case()
    result = get_ablation_method("w/o_graph_orchestration", FailingConstructionAdapter()).construct(case.request())
    assert result.telemetry.fallback is True
    assert "construction_json_fallback=true" in result.telemetry.notes
    result.validate(case.request())


def test_pubmedqa_output_contract_uses_author_conclusion_not_generic_uncertainty():
    executor = __import__(
        "openmas_bench.application_executor", fromlist=["_benchmark_output_contract"])
    contract = executor._benchmark_output_contract("pubmedqa_accuracy")
    assert "exactly one lowercase label" in contract
    assert "authors' conclusion" in contract
    assert "do not change" in contract
    assert executor._benchmark_output_contract("f1") == ""


def test_pubmedqa_normalization_requires_an_explicit_leading_label():
    adapter = __import__(
        "openmas_bench.dataset_adapters", fromlist=["PubMedQAAdapter"]).PubMedQAAdapter()
    assert adapter.normalize_prediction("Answer: yes") == "yes"
    assert adapter.normalize_prediction("No, the authors found no meaningful difference.") == "no"
    assert adapter.normalize_prediction("The result is not conclusive") == "the result is not conclusive"
    assert adapter.normalize_prediction("Snellen E was modestly higher") == "snellen e was modestly higher"


def test_pubmedqa_reasoning_contract_is_visible_before_terminal_stage():
    executor = __import__(
        "openmas_bench.application_executor", fromlist=[
            "_benchmark_reasoning_contract", "_node_system_prompt"])
    contract = executor._benchmark_reasoning_contract("pubmedqa_accuracy")
    prompt = executor._node_system_prompt(
        "agent", "Interpret findings", False, "answer", "Use evidence.", "", contract)
    assert "Benchmark reasoning contract" in prompt
    assert "Preserve the original question verbatim" in prompt
    assert "authors' conclusion" in prompt
    assert "Return exactly one" not in prompt


def test_finqa_decimal_tool_normalizes_percent_and_preserves_negative_values():
    divided = _execute_finqa_tool_call({
        "selected_evidence": [{"type": "cell", "location": "r/c", "raw_value": "23.6%"}],
        "steps": [{"operation": "divide", "operands": ["9,896", "23.6%"],
                   "result_unit": "million"}],
    })
    assert divided["steps"][0]["normalized_operands"] == ["9896", "0.236"]
    assert divided["final_value"].startswith("41932.2033898305")
    changed = _execute_finqa_tool_call({
        "selected_evidence": [{"type": "cell", "location": "r/c", "raw_value": "-3.2%"}],
        "steps": [{"operation": "percent_point_change", "operands": ["-3.2%", "2.1%"],
                   "result_unit": "percentage points"}],
    })
    assert changed["final_value"] == "-5.3"


def test_finqa_multistep_trace_resolves_percent_intermediate():
    result = _execute_finqa_tool_call({
        "selected_evidence": [{
            "type": "text", "location": "pre_text[1]", "raw_value": "3.8 million",
        }],
        "steps": [
            {"operation": "divide", "operands": ["100", "100"], "result_unit": "percent"},
            {"operation": "divide", "operands": ["3.8", "$step_0"], "result_unit": "number"},
        ],
    })
    assert result["steps"][0]["computed_value"] == "1"
    assert result["steps"][1]["normalized_operands"] == ["3.8", "0.01"]
    assert result["final_value"] == "380"
    assert result["tool_status"] == "ok"


def test_finqa_selected_evidence_must_exist_in_public_task():
    call = {
        "selected_evidence": [{
            "type": "text", "location": "pre_text[9]", "raw_value": "invented 99.9 million",
        }],
        "steps": [{"operation": "divide", "operands": ["1", "2"], "result_unit": "number"}],
    }
    try:
        _execute_finqa_tool_call(call, {"question": "q", "context": "real value is 3.8 million"})
    except ValueError as exc:
        assert "not present in the public task" in str(exc)
    else:
        raise AssertionError("hallucinated evidence passed public-task validation")


def test_finqa_decimal_tool_rejects_malformed_or_unapproved_calls():
    base = {
        "selected_evidence": [{"type": "cell", "location": "r/c", "raw_value": "1"}],
        "steps": [{"operation": "eval", "operands": ["1", "2"], "result_unit": "number"}],
    }
    try:
        _execute_finqa_tool_call(base)
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("unapproved operation was accepted")
    base["steps"][0]["operation"] = "divide"
    base["steps"][0]["operands"] = ["not-a-number", "2"]
    try:
        _execute_finqa_tool_call(base)
    except ValueError as exc:
        assert "numeric operand" in str(exc)
    else:
        raise AssertionError("malformed operand was accepted")
    base["steps"][0]["operands"] = ["1", "$step_0"]
    try:
        _execute_finqa_tool_call(base)
    except ValueError as exc:
        assert "earlier existing step" in str(exc)
    else:
        raise AssertionError("forward/self step reference was accepted")


def _finqa_case():
    row = {
        "id": "finqa-tool-1",
        "question": "What amount corresponds to 23.6 percent of the total?",
        "context": "FINQA_TABLE:\nmetric | 2018\nexpense | 9,896\nrate | 23.6%",
        "choices": None,
        "answer": "GOLD_MUST_NOT_APPEAR",
        "raw": {"metadata": {"program": "GOLD_PROGRAM_MUST_NOT_APPEAR"}},
    }
    dataset = DATASET_ADAPTERS["finqa"]
    return dataset, row, build_dataset_case(dataset, row, 0)


def test_finqa_compiled_variants_execute_tool_and_answer_copies_result():
    dataset, row, case = _finqa_case()
    for variant in ("full_graph_harness", "w/o_requirement_grounding",
                    "w/o_graph_orchestration", "w/o_constraint_aware_orchestration"):
        construction = get_ablation_method(variant).construct(case.request())
        adapter = FinQAStructuredAdapter()
        result = ApplicationTaskExecutor(adapter, Path(__file__).parents[3]).execute(
            dataset, row, case, construction, 11)
        audit = result.metadata["finqa_tool_audit"]
        assert audit["eligible"] is True
        assert audit["invoked"] is True
        assert audit["gold_used"] is False
        assert result.prediction == audit["computed_value"]
        assert audit["step_count"] == 1
        assert len(adapter.prompts) == len(construction.application.nodes) - 1
        serialized = json.dumps([(system, user, sorted(fields))
                                 for system, user, fields in adapter.prompts])
        assert "GOLD_MUST_NOT_APPEAR" not in serialized
        assert "GOLD_PROGRAM_MUST_NOT_APPEAR" not in serialized
        verifier = next(prompt for prompt in adapter.prompts if "verifier boundary" in prompt[0])
        assert "TOOL_TRACE" in verifier[1]
        assert "FINQA_TABLE" not in verifier[1]


def test_finqa_blueprint_and_realization_ablations_receive_no_tool_interface():
    dataset, row, case = _finqa_case()
    for variant in ("w/o_blueprint", "w/o_realization"):
        construction = get_ablation_method(variant).construct(case.request())
        adapter = FinQAStructuredAdapter()
        result = ApplicationTaskExecutor(adapter, Path(__file__).parents[3]).execute(
            dataset, row, case, construction, 11)
        audit = result.metadata["finqa_tool_audit"]
        assert audit == {"eligible": False, "invoked": False, "gold_used": False}
        serialized = json.dumps([(system, user, sorted(fields))
                                 for system, user, fields in adapter.prompts])
        assert "finqa_structured_tool_schema" not in serialized
        assert "TOOL_TRACE" not in serialized
        assert all(fields != {"selected_evidence", "steps"}
                   for _, _, fields in adapter.prompts)
