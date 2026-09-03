import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from openmas_bench.construction import Q1_METHODS, get_construction_method
from openmas_bench.evaluate import evaluate_construction
from openmas_bench.io import load_construction_case
from openmas_bench.schema import ApplicationBlueprint, BlueprintNode
from scripts.prepare_construction_cases import build_suite


ROOT = Path(__file__).parents[1]


def cases():
    if not (ROOT / "cleaned" / "finance_tasks.jsonl").exists():
        pytest.skip("Q1 construction fixtures are external to the source distribution")
    return build_suite()


def test_formal_suite_has_balanced_splits_and_families():
    suite = cases()
    assert len(suite) == 20
    assert sum(x.split == "dev" for x in suite) == 8
    assert sum(x.split == "validation" for x in suite) == 12
    assert {family: sum(x.family == family for x in suite) for family in {x.family for x in suite}} == {
        "sequential": 5, "multi_branch": 5, "feedback_driven": 5, "constraint_heavy": 5,
    }
    assert all(len(x.execution_tasks) == 3 for x in suite)


def test_all_q1_methods_share_contract_and_stay_within_budget():
    for case in cases():
        request = case.request()
        for name in Q1_METHODS:
            result = get_construction_method(name).construct(request)
            result.validate(request)
            assert result.method == name
            assert result.application.metadata["blueprint_preserving"] is True


def test_graph_harness_preserves_constraint_controls():
    constrained = next(x for x in cases() if x.case_id == "constraint_medical")
    result = get_construction_method("graph_harness").construct(constrained.request())
    assert {x.id for x in result.requirement_model.constraints} == {"human_approval", "approved_source"}
    assert {x.id for x in result.blueprint.nodes if x.kind == "control"} == {
        "control_human_approval", "control_approved_source",
    }


def test_blueprint_rejects_concrete_runtime_binding():
    blueprint = ApplicationBlueprint("x", "test", [BlueprintNode("n", "component_requirement", "x", binding_constraints={"agent_id": "agent-1"})], [])
    try:
        blueprint.validate()
    except ValueError as exc:
        assert "concrete bindings" in str(exc)
    else:
        raise AssertionError("concrete runtime binding leaked into Blueprint")


def test_q1_metrics_cover_core_causal_chain():
    for case in cases():
        score = evaluate_construction(case, get_construction_method("graph_harness").construct(case.request()))
        assert score["requirement_task_f1"] == 1.0
        assert score["orchestration_relation_recall"] == 1.0
        assert score["realization_fidelity"] == 1.0
