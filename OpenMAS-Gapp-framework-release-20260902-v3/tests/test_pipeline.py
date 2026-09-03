import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from openmas_bench.baselines import BUILDERS, get_builder
from openmas_bench.evaluate import evaluate_spec
from openmas_bench.schema import Capability, Contract, DomainPackage, ExecutionTask


def package():
    return DomainPackage(
        "test_001", "financial_analysis", "Test Assistant", {"dataset": "fixture"},
        {"goal": "answer", "process": ["retrieve then calculate"], "resources": ["report"], "governance": ["verify"], "output": ["answer"]},
        [Capability("report_retrieval", "tool", "retrieve report", tags=["retrieve"]), Capability("numerical_reasoning", "reasoning", "calculate answer", tags=["calculate"]), Capability("calculation_verification", "verification", "verify calculation", tags=["verify"])],
        [Contract("r", "capability_required", "report_retrieval", "selected_capability", "required"), Contract("o", "order_required", "report_retrieval<numerical_reasoning", "reachable", "order")],
        [ExecutionTask("q1", "calculate from report", "42")],
    )


def test_all_builders_emit_valid_specs():
    p = package()
    for name in BUILDERS:
        spec = get_builder(name).build(p)
        spec.validate()
        assert spec.package_id == p.package_id
        assert spec.nodes


def test_evaluator_has_contract_metrics():
    p = package()
    score = evaluate_spec(p, get_builder("rule_compiler").build(p))
    assert "capability_recall" in score
    assert score["package_id"] == "test_001"
