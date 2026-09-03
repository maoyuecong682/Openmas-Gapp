from pathlib import Path

import pytest

from openmas_bench.dataset_adapters import DATASET_ADAPTERS
from openmas_bench.dataset_cases import build_dataset_case, load_normalized_rows
from openmas_bench.engine import GraphHarnessEngine
from openmas_bench.llm import DeterministicAdapter
from openmas_bench.q3.baselines import get_q3_baseline
from openmas_bench.q3.suite import build_q3_suite
from openmas_bench.schema import ApplicationBlueprint


def test_engine_runs_shared_q2_pipeline_and_emits_audit():
    root = Path(__file__).resolve().parents[2]
    dataset = DATASET_ADAPTERS["bbh_full"]
    if not (root / dataset.source_file).exists():
        pytest.skip("BBH fixture is external to the source distribution")
    row = load_normalized_rows(root, dataset, 1)[0]
    case = build_dataset_case(dataset, row, 0)
    result = GraphHarnessEngine(DeterministicAdapter(), root).run_case(
        dataset, row, case, seed=11, intervention="full_graph_harness")
    assert result.audit["pipeline"] == "R+G_H->M_R->B->A->trace"
    assert result.audit["gold_used"] is False
    assert result.to_experiment_record()["engine_audit"] == result.audit


def test_q3_baselines_emit_shared_application_blueprint():
    case = build_q3_suite()[0]
    for baseline in ("flat_component_selection", "graph_harness"):
        blueprint = get_q3_baseline(baseline).build_blueprint(case)
        assert isinstance(blueprint, ApplicationBlueprint)
        blueprint.validate()
