"""Minimal public GraphHarnessEngine example."""
from pathlib import Path

from openmas_bench.dataset_adapters import DATASET_ADAPTERS
from openmas_bench.dataset_cases import build_dataset_case, load_normalized_rows
from openmas_bench.engine import GraphHarnessEngine
from openmas_bench.llm import DeterministicAdapter


data_root = Path(__file__).resolve().parents[3]
dataset = DATASET_ADAPTERS["bbh_full"]
row = load_normalized_rows(data_root, dataset, 1)[0]
case = build_dataset_case(dataset, row, 0)

engine = GraphHarnessEngine(DeterministicAdapter(), data_root)
result = engine.run_case(dataset, row, case, seed=11)

print(result.prediction)
print(result.audit)

