from __future__ import annotations

from collections import Counter
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openmas_bench.construction import Q1_METHODS, get_construction_method
from openmas_bench.evaluate import evaluate_construction, evaluate_execution
from openmas_bench.runtime import MinimalMARRuntime
from scripts.prepare_construction_cases import build_suite


def main():
    cases = build_suite()
    assert len(cases) == 20
    assert Counter(x.split for x in cases) == {"validation": 12, "dev": 8}
    assert Counter(x.family for x in cases) == {"sequential": 5, "multi_branch": 5, "feedback_driven": 5, "constraint_heavy": 5}
    assert all(len(x.execution_tasks) == 3 for x in cases)
    runtime = MinimalMARRuntime()
    runs = 0
    for case in cases:
        case.validate()
        for method_name in Q1_METHODS:
            result = get_construction_method(method_name, seed=0).construct(case.request())
            result.validate(case.request())
            construction_score = evaluate_construction(case, result)
            assert 0 <= construction_score["construction_quality"] <= 1
            for task in case.execution_tasks:
                execution = runtime.execute(case, result, task, seed=0)
                execution_score = evaluate_execution(case, execution)
                assert 0 <= execution_score["execution_performance"] <= 1
            runs += 1
    print(f"Q1 protocol validation passed: {len(cases)} cases, {runs} constructions, {runs * 3} executions")


if __name__ == "__main__":
    main()
