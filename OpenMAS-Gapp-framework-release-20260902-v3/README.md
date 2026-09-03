# OpenMAS-Gapp Construction Benchmark

The reusable framework entry point is `openmas_bench.engine.GraphHarnessEngine`.
See [ARCHITECTURE.md](ARCHITECTURE.md) and [examples/run_engine.py](examples/run_engine.py)
before using the Q1/Q2/Q3 experiment runners.

The current Q1-Q4 protocol is frozen in [Q1_Q4_PROTOCOL.md](Q1_Q4_PROTOCOL.md).
The authoritative construction chain is `R + G_H -> M_R -> B -> A`.

This directory contains the self-implemented controlled reference baselines and the first domain-package construction pipeline. It is intentionally separate from `准备工作/baseline/`, which contains external paper repositories kept for related-work inspection.

## Data layers

```text
raw/       immutable downloaded source snapshots + SHA-256 manifest
cleaned/   normalized execution tasks (JSONL)
packages/  Application Packages: requirement + ecosystem + contracts + tasks + mutations
```

One Application Package contains multiple execution tasks. A QA item is not treated as a complete application by itself.

## Pilot domains

- Software engineering: SWE-bench Verified task rows
- Financial analysis: FinQA task rows
- Biomedical evidence: PubMedQA labeled task rows

Each package defines five requirement facets (`goal`, `process`, `resources`, `governance`, `output`), a typed capability catalog, requirement-to-MAS contracts, execution tasks, and mutations. There is no unique gold MAS graph.

## Reference baselines

`single_agent`, `universal_fixed`, `domain_template`, `direct_prompt`, `json_spec`, `rag_example`, `flat_capability`, `rule_compiler`, and `search_composer` are transparent self-implemented reference baselines. They all emit the same `MASSpec` schema and can be evaluated by the same evaluator.

The current direct-prompt baseline uses a deterministic keyword proxy so the pipeline is runnable without an API key. A model adapter can replace that component without changing package or result schemas.

`openmas_bench.trace.planned_trace` is only a plan-level adapter smoke test. It must not be reported as dynamic execution evidence; real runtime traces require a domain sandbox.

## Build

From this directory, with the repository root on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = (Resolve-Path '.').Path
python scripts/prepare_pilot.py --download --clean --packages --limit 30
python -m openmas_bench.cli build --baseline rule_compiler --package packages/finance_000.json --output /tmp/spec.json
python -m openmas_bench.cli evaluate --package packages/finance_000.json --spec /tmp/spec.json --output /tmp/score.json
```

## Q1-Q4 minimal causal-chain suite

```powershell
python scripts/prepare_construction_cases.py
python scripts/run_q1_smoke.py
python -m pytest -q
```

`construction_cases/` contains eight gold development cases. `q1_smoke/`
contains 8 cases x 5 controlled construction methods. These deterministic runs
verify interfaces and metrics only; `summary.json` therefore records
`formal_result: false`.

## Q1 DeepSeek controlled pilot

The completed 20-case, five-method, three-seed run is stored in
`q1_deepseek_300_final.json`. A concise analysis and interpretation boundary are
documented in `Q1_FORMAL_REPORT.md`. The final dataset contains zero fallback
runs; six outputs required bounded ecosystem-ID repair.

Raw files are not modified during cleaning. All derived records keep source dataset and task identifiers.
