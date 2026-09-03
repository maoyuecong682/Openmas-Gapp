# OpenMAS-Gapp Public Architecture

The public pipeline has one canonical path:

```text
Requirement R + Harness Graph G_H
        -> Application Requirement Model M_R (ARG)
        -> Application Blueprint B (CCG + validation-only candidate search)
        -> Executable MAS Application A (MEG)
        -> Runtime Trace + Prediction
```

## Public API

`openmas_bench.engine.GraphHarnessEngine` is the only façade that owns a full
construction and execution run. Experiment runners enumerate cases and aggregate
results; they do not reimplement construction, realization, scoring, or auditing.

```python
engine = GraphHarnessEngine(adapter, data_root)
result = engine.run_case(
    dataset,
    normalized_row,
    construction_case,
    seed=11,
    intervention="full_graph_harness",
)
```

The returned `EngineRunResult` exposes the requirement model, selected Blueprint,
executable application, execution trace, prediction, metric, resource audit, tool
audit, candidate scores, and `gold_used=false` declaration.

## Module Ownership

| Module | Responsibility |
|---|---|
| `engine.py` | Public façade and end-to-end run result |
| `schema.py` | Canonical ARG, Blueprint, executable MAS, and trace data structures |
| `construction.py` | ARG, CCG, candidate search, Blueprint realization |
| `application_executor.py` | Dataset-neutral graph scheduling and node execution |
| `domains/` | Gold-blind domain resource routing, output contracts, and tools |
| `ablation.py` | Q2 interventions over the shared construction pipeline |
| `q3/` | Orchestration baselines and structural evaluation using `ApplicationBlueprint` |
| `scripts/` | Checkpointing, experiment enumeration, aggregation, and reporting |

## Experiment Boundaries

- Q1 changes the construction method.
- Q2 changes exactly one intervention in the shared Engine pipeline.
- Q3 changes the orchestration representation but emits the shared
  `ApplicationBlueprint` and uses the shared MEG and executor.
- Domain plugins may inspect only the public normalized row. They must never read
  the gold answer, annotated gold program, or evaluation result.

## Extension Contract

Add a domain by implementing `DomainPlugin` and registering it in
`domains/registry.py`. Keep dataset-specific evidence routing and output semantics
inside the plugin. Do not add dataset switches to `GraphHarnessEngine`.

