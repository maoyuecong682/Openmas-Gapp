# Q10 Financial MAS Dataset Space

Q10 tests whether Graph Harness can construct and execute a finance-specific
multi-agent application for financial analysis, risk assessment, compliance
review and auditable final reporting.

## Dataset Inputs

- FinanceBench-style rows: filing evidence snippets, metric reasoning labels,
  risk factors and disclosure caveats.
- FinQA-style rows: financial tables, report narrative, numeric calculations,
  units and audit metadata.

## Storage Contract

- `raw/financial/<dataset>.jsonl` stores source rows or local fixture sources.
- `normalized/*.jsonl` stores the stable benchmark row schema.
- `pilot/` stores frozen small subsets used for qualification and smoke runs.
- `manifests/` stores source metadata, split, license, row counts and SHA-256.

Q10 smoke outputs are diagnostic and must not be reported as formal model
performance.
