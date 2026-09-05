# Q10 Financial Dataset Analysis

## Task

Q10 asks:

> Can Graph Harness construct a specialized MAS application for the financial
> domain?

This covers financial analysis, risk assessment and compliance review. The
benchmark should show how Graph Harness handles data analysis, risk controls,
regulatory constraints and auditable final reporting.

## Current Pilot Inputs

| Dataset | Q10 Role | Current Source | Pilot Rows |
|---|---|---|---:|
| FinanceBench-style | filing evidence, financial metrics, risk factors, disclosure caveats | local frozen fixture | 3 |
| FinQA-style | financial tables, arithmetic program, units, auditability | local frozen fixture | 3 |

These pilot rows are protocol fixtures, not formal benchmark evidence. Replace
them with a frozen public split before reporting model performance.

## Expected Graph Shape

```text
filing/table evidence branch ─┐
                              ├─ financial analysis ─┬─ risk assessment ─┐
risk/disclosure evidence ─────┘                       └─ compliance review ├─ audit trail ─ controls ─ final report
                                                                   feedback ┘
```

The important claim is construction structure, not deterministic answer
accuracy. A passing smoke run should show completed status, non-empty Harness,
Blueprint and executable MAS graphs, `requires_multi_branch=true`,
`requires_constraint_gate=true`, and construction/runtime metrics equal to 1.0
for the structural contracts.
