# OpenMAS-Gapp Q10 Workspace

## Purpose

Q10 is the Financial Application Case Study:

> Can Graph Harness construct a specialized MAS application for financial
> analysis by organizing filing/table evidence, quantitative analysis, risk
> assessment, compliance review, auditability, and final reporting?

The input is a financial dataset row plus the shared Graph Harness ecosystem.
The application output is a governed financial answer produced by a graph of
bound agents, resources and controls. The experiment records the application
graph, abstract Blueprint, construction metrics, execution trace, answer score,
and source audit. It is not investment advice and does not claim production
regulatory compliance.

## Q10 Management Shape

Q10 follows Q9's operational separation while using financial roots:

1. `config/q10.paths.ps1` establishes paths, proxy and provider key aliases.
2. `scripts/q10/prepare_q10_financial.py` creates frozen local financial pilots.
3. `scripts/q10/qualify_q10_datasets.py` checks row contracts before model calls.
4. `scripts/q10/run_q10_smoke.py` runs deterministic or explicitly selected model cases.
5. `q10_datasets/manifests/` records provenance and hashes.
6. `outputs/q10_financial/` stores runs, graphs, tables, traces and audits.

## Stable Row Contract

Every normalized row exposes `id`, `question`, `answer`, `context`, `choices`,
`source`, and `raw`.

FinanceBench-style rows keep filing evidence snippets under
`raw.evidence[*].evidence_text`. FinQA-style rows keep table, pre-text,
post-text and numeric-program audit metadata under `raw.context` and
`raw.metadata.program`.

## Graph Harness Claim

The Q10 graph is intentionally not a linear QA pipeline. The financial Harness
must preserve:

- parallel evidence branches for filing/table facts and risk/disclosure facts;
- a merge point for financial analysis;
- risk assessment and compliance review as separate governed tasks;
- audit trail construction before final reporting;
- risk, regulatory-compliance and auditability controls gating the final report.

Deterministic smoke validates schema, construction parity, branch/resource
routing, runtime trace plumbing, and control placement. A formal model run
requires a frozen public split, declared provider/model, temperature, key
source, endpoint, and a separate compliance review policy.
