# Q2 Dataset Selection and Qualification

## Pre-registered inclusion rule

A dataset enters the primary Q2 table only when all of the following hold:

1. The normalized rows contain a non-empty gold contract appropriate to the
   evaluator (answer text/choice, executable tests, or a validated patch).
2. All six variants construct and execute the same number of application
   stages on the qualification sample.
3. The evaluator is deterministic and independent of the construction
   metadata. Runtime validity and trace diagnostics are reported separately.
4. There is no dataset-specific environment blocker that can create a
   systematic zero unrelated to the ablated module.

The rule is applied before looking at formal model scores. A low score is not
itself a reason to exclude a dataset; an invalid or un-frozen evaluator is.

## Current decision

| Dataset | Decision | Reason |
|---|---|---|
| GSM8K | Primary Q2 | Stable numeric answer contract and sequential reasoning stages |
| MATH-500 | Primary Q2 | Exact normalized math answer and strategy/verification stages |
| MMLU | Primary Q2 | Stable multiple-choice contract; useful grounding/constraint contrast |
| HotpotQA | Primary Q2 | Multi-hop evidence and synthesis expose graph orchestration effects |
| MedQA | Primary Q2 | Constraint-heavy clinical choice task; report license caveat |
| HumanEval | Separate code track | Executable sandbox is available, but unit-test pass is not comparable to QA accuracy |
| MBPP | Separate code track | Same code-track treatment; re-run after function-name/test contract smoke |
| DROP | Separate reading/numeric track | Span and discrete-operation evaluator needs a dedicated qualification smoke; do not pool with QA means |
| SWE-bench | Temporarily excluded | Repository snapshots exist, but per-instance Python/test environments are not uniformly frozen |

`PubMedQA` and `FinQA` are now registered as conditional candidates. Their
current 30-row files are pilot subsets, so a paper-level run must replace them
with article/report-level held-out splits. FinQA additionally requires an
audit of the annotated arithmetic program executor; its current primary score
is numeric answer accuracy, not program-execution success.

## Reproducible check

Run the API-free qualification before any LLM experiment:

```cmd
python "准备工作\openmas_benchmark\scripts\qualify_q2_datasets.py" --rows-per-dataset 3 --output "q2_dataset_qualification.json"
```

The generated JSON records source checks, gold checks, stage parity, warnings,
and blockers. The current result is stored at the repository root as
`q2_dataset_qualification.json`.

## Formal primary-table command

Use only the five primary candidates for the first formal cross-dataset run:

```cmd
set "Q1_LLM_API_KEY=YOUR_NEW_KEY" && python "准备工作\openmas_benchmark\scripts\run_q2_cross_dataset.py" --provider openai_compatible --datasets GSM8K,MATH-500,MMLU,HotpotQA,MedQA --rows-per-dataset 5 --seeds 11,22,33 --temperature 0.0 --max-output-tokens 16384 --workers 4 --output "q2_formal_5ds_5row_3seed.json" --stage formal --resume
```

Code and DROP tracks should be reported in separate tables with their native
metrics and failure rates, never combined into a single accuracy average.

After the source-level split and FinQA program audit, the extended primary
table can use:

```text
GSM8K, MATH-500, MMLU, HotpotQA, MedQA, PubMedQA, FinQA
```

## Other datasets in the catalogue

These are good conceptual matches, but are not yet local Q2 adapters and
should be added only with the same qualification script:

| Dataset | Best ablation signal | Current recommendation |
|---|---|---|
| MuSiQue | Multi-hop retrieval, typed dependency, graph orchestration | Strong next addition after a passage/evidence adapter |
| StrategyQA | Requirement decomposition and implicit sub-question planning | Good grounding/control addition; needs yes/no + rationale normalization |
| ARC / SciQ | Evidence retrieval plus compact science reasoning | Stable low-cost control set; weaker graph signal than HotpotQA |
| MMLU-Pro | More difficult choice reasoning and reduced guessing | Useful replacement/extension for MMLU, not a separate domain |
| MathQA | Program selection and arithmetic execution | Good FinQA companion if a program executor is available |
| AQuA | Programmatic math with multiple-choice output | Candidate only after a reliable option/program parser |

The catalogue's agent-environment and web benchmarks (GAIA, WebArena,
AgentBench) are not suitable for the first Q2 table because their external
environment state would dominate the component ablation. They can be a later
separate environment track.
