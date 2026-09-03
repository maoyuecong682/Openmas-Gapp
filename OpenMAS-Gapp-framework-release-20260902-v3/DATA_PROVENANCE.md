# Pilot Data Provenance and Limitations

## Downloaded snapshots

| Name | URL | Use | Current snapshot |
|---|---|---|---|
| FinQA | `https://github.com/czyssrs/FinQA` | financial report QA execution tasks | repository `train.json`, 30 rows used in pilot |
| PubMedQA | `https://github.com/pubmedqa/pubmedqa` | biomedical evidence QA execution tasks | `ori_pqal.json`, 30 rows used in pilot |
| SWE-bench Verified | `https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified` | repository issue/patch execution tasks | HF datasets-server test rows, 30 rows used in pilot |

The exact URLs, byte sizes, and SHA-256 hashes are in `raw/sources.manifest.json`. Raw files are preserved and should not be edited in place.

## Construction policy

- The source QA/task item is an execution task, not a complete application requirement.
- Multiple tasks are grouped into one Application Package.
- Requirements are domain-level construction contracts authored from the source task family and domain workflow assumptions.
- No unique gold MAS graph is stored. Evaluation is contract-based and allows multiple valid topologies.
- Source answers/patches are retained only as execution references; they do not define a gold architecture.

## Pilot limitations

This is a construction and interface pilot, not the final benchmark release.

1. FinQA currently snapshots the training file and uses only its first 30 rows. A final experiment must use source-level train/dev/test separation and hold out report templates.
2. PubMedQA uses a deterministic subset of the labeled file. A final experiment should split by article/topic and document the evidence policy with domain reviewers.
3. SWE-bench Verified is downloaded through the Hugging Face rows API for a bounded pilot. Repository checkout, patch application, and test execution are not yet wired into the OpenMAS runtime.
4. The generated `planned_trace` is a schema smoke test, not runtime evidence. Dynamic Trace scores must only be computed after a real tool sandbox executes the task.
5. The current contracts are initial author annotations and require independent domain review before publication.

