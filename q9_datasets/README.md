# Q9 Dataset Space

Canonical root: `D:\Openmas-Gapp\q9_datasets`

Compatibility alias: `D:\Openmas-Gapp\q2_datasets`

Q9 currently stores frozen local medical samples for:

- `MedQA`
- `MedMCQA`
- `PubMedQA`
- `MMLU medical subset`

Current files:

- `raw\medical\medqa.jsonl`
- `raw\medical\medmcqa.jsonl`
- `raw\medical\pubmedqa.jsonl`
- `raw\medical\mmlu_medical.jsonl`
- `normalized\medqa.jsonl`
- `normalized\medmcqa.jsonl`
- `normalized\pubmedqa.jsonl`
- `normalized\mmlu_medical.jsonl`
- `pilot\medqa_pilot.jsonl`
- `pilot\medmcqa_pilot.jsonl`
- `pilot\pubmedqa_pilot.jsonl`
- `pilot\mmlu_medical_pilot.jsonl`

Each normalized row should expose at least:

- `id`
- `question`
- `answer`
- `context`
- `choices` for multiple-choice items
- `raw` with the original dataset row

Status: local frozen rows are present. The full public benchmark splits can be
expanded later if publication-grade evaluation is needed.
