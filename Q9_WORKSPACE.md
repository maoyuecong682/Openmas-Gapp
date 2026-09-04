# OpenMAS-Gapp Q9 Workspace

Q9 is the Medical Application Case Study from `OpenMAS-Gapp实验方案.md`.
The Q9 data space is independent and uses `q9_datasets` as the canonical
storage root.

## Roots

- Workspace root: `D:\Openmas-Gapp`
- Code root: `D:\Openmas-Gapp\OpenMAS-Gapp-framework-release-20260902-v3`
- Dataset root passed to CLI / engine: `D:\Openmas-Gapp`
- Q9 dataset storage: `D:\Openmas-Gapp\q9_datasets`
- Q9 normalized rows: `D:\Openmas-Gapp\q9_datasets\normalized`
- Q9 raw medical rows: `D:\Openmas-Gapp\q9_datasets\raw\medical`
- Hugging Face cache: `D:\Openmas-Gapp\.cache\huggingface`
- Output root: `D:\Openmas-Gapp\outputs\q9_medical`
- Logs: `D:\Openmas-Gapp\logs\q9_medical`
- Temporary files: `D:\Openmas-Gapp\tmp\q9_medical`

Compatibility alias:

- `D:\Openmas-Gapp\q2_datasets` is a junction to `D:\Openmas-Gapp\q9_datasets`

## Q9 Datasets

Q9 needs four medical datasets:

- `MedQA`
- `MedMCQA`
- `PubMedQA`
- `MMLU 医学子集`

Planned normalized files:

- `D:\Openmas-Gapp\q9_datasets\normalized\medqa.jsonl`
- `D:\Openmas-Gapp\q9_datasets\normalized\medmcqa.jsonl`
- `D:\Openmas-Gapp\q9_datasets\normalized\pubmedqa.jsonl`
- `D:\Openmas-Gapp\q9_datasets\normalized\mmlu_medical.jsonl`

Each normalized row should expose at least:

- `id`
- `question`
- `answer`
- `context`
- `choices` for multiple-choice items
- `raw` with the original dataset row

## Environment Setup

From PowerShell:

```powershell
. D:\Openmas-Gapp\config\q9.paths.ps1
cd $env:OPENMAS_CODE_ROOT
```

The local secrets file is:

```text
D:\Openmas-Gapp\secrets\q9_api_keys.local.ps1
```

Keep real API keys in the shell or in that local secrets file, not in source
docs.

## Next Steps

1. Keep `q9_datasets` as the canonical storage root.
2. Add raw dataset folders for the four Q9 datasets under
   `q9_datasets\raw\medical`.
3. Normalize the four datasets into `q9_datasets\normalized`.
4. Add or map the missing medical adapters for `MedMCQA` and
   `MMLU 医学子集`.
5. Run one-row smoke checks for each dataset once the normalized files exist.

## Minimal Q9 Run Shape

Once normalized data exists, one-case smoke runs can use the public CLI:

```powershell
python -m openmas_bench.cli run `
  --dataset MedQA `
  --data-root $env:OPENMAS_DATA_ROOT `
  --row-index 0 `
  --provider deterministic `
  --output "$env:OPENMAS_Q9_RUN_ROOT\medqa_row0_deterministic.json"
```

For a model-backed run, set `DASHSCOPE_API_KEY` or another compatible API key
and switch provider/model/base-url accordingly.
