# Q2 Extra Dataset Download Plan

The current Windows workspace cannot reach Hugging Face or GitHub
(`WinError 10060`), so no unverified partial download is stored. The
following datasets are the recommended additions from `QA 数据集汇总表(1).md`.

## Priority order

1. **MuSiQue**: strongest signal for multi-hop retrieval, typed dependencies,
   graph orchestration, and evidence synthesis.
2. **StrategyQA**: strongest signal for requirement grounding and explicit
   sub-question decomposition.
3. **SciQ**: stable, inexpensive evidence-backed multiple choice control.
4. **MathQA**: program-labelled arithmetic and operation selection; useful
   companion to FinQA.

Do not add GAIA/WebArena/AgentBench to the first Q2 table. Their external
environment state would dominate a component-wise construction ablation.

## Download on a server with network access

Install the dataset client in the server environment:

```bash
pip install -U datasets huggingface_hub
mkdir -p /data2/jiangjiaqi/openmas/q2_datasets/raw_extra
```

Download only the frozen pilot splits, then export JSONL records:

```bash
python - <<'PY'
from datasets import load_dataset
from pathlib import Path
import json

out = Path('/data2/jiangjiaqi/openmas/q2_datasets/raw_extra')
out.mkdir(parents=True, exist_ok=True)
specs = [
    ('musique', 'StonyBrookNLP/musique', 'default', 'validation', 100),
    ('strategyqa', 'Chiahuali/StrategyQA', 'default', 'validation', 100),
    ('sciq', 'allenai/sciq', 'default', 'test', 100),
    ('mathqa', 'MathQA/MathQA', 'default', 'test', 100),
]
for name, repo, config, split, limit in specs:
    ds = load_dataset(repo, config, split=split)
    rows = [dict(ds[i]) for i in range(min(limit, len(ds)))]
    (out / f'{name}_{split}_pilot.jsonl').write_text(
        '\n'.join(json.dumps(x, ensure_ascii=False) for x in rows) + '\n',
        encoding='utf-8')
    print(name, len(rows))
PY
```

Copy the resulting files back to `q2_datasets/raw_extra/` and record SHA-256,
dataset revision, config, split, and license before adding an adapter. The
qualification script must pass before any LLM run.

## Why these four

| Dataset | Graph Harness signal | Evaluator |
|---|---|---|
| MuSiQue | multi-hop typed evidence and branch merging | answer EM/F1 plus evidence coverage |
| StrategyQA | requirement decomposition and hidden subgoals | yes/no accuracy and decomposition coverage |
| SciQ | evidence retrieval under a compact choice contract | accuracy |
| MathQA | operation selection and executable arithmetic program | option accuracy plus program execution |

The existing local `PubMedQA` and `FinQA` pilot files remain usable as
conditional candidates; they do not depend on this download.
