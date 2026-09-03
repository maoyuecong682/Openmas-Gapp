from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests


SPECS = [
    ("musique", "bdsaglam/musique", "default", "train"),
    ("strategyqa", "tasksource/strategy-qa", "default", "train"),
    ("sciq", "allenai/sciq", "default", "test"),
    ("mathqa", "rootacess/math-qa-classification", "default", "test"),
]


def main() -> None:
    out = Path(__file__).resolve().parents[3] / "q2_datasets" / "raw_extra"
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, dataset, config, split in SPECS:
        if name == "strategyqa":
            response = requests.get("https://huggingface.co/datasets/tasksource/strategy-qa/resolve/main/strategyQA_train.json", timeout=(20, 120))
            response.raise_for_status()
            raw = response.json()
            rows = raw[:100] if isinstance(raw, list) else []
        else:
            response = requests.get(
                "https://datasets-server.huggingface.co/rows",
                params={"dataset": dataset, "config": config, "split": split,
                        "offset": 0, "length": 100},
                timeout=(20, 120),
            )
            response.raise_for_status()
            payload = response.json()
            rows = [item["row"] for item in payload.get("rows", [])]
        if not rows:
            raise RuntimeError(f"{name}: endpoint returned no rows")
        path = out / f"{name}_{split}_pilot.jsonl"
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append({"name": name, "dataset": dataset, "config": config,
                         "split": split, "rows": len(rows), "sha256": digest,
                         "source": "https://datasets-server.huggingface.co/rows"})
        print(f"downloaded {name}: {len(rows)} rows sha256={digest}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
