"""Normalize downloaded MuSiQue, StrategyQA, and SciQ pilot rows."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "q2_datasets" / "raw_extra"
OUT = ROOT / "q2_datasets" / "normalized"


def read(name: str) -> list[dict]:
    return [json.loads(x) for x in (RAW / name).read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {
        "musique": read("musique_train_pilot.jsonl"),
        "strategyqa": read("strategyqa_train_pilot.jsonl"),
        "sciq": read("sciq_test_pilot.jsonl"),
    }
    normalized = {}
    for i, row in enumerate(rows["musique"]):
        paragraphs = row.get("paragraphs") or []
        context = "\n\n".join(
            f"[{p.get('title', '')}] {p.get('paragraph_text', p.get('paragraph', ''))}"
            for p in paragraphs if isinstance(p, dict)
        )
        normalized.setdefault("musique", []).append({
            "id": str(row.get("id", f"musique_{i:04d}")),
            "question": row.get("question", ""), "context": context,
            "choices": None, "answer": row.get("answer"), "source": "musique",
            "raw": row,
        })
    for i, row in enumerate(rows["strategyqa"]):
        facts = row.get("facts") or []
        decomposition = row.get("decomposition") or []
        context = "Facts:\n" + "\n".join(map(str, facts)) + "\nDecomposition:\n" + "\n".join(map(str, decomposition))
        normalized.setdefault("strategyqa", []).append({
            "id": str(row.get("qid", f"strategyqa_{i:04d}")),
            "question": row.get("question", ""), "context": context,
            "choices": ["yes", "no"], "answer": "yes" if row.get("answer") else "no",
            "source": "strategyqa", "raw": row,
        })
    for i, row in enumerate(rows["sciq"]):
        choices = [row.get("correct_answer"), row.get("distractor1"), row.get("distractor2"), row.get("distractor3")]
        normalized.setdefault("sciq", []).append({
            "id": f"sciq_{i:04d}", "question": row.get("question", ""),
            "context": row.get("support", ""), "choices": choices,
            "answer": row.get("correct_answer"), "source": "sciq", "raw": row,
        })
    for name, items in normalized.items():
        path = OUT / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n", encoding="utf-8")
        print(f"normalized {name}: {len(items)} rows -> {path}")


if __name__ == "__main__":
    main()
