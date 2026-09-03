from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openmas_bench.llm import LLMConfig, OpenAICompatibleAdapter
from openmas_bench.sandbox import run_swebench_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--seeds", default="11,22,33")
    p.add_argument("--output", default=str(ROOT.parents[1] / "swebench_real_patch_eval.json"))
    p.add_argument("--base-url", default="https://api.deepseek.com")
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--api-key-env", default="Q1_LLM_API_KEY")
    args = p.parse_args()
    key = os.environ.get(args.api_key_env)
    if not key:
        raise RuntimeError(f"missing {args.api_key_env}")
    adapter = OpenAICompatibleAdapter(LLMConfig(provider="openai_compatible", model=args.model,
        base_url=args.base_url, api_key=key, temperature=0.0, max_output_tokens=2048))
    path = ROOT.parents[1] / "q2_datasets" / "normalized" / "swebench_verified.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()[:args.rows]]
    results = []
    for row in rows:
        raw = row["raw"]
        for seed in [int(x) for x in args.seeds.split(",")]:
            prompt = ("Produce a unified diff patch for the issue below. Return JSON with exactly one field `patch`. "
                      "Do not include markdown fences.\nISSUE:\n" + str(raw.get("problem_statement", "")) +
                      "\nHINTS:\n" + str(raw.get("hints_text", ""))[:8000])
            response = adapter.generate_json("You are a careful software maintenance solver.", prompt, seed, {"patch"})
            evaluation = run_swebench_tests(row, response.value.get("patch"), timeout=180)
            results.append({"instance_id": raw.get("instance_id"), "seed": seed,
                "passed": evaluation.get("passed", False), "evaluation": evaluation,
                "patch": response.value.get("patch"), "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens, "retries": response.retry_count,
                "json_repaired": response.json_repaired})
            print(raw.get("instance_id"), seed, evaluation.get("passed"), evaluation.get("error"), flush=True)
    payload = {"protocol": "SWE-bench real patch application/test v1", "model": args.model,
        "rows": args.rows, "seeds": args.seeds, "gold_patch_in_prompt": False,
        "passed": sum(x["passed"] for x in results), "total": len(results), "results": results}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
