from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openmas_bench.io import write_json
from openmas_bench.q3 import build_q3_suite, render_markdown_tables, run_q3_experiment


def main() -> None:
    parser = argparse.ArgumentParser(prog="run_q3_experiment")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=False)
    parser.add_argument("--seeds", default="11,22,33")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    payload = run_q3_experiment(seeds=seeds, cases=build_q3_suite())
    write_json(args.output, payload)
    if args.markdown:
        Path(args.markdown).write_text(render_markdown_tables(payload["tables"]), encoding="utf-8")


if __name__ == "__main__":
    main()

