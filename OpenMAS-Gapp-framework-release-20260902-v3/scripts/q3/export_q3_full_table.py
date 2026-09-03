from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Q3 result JSON file")
    parser.add_argument("--output", required=True, help="LaTeX table output path")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = payload.get("summary", [])
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Q3 structural benchmark results.}")
    lines.append(r"\label{tab:q3_full_results}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{Baseline} & \textbf{E2E (mean [95\% CI])} & \textbf{GraphPres.} & \textbf{OSV} & \textbf{Arch.} \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(
            f"{_escape(row.get('dataset'))} & {_escape(row.get('baseline'))} & "
            f"{_fmt_mean_ci(row.get('primary_score_mean'), row.get('primary_score_ci95_low'), row.get('primary_score_ci95_high'))} & {_fmt(row.get('graph_structural_preservation_mean'))} & "
            f"{_fmt(row.get('osv_mean'))} & {_fmt(row.get('architecture_validity_mean'))} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value) -> str:
    if value is None:
        return "--"
    return f"{float(value):.3f}"


def _fmt_mean_ci(mean, low, high) -> str:
    if mean is None:
        return "--"
    if low is None or high is None:
        return f"{float(mean):.3f}"
    return f"{float(mean):.3f} [{float(low):.3f}, {float(high):.3f}]"


def _escape(value) -> str:
    text = str(value)
    return text.replace("_", r"\_")


if __name__ == "__main__":
    main()
