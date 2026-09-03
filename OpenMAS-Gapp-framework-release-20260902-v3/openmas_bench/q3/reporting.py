from __future__ import annotations

from typing import Any

from .baselines import Q3_BASELINES


def render_markdown_tables(tables: dict[str, list[dict[str, Any]]]) -> str:
    lines = []
    for dataset_id, rows in sorted(tables.items()):
        lines.append(f"## {dataset_id}")
        lines.append("")
        lines.append("| Orchestration Representation | Seq Success ↑ | Branch Success ↑ | Loop Success ↑ | Constraint Success ↑ | Overall E2E Success ↑ | OSV ↑ | Graph Preservation ↑ |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            lines.append(
                f"| {row['baseline']} | {_fmt(row['seq_success'])} | {_fmt(row['branch_success'])} | "
                f"{_fmt(row['loop_success'])} | {_fmt(row['constraint_success'])} | {_fmt(row['overall_e2e_success'])} | {_fmt(row['osv'])} | {_fmt(row.get('graph_structural_preservation', 0.0))} |"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _fmt(value: Any) -> str:
    if isinstance(value, dict):
        return f"{value['mean']:.2f} ± {value['std']:.2f}"
    return f"{float(value):.2f}"
