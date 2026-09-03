"""Render publication-style Q2 tables from a completed cross-dataset run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value, digits=3):
    return "NA" if value is None else f"{float(value):.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    expected_protocol = "q2-graph-resource-isolation-v3"
    actual_protocol = payload.get("causal_protocol_version")
    if actual_protocol != expected_protocol:
        raise ValueError(
            f"refusing to render stale Q2 result: causal protocol {actual_protocol!r}; "
            f"expected {expected_protocol!r}"
        )
    primary = payload["primary_table"]
    structural = payload["structural_table"]
    variants = payload["variants"]
    labels = {
        "full_graph_harness": "Full Graph Harness",
        "w/o_requirement_grounding": "w/o Requirement Grounding",
        "w/o_graph_orchestration": "w/o Graph Orchestration",
        "w/o_blueprint": "w/o Blueprint",
        "w/o_constraint_aware_orchestration": "w/o Constraint-aware",
        "w/o_realization": "w/o Realization",
    }
    lines = [
        "# Q2 Ablation Results",
        "",
        f"Source: `{args.input.name}`; provider: `{payload.get('provider')}`; model: `{payload.get('model')}`.",
        f"Completed runs: {payload.get('completed_runs')}/{payload.get('expected_runs')}; failed: {payload.get('failed_runs', 0)}.",
        "",
        "## Table 1. Primary task performance",
        "",
        "Values are mean +/- SD over rows and seeds. Delta is paired variant minus Full; positive values are allowed and are reported without correction.",
        "",
        "| Dataset | Metric | Full | w/o Requirement | w/o Graph | w/o Blueprint | w/o Constraint | w/o Realization |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_dataset = {}
    for row in primary:
        by_dataset.setdefault(row["dataset"], {})[row["variant"]] = row
    order = ["full_graph_harness", "w/o_requirement_grounding", "w/o_graph_orchestration", "w/o_blueprint", "w/o_constraint_aware_orchestration", "w/o_realization"]
    for dataset, rows in by_dataset.items():
        metric = rows["full_graph_harness"]["metric"]
        cells = []
        for variant in order:
            row = rows.get(variant)
            if row is None:
                cells.append("NA")
            else:
                delta = row.get("paired_delta_vs_full")
                suffix = "" if variant == "full_graph_harness" else f" ({delta:+.3f})"
                cells.append(f"{fmt(row['mean'])} +/- {fmt(row.get('std'))}{suffix}")
        lines.append(f"| {dataset} | {metric} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Table 2. Structural module fingerprints",
        "",
        "| Dataset | Variant | Req. coverage | Relation recall | Blueprint fidelity | Constraint detection | Constraint orchestration | Construction quality |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in structural:
        lines.append(
            f"| {row['dataset']} | {labels.get(row['variant'], row['variant'])} | "
            f"{fmt(row.get('requirement_coverage'))} | {fmt(row.get('orchestration_relation_recall'))} | "
            f"{fmt(row.get('realization_fidelity'))} | {fmt(row.get('constraint_recall'))} | "
            f"{fmt(row.get('constraint_orchestration_recall'))} | {fmt(row.get('construction_quality'))} |"
        )

    lines += [
        "",
        "## Table 3. Paired effect summary",
        "",
        "| Dataset | Variant | Paired delta vs Full | 95% CI | Full wins / ties / losses |",
        "|---|---|---:|---|---:|",
    ]
    for row in primary:
        if row["variant"] == "full_graph_harness":
            continue
        ci = row.get("paired_delta_ci95")
        wtl = row.get("wins_ties_losses_vs_full", {})
        ci_text = "NA" if not ci else f"[{fmt(ci[0])}, {fmt(ci[1])}]"
        lines.append(
            f"| {row['dataset']} | {labels.get(row['variant'], row['variant'])} | "
            f"{fmt(row.get('paired_delta_vs_full'))} | {ci_text} | "
            f"{wtl.get('wins', 0)} / {wtl.get('ties', 0)} / {wtl.get('losses', 0)} |"
        )

    lines += [
        "",
        "*Note.* Structural metrics are contract-based diagnostics. Primary task scores are dataset-specific and should not be interpreted as requiring Full Graph Harness to dominate every dataset. Formal claims should use paired deltas and confidence intervals.",
        "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
