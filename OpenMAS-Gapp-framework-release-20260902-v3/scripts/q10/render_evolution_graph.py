"""Render the Q10 financial MAS construction stages as a PNG."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from openmas_bench.q10.financial_profiles import get_financial_profile  # noqa: E402

CANVAS_W = 1920
CANVAS_H = 1080
PANEL_W = 430
PANEL_H = 930
PANEL_GAP = 24
PANEL_X0 = 40
PANEL_Y0 = 70
RADIUS = 31
INPUT_W = 150
INPUT_H = 72

COLORS = {
    "input": ("#fff7ed", "#ea580c"),
    "task_pattern": ("#ecfeff", "#0891b2"),
    "component_requirement": ("#eef2ff", "#4f46e5"),
    "agent": ("#eff6ff", "#2563eb"),
    "control": ("#fef2f2", "#dc2626"),
    "resource": ("#f0fdf4", "#16a34a"),
    "capability": ("#eef2ff", "#4f46e5"),
    "constraint": ("#fef2f2", "#dc2626"),
    "default": ("#f8fafc", "#64748b"),
}


def _fonts() -> dict[str, ImageFont.FreeTypeFont]:
    paths = [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"]
    selected = next((path for path in paths if Path(path).exists()), None)
    if selected:
        return {
            "title": ImageFont.truetype(selected, 30),
            "subtitle": ImageFont.truetype(selected, 16),
            "panel_title": ImageFont.truetype(selected, 18),
            "panel_subtitle": ImageFont.truetype(selected, 13),
            "node": ImageFont.truetype(selected, 14),
            "edge": ImageFont.truetype(selected, 10),
            "large": ImageFont.truetype(selected, 14),
        }
    fallback = ImageFont.load_default()
    return {key: fallback for key in ("title", "subtitle", "panel_title", "panel_subtitle", "node", "edge", "large")}


def _wrap(value: object, width: int, limit: int = 3) -> list[str]:
    text = str(value).replace("\r", " ").strip()
    if not text:
        return [""]
    lines: list[str] = []
    for part in text.splitlines():
        chunk = re.sub(r"\s+", " ", part.replace("_", " ").replace("-", " ")).strip()
        if not chunk:
            continue
        wrapped = textwrap.wrap(chunk, width=width, break_long_words=False, break_on_hyphens=False) or [chunk]
        lines.extend(wrapped)
    if not lines:
        lines = [text]
    if len(lines) > limit:
        lines = lines[:limit]
        lines[-1] = lines[-1][: max(3, width - 3)] + "..." if len(lines[-1]) > width else lines[-1]
    return lines


def _pretty(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value).replace("_", " ").replace("-", " ")).strip()
    return text.title() if text else ""


def _dataset(record: dict[str, Any]) -> str:
    blueprint = record.get("blueprint_metadata")
    task_profile = blueprint.get("task_profile", {}) if isinstance(blueprint, dict) else {}
    return str(record.get("dataset") or task_profile.get("dataset") or "FinanceBench")


def _question(record: dict[str, Any]) -> str:
    executions = record.get("task_execution") or {}
    if isinstance(executions, dict):
        for execution in executions.get("node_executions", []):
            artifact = execution.get("artifact")
            if not isinstance(artifact, str):
                continue
            try:
                payload = json.loads(artifact)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                for key in ("original_question", "question", "prompt"):
                    if payload.get(key):
                        return str(payload[key])
    for key in ("question", "prompt"):
        if record.get(key):
            return str(record[key])
    return str(record.get("dataset") or "dataset item")


def _analysis(record: dict[str, Any]) -> dict[str, Any]:
    analysis = record.get("q10_analysis")
    if isinstance(analysis, dict) and analysis.get("tasks") and analysis.get("edges") and analysis.get("constraints"):
        return analysis
    dataset = _dataset(record)
    profile = get_financial_profile(dataset)
    if profile["dataset"] == "FinanceBench":
        objectives = {
            "filing_evidence": "Retrieve audited filing evidence and metric definitions",
            "market_risk_evidence": "Retrieve risk-factor, uncertainty and forward-looking evidence",
            "financial_analysis": "Analyze financial metrics, trend drivers and evidence consistency",
            "risk_assessment": "Assess material risk, sensitivity and unsupported inference risk",
            "compliance_review": "Check regulatory, disclosure and non-advice constraints",
            "audit_trail": "Record evidence lineage, assumptions and control decisions",
            "final_report": "Return the dataset-specific financial answer contract",
        }
        edges = [
            {"source": "filing_evidence", "target": "financial_analysis", "relation": "precedes"},
            {"source": "market_risk_evidence", "target": "risk_assessment", "relation": "precedes"},
            {"source": "financial_analysis", "target": "risk_assessment", "relation": "requires"},
            {"source": "financial_analysis", "target": "compliance_review", "relation": "requires"},
            {"source": "risk_assessment", "target": "audit_trail", "relation": "reviews"},
            {"source": "compliance_review", "target": "audit_trail", "relation": "reviews"},
            {"source": "audit_trail", "target": "final_report", "relation": "precedes"},
            {"source": "compliance_review", "target": "risk_assessment", "relation": "feedback"},
        ]
        evidence_mode = "filing"
        risk_level = "low"
    else:
        objectives = {
            "report_table_evidence": "Retrieve table cells and line items from the financial report",
            "narrative_disclosure_evidence": "Retrieve relevant narrative disclosures and footnotes",
            "calculation_analysis": "Compute the numeric financial result with unit handling",
            "risk_assessment": "Assess material risk, sensitivity and unsupported inference risk",
            "compliance_review": "Check regulatory, disclosure and non-advice constraints",
            "audit_trail": "Record evidence lineage, assumptions and control decisions",
            "final_report": "Return the dataset-specific financial answer contract",
        }
        edges = [
            {"source": "report_table_evidence", "target": "calculation_analysis", "relation": "precedes"},
            {"source": "narrative_disclosure_evidence", "target": "risk_assessment", "relation": "precedes"},
            {"source": "calculation_analysis", "target": "risk_assessment", "relation": "requires"},
            {"source": "calculation_analysis", "target": "compliance_review", "relation": "requires"},
            {"source": "risk_assessment", "target": "audit_trail", "relation": "reviews"},
            {"source": "compliance_review", "target": "audit_trail", "relation": "reviews"},
            {"source": "audit_trail", "target": "final_report", "relation": "precedes"},
            {"source": "compliance_review", "target": "calculation_analysis", "relation": "feedback"},
        ]
        evidence_mode = "table"
        risk_level = "medium"
    return {
        "dataset": profile["dataset"],
        "task_family": profile["task_family"],
        "financial_focus": "dataset-driven financial analysis, risk assessment and compliance review",
        "risk_level": risk_level,
        "evidence_mode": evidence_mode,
        "tasks": [{"id": task, "objective": objectives[task]} for task in profile["baseline_tasks"]],
        "edges": edges,
        "constraints": [{"id": control, "target": "final_report", "predicate": "required"} for control in profile["required_controls"]],
    }


def _task_label(task_id: str) -> str:
    return _pretty(task_id)


def _control_label(control_id: str) -> str:
    return _pretty(control_id)


def _build_panels(record: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = _dataset(record)
    analysis = _analysis(record)
    task_ids = [str(item["id"]) for item in analysis["tasks"]]
    task_labels = {item["id"]: _task_label(item["id"]) for item in analysis["tasks"]}
    final_task = task_labels.get(task_ids[-1], _task_label(task_ids[-1])) if task_ids else "Final Report"

    input_panel = {
        "title": "1  Input",
        "subtitle": "Dataset row becomes a construction case",
        "nodes": [
            {"id": "question", "kind": "input", "label": "\n".join(["Question", *_wrap(_question(record), 19, 4)])},
            {"id": "context", "kind": "input", "label": "Context / choices"},
            {"id": "output", "kind": "input", "label": "\n".join(["Required output", *_wrap(final_task, 18, 2)])},
            {"id": "case", "kind": "task_pattern", "label": f"{dataset}\ncase input"},
        ],
        "edges": [
            {"source": "question", "target": "case", "relation": "input"},
            {"source": "context", "target": "case", "relation": "input"},
            {"source": "output", "target": "case", "relation": "contract"},
        ],
    }

    requirement_panel = {
        "title": "2  Requirement Model",
        "subtitle": "Financial tasks and dependency edges are made explicit",
        "nodes": [{"id": task_id, "kind": "task_pattern", "label": task_labels[task_id]} for task_id in task_ids],
        "edges": [{"source": str(edge["source"]), "target": str(edge["target"]), "relation": str(edge["relation"])} for edge in analysis["edges"]],
    }

    blueprint_nodes: list[dict[str, Any]] = []
    blueprint_edges: list[dict[str, Any]] = []
    for task_id in task_ids:
        blueprint_nodes.append({"id": task_id, "kind": "task_pattern", "label": task_labels[task_id]})
        blueprint_nodes.append({"id": f"req_{task_id}", "kind": "component_requirement", "label": f"Req {_task_label(task_id)}"})
        blueprint_edges.append({"source": task_id, "target": f"req_{task_id}", "relation": "requires"})
    for index, task_id in enumerate(task_ids[:2]):
        resource_id = f"resource_{task_id}"
        blueprint_nodes.append({"id": resource_id, "kind": "resource", "label": f"Branch {index}"})
        blueprint_edges.append({"source": resource_id, "target": f"req_{task_id}", "relation": "uses"})
    for edge in analysis["edges"]:
        blueprint_edges.append({"source": str(edge["source"]), "target": str(edge["target"]), "relation": str(edge["relation"])})
    for constraint in analysis["constraints"]:
        blueprint_nodes.append({"id": str(constraint["id"]), "kind": "control", "label": _control_label(constraint["id"])})
        blueprint_edges.append({"source": "final_report", "target": str(constraint["id"]), "relation": "constrained_by"})
    blueprint_panel = {
        "title": "3  Blueprint",
        "subtitle": "Tasks bind to requirements, resources and controls",
        "nodes": blueprint_nodes,
        "edges": blueprint_edges,
    }

    application = record.get("application") or record.get("q10_harness") or {}
    app_nodes = [node for node in application.get("nodes", []) if isinstance(node, dict)]
    app_edges = [edge for edge in application.get("edges", []) if isinstance(edge, dict)]
    executable_nodes: list[dict[str, Any]] = []
    for node in app_nodes:
        kind = str(node.get("kind") or "default")
        node_id = str(node.get("id") or "")
        impl = str(node.get("implementation_ref") or node_id)
        meta = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        if kind == "agent":
            core = impl.removeprefix("component_")
            label = f"{_pretty(core)}\nAgent"
        elif kind == "control":
            core = impl.removeprefix("component_")
            label = f"{_pretty(core)}\nControl"
        elif kind == "resource":
            core = str(meta.get("resource_key") or node_id.removeprefix("resource_") or node_id)
            label = f"{_pretty(core)}\nResource"
        elif kind == "component":
            core = impl.removeprefix("component_") if impl else node_id
            label = f"{_pretty(core)}\nComponent"
        else:
            label = _pretty(impl or node_id)
        executable_nodes.append({"id": node_id, "kind": kind, "label": label})
    executable_panel = {
        "title": "4  Executable MAS",
        "subtitle": "Concrete agents, resources, controls and execution edges",
        "nodes": executable_nodes,
        "edges": [{"source": str(edge.get("source")), "target": str(edge.get("target")), "relation": str(edge.get("relation") or "execution")} for edge in app_edges],
    }

    return [input_panel, requirement_panel, blueprint_panel, executable_panel]


def _layout(node_ids: list[str], edges: list[dict[str, Any]] | list[tuple[str, str, str]], x0: int, y0: int) -> dict[str, tuple[float, float]]:
    outgoing: dict[str, list[str]] = defaultdict(list)
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        source = str(edge.get("source")) if isinstance(edge, dict) else str(edge[0])
        target = str(edge.get("target")) if isinstance(edge, dict) else str(edge[1])
        if source in indegree and target in indegree:
            outgoing[source].append(target)
            indegree[target] += 1
    levels = {node_id: 0 for node_id in node_ids}
    queue = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    seen = 0
    while queue:
        source = queue.popleft()
        seen += 1
        for target in outgoing[source]:
            levels[target] = max(levels[target], levels[source] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if seen != len(node_ids):
        return _circle_layout(node_ids, x0, y0)
    by_level: dict[int, list[str]] = defaultdict(list)
    for node_id in node_ids:
        by_level[levels[node_id]].append(node_id)
    max_level = max(by_level, default=0)
    positions: dict[str, tuple[float, float]] = {}
    usable_w = PANEL_W - 140
    usable_h = PANEL_H - 180
    for level in range(max_level + 1):
        column = by_level[level]
        x = x0 + 45 + level * (usable_w / max(1, max_level))
        if len(column) == 1:
            positions[column[0]] = (x, y0 + 40 + usable_h / 2)
            continue
        gap = min(96, max(62, usable_h / max(1, len(column) - 1)))
        total = gap * (len(column) - 1)
        start = y0 + 40 + max(0, (usable_h - total) / 2)
        for index, node_id in enumerate(column):
            positions[node_id] = (x, start + index * gap)
    return positions


def _circle_layout(node_ids: list[str], x0: int, y0: int) -> dict[str, tuple[float, float]]:
    count = max(1, len(node_ids))
    center = (x0 + 45 + (PANEL_W - 140) / 2, y0 + 40 + (PANEL_H - 180) / 2)
    radius = min((PANEL_W - 170) / 2, (PANEL_H - 240) / 2)
    radius = max(96, radius)
    return {
        node_id: (
            center[0] + radius * math.cos(2 * math.pi * index / count - math.pi / 2),
            center[1] + radius * math.sin(2 * math.pi * index / count - math.pi / 2),
        )
        for index, node_id in enumerate(node_ids)
    }


def _draw_panel(draw: ImageDraw.ImageDraw, x: int, y: int, panel: dict[str, Any], fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    draw.rounded_rectangle((x, y, x + PANEL_W, y + PANEL_H), radius=10, fill="#ffffff", outline="#cbd5e1", width=2)
    draw.text((x + 16, y + 14), panel["title"], fill="#0f172a", font=fonts["panel_title"])
    draw.multiline_text((x + 16, y + 45), "\n".join(_wrap(panel["subtitle"], 37)), fill="#64748b", font=fonts["panel_subtitle"], spacing=2)
    nodes = panel["nodes"]
    edges = panel["edges"]
    positions = _layout([str(node["id"]) for node in nodes], edges, x + 30, y + 118)
    by_id = {str(node["id"]): node for node in nodes}
    for edge in edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source in positions and target in positions:
            _draw_edge(draw, positions[source], positions[target], str(edge.get("relation") or ""), fonts["edge"])
    for node_id, (cx, cy) in positions.items():
        _draw_node(draw, by_id[node_id], (cx, cy), fonts)


def _draw_node(draw: ImageDraw.ImageDraw, node: dict[str, Any], position: tuple[float, float], fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    x, y = position
    kind = str(node.get("kind") or "default")
    fill, stroke = COLORS.get(kind, COLORS["default"])
    label = str(node.get("label") or node.get("id") or "")
    if kind == "input":
        draw.rounded_rectangle((x - INPUT_W / 2, y - INPUT_H / 2, x + INPUT_W / 2, y + INPUT_H / 2), radius=10, fill=fill, outline=stroke, width=2)
        lines = _wrap(label, 18, 4)
    else:
        draw.ellipse((x - RADIUS, y - RADIUS, x + RADIUS, y + RADIUS), fill=fill, outline=stroke, width=2)
        lines = _wrap(label, 14, 3)
    heights = [draw.textbbox((0, 0), line, font=fonts["node"])[3] - draw.textbbox((0, 0), line, font=fonts["node"])[1] for line in lines]
    total = sum(heights) + 2 * max(0, len(lines) - 1)
    current_y = y - total / 2
    for index, line in enumerate(lines):
        bounds = draw.textbbox((0, 0), line, font=fonts["node"])
        draw.text((x - (bounds[2] - bounds[0]) / 2, current_y), line, fill="#0f172a", font=fonts["node"])
        current_y += heights[index] + 2


def _draw_edge(draw: ImageDraw.ImageDraw, source: tuple[float, float], target: tuple[float, float], relation: str, font: ImageFont.FreeTypeFont) -> None:
    sx, sy = source
    tx, ty = target
    dx, dy = tx - sx, ty - sy
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    start = (sx + ux * RADIUS, sy + uy * RADIUS)
    end = (tx - ux * RADIUS, ty - uy * RADIUS)
    draw.line((*start, *end), fill="#475569", width=2)
    px, py = -uy, ux
    size = 8
    draw.polygon(
        [
            (end[0], end[1]),
            (end[0] - ux * size + px * 4, end[1] - uy * size + py * 4),
            (end[0] - ux * size - px * 4, end[1] - uy * size - py * 4),
        ],
        fill="#475569",
    )
    if relation:
        label = relation.replace("_", " ")
        bounds = draw.textbbox((0, 0), label, font=font)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        draw.rounded_rectangle((mx - width / 2 - 5, my - height / 2 - 3, mx + width / 2 + 5, my + height / 2 + 3), radius=4, fill="#ffffff", outline="#ffffff")
        draw.text((mx - width / 2, my - height / 2), label, fill="#64748b", font=font)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Q10 Input/Requirement/Blueprint/Executable MAS stages.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record = json.loads(args.input.read_text(encoding="utf-8"))
    panels = _build_panels(record)
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), "#e2e8f0")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    for index, panel in enumerate(panels):
        x = PANEL_X0 + index * (PANEL_W + PANEL_GAP)
        _draw_panel(draw, x, PANEL_Y0, panel, fonts)
        if index < len(panels) - 1:
            _draw_transition(draw, x + PANEL_W + 4, PANEL_Y0 + PANEL_H / 2, fonts["large"])
    dataset = str(record.get("dataset") or "dataset")
    draw.text((24, 18), f"Q10 Financial MAS Graph Evolution | {dataset}", fill="#0f172a", font=fonts["title"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, "PNG")
    print(f"wrote {args.output} {image.size[0]}x{image.size[1]}")


def _draw_transition(draw: ImageDraw.ImageDraw, x: float, y: float, font: ImageFont.FreeTypeFont) -> None:
    draw.line((x, y, x + 28, y), fill="#0f766e", width=3)
    draw.polygon([(x + 36, y), (x + 26, y - 6), (x + 26, y + 6)], fill="#0f766e")
    draw.text((x + 2, y + 10), "compile", fill="#0f766e", font=font)


if __name__ == "__main__":
    main()
