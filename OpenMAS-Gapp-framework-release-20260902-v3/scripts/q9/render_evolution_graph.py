from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


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
    "task": ("#ecfeff", "#0891b2"),
    "component_requirement": ("#eef2ff", "#4f46e5"),
    "agent": ("#eff6ff", "#2563eb"),
    "control": ("#fef2f2", "#dc2626"),
    "resource": ("#f0fdf4", "#16a34a"),
    "default": ("#f8fafc", "#64748b"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render four structural stages of a Q9 MAS graph as one PNG."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    record = json.loads(Path(args.input).read_text(encoding="utf-8"))
    panels = _build_panels(record)
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), "#e2e8f0")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()

    for index, panel in enumerate(panels):
        x = PANEL_X0 + index * (PANEL_W + PANEL_GAP)
        _draw_panel(draw, x, PANEL_Y0, panel, fonts)
        if index < len(panels) - 1:
            _draw_transition(draw, x + PANEL_W + 4, PANEL_Y0 + PANEL_H / 2, fonts["large"])

    title = f"Q9 MAS Graph Evolution | {record.get('dataset', 'dataset')}"
    draw.text((24, 18), title, fill="#0f172a", font=fonts["title"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG")
    print(f"wrote {output} {image.size[0]}x{image.size[1]}")


def _build_panels(record: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = str(record.get("dataset") or "dataset")
    question = _question(record)
    task_ids = _task_ids(dataset)
    task_labels = {task_id: _task_label(task_id) for task_id in task_ids}
    dependency_edges = list(zip(task_ids, task_ids[1:]))

    application = record.get("application") or {}
    app_nodes = [node for node in application.get("nodes", []) if isinstance(node, dict)]
    app_edges = [edge for edge in application.get("edges", []) if isinstance(edge, dict)]

    control_nodes = [node for node in app_nodes if str(node.get("kind")) == "control"]
    control_label = _short(
        str(control_nodes[0].get("implementation_ref", "review_control")).replace("component_", "").replace("_", " ")
        if control_nodes
        else "review_control",
        24,
    )

    input_nodes = [
        {"id": "question", "label": "\n".join(["Question", *_wrap(question, 19, 4)]), "kind": "input"},
        {"id": "context", "label": "Context / choices", "kind": "input"},
        {"id": "output", "label": "\n".join(["Required output", *_wrap(task_labels[task_ids[-1]], 18, 2)]), "kind": "input"},
        {"id": "case", "label": f"{dataset}\ncase input", "kind": "task"},
    ]
    input_edges = [
        ("question", "case", "input"),
        ("context", "case", "input"),
        ("output", "case", "contract"),
    ]

    requirement_nodes = [{"id": task_id, "label": task_labels[task_id], "kind": "task"} for task_id in task_ids]
    requirement_edges = [
        {"source": source, "target": target, "relation": "dependency"}
        for source, target in dependency_edges
    ]

    blueprint_nodes = [{"id": task_id, "label": task_labels[task_id], "kind": "task"} for task_id in task_ids]
    blueprint_nodes.extend(
        {
            "id": f"req_{task_id}",
            "label": f"req {task_labels[task_id].lower()}",
            "kind": "component_requirement",
        }
        for task_id in task_ids
    )
    if control_nodes:
        blueprint_nodes.append({"id": "blueprint_control", "label": control_label, "kind": "control"})
    blueprint_edges = [
        {"source": source, "target": target, "relation": "precedes"}
        for source, target in dependency_edges
    ]
    blueprint_edges.extend(
        {"source": task_id, "target": f"req_{task_id}", "relation": "requires"}
        for task_id in task_ids
    )
    if control_nodes:
        blueprint_edges.extend(
            [
                {"source": "review", "target": "blueprint_control", "relation": "constrained_by"},
                {"source": "blueprint_control", "target": "answer", "relation": "precedes"},
            ]
        )

    executable_nodes = [
        {
            "id": str(node.get("id")),
            "label": _agent_label(node),
            "kind": str(node.get("kind") or "default"),
        }
        for node in app_nodes
    ]
    executable_edges = [
        {
            "source": str(edge.get("source")),
            "target": str(edge.get("target")),
            "relation": str(edge.get("relation") or "execution"),
        }
        for edge in app_edges
    ]

    return [
        {
            "title": "1  Input",
            "subtitle": "Dataset row becomes a construction case",
            "nodes": input_nodes,
            "edges": input_edges,
        },
        {
            "title": "2  Requirement Model",
            "subtitle": "Tasks and dependencies are made explicit",
            "nodes": requirement_nodes,
            "edges": requirement_edges,
        },
        {
            "title": "3  Blueprint",
            "subtitle": "Tasks bind to components and controls",
            "nodes": blueprint_nodes,
            "edges": blueprint_edges,
        },
        {
            "title": "4  Executable MAS",
            "subtitle": "Concrete agents and execution edges",
            "nodes": executable_nodes,
            "edges": executable_edges,
        },
    ]


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    panel: dict[str, Any],
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> None:
    draw.rounded_rectangle((x, y, x + PANEL_W, y + PANEL_H), radius=10, fill="#ffffff", outline="#cbd5e1", width=2)
    draw.text((x + 16, y + 14), panel["title"], fill="#0f172a", font=fonts["heading"])
    draw.multiline_text(
        (x + 16, y + 45),
        "\n".join(_wrap(panel["subtitle"], 37)),
        fill="#64748b",
        font=fonts["small"],
        spacing=2,
    )

    nodes = panel["nodes"]
    edges = panel["edges"]
    positions = _layout([str(node["id"]) for node in nodes], edges, x + 30, y + 118)
    by_id = {str(node["id"]): node for node in nodes}

    for edge in edges:
        if isinstance(edge, dict):
            source = str(edge.get("source"))
            target = str(edge.get("target"))
            relation = str(edge.get("relation") or "")
        else:
            source = str(edge[0])
            target = str(edge[1])
            relation = str(edge[2]) if len(edge) > 2 else ""
        if source in positions and target in positions:
            _draw_edge(draw, positions[source], positions[target], relation, fonts["edge"])

    for node_id in [str(node["id"]) for node in nodes]:
        _draw_node(draw, by_id[node_id], positions[node_id], fonts)


def _draw_node(
    draw: ImageDraw.ImageDraw,
    node: dict[str, Any],
    position: tuple[float, float],
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> None:
    x, y = position
    kind = str(node.get("kind") or "default")
    fill, stroke = COLORS.get(kind, COLORS["default"])
    if kind == "input":
        draw.rounded_rectangle((x - INPUT_W / 2, y - INPUT_H / 2, x + INPUT_W / 2, y + INPUT_H / 2), radius=10, fill=fill, outline=stroke, width=2)
        labels = _wrap(str(node.get("label") or node.get("id")), 18, 4)
        node_font = fonts["node"]
    elif kind == "component_requirement":
        draw.ellipse((x - RADIUS, y - RADIUS, x + RADIUS, y + RADIUS), fill=fill, outline=stroke, width=2)
        labels = _wrap(str(node.get("label") or node.get("id")), 8, 2)
        node_font = fonts["small"]
    else:
        draw.ellipse((x - RADIUS, y - RADIUS, x + RADIUS, y + RADIUS), fill=fill, outline=stroke, width=2)
        labels = _wrap(str(node.get("label") or node.get("id")), 14, 3)
        node_font = fonts["node"]

    heights = [draw.textbbox((0, 0), label, font=node_font)[3] - draw.textbbox((0, 0), label, font=node_font)[1] for label in labels]
    total = sum(heights) + 2 * max(0, len(labels) - 1)
    current_y = y - total / 2
    for index, label in enumerate(labels):
        bounds = draw.textbbox((0, 0), label, font=node_font)
        draw.text((x - (bounds[2] - bounds[0]) / 2, current_y), label, fill="#0f172a", font=node_font)
        current_y += heights[index] + 2


def _draw_edge(
    draw: ImageDraw.ImageDraw,
    source: tuple[float, float],
    target: tuple[float, float],
    relation: str,
    font: ImageFont.FreeTypeFont,
) -> None:
    sx, sy = source
    tx, ty = target
    dx, dy = tx - sx, ty - sy
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    start = (sx + ux * RADIUS, sy + uy * RADIUS)
    end = (tx - ux * (RADIUS + 8), ty - uy * (RADIUS + 8))
    draw.line((start, end), fill="#475569", width=2)
    px, py = -uy, ux
    arrow_size = 8
    arrow_width = 4
    ex, ey = end
    draw.polygon(
        [
            (ex, ey),
            (ex - ux * arrow_size + px * arrow_width, ey - uy * arrow_size + py * arrow_width),
            (ex - ux * arrow_size - px * arrow_width, ey - uy * arrow_size - py * arrow_width),
        ],
        fill="#475569",
    )
    if relation:
        label = _short(relation, 16)
        bounds = draw.textbbox((0, 0), label, font=font)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        draw.rounded_rectangle(
            (mx - width / 2 - 5, my - height / 2 - 3, mx + width / 2 + 5, my + height / 2 + 3),
            radius=4,
            fill="#ffffff",
            outline="#ffffff",
        )
        draw.text((mx - width / 2, my - height / 2), label, fill="#64748b", font=font)


def _draw_transition(draw: ImageDraw.ImageDraw, x: int, y: float, font: ImageFont.FreeTypeFont) -> None:
    draw.line((x, y, x + 30, y), fill="#0f766e", width=3)
    draw.polygon([(x + 38, y), (x + 28, y - 6), (x + 28, y + 6)], fill="#0f766e")
    draw.text((x + 2, y + 10), "compile", fill="#0f766e", font=font)


def _layout(
    node_ids: list[str],
    edges: list[dict[str, Any]] | list[tuple[str, str, str]],
    x0: int,
    y0: int,
) -> dict[str, tuple[float, float]]:
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


def _task_ids(dataset: str) -> list[str]:
    if dataset == "PubMedQA":
        return ["retrieve", "interpret", "uncertainty", "review", "answer"]
    return ["retrieve", "reason", "safety", "review", "answer"]


def _task_label(task_id: str) -> str:
    return {
        "retrieve": "Retrieve",
        "reason": "Reason",
        "interpret": "Interpret",
        "safety": "Safety",
        "uncertainty": "Uncertain",
        "review": "Review",
        "answer": "Answer",
    }.get(task_id, task_id)


def _agent_label(node: dict[str, Any]) -> str:
    implementation = str(node.get("implementation_ref") or node.get("id") or "")
    if implementation.startswith("component_"):
        implementation = implementation[len("component_") :]
    kind = str(node.get("kind") or "agent")
    return f"{implementation}\n{kind}"


def _question(record: dict[str, Any]) -> str:
    executions = record.get("task_execution") or {}
    executions = executions if isinstance(executions, dict) else {}
    for execution in executions.get("node_executions", []):
        artifact = execution.get("artifact")
        if isinstance(artifact, str) and "original_question" in artifact:
            try:
                payload = json.loads(artifact)
                if payload.get("original_question"):
                    return str(payload["original_question"])
            except json.JSONDecodeError:
                pass
    return str(record.get("dataset") or "dataset item")


def _fonts() -> dict[str, ImageFont.FreeTypeFont]:
    paths = [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"]
    selected = next((path for path in paths if Path(path).exists()), None)
    if selected:
        return {
            "title": ImageFont.truetype(selected, 30),
            "heading": ImageFont.truetype(selected, 18),
            "large": ImageFont.truetype(selected, 14),
            "small": ImageFont.truetype(selected, 13),
            "node": ImageFont.truetype(selected, 14),
            "edge": ImageFont.truetype(selected, 10),
        }
    fallback = ImageFont.load_default()
    return {key: fallback for key in ("title", "heading", "large", "small", "node", "edge")}


def _wrap(value: str, width: int, limit: int = 3) -> list[str]:
    text = str(value).replace("\r", " ").strip()
    if not text:
        return [""]
    lines: list[str] = []
    for part in text.splitlines():
        chunk = textwrap.fill(part.replace("_", " ").replace("-", " "), width=width, break_long_words=False, break_on_hyphens=False)
        lines.extend(chunk.splitlines())
    if not lines:
        lines = [text]
    if len(lines) > limit:
        lines = lines[:limit]
        lines[-1] = lines[-1][: max(3, width - 3)] + "..." if len(lines[-1]) > width else lines[-1]
    return lines


def _short(value: Any, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


if __name__ == "__main__":
    main()
