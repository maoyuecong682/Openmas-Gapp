"""Render Q10 financial Harness Graphs as PNG files."""
from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from openmas_bench.dataset_adapters import all_adapters  # noqa: E402
from openmas_bench.dataset_cases import build_dataset_case  # noqa: E402


WIDTH = 2200
HEIGHT = 1120
MARGIN_X = 90
TOP = 175
COL_GAP = 370
ROW_GAP = 92
NODE_W = 270
NODE_H = 58

STYLE = {
    "task_pattern": ("#ecfeff", "#0891b2"),
    "capability": ("#eef2ff", "#4f46e5"),
    "component": ("#eff6ff", "#2563eb"),
    "constraint": ("#fef2f2", "#dc2626"),
    "control": ("#fff7ed", "#ea580c"),
    "resource": ("#f0fdf4", "#16a34a"),
}


def fonts() -> dict[str, ImageFont.FreeTypeFont]:
    path = next((Path(x) for x in (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf") if Path(x).exists()), None)
    if path:
        return {"title": ImageFont.truetype(path, 28), "subtitle": ImageFont.truetype(path, 15), "node": ImageFont.truetype(path, 13), "edge": ImageFont.truetype(path, 11), "legend": ImageFont.truetype(path, 12)}
    fallback = ImageFont.load_default()
    return {key: fallback for key in ("title", "subtitle", "node", "edge", "legend")}


def short(value: object, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def node_label(node: dict) -> str:
    kind = str(node.get("kind") or "")
    node_id = str(node.get("id") or "")
    if kind == "task_pattern":
        return f"TASK\n{short(node_id, 22)}"
    if kind == "capability":
        return f"CAPABILITY\n{short(node_id.removeprefix('cap_'), 22)}"
    if kind == "component":
        return f"COMPONENT\n{short(node_id.removeprefix('component_'), 22)}"
    if kind == "constraint":
        return f"CONSTRAINT\n{short(node_id, 22)}"
    if kind == "control":
        return f"CONTROL\n{short(node_id.removeprefix('component_'), 22)}"
    return f"{kind.upper()}\n{short(node_id, 22)}"


def layout(nodes: list[dict]) -> dict[str, tuple[float, float]]:
    order = ["task_pattern", "capability", "component", "constraint", "control", "resource"]
    columns: dict[str, list[dict]] = defaultdict(list)
    for node in nodes:
        columns[str(node.get("kind") or "")].append(node)
    positions: dict[str, tuple[float, float]] = {}
    for column, kind in enumerate(order):
        values = columns.get(kind, [])
        if not values:
            continue
        x = MARGIN_X + column * COL_GAP
        total = len(values) * NODE_H + max(0, len(values) - 1) * (ROW_GAP - NODE_H)
        y0 = TOP + max(0, (HEIGHT - TOP - 90 - total) / 2)
        for index, node in enumerate(values):
            positions[str(node["id"])] = (x, y0 + index * ROW_GAP)
    return positions


def draw_arrow(draw: ImageDraw.ImageDraw, source: tuple[float, float], target: tuple[float, float], relation: str, font: ImageFont.FreeTypeFont) -> None:
    sx, sy = source
    tx, ty = target
    dx, dy = tx - sx, ty - sy
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    start = (sx + ux * (NODE_W / 2), sy + uy * (NODE_H / 2))
    end = (tx - ux * (NODE_W / 2), ty - uy * (NODE_H / 2))
    draw.line((*start, *end), fill="#64748b", width=2)
    px, py = -uy, ux
    size = 10
    draw.polygon([(end[0], end[1]), (end[0] - ux * size + px * 4, end[1] - uy * size + py * 4), (end[0] - ux * size - px * 4, end[1] - uy * size - py * 4)], fill="#64748b")
    if relation:
        label = short(relation, 15)
        bounds = draw.textbbox((0, 0), label, font=font)
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        draw.rectangle((mx - (bounds[2] - bounds[0]) / 2 - 3, my - 9, mx + (bounds[2] - bounds[0]) / 2 + 3, my + 4), fill="#ffffff")
        draw.text((mx - (bounds[2] - bounds[0]) / 2, my - 8), label, fill="#475569", font=font)


def render(dataset: str, row: dict, output: Path, harness_override: dict | None = None) -> None:
    adapters = {item.dataset_id.casefold(): item for item in all_adapters()}
    adapter = adapters[dataset.casefold()]
    if harness_override:
        harness = harness_override
    else:
        case = build_dataset_case(adapter, row, 0)
        harness = asdict(case.harness)
    nodes = harness["nodes"]
    edges = harness["edges"]
    positions = layout(nodes)
    image = Image.new("RGB", (WIDTH, HEIGHT), "#e2e8f0")
    draw = ImageDraw.Draw(image)
    face = fonts()
    draw.text((MARGIN_X, 28), f"Q10 Financial Graph Harness | {adapter.dataset_id}", fill="#0f172a", font=face["title"])
    draw.text((MARGIN_X, 75), "Parallel financial evidence, risk assessment, compliance controls, auditability and final reporting", fill="#475569", font=face["subtitle"])
    draw.text((MARGIN_X, 103), f"nodes={len(nodes)}  edges={len(edges)}  family={adapter.template.family}  domain={adapter.template.domain}", fill="#64748b", font=face["subtitle"])

    by_id = {str(node["id"]): node for node in nodes}
    for edge in edges:
        source, target = str(edge["source"]), str(edge["target"])
        if source in positions and target in positions:
            draw_arrow(draw, positions[source], positions[target], str(edge.get("relation") or ""), face["edge"])
    for node_id, (x, y) in positions.items():
        node = by_id[node_id]
        kind = str(node.get("kind") or "")
        fill, outline = STYLE.get(kind, ("#f8fafc", "#64748b"))
        draw.rounded_rectangle((x - NODE_W / 2, y - NODE_H / 2, x + NODE_W / 2, y + NODE_H / 2), radius=9, fill=fill, outline=outline, width=3)
        lines = node_label(node).splitlines()
        for index, line in enumerate(lines):
            bounds = draw.textbbox((0, 0), line, font=face["node"])
            draw.text((x - (bounds[2] - bounds[0]) / 2, y - 22 + index * 18), line, fill="#0f172a" if index else outline, font=face["node"])

    legend_x = MARGIN_X
    legend_y = HEIGHT - 62
    for kind in ("task_pattern", "capability", "component", "constraint", "control"):
        fill, outline = STYLE[kind]
        label = kind.replace("_", " ")
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 18, legend_y + 18), radius=4, fill=fill, outline=outline, width=2)
        draw.text((legend_x + 25, legend_y + 1), label, fill="#475569", font=face["legend"])
        legend_x += 175
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG")
    print(f"wrote {output} {WIDTH}x{HEIGHT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Q10 financial Harness Graphs.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--dataset", required=False, default="")
    parser.add_argument("--input", type=Path, default=None, help="Q10 run JSON containing q10_harness")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    harness_override = None
    if args.input:
        record = json.loads(args.input.read_text(encoding="utf-8"))
        dataset = str(record.get("dataset") or args.dataset)
        key = dataset.casefold().replace("-", "_")
        path = args.data_root / "q10_datasets" / "normalized" / f"{key}.jsonl"
        row = json.loads(next(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))
        harness_override = record.get("q10_harness")
    else:
        dataset = args.dataset
        key = dataset.casefold().replace("-", "_")
        path = args.data_root / "q10_datasets" / "normalized" / f"{key}.jsonl"
        row = json.loads(next(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))
    if not dataset:
        raise ValueError("--dataset or --input is required")
    render(dataset, row, args.output, harness_override=harness_override)


if __name__ == "__main__":
    main()
