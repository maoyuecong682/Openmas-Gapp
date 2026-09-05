from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections import defaultdict, deque
from html import escape
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


RADIUS = 42
MARGIN = 90
X_GAP = 170
Y_GAP = 105

PNG_W = 1920
PNG_H = 1080
PNG_MARGIN_X = 140
PNG_MARGIN_Y = 180
PNG_RADIUS = 46
PNG_NODE_FONT = 14
PNG_EDGE_FONT = 10
PNG_TITLE_FONT = 30
PNG_SUBTITLE_FONT = 16

NODE_STYLE = {
    "agent": ("#ffffff", "#2563eb"),
    "control": ("#ffffff", "#dc2626"),
    "resource": ("#ffffff", "#16a34a"),
    "tool": ("#ffffff", "#7c3aed"),
    "task": ("#ffffff", "#0891b2"),
    "component_requirement": ("#ffffff", "#4f46e5"),
    "resource_requirement": ("#ffffff", "#16a34a"),
}
DEFAULT_STYLE = ("#ffffff", "#64748b")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a JSON-described node/edge data structure as a graph."
    )
    parser.add_argument("--input", required=True, help="Input experiment JSON file")
    parser.add_argument("--output", default="", help="Output SVG path")
    parser.add_argument(
        "--graph",
        choices=["application", "blueprint", "harness"],
        default="application",
        help="Which node/edge object to render from the JSON",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help="Run index when input is an aggregate JSON with a runs array",
    )
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="Render edges as undirected lines instead of arrows",
    )
    parser.add_argument(
        "--show-kind",
        action="store_true",
        help="Show each node kind under the node id",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    record = _select_record(payload, args.run_index)
    graph = _extract_graph(record, args.graph)
    nodes = graph["nodes"]
    edges = graph["edges"]
    if not nodes:
        raise ValueError(f"{args.graph} graph has no nodes")

    output = Path(args.output) if args.output else input_path.with_suffix(f".{args.graph}.svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".png":
        image = render_graph_png(
            nodes,
            edges,
            directed=not args.undirected,
            show_kind=args.show_kind,
            graph_name=args.graph,
            dataset=str(record.get("dataset") or "dataset"),
        )
        image.save(output, "PNG")
        print(f"wrote {output} {image.size[0]}x{image.size[1]}")
    else:
        output.write_text(
            render_graph_svg(nodes, edges, directed=not args.undirected, show_kind=args.show_kind),
            encoding="utf-8",
        )
        print(f"wrote {output}")


def _select_record(payload: dict[str, Any], run_index: int) -> dict[str, Any]:
    runs = payload.get("runs")
    if isinstance(runs, list):
        if run_index < 0 or run_index >= len(runs):
            raise IndexError(f"run index {run_index} out of range 0..{len(runs) - 1}")
        run = runs[run_index]
        if not isinstance(run, dict):
            raise ValueError(f"runs[{run_index}] is not an object")
        return run
    return payload


def _extract_graph(record: dict[str, Any], graph_name: str) -> dict[str, list[dict[str, Any]]]:
    if graph_name == "application":
        graph = record.get("application") or {}
    elif graph_name == "blueprint":
        graph = record.get("blueprint") or record.get("blueprint_metadata") or {}
    else:
        graph = (
            record.get("harness")
            or record.get("application")
            or record.get("q9_harness")
            or record.get("q9_application")
            or {}
        )

    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    edges = graph.get("edges") if isinstance(graph, dict) else None

    if graph_name == "blueprint" and not nodes:
        construction = record.get("construction") or record.get("construction_result") or {}
        blueprint = construction.get("blueprint") if isinstance(construction, dict) else None
        if isinstance(blueprint, dict):
            nodes = blueprint.get("nodes")
            edges = blueprint.get("edges")

    return {
        "nodes": [node for node in (nodes or []) if isinstance(node, dict)],
        "edges": [edge for edge in (edges or []) if isinstance(edge, dict)],
    }


def render_graph_svg(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    directed: bool,
    show_kind: bool,
) -> str:
    node_ids = [str(node.get("id")) for node in nodes]
    positions = _hierarchical_layout(node_ids, edges)
    width, height = _canvas_size(positions)
    id_to_node = {str(node.get("id")): node for node in nodes}

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto" markerUnits="strokeWidth">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#334155"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    for edge in edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source not in positions or target not in positions:
            continue
        lines.extend(_edge_svg(edge, positions[source], positions[target], directed))

    for node_id in node_ids:
        lines.extend(_node_svg(id_to_node[node_id], positions[node_id], show_kind))

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _hierarchical_layout(node_ids: list[str], edges: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    outgoing: dict[str, list[str]] = defaultdict(list)
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source in indegree and target in indegree:
            outgoing[source].append(target)
            indegree[target] += 1

    levels = {node_id: 0 for node_id in node_ids}
    queue = deque([node_id for node_id in node_ids if indegree[node_id] == 0])
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
        return _circular_layout(node_ids)

    by_level: dict[int, list[str]] = defaultdict(list)
    for node_id in node_ids:
        by_level[levels[node_id]].append(node_id)

    max_rows = max((len(values) for values in by_level.values()), default=1)
    graph_height = max(1, max_rows - 1) * Y_GAP
    positions: dict[str, tuple[float, float]] = {}
    for level in range(max(by_level) + 1):
        column = by_level.get(level, [])
        column_height = max(0, len(column) - 1) * Y_GAP
        y0 = MARGIN + (graph_height - column_height) / 2
        x = MARGIN + level * X_GAP
        for index, node_id in enumerate(column):
            positions[node_id] = (x, y0 + index * Y_GAP)
    return positions


def _circular_layout(node_ids: list[str]) -> dict[str, tuple[float, float]]:
    count = max(1, len(node_ids))
    radius = max(150, count * 28)
    center = (MARGIN + radius, MARGIN + radius)
    positions = {}
    for index, node_id in enumerate(node_ids):
        angle = (2 * math.pi * index / count) - math.pi / 2
        positions[node_id] = (
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
        )
    return positions


def _canvas_size(positions: dict[str, tuple[float, float]]) -> tuple[int, int]:
    max_x = max((x for x, _ in positions.values()), default=0)
    max_y = max((y for _, y in positions.values()), default=0)
    return int(max_x + MARGIN), int(max_y + MARGIN)


def _edge_svg(
    edge: dict[str, Any],
    source: tuple[float, float],
    target: tuple[float, float],
    directed: bool,
) -> list[str]:
    sx, sy = source
    tx, ty = target
    dx = tx - sx
    dy = ty - sy
    length = math.hypot(dx, dy) or 1.0
    ux = dx / length
    uy = dy / length
    start_x = sx + ux * RADIUS
    start_y = sy + uy * RADIUS
    end_x = tx - ux * (RADIUS + (6 if directed else 0))
    end_y = ty - uy * (RADIUS + (6 if directed else 0))
    marker = ' marker-end="url(#arrow)"' if directed else ""
    relation = str(edge.get("relation") or edge.get("kind") or "")
    label_x = (start_x + end_x) / 2
    label_y = (start_y + end_y) / 2 - 7
    return [
        f'<line x1="{start_x:.1f}" y1="{start_y:.1f}" x2="{end_x:.1f}" y2="{end_y:.1f}" '
        f'stroke="#334155" stroke-width="1.5"{marker}/>',
        f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="11" fill="#475569">{escape(_short(relation, 22))}</text>',
    ]


def _node_svg(node: dict[str, Any], position: tuple[float, float], show_kind: bool) -> list[str]:
    x, y = position
    node_id = str(node.get("id"))
    kind = str(node.get("kind") or "")
    fill, stroke = NODE_STYLE.get(kind, DEFAULT_STYLE)
    lines = [
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{RADIUS}" fill="{fill}" stroke="{stroke}" stroke-width="2.2"/>',
    ]
    label_lines = _wrap_node_label(node_id)
    start_y = y - (len(label_lines) - 1) * 6 - (7 if show_kind else 0)
    for index, label in enumerate(label_lines):
        lines.append(
            f'<text x="{x:.1f}" y="{start_y + index * 13:.1f}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="11" font-weight="700" fill="#0f172a">{escape(label)}</text>'
        )
    if show_kind and kind:
        lines.append(
            f'<text x="{x:.1f}" y="{y + 25:.1f}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="10" fill="{stroke}">{escape(_short(kind, 18))}</text>'
        )
    return lines


def _wrap_node_label(value: str) -> list[str]:
    short = _short(value, 28)
    wrapped = textwrap.wrap(
        short.replace("_", " "),
        width=11,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        wrapped = [short]
    return wrapped[:2]


def _short(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def render_graph_png(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    directed: bool,
    show_kind: bool,
    graph_name: str,
    dataset: str,
) -> Image.Image:
    node_ids = [str(node.get("id")) for node in nodes]
    positions = _hierarchical_layout(node_ids, edges)
    positions = _fit_positions(positions, PNG_W, PNG_H)
    id_to_node = {str(node.get("id")): node for node in nodes}
    image = Image.new("RGB", (PNG_W, PNG_H), "#e2e8f0")
    draw = ImageDraw.Draw(image)
    fonts = _png_fonts()

    title = f"Q9 {graph_name.title()} Graph | {dataset}"
    draw.text((PNG_MARGIN_X, 28), title, fill="#0f172a", font=fonts["title"])
    subtitle = f"nodes={len(nodes)}  edges={len(edges)}  graph={graph_name}"
    draw.text((PNG_MARGIN_X, 74), subtitle, fill="#475569", font=fonts["subtitle"])

    for edge in edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source not in positions or target not in positions:
            continue
        _edge_png(draw, edge, positions[source], positions[target], directed, fonts["edge"])

    for node_id in node_ids:
        _node_png(draw, id_to_node[node_id], positions[node_id], show_kind, fonts)

    return image


def _fit_positions(
    positions: dict[str, tuple[float, float]],
    width: int,
    height: int,
) -> dict[str, tuple[float, float]]:
    if not positions:
        return positions
    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    content_w = max_x - min_x
    content_h = max_y - min_y
    available_w = max(1.0, width - 2 * PNG_MARGIN_X)
    available_h = max(1.0, height - 2 * PNG_MARGIN_Y)
    shift_x = PNG_MARGIN_X - min_x + max(0.0, (available_w - content_w) / 2)
    shift_y = PNG_MARGIN_Y - min_y + max(0.0, (available_h - content_h) / 2)
    return {node_id: (x + shift_x, y + shift_y) for node_id, (x, y) in positions.items()}


def _png_fonts() -> dict[str, ImageFont.FreeTypeFont]:
    paths = [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"]
    selected = next((path for path in paths if Path(path).exists()), None)
    if selected:
        return {
            "title": ImageFont.truetype(selected, PNG_TITLE_FONT),
            "subtitle": ImageFont.truetype(selected, PNG_SUBTITLE_FONT),
            "node": ImageFont.truetype(selected, PNG_NODE_FONT),
            "kind": ImageFont.truetype(selected, 11),
            "edge": ImageFont.truetype(selected, PNG_EDGE_FONT),
        }
    fallback = ImageFont.load_default()
    return {key: fallback for key in ("title", "subtitle", "node", "kind", "edge")}


def _edge_png(
    draw: ImageDraw.ImageDraw,
    edge: dict[str, Any],
    source: tuple[float, float],
    target: tuple[float, float],
    directed: bool,
    font: ImageFont.FreeTypeFont,
) -> None:
    sx, sy = source
    tx, ty = target
    dx = tx - sx
    dy = ty - sy
    length = math.hypot(dx, dy) or 1.0
    ux = dx / length
    uy = dy / length
    start_x = sx + ux * PNG_RADIUS
    start_y = sy + uy * PNG_RADIUS
    end_x = tx - ux * (PNG_RADIUS + (6 if directed else 0))
    end_y = ty - uy * (PNG_RADIUS + (6 if directed else 0))
    draw.line((start_x, start_y, end_x, end_y), fill="#334155", width=2)
    px, py = -uy, ux
    size = 8
    draw.polygon(
        [
            (end_x, end_y),
            (end_x - ux * size + px * 4, end_y - uy * size + py * 4),
            (end_x - ux * size - px * 4, end_y - uy * size - py * 4),
        ],
        fill="#334155",
    )
    relation = str(edge.get("relation") or edge.get("kind") or "")
    if relation:
        label = _short(relation.replace("_", " "), 16)
        bounds = draw.textbbox((0, 0), label, font=font)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        mx, my = (start_x + end_x) / 2, (start_y + end_y) / 2
        draw.rounded_rectangle(
            (mx - width / 2 - 5, my - height / 2 - 3, mx + width / 2 + 5, my + height / 2 + 3),
            radius=4,
            fill="#ffffff",
            outline="#ffffff",
        )
        draw.text((mx - width / 2, my - height / 2), label, fill="#475569", font=font)


def _node_png(
    draw: ImageDraw.ImageDraw,
    node: dict[str, Any],
    position: tuple[float, float],
    show_kind: bool,
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> None:
    x, y = position
    kind = str(node.get("kind") or "")
    fill, stroke = NODE_STYLE.get(kind, DEFAULT_STYLE)
    draw.ellipse((x - PNG_RADIUS, y - PNG_RADIUS, x + PNG_RADIUS, y + PNG_RADIUS), fill=fill, outline=stroke, width=3)
    label_lines = _wrap_node_label(_display_node_label(node))
    heights = [draw.textbbox((0, 0), label, font=fonts["node"])[3] - draw.textbbox((0, 0), label, font=fonts["node"])[1] for label in label_lines]
    total = sum(heights) + 2 * max(0, len(label_lines) - 1)
    start_y = y - total / 2 - (7 if show_kind else 0)
    for index, label in enumerate(label_lines):
        bounds = draw.textbbox((0, 0), label, font=fonts["node"])
        draw.text((x - (bounds[2] - bounds[0]) / 2, start_y), label, fill="#0f172a", font=fonts["node"])
        start_y += heights[index] + 2
    if show_kind and kind:
        bounds = draw.textbbox((0, 0), kind, font=fonts["kind"])
        draw.text((x - (bounds[2] - bounds[0]) / 2, y + 23), kind, fill=stroke, font=fonts["kind"])


def _display_node_label(node: dict[str, Any]) -> str:
    kind = str(node.get("kind") or "")
    node_id = str(node.get("id") or "")
    impl = str(node.get("implementation_ref") or node_id)
    if kind == "agent":
        core = impl.removeprefix("component_")
    elif kind == "control":
        core = impl.removeprefix("control_").removeprefix("component_")
    elif kind == "resource":
        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        core = str(metadata.get("resource_key") or node_id.removeprefix("resource_") or node_id)
    elif kind == "component":
        core = impl.removeprefix("component_")
    elif kind == "task":
        core = node_id
    else:
        core = impl or node_id
    return core.replace("_", " ").strip() or node_id


if __name__ == "__main__":
    main()
