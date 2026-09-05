from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 1080
MARGIN_X = 140
TOP = 180
COL_GAP = 300
ROW_GAP = 88
NODE_W = 260
NODE_H = 76

EV_PANEL_W = 430
EV_PANEL_H = 930
EV_PANEL_GAP = 24
EV_PANEL_X0 = 40
EV_PANEL_Y0 = 70
EV_RADIUS = 31
EV_INPUT_W = 150
EV_INPUT_H = 72

STYLE = {
    "task_pattern": ("#ecfeff", "#0891b2"),
    "capability": ("#eef2ff", "#4f46e5"),
    "component": ("#eff6ff", "#2563eb"),
    "constraint": ("#fef2f2", "#dc2626"),
    "control": ("#fff7ed", "#ea580c"),
    "resource": ("#f0fdf4", "#16a34a"),
}


def dataset_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return key or "dataset"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_dataset_space(
    *,
    dataset_root: Path,
    raw_subdir: str,
    schema_version: str,
    purpose: str,
    rows_by_dataset: dict[str, list[dict[str, Any]]],
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest_items = []
    for key, rows in rows_by_dataset.items():
        profile = profiles[key]
        raw_path = dataset_root / "raw" / raw_subdir / f"{key}.jsonl"
        normalized_path = dataset_root / "normalized" / f"{key}.jsonl"
        pilot_path = dataset_root / "pilot" / f"{key}_pilot.jsonl"
        write_jsonl(raw_path, [row.get("raw", row) for row in rows])
        write_jsonl(normalized_path, rows)
        write_jsonl(pilot_path, rows)
        manifest_items.append(
            {
                "dataset": profile["dataset"],
                "dataset_key": key,
                "source_dataset": profile["source_dataset"],
                "split": profile["split"],
                "rows": len(rows),
                "pilot_rows": len(rows),
                "license": profile["license"],
                "download_url": profile["download_url"],
                "problem_type": profile["problem_type"],
                "raw_file": str(raw_path.relative_to(dataset_root)),
                "normalized_file": str(normalized_path.relative_to(dataset_root)),
                "pilot_file": str(pilot_path.relative_to(dataset_root)),
                "sha256_raw": sha256(raw_path),
                "sha256_normalized": sha256(normalized_path),
                "source_status": "local_frozen_pilot",
            }
        )
    manifest = {
        "schema_version": schema_version,
        "purpose": purpose,
        "datasets": manifest_items,
    }
    write_json(dataset_root / "manifests" / f"{schema_version}_dataset_manifest.json", manifest)
    return manifest


def build_domain_record(
    *,
    question_id: str,
    domain_slug: str,
    domain_title: str,
    case_study: str,
    question_description: str,
    profile: dict[str, Any],
    row: dict[str, Any],
    row_index: int,
) -> dict[str, Any]:
    dataset = str(profile["dataset"])
    key = dataset_key(dataset)
    qkey = question_id.casefold()
    case_id = f"{qkey}_{key}_{row_index:04d}"
    analysis = {
        "dataset": dataset,
        "domain": profile["domain"],
        "task_family": profile["task_family"],
        "problem_type": profile["problem_type"],
        "domain_focus": profile["domain_focus"],
        "tasks": profile["tasks"],
        "edges": profile["edges"],
        "resources": profile["resources"],
        "constraints": profile["constraints"],
    }
    harness = _build_harness(case_id, dataset, profile)
    blueprint = _build_blueprint(case_id, dataset, profile)
    record = {
        "status": "harness_completed",
        "question_id": question_id,
        "domain_slug": domain_slug,
        "domain_title": domain_title,
        "case_study": case_study,
        "question_description": question_description,
        "dataset": dataset,
        "source_id": row.get("id"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "context": row.get("context"),
        "choices": row.get("choices", []),
        "primary_metric": profile["metric"],
        f"{qkey}_analysis": analysis,
        f"{qkey}_harness": harness,
        f"{qkey}_blueprint": blueprint,
        f"{qkey}_three_layer_pipeline": [
            "dataset_profile",
            "row_problem_type",
            "constrained_harness_compilation",
        ],
        "harness": harness,
        "blueprint": blueprint,
    }
    return record


def _build_harness(case_id: str, dataset: str, profile: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for task in profile["tasks"]:
        task_id = str(task["id"])
        objective = str(task["objective"])
        cap_id = f"cap_{task_id}"
        component_id = f"component_{task_id}"
        tags = [profile["domain"], task_id, profile["problem_type"]]
        nodes.extend(
            [
                {
                    "id": task_id,
                    "kind": "task_pattern",
                    "description": objective,
                    "tags": tags,
                },
                {
                    "id": cap_id,
                    "kind": "capability",
                    "description": objective,
                    "tags": tags,
                },
                {
                    "id": component_id,
                    "kind": "component",
                    "description": f"Domain component for {objective.lower()}",
                    "capabilities": [cap_id],
                    "tags": tags,
                    "metadata": {"runtime_kind": "agent"},
                },
            ]
        )
        edges.extend(
            [
                {"source": task_id, "target": cap_id, "relation": "requires"},
                {"source": cap_id, "target": component_id, "relation": "realizes"},
            ]
        )
    for edge in profile["edges"]:
        edges.append(
            {
                "source": str(edge["source"]),
                "target": str(edge["target"]),
                "relation": str(edge["relation"]),
            }
        )
    for resource in profile["resources"]:
        nodes.append(
            {
                "id": str(resource["id"]),
                "kind": "resource",
                "description": str(resource["description"]),
                "tags": [profile["domain"], str(resource["task_id"]), "branch_resource"],
                "metadata": {"resource_key": str(resource["resource_key"])},
            }
        )
        edges.append(
            {
                "source": str(resource["id"]),
                "target": f"component_{resource['task_id']}",
                "relation": "uses",
            }
        )
    for constraint in profile["constraints"]:
        constraint_id = str(constraint["id"])
        target = str(constraint["target"])
        kind = str(constraint["kind"])
        predicate = str(constraint["predicate"])
        nodes.extend(
            [
                {
                    "id": constraint_id,
                    "kind": "constraint",
                    "description": f"{kind} {predicate}",
                    "tags": [profile["domain"], constraint_id, kind],
                    "metadata": {
                        "constraint_kind": kind,
                        "target": target,
                        "predicate": predicate,
                        "severity": "required",
                    },
                },
                {
                    "id": f"component_{constraint_id}",
                    "kind": "control",
                    "description": f"Control enforcing {constraint_id}",
                    "capabilities": [f"cap_{constraint_id}"],
                    "tags": [profile["domain"], constraint_id, kind],
                    "metadata": {"runtime_kind": "control"},
                },
            ]
        )
        edges.extend(
            [
                {"source": target, "target": constraint_id, "relation": "constrained_by"},
                {"source": constraint_id, "target": f"component_{constraint_id}", "relation": "realizes"},
            ]
        )
    forbidden = [
        f"forbidden_{case_id}_single_agent_shortcut",
        f"forbidden_{case_id}_unsupported_source",
    ]
    nodes.extend(
        [
            {
                "id": forbidden[0],
                "kind": "component",
                "description": "Unsafe single-agent shortcut component",
                "capabilities": ["cap_shortcut"],
                "risk": "high",
            },
            {
                "id": forbidden[1],
                "kind": "component",
                "description": "Untrusted or unsupported source component",
                "capabilities": ["cap_untrusted"],
                "risk": "high",
            },
        ]
    )
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "dataset": dataset,
            "case_id": case_id,
            "domain": profile["domain"],
            "problem_type": profile["problem_type"],
            "forbidden_components": forbidden,
        },
    }


def _build_blueprint(case_id: str, dataset: str, profile: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for task in profile["tasks"]:
        task_id = str(task["id"])
        objective = str(task["objective"])
        nodes.append({"id": task_id, "kind": "task", "description": objective, "tags": [task_id]})
        nodes.append(
            {
                "id": f"req_component_{task_id}",
                "kind": "component_requirement",
                "description": objective,
                "capability_refs": [f"cap_{task_id}"],
                "metadata": {"candidates": [f"component_{task_id}"]},
            }
        )
        edges.append({"source": task_id, "target": f"req_component_{task_id}", "relation": "requires"})
    for edge in profile["edges"]:
        edges.append(
            {
                "source": str(edge["source"]),
                "target": str(edge["target"]),
                "relation": str(edge["relation"]),
            }
        )
    for resource in profile["resources"]:
        nodes.append(
            {
                "id": str(resource["id"]),
                "kind": "resource_requirement",
                "description": str(resource["description"]),
                "metadata": {"resource_key": str(resource["resource_key"])},
            }
        )
        edges.append(
            {
                "source": str(resource["id"]),
                "target": f"req_component_{resource['task_id']}",
                "relation": "uses",
            }
        )
    for constraint in profile["constraints"]:
        nodes.append(
            {
                "id": f"control_{constraint['id']}",
                "kind": "control",
                "description": f"Enforce {constraint['id']}",
                "capability_refs": [f"cap_{constraint['id']}"],
                "metadata": {"candidate": f"component_{constraint['id']}"},
            }
        )
        edges.append(
            {
                "source": str(constraint["target"]),
                "target": f"control_{constraint['id']}",
                "relation": "constrained_by",
            }
        )
    return {
        "case_id": case_id,
        "strategy": "domain_profile_compilation",
        "nodes": nodes,
        "edges": edges,
        "metadata": {"dataset": dataset, "domain": profile["domain"]},
    }


def render_harness_png(record: dict[str, Any], output: Path) -> None:
    harness = record.get("harness") or record.get(f"{str(record.get('question_id', '')).casefold()}_harness") or {}
    nodes = [node for node in harness.get("nodes", []) if isinstance(node, dict)]
    edges = [edge for edge in harness.get("edges", []) if isinstance(edge, dict)]
    if not nodes:
        raise ValueError(f"{record.get('dataset', 'dataset')} harness has no nodes")

    positions = _layout(nodes)
    image = Image.new("RGB", (WIDTH, HEIGHT), "#e2e8f0")
    draw = ImageDraw.Draw(image)
    face = _fonts()
    dataset = str(record.get("dataset") or "dataset")
    title = f"{record.get('question_id')} {record.get('domain_title')} Graph Harness | {dataset}"
    subtitle = str(record.get("question_description") or record.get("case_study") or "")
    draw.text((MARGIN_X, 28), title, fill="#0f172a", font=face["title"])
    draw.text((MARGIN_X, 75), _short(subtitle, 120), fill="#475569", font=face["subtitle"])
    draw.text(
        (MARGIN_X, 103),
        f"nodes={len(nodes)}  edges={len(edges)}  case={record.get('case_study')}  metric={record.get('primary_metric')}",
        fill="#64748b",
        font=face["subtitle"],
    )

    by_id = {str(node["id"]): node for node in nodes}
    for edge in edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source in positions and target in positions:
            _draw_arrow(draw, positions[source], positions[target], str(edge.get("relation") or ""), face["edge"])
    for node_id, position in positions.items():
        _draw_node(draw, by_id[node_id], position, face)

    legend_x = MARGIN_X
    legend_y = HEIGHT - 66
    for kind in ("task_pattern", "capability", "component", "constraint", "control", "resource"):
        fill, outline = STYLE[kind]
        label = kind.replace("_", " ")
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 18, legend_y + 18), radius=4, fill=fill, outline=outline, width=2)
        draw.text((legend_x + 25, legend_y + 1), label, fill="#475569", font=face["legend"])
        legend_x += 155
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG")
    print(f"wrote {output} {WIDTH}x{HEIGHT}")


def _fonts() -> dict[str, ImageFont.FreeTypeFont]:
    path = next(
        (Path(x) for x in (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf") if Path(x).exists()),
        None,
    )
    if path:
        return {
            "title": ImageFont.truetype(path, 30),
            "subtitle": ImageFont.truetype(path, 16),
            "panel_title": ImageFont.truetype(path, 18),
            "panel_subtitle": ImageFont.truetype(path, 13),
            "node": ImageFont.truetype(path, 14),
            "small": ImageFont.truetype(path, 13),
            "edge": ImageFont.truetype(path, 11),
            "legend": ImageFont.truetype(path, 13),
            "large": ImageFont.truetype(path, 14),
        }
    fallback = ImageFont.load_default()
    return {key: fallback for key in ("title", "subtitle", "panel_title", "panel_subtitle", "node", "small", "edge", "legend", "large")}


def _layout(nodes: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    order = ["task_pattern", "capability", "component", "constraint", "control", "resource"]
    columns: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        columns[str(node.get("kind") or "")].append(node)
    positions: dict[str, tuple[float, float]] = {}
    available_h = HEIGHT - TOP - 110
    for column, kind in enumerate(order):
        values = columns.get(kind, [])
        if not values:
            continue
        x = MARGIN_X + column * COL_GAP
        span = max(0, len(values) - 1) * ROW_GAP
        y0 = TOP + max(0, (available_h - span) / 2)
        for index, node in enumerate(values):
            positions[str(node["id"])] = (x, y0 + index * ROW_GAP)
    return positions


def _draw_arrow(
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
    start = (sx + ux * (NODE_W / 2), sy + uy * (NODE_H / 2))
    end = (tx - ux * (NODE_W / 2 + 6), ty - uy * (NODE_H / 2 + 6))
    draw.line((*start, *end), fill="#64748b", width=2)
    px, py = -uy, ux
    size = 10
    draw.polygon(
        [
            (end[0], end[1]),
            (end[0] - ux * size + px * 4, end[1] - uy * size + py * 4),
            (end[0] - ux * size - px * 4, end[1] - uy * size - py * 4),
        ],
        fill="#64748b",
    )
    if relation:
        label = _short(relation.replace("_", " "), 17)
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
        draw.text((mx - width / 2, my - height / 2), label, fill="#475569", font=font)


def _draw_node(
    draw: ImageDraw.ImageDraw,
    node: dict[str, Any],
    position: tuple[float, float],
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> None:
    x, y = position
    kind = str(node.get("kind") or "")
    fill, outline = STYLE.get(kind, ("#f8fafc", "#64748b"))
    draw.rounded_rectangle(
        (x - NODE_W / 2, y - NODE_H / 2, x + NODE_W / 2, y + NODE_H / 2),
        radius=9,
        fill=fill,
        outline=outline,
        width=3,
    )
    lines = _node_label(node).splitlines()
    total_h = len(lines) * 17
    start_y = y - total_h / 2 + 2
    for index, line in enumerate(lines):
        bounds = draw.textbbox((0, 0), line, font=fonts["node"])
        draw.text(
            (x - (bounds[2] - bounds[0]) / 2, start_y + index * 17),
            line,
            fill=outline if index == 0 else "#0f172a",
            font=fonts["node"],
        )


def _node_label(node: dict[str, Any]) -> str:
    kind = str(node.get("kind") or "")
    node_id = str(node.get("id") or "")
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    if kind == "task_pattern":
        core = node_id
    elif kind == "capability":
        core = node_id.removeprefix("cap_")
    elif kind == "component":
        core = node_id.removeprefix("component_")
    elif kind == "control":
        core = node_id.removeprefix("component_")
    elif kind == "resource":
        core = str(metadata.get("resource_key") or node_id.removeprefix("resource_") or node_id)
    else:
        core = node_id
    heading = kind.replace("_", " ").upper() if kind else "NODE"
    wrapped = _wrap(core, 20, 3)
    return "\n".join([heading, *wrapped])


def _wrap(value: object, width: int, limit: int) -> list[str]:
    text = re.sub(r"\s+", " ", str(value).replace("_", " ").replace("-", " ")).strip()
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [text]
    if len(lines) > limit:
        lines = lines[:limit]
        lines[-1] = _short(lines[-1], width)
    return lines


def _short(value: object, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def render_evolution_png(record: dict[str, Any], output: Path) -> None:
    panels = _evolution_panels(record)
    image = Image.new("RGB", (WIDTH, HEIGHT), "#e2e8f0")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    question_id = str(record.get("question_id") or "Q")
    domain_title = str(record.get("domain_title") or "Application")
    dataset = str(record.get("dataset") or "dataset")

    draw.text((24, 18), f"{question_id} {domain_title} Graph Evolution | {dataset}", fill="#0f172a", font=fonts["title"])
    draw.text((24, 56), _short(str(record.get("question_description") or record.get("case_study") or ""), 120), fill="#475569", font=fonts["subtitle"])

    for index, panel in enumerate(panels):
        x = EV_PANEL_X0 + index * (EV_PANEL_W + EV_PANEL_GAP)
        _draw_evolution_panel(draw, x, EV_PANEL_Y0, panel, fonts)
        if index < len(panels) - 1:
            _draw_evolution_transition(draw, x + EV_PANEL_W + 4, EV_PANEL_Y0 + EV_PANEL_H / 2, fonts["large"])

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG")
    print(f"wrote {output} {WIDTH}x{HEIGHT}")


def _evolution_panels(record: dict[str, Any]) -> list[dict[str, Any]]:
    question_id = str(record.get("question_id") or "")
    qkey = question_id.casefold()
    analysis = record.get(f"{qkey}_analysis") or record.get("analysis") or {}
    if not isinstance(analysis, dict):
        analysis = {}
    blueprint = record.get(f"{qkey}_blueprint") or record.get("blueprint") or {}
    if not isinstance(blueprint, dict):
        blueprint = {}
    harness = record.get(f"{qkey}_harness") or record.get("harness") or {}
    if not isinstance(harness, dict):
        harness = {}

    tasks = [item for item in analysis.get("tasks", []) if isinstance(item, dict)]
    if not tasks:
        tasks = [{"id": "answer", "objective": "Return the answer"}]
    final_label = _evo_task_label(tasks[-1])

    input_panel = {
        "title": "1  Input",
        "subtitle": "Dataset row becomes a construction case",
        "nodes": [
            {"id": "question", "kind": "input", "label": "\n".join(["Question", *_wrap(record.get("question") or "", 19, 4)])},
            {"id": "context", "kind": "input", "label": "Context / choices"},
            {"id": "output", "kind": "input", "label": "\n".join(["Required output", *_wrap(final_label, 18, 2)])},
            {"id": "case", "kind": "task_pattern", "label": f"{question_id}\ncase input"},
        ],
        "edges": [
            {"source": "question", "target": "case", "relation": "input"},
            {"source": "context", "target": "case", "relation": "input"},
            {"source": "output", "target": "case", "relation": "contract"},
        ],
    }

    requirement_panel = {
        "title": "2  Requirement Model",
        "subtitle": "Tasks and dependency edges are made explicit",
        "nodes": [{"id": str(item["id"]), "kind": "task_pattern", "label": _evo_task_label(item)} for item in tasks],
        "edges": [{"source": str(edge.get("source")), "target": str(edge.get("target")), "relation": str(edge.get("relation") or "precedes")} for edge in analysis.get("edges", []) if isinstance(edge, dict)],
    }

    blueprint_nodes, blueprint_edges = _build_evolution_blueprint(blueprint, tasks)
    executable_nodes, executable_edges = _build_evolution_executable(harness, analysis, tasks)

    return [
        input_panel,
        requirement_panel,
        {
            "title": "3  Blueprint",
            "subtitle": "Tasks bind to components, resources and controls",
            "nodes": blueprint_nodes,
            "edges": blueprint_edges,
        },
        {
            "title": "4  Executable MAS",
            "subtitle": "Concrete agents, resources, controls and execution edges",
            "nodes": executable_nodes,
            "edges": executable_edges,
        },
    ]


def _evo_task_label(task: dict[str, Any]) -> str:
    text = str(task.get("objective") or task.get("id") or "")
    return _short(_friendly(text), 32)


def _build_evolution_blueprint(blueprint: dict[str, Any], tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_ids = {str(item["id"]) for item in tasks}
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for node in blueprint.get("nodes", []):
        if not isinstance(node, dict):
            continue
        kind = str(node.get("kind") or "")
        node_id = str(node.get("id") or "")
        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        label = _blueprint_label(kind, node_id, metadata)
        nodes.append({"id": node_id, "kind": kind, "label": label})
    for edge in blueprint.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        relation = str(edge.get("relation") or "precedes")
        if source in task_ids or target.startswith("req_") or target.startswith("control_") or target.startswith("resource_"):
            edges.append({"source": source, "target": target, "relation": relation})
        elif relation in {"precedes", "requires", "reviews", "feedback", "constrained_by", "uses"}:
            edges.append({"source": source, "target": target, "relation": relation})
    return nodes, edges


def _blueprint_label(kind: str, node_id: str, metadata: dict[str, Any]) -> str:
    if kind == "task":
        core = _friendly(node_id)
    elif kind == "component_requirement":
        candidate = ""
        candidates = metadata.get("candidates")
        if isinstance(candidates, list) and candidates:
            candidate = str(candidates[0])
        core = _friendly(candidate.removeprefix("component_") or node_id.removeprefix("req_component_") or node_id)
        core = f"req {core}".strip()
    elif kind == "resource_requirement":
        core = _friendly(str(metadata.get("resource_key") or node_id.removeprefix("resource_") or node_id))
    elif kind == "control":
        core = _friendly(node_id.removeprefix("control_").removeprefix("component_"))
    else:
        core = _friendly(node_id)
    return _short(core, 28)


def _build_evolution_executable(
    harness: dict[str, Any],
    analysis: dict[str, Any],
    tasks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = harness.get("metadata") if isinstance(harness.get("metadata"), dict) else {}
    forbidden = {str(item) for item in metadata.get("forbidden_components", [])} if isinstance(metadata.get("forbidden_components"), list) else set()
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    task_ids = [str(item["id"]) for item in tasks]
    final_task = task_ids[-1] if task_ids else ""
    final_component = f"component_{final_task}" if final_task else ""

    for node in harness.get("nodes", []):
        if not isinstance(node, dict):
            continue
        kind = str(node.get("kind") or "")
        node_id = str(node.get("id") or "")
        if node_id in forbidden:
            continue
        if kind not in {"component", "control", "resource", "constraint"}:
            continue
        nodes.append({"id": node_id, "kind": kind, "label": _evo_exec_label(kind, node_id, node)})

    component_ids = {f"component_{task_id}" for task_id in task_ids}
    for edge in analysis.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = f"component_{str(edge.get('source') or '')}"
        target = f"component_{str(edge.get('target') or '')}"
        relation = str(edge.get("relation") or "precedes")
        if source in component_ids and target in component_ids:
            edges.append({"source": source, "target": target, "relation": relation})

    for edge in harness.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        relation = str(edge.get("relation") or "")
        if source in forbidden or target in forbidden:
            continue
        if relation == "uses" and _node_kind(harness, source) == "resource" and _node_kind(harness, target) == "component":
            edges.append({"source": source, "target": target, "relation": relation})

    for node in harness.get("nodes", []):
        if not isinstance(node, dict) or str(node.get("kind") or "") != "constraint":
            continue
        constraint_id = str(node.get("id") or "")
        if constraint_id in forbidden:
            continue
        if final_component:
            edges.append({"source": final_component, "target": constraint_id, "relation": "constrained_by"})
        control_id = f"component_{constraint_id}"
        if _node_kind(harness, control_id) == "control":
            edges.append({"source": constraint_id, "target": control_id, "relation": "realizes"})

    return nodes, edges


def _evo_exec_label(kind: str, node_id: str, node: dict[str, Any]) -> str:
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    if kind == "component":
        core = node_id.removeprefix("component_")
    elif kind == "control":
        core = node_id.removeprefix("component_")
    elif kind == "resource":
        core = str(metadata.get("resource_key") or node_id.removeprefix("resource_") or node_id)
    elif kind == "constraint":
        core = node_id
    else:
        core = node_id
    return _short(_friendly(core), 24)


def _node_kind(harness: dict[str, Any], node_id: str) -> str:
    for node in harness.get("nodes", []):
        if isinstance(node, dict) and str(node.get("id") or "") == node_id:
            return str(node.get("kind") or "")
    return ""


def _friendly(value: str) -> str:
    text = value.replace("_", " ").replace("-", " ").strip()
    text = text.removeprefix("component ").removeprefix("resource ").removeprefix("control ")
    return " ".join(text.split())


def _draw_evolution_panel(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    panel: dict[str, Any],
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> None:
    draw.rounded_rectangle((x, y, x + EV_PANEL_W, y + EV_PANEL_H), radius=10, fill="#ffffff", outline="#cbd5e1", width=2)
    draw.text((x + 16, y + 14), panel["title"], fill="#0f172a", font=fonts["panel_title"])
    draw.multiline_text((x + 16, y + 45), "\n".join(_wrap(panel["subtitle"], 37, 2)), fill="#64748b", font=fonts["panel_subtitle"], spacing=2)

    nodes = panel["nodes"]
    edges = panel["edges"]
    positions = _evo_layout([str(node["id"]) for node in nodes], edges, x + 30, y + 118)
    by_id = {str(node["id"]): node for node in nodes}
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        relation = str(edge.get("relation") or "")
        if source in positions and target in positions:
            _draw_evolution_edge(draw, positions[source], positions[target], relation, fonts["edge"])
    for node_id in [str(node["id"]) for node in nodes]:
        _draw_evolution_node(draw, by_id[node_id], positions[node_id], fonts)


def _draw_evolution_node(
    draw: ImageDraw.ImageDraw,
    node: dict[str, Any],
    position: tuple[float, float],
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> None:
    x, y = position
    kind = str(node.get("kind") or "default")
    fill, stroke = STYLE.get(kind, STYLE["component"])
    if kind == "input":
        draw.rounded_rectangle((x - EV_INPUT_W / 2, y - EV_INPUT_H / 2, x + EV_INPUT_W / 2, y + EV_INPUT_H / 2), radius=10, fill=fill, outline=stroke, width=2)
        labels = _wrap(str(node.get("label") or node.get("id") or ""), 18, 4)
        font = fonts["node"]
    elif kind in {"component_requirement", "constraint"}:
        draw.ellipse((x - EV_RADIUS, y - EV_RADIUS, x + EV_RADIUS, y + EV_RADIUS), fill=fill, outline=stroke, width=2)
        labels = _wrap(str(node.get("label") or node.get("id") or ""), 10, 2)
        font = fonts["small"]
    else:
        draw.ellipse((x - EV_RADIUS, y - EV_RADIUS, x + EV_RADIUS, y + EV_RADIUS), fill=fill, outline=stroke, width=2)
        labels = _wrap(str(node.get("label") or node.get("id") or ""), 13, 3)
        font = fonts["node"]

    heights = [draw.textbbox((0, 0), label, font=font)[3] - draw.textbbox((0, 0), label, font=font)[1] for label in labels]
    total = sum(heights) + 2 * max(0, len(labels) - 1)
    current_y = y - total / 2
    for index, label in enumerate(labels):
        bounds = draw.textbbox((0, 0), label, font=font)
        draw.text((x - (bounds[2] - bounds[0]) / 2, current_y), label, fill="#0f172a", font=font)
        current_y += heights[index] + 2


def _draw_evolution_edge(
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
    start = (sx + ux * EV_RADIUS, sy + uy * EV_RADIUS)
    end = (tx - ux * (EV_RADIUS + 8), ty - uy * (EV_RADIUS + 8))
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
        label = _short(_friendly(relation), 16)
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


def _evo_layout(
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
    queue = [node_id for node_id in node_ids if indegree[node_id] == 0]
    seen = 0
    while queue:
        source = queue.pop(0)
        seen += 1
        for target in outgoing[source]:
            levels[target] = max(levels[target], levels[source] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    if seen != len(node_ids):
        return _evo_circle_layout(node_ids, x0, y0)

    by_level: dict[int, list[str]] = defaultdict(list)
    for node_id in node_ids:
        by_level[levels[node_id]].append(node_id)

    max_level = max(by_level, default=0)
    positions: dict[str, tuple[float, float]] = {}
    usable_w = EV_PANEL_W - 140
    usable_h = EV_PANEL_H - 180
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


def _evo_circle_layout(node_ids: list[str], x0: int, y0: int) -> dict[str, tuple[float, float]]:
    count = max(1, len(node_ids))
    center = (x0 + 45 + (EV_PANEL_W - 140) / 2, y0 + 40 + (EV_PANEL_H - 180) / 2)
    radius = min((EV_PANEL_W - 170) / 2, (EV_PANEL_H - 240) / 2)
    radius = max(96, radius)
    return {
        node_id: (
            center[0] + radius * math.cos(2 * math.pi * index / count - math.pi / 2),
            center[1] + radius * math.sin(2 * math.pi * index / count - math.pi / 2),
        )
        for index, node_id in enumerate(node_ids)
    }


def _draw_evolution_transition(draw: ImageDraw.ImageDraw, x: float, y: float, font: ImageFont.FreeTypeFont) -> None:
    draw.line((x, y, x + 28, y), fill="#0f766e", width=3)
    draw.polygon([(x + 36, y), (x + 26, y - 6), (x + 26, y + 6)], fill="#0f766e")
    draw.text((x + 2, y + 10), "compile", fill="#0f766e", font=font)
