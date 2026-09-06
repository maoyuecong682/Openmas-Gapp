from __future__ import annotations

from collections import defaultdict, deque
from typing import Any


BACKWARD_RELATIONS = {"feedback", "reviews", "review", "reviewed_by"}


def ordered_layer_positions(
    node_ids: list[str],
    edges: list[dict[str, Any]] | list[tuple[str, ...]],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    padding_x: float = 45.0,
    padding_y: float = 40.0,
    order_hint: list[str] | None = None,
    backward_relations: set[str] | None = None,
) -> dict[str, tuple[float, float]]:
    ordered_ids = _preferred_order(node_ids, order_hint)
    if not ordered_ids:
        return {}

    backward = {str(item).casefold() for item in (backward_relations or BACKWARD_RELATIONS)}

    outgoing: dict[str, list[str]] = defaultdict(list)
    indegree = {node_id: 0 for node_id in ordered_ids}
    for edge in edges:
        source, target, relation = _edge_parts(edge)
        if source not in indegree or target not in indegree:
            continue
        if relation.casefold() in backward:
            continue
        if target not in outgoing[source]:
            outgoing[source].append(target)
            indegree[target] += 1

    levels = {node_id: 0 for node_id in ordered_ids}
    order_index = {node_id: index for index, node_id in enumerate(ordered_ids)}
    queue = deque(sorted((node_id for node_id, degree in indegree.items() if degree == 0), key=order_index.get))
    seen = 0
    while queue:
        source = queue.popleft()
        seen += 1
        for target in outgoing.get(source, []):
            levels[target] = max(levels[target], levels[source] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    if seen != len(ordered_ids) or (len(ordered_ids) > 1 and max(levels.values(), default=0) == 0):
        levels = {node_id: index for index, node_id in enumerate(ordered_ids)}

    by_level: dict[int, list[str]] = defaultdict(list)
    for node_id in ordered_ids:
        by_level[levels[node_id]].append(node_id)

    max_level = max(by_level, default=0)
    usable_w = max(1.0, width - 2 * padding_x)
    usable_h = max(1.0, height - 2 * padding_y)
    positions: dict[str, tuple[float, float]] = {}
    for level in range(max_level + 1):
        column = sorted(by_level.get(level, []), key=lambda node_id: order_index[node_id])
        if not column:
            continue
        if max_level == 0:
            x = left + padding_x + usable_w / 2
        else:
            x = left + padding_x + level * (usable_w / max(1, max_level))
        if len(column) == 1:
            positions[column[0]] = (x, top + padding_y + usable_h / 2)
            continue
        gap = min(96.0, max(56.0, usable_h / max(1, len(column) - 1)))
        total = gap * (len(column) - 1)
        if total > usable_h:
            gap = usable_h / max(1, len(column) - 1)
            total = gap * (len(column) - 1)
        start = top + padding_y + max(0.0, (usable_h - total) / 2)
        for index, node_id in enumerate(column):
            positions[node_id] = (x, start + index * gap)
    return positions


def _preferred_order(node_ids: list[str], order_hint: list[str] | None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    hinted = [str(node_id) for node_id in order_hint or []]
    for node_id in hinted:
        if node_id in node_ids and node_id not in seen:
            ordered.append(node_id)
            seen.add(node_id)
    for node_id in node_ids:
        if node_id not in seen:
            ordered.append(node_id)
            seen.add(node_id)
    return ordered


def _edge_parts(edge: dict[str, Any] | tuple[str, ...]) -> tuple[str, str, str]:
    if isinstance(edge, dict):
        return str(edge.get("source") or ""), str(edge.get("target") or ""), str(edge.get("relation") or edge.get("kind") or "")
    source = str(edge[0]) if len(edge) > 0 else ""
    target = str(edge[1]) if len(edge) > 1 else ""
    relation = str(edge[2]) if len(edge) > 2 else ""
    return source, target, relation
