from __future__ import annotations

from datetime import datetime, timezone

from .schema import MASSpec, TraceEvent


def planned_trace(spec: MASSpec, task_id: str) -> list[TraceEvent]:
    """Emit a construction-plan trace for schema and adapter smoke tests.

    This is deliberately not a task-execution result. It records the nodes and
    edges a runtime should execute; domain tool calls must be produced by a
    real sandbox adapter before runtime contracts are scored.
    """
    now = datetime.now(timezone.utc).isoformat()
    events = []
    for index, node in enumerate(spec.nodes):
        events.append(TraceEvent(
            event_id=f"plan_{index:04d}", package_id=spec.package_id, task_id=task_id,
            actor=node.id, action="planned_node", timestamp=now,
            capability_refs=node.capabilities, payload={"role": node.role, "trace_mode": "plan_only"},
        ))
    return events

