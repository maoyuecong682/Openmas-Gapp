from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .llm import LLMConfig, OpenAICompatibleAdapter


PLANNER_SYSTEM_PROMPT = """You are a DeepSeek planner for dynamic multi-agent graphs.
You do not answer the task.

Your job is to:
- understand the task,
- choose the agents needed,
- define each agent's role, objective, capabilities, and tools,
- decide the dependencies between agents,
- produce a graph topology that expresses parallel and serial work.

Rules:
- Return exactly one JSON object.
- Use only the keys "agents" and "edges".
- Do not include prose, markdown, code fences, or extra keys.
- Prefer the smallest graph that fully covers the task.
- Leave independent agents disconnected so they can run in parallel.
- Create edges only for true dependencies.
- Make agent ids unique, stable, snake_case, and task-specific.

Output schema:
{
  "agents": [
    {
      "id": "snake_case_identifier",
      "role": "short role name",
      "objective": "what this agent must achieve",
      "capabilities": ["capability_name"],
      "tools": ["tool_name"]
    }
  ],
  "edges": [
    {
      "source": "agent_id",
      "target": "agent_id"
    }
  ]
}
"""


AGENT_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


@dataclass
class PlannerAgentSpec:
    id: str
    role: str
    objective: str
    capabilities: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "objective": self.objective,
            "capabilities": list(self.capabilities),
            "tools": list(self.tools),
        }


@dataclass
class PlannerEdgeSpec:
    source: str
    target: str
    relation: str = "precedes"

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target}


@dataclass
class PlannerGraphSpec:
    agents: list[PlannerAgentSpec]
    edges: list[PlannerEdgeSpec]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.agents:
            raise ValueError("planner produced no agents")
        seen: set[str] = set()
        for agent in self.agents:
            if not AGENT_ID_RE.match(agent.id):
                raise ValueError(f"invalid agent id: {agent.id!r}")
            if agent.id in seen:
                raise ValueError(f"duplicate agent id: {agent.id!r}")
            if not agent.role.strip():
                raise ValueError(f"agent role is required: {agent.id!r}")
            if not agent.objective.strip():
                raise ValueError(f"agent objective is required: {agent.id!r}")
            seen.add(agent.id)
        for edge in self.edges:
            if edge.source not in seen:
                raise ValueError(f"edge source not declared: {edge.source!r}")
            if edge.target not in seen:
                raise ValueError(f"edge target not declared: {edge.target!r}")
            if edge.source == edge.target:
                raise ValueError(f"self edge not allowed: {edge.source!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "agents": [agent.to_dict() for agent in self.agents],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DeepSeekPlannerConfig:
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    api_key: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 4096
    timeout_seconds: int = 120
    max_retries: int = 2
    repair_retries: int = 1

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        model: str = "deepseek-chat",
        api_key_envs: tuple[str, ...] = ("OPENMAS_LLM_API_KEY", "DEEPSEEK_API_KEY"),
    ) -> "DeepSeekPlannerConfig":
        api_key = None
        for env_name in api_key_envs:
            api_key = os.environ.get(env_name)
            if api_key:
                break
        return cls(
            base_url=base_url or "https://api.deepseek.com",
            model=model,
            api_key=api_key,
        )


class DeepSeekPlannerClient:
    def __init__(self, config: DeepSeekPlannerConfig | None = None):
        self.config = config or DeepSeekPlannerConfig.from_env()
        self.adapter = OpenAICompatibleAdapter(
            LLMConfig(
                provider="deepseek",
                model=self.config.model,
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                timeout_seconds=self.config.timeout_seconds,
                max_retries=self.config.max_retries,
                repair_retries=self.config.repair_retries,
            )
        )

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        model: str = "deepseek-chat",
        api_key_envs: tuple[str, ...] = ("OPENMAS_LLM_API_KEY", "DEEPSEEK_API_KEY"),
    ) -> "DeepSeekPlannerClient":
        return cls(DeepSeekPlannerConfig.from_env(base_url=base_url, model=model, api_key_envs=api_key_envs))

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        seed: int,
        required_fields: set[str] | None = None,
    ):
        return self.adapter.generate_json(system_prompt, user_prompt, seed, required_fields)


def _normalize_agent_id(value: str, fallback: str) -> str:
    text = re.sub(r"[^a-z0-9_]+", "_", str(value).strip().casefold())
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = fallback
    if not text[0].isalpha():
        text = f"a_{text}"
    return text[:64]


class DeepSeekGraphPlanner:
    def __init__(
        self,
        client: DeepSeekPlannerClient | None = None,
        *,
        config: DeepSeekPlannerConfig | None = None,
    ):
        self.client = client or DeepSeekPlannerClient(config)

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        model: str = "deepseek-chat",
        api_key_envs: tuple[str, ...] = ("OPENMAS_LLM_API_KEY", "DEEPSEEK_API_KEY"),
    ) -> "DeepSeekGraphPlanner":
        return cls(client=DeepSeekPlannerClient.from_env(base_url=base_url, model=model, api_key_envs=api_key_envs))

    def plan(self, task: str, seed: int = 11, context: dict[str, Any] | None = None) -> PlannerGraphSpec:
        if not task.strip():
            raise ValueError("task is required")
        user_prompt = json.dumps(
            {
                "task": task,
                "context": context or {},
                "instruction": "Design the task-specific agent graph. Use parallel branches where work is independent. Do not answer the task.",
                "required_output": {
                    "agents": [
                        {
                            "id": "snake_case_identifier",
                            "role": "short role name",
                            "objective": "what this agent must achieve",
                            "capabilities": ["capability_name"],
                            "tools": ["tool_name"],
                        }
                    ],
                    "edges": [
                        {
                            "source": "agent_id",
                            "target": "agent_id",
                        }
                    ],
                },
                "constraints": [
                    "Different tasks should be able to yield different graph topologies.",
                    "Do not reuse a fixed template.",
                    "Do not let prompt wording alone represent the topology.",
                    "Return JSON only.",
                ],
            },
            ensure_ascii=False,
        )
        response = self.client.generate_json(PLANNER_SYSTEM_PROMPT, user_prompt, seed, {"agents", "edges"})
        plan = self._parse_plan(response.value, response.provider, response.model, seed, task, response.raw_text)
        try:
            plan.validate()
        except ValueError as exc:
            plan = self._repair_plan(task, context or {}, seed, response.value, str(exc))
        plan.metadata.update(
            {
                "planner_provider": response.provider,
                "planner_model": response.model,
                "planner_seed": seed,
                "planner_raw_text": response.raw_text,
            }
        )
        return plan

    def _repair_plan(
        self,
        task: str,
        context: dict[str, Any],
        seed: int,
        previous: dict[str, Any],
        validation_error: str,
    ) -> PlannerGraphSpec:
        repair_prompt = json.dumps(
            {
                "task": task,
                "context": context,
                "validation_error": validation_error,
                "previous_output": previous,
                "instruction": "Return a corrected JSON object that contains only agents and edges and satisfies the schema exactly.",
            },
            ensure_ascii=False,
        )
        response = self.client.generate_json(PLANNER_SYSTEM_PROMPT, repair_prompt, seed + 1, {"agents", "edges"})
        plan = self._parse_plan(response.value, response.provider, response.model, seed + 1, task, response.raw_text)
        plan.validate()
        plan.metadata.update(
            {
                "planner_provider": response.provider,
                "planner_model": response.model,
                "planner_seed": seed + 1,
                "planner_raw_text": response.raw_text,
                "planner_repaired": True,
                "planner_validation_error": validation_error,
            }
        )
        return plan

    def _parse_plan(
        self,
        raw: dict[str, Any],
        provider: str,
        model: str,
        seed: int,
        task: str,
        raw_text: str,
    ) -> PlannerGraphSpec:
        raw_agents = raw.get("agents") if isinstance(raw.get("agents"), list) else []
        raw_edges = raw.get("edges") if isinstance(raw.get("edges"), list) else []
        agents: list[PlannerAgentSpec] = []
        seen: set[str] = set()
        id_map: dict[str, str] = {}
        for index, item in enumerate(raw_agents):
            if not isinstance(item, dict):
                continue
            raw_id = str(item.get("id", "")).strip()
            agent_id = _normalize_agent_id(raw_id, f"agent_{index + 1}")
            if agent_id in seen:
                suffix = 2
                candidate = f"{agent_id}_{suffix}"
                while candidate in seen:
                    suffix += 1
                    candidate = f"{agent_id}_{suffix}"
                agent_id = candidate
            seen.add(agent_id)
            if raw_id:
                id_map[raw_id.casefold()] = agent_id
            id_map[agent_id] = agent_id
            agents.append(
                PlannerAgentSpec(
                    agent_id,
                    str(item.get("role") or "agent").strip()[:80],
                    str(item.get("objective") or task).strip()[:220],
                    [str(x).strip() for x in item.get("capabilities", []) if str(x).strip()],
                    [str(x).strip() for x in item.get("tools", []) if str(x).strip()],
                )
            )
        edges: list[PlannerEdgeSpec] = []
        for item in raw_edges:
            if not isinstance(item, dict):
                continue
            raw_source = str(item.get("source", "")).strip()
            raw_target = str(item.get("target", "")).strip()
            source = id_map.get(raw_source.casefold(), _normalize_agent_id(raw_source, raw_source or "agent"))
            target = id_map.get(raw_target.casefold(), _normalize_agent_id(raw_target, raw_target or "agent"))
            if source and target and source != target:
                edges.append(PlannerEdgeSpec(source, target))
        metadata = {
            "planner_provider": provider,
            "planner_model": model,
            "planner_seed": seed,
            "task": task,
            "raw_response": raw_text,
        }
        return PlannerGraphSpec(agents=agents, edges=edges, metadata=metadata)
