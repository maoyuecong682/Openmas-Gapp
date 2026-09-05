from __future__ import annotations

import http.client
import json
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "deterministic"
    model: str = "deterministic-q1-proxy"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 4096
    timeout_seconds: int = 120
    max_retries: int = 2
    repair_retries: int = 1


@dataclass
class AdapterResponse:
    value: dict[str, Any]
    provider: str
    model: str
    seed: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw_text: str = ""
    retry_count: int = 0
    json_repaired: bool = False
    finish_reason: str = ""


class LLMAdapter(ABC):
    config: LLMConfig

    @abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str, seed: int, required_fields: set[str] | None = None) -> AdapterResponse:
        raise NotImplementedError

    def generate_text(self, system_prompt: str, user_prompt: str, seed: int) -> AdapterResponse:
        """Long-form output path used for code and unified patches."""
        return self.generate_json(system_prompt, user_prompt, seed, {"text"})


class DeterministicAdapter(LLMAdapter):
    """Offline protocol adapter. It validates fairness plumbing, not LLM quality."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    def generate_json(self, system_prompt: str, user_prompt: str, seed: int, required_fields: set[str] | None = None) -> AdapterResponse:
        start = time.perf_counter()
        text = json.dumps({"adapter_mode": "offline_protocol_proxy", "seed": seed})
        return AdapterResponse(
            {"adapter_mode": "offline_protocol_proxy", "seed": seed}, self.config.provider,
            self.config.model, seed, _token_estimate(system_prompt + user_prompt),
            _token_estimate(text), (time.perf_counter() - start) * 1000, text,
        )

    def generate_text(self, system_prompt: str, user_prompt: str, seed: int) -> AdapterResponse:
        start = time.perf_counter()
        text = "# deterministic protocol output"
        return AdapterResponse({"text": text}, self.config.provider, self.config.model, seed,
                                _token_estimate(system_prompt + user_prompt), _token_estimate(text),
                                (time.perf_counter() - start) * 1000, text)


class OpenAICompatibleAdapter(LLMAdapter):
    """Dependency-free adapter for OpenAI-compatible chat completion endpoints."""

    def __init__(self, config: LLMConfig):
        if not config.base_url or not config.api_key or not config.model:
            raise ValueError("base_url, api_key, and model are required")
        self.config = config

    def generate_json(self, system_prompt: str, user_prompt: str, seed: int, required_fields: set[str] | None = None) -> AdapterResponse:
        return self._generate(system_prompt, user_prompt, seed, required_fields, as_json=True)

    def generate_text(self, system_prompt: str, user_prompt: str, seed: int) -> AdapterResponse:
        return self._generate(system_prompt, user_prompt, seed, None, as_json=False)

    def _generate(self, system_prompt: str, user_prompt: str, seed: int,
                  required_fields: set[str] | None, as_json: bool) -> AdapterResponse:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "seed": seed,
        }
        if as_json:
            body["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        last_error = None
        total_input = total_output = 0
        repaired = False
        previous_raw = ""
        finish_reason = ""
        truncation_retries = 0
        for attempt in range(self.config.max_retries + self.config.repair_retries + 1):
            try:
                current_body = dict(body)
                # A length finish is not fixed by a wording-only repair. Grow
                # the completion budget on the next attempt, within a bounded
                # ceiling, so long plans and patches can complete.
                if truncation_retries:
                    current_body["max_tokens"] = min(
                        self.config.max_output_tokens * (2 ** truncation_retries), 32768
                    )
                if attempt:
                    tail = previous_raw[-4000:] if previous_raw else "(no previous response captured)"
                    repair = ("Return a complete response only. The previous response may have been truncated. "
                              + ((f"Return one JSON object containing these required fields exactly: {sorted(required_fields)}. "
                                  "For intermediate work, put all useful content in the artifact field. ")
                                 if as_json else "Return only the complete requested text, without commentary. ")
                              + "Previous response tail:\n" + tail)
                    current_body["messages"] = list(body["messages"]) + [{"role": "system", "content": repair}]
                request = urllib.request.Request(url, json.dumps(current_body).encode("utf-8"), {
                    "Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json",
                })
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                choice = payload["choices"][0]
                finish_reason = str(choice.get("finish_reason") or "")
                raw = _content_text(choice.get("message", {}).get("content", ""))
                previous_raw = raw
                usage = payload.get("usage", {})
                total_input += usage.get("prompt_tokens", _token_estimate(system_prompt + user_prompt))
                total_output += usage.get("completion_tokens", _token_estimate(raw))
                if not as_json:
                    if not raw.strip():
                        raise ValueError("empty text response")
                    if finish_reason == "length":
                        raise ValueError("response truncated (finish_reason=length)")
                    raw = _strip_code_fences(raw)
                    return AdapterResponse({"text": raw}, self.config.provider, self.config.model, seed,
                                           total_input, total_output, (time.perf_counter() - start) * 1000,
                                           raw, attempt, False, finish_reason)
                value, was_repaired = _parse_json_object(raw)
                repaired = repaired or was_repaired
                value = _fill_common_aliases(value, required_fields)
                if required_fields and not required_fields.issubset(value):
                    raise ValueError(f"missing required JSON fields: {sorted(required_fields.difference(value))}")
                if finish_reason == "length":
                    raise ValueError("response truncated (finish_reason=length)")
                return AdapterResponse(value, self.config.provider, self.config.model, seed, total_input, total_output, (time.perf_counter() - start) * 1000, raw, attempt, repaired, finish_reason)
            except (ValueError, KeyError, json.JSONDecodeError, OSError,
                    http.client.HTTPException, http.client.IncompleteRead) as exc:
                last_error = exc
                if "finish_reason=length" in str(exc):
                    truncation_retries += 1
        raise RuntimeError(f"LLM JSON generation failed after retries: {last_error}") from last_error


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _parse_json_object(raw: str) -> tuple[dict[str, Any], bool]:
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("response JSON is not an object")
        return value, False
    except json.JSONDecodeError:
        start = raw.find("{")
        if start >= 0:
            # raw_decode stops at the first complete object and tolerates
            # accidental trailing commentary or a second JSON object.
            try:
                value, _ = json.JSONDecoder().raw_decode(raw[start:])
                if isinstance(value, dict):
                    return value, True
            except json.JSONDecodeError:
                pass
        raise


def _strip_code_fences(raw: str) -> str:
    """Normalize long-form code/patch output without parsing it as JSON."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _fill_common_aliases(value: dict[str, Any], required_fields: set[str] | None) -> dict[str, Any]:
    if not required_fields:
        return value
    aliases = {
        "artifact": ("artifact", "answer", "text", "output"),
        "answer": ("answer", "artifact", "text", "output"),
        "text": ("text", "artifact", "answer", "output"),
        "output": ("output", "artifact", "answer", "text"),
    }
    filled = dict(value)
    for field in required_fields:
        if field in filled:
            continue
        for alias in aliases.get(field, (field,)):
            if alias in value:
                filled[field] = value[alias]
                break
        else:
            # If the model returned a different single-field JSON object
            # such as {"analysis": "..."}, treat that value as the main
            # payload. This keeps the executor tolerant while still requiring
            # the response to be structured JSON.
            if len(value) == 1:
                only_value = next(iter(value.values()))
                if isinstance(only_value, (str, int, float, bool)) or only_value is not None:
                    filled[field] = only_value
            else:
                # Prefer the first non-empty scalar value as a best-effort
                # compatibility fallback for models that use unexpected field
                # names.
                for candidate in value.values():
                    if isinstance(candidate, (str, int, float, bool)) and str(candidate).strip():
                        filled[field] = candidate
                        break
                # Intermediate artifacts may be structured lists/dicts. They
                # are valid work products even when the model chose a field
                # such as analysis, steps, result, or content. Serialize only
                # for the artifact contract; terminal contracts stay strict.
                if field == "artifact" and field not in filled:
                    for key in ("analysis", "reasoning", "steps", "result", "data", "content", "message"):
                        candidate = value.get(key)
                        if candidate is not None:
                            filled[field] = (candidate if isinstance(candidate, str)
                                             else json.dumps(candidate, ensure_ascii=False))
                            break
    return filled


def _token_estimate(text: str) -> int:
    return max(1, (len(text) + 3) // 4)
