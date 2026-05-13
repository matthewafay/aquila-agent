from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import httpx
from openai import OpenAI


DEFAULT_BASE_URL = os.environ.get("AQUILA_BASE_URL", "http://localhost:1234/v1")
DEFAULT_API_KEY = os.environ.get("AQUILA_API_KEY", "lm-studio")


@dataclass
class ModelInfo:
    id: str
    owned_by: str | None = None


class LMStudioClient:
    """Thin wrapper around the OpenAI SDK pointed at a local LM Studio server."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, api_key: str = DEFAULT_API_KEY):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._sdk = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=600)

    def list_models(self) -> list[ModelInfo]:
        try:
            resp = self._sdk.models.list()
        except Exception as e:
            raise RuntimeError(
                f"Could not reach LM Studio at {self.base_url}. "
                f"Is the local server running? ({e})"
            ) from e
        return [ModelInfo(id=m.id, owned_by=getattr(m, "owned_by", None)) for m in resp.data]

    def ping(self) -> bool:
        try:
            with httpx.Client(timeout=5) as c:
                r = c.get(f"{self.base_url}/models", headers={"Authorization": f"Bearer {self.api_key}"})
            return r.status_code < 500
        except Exception:
            return False

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        stream: bool = False,
    ):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = "auto"
        if stream:
            kwargs["stream"] = True
        return self._sdk.chat.completions.create(**kwargs)
