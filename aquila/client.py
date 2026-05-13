from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import httpx
from openai import OpenAI


DEFAULT_BASE_URL = os.environ.get("AQUILA_BASE_URL", "http://localhost:1234/v1")
DEFAULT_API_KEY = os.environ.get("AQUILA_API_KEY", "lm-studio")


def _default_request_timeout() -> float:
    """Read AQUILA_REQUEST_TIMEOUT from the environment, falling back to 1200s.

    Local-model first-token latency on big-context turns can easily push past the
    OpenAI SDK's stock 600s ceiling, so the default is generous and the CLI lets
    you crank it further with --request-timeout.
    """
    raw = os.environ.get("AQUILA_REQUEST_TIMEOUT")
    if raw is None:
        return 1200.0
    try:
        return float(raw)
    except ValueError:
        return 1200.0


DEFAULT_REQUEST_TIMEOUT = _default_request_timeout()


@dataclass
class ModelInfo:
    id: str
    owned_by: str | None = None


class LMStudioClient:
    """Thin wrapper around the OpenAI SDK pointed at a local LM Studio server."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.request_timeout = request_timeout
        self._sdk = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=request_timeout)
        # Per-model context length, cached after first lookup.
        self._ctx_cache: dict[str, int | None] = {}

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

    def get_model_context_length(self, model_id: str) -> int | None:
        """Look up the loaded context length for a model via LM Studio's
        /v1/models/{id} endpoint, which adds non-standard fields beyond the
        OpenAI schema. Returns None if the server doesn't expose it (other
        OpenAI-compatible servers usually don't). Result is cached per-model."""
        if model_id in self._ctx_cache:
            return self._ctx_cache[model_id]
        try:
            with httpx.Client(timeout=5) as c:
                r = c.get(
                    f"{self.base_url}/models/{model_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            if r.status_code >= 400:
                self._ctx_cache[model_id] = None
                return None
            data = r.json()
        except Exception:
            self._ctx_cache[model_id] = None
            return None
        # LM Studio exposes loaded_context_length; fall back to a few other
        # plausible names that other OpenAI-compatible servers might use.
        for key in ("loaded_context_length", "max_context_length", "context_length", "n_ctx"):
            v = data.get(key) if isinstance(data, dict) else None
            if isinstance(v, int) and v > 0:
                self._ctx_cache[model_id] = v
                return v
        self._ctx_cache[model_id] = None
        return None

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
            # Request a final usage chunk so we can track context fill across
            # streaming calls. Modern OpenAI-compatible servers (LM Studio,
            # vLLM, etc.) honor this; older / stricter ones may ignore it,
            # in which case usage stays None and the context display reads "--".
            kwargs["stream_options"] = {"include_usage": True}
        return self._sdk.chat.completions.create(**kwargs)
