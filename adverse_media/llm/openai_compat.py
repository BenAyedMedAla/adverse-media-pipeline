"""OpenAI-compatible asynchronous chat-completions backend."""

from typing import Any
import re

import httpx

from adverse_media.config import Settings
from adverse_media.llm.base import LLMBackend, LLMError


class OpenAICompatibleLLM(LLMBackend):
    """Call providers exposing the OpenAI chat-completions shape."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        json_mode: bool,
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.json_mode = json_mode
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def complete(self, system: str, user: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": "Bearer " + self.api_key},
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            if choice.get("finish_reason") == "length":
                raise LLMError(
                    "truncated: finish_reason=length",
                    retryable=True,
                    status=response.status_code,
                )
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise KeyError("content")
            return content
        except httpx.TimeoutException as exc:
            raise LLMError("LLM request timed out", retryable=True) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            code, message = _provider_error(exc.response, self.api_key)
            raise LLMError(
                f"LLM request failed with status {status}: {message or 'provider error'}",
                retryable=status == 429 or status >= 500,
                status=status,
                provider_code=code,
                provider_message=message,
            ) from exc
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            status = response.status_code if "response" in locals() else "unknown"
            raise LLMError(
                f"LLM response was malformed (status {status})",
                retryable=isinstance(exc, httpx.HTTPError),
                status=status if isinstance(status, int) else None,
            ) from exc

    async def close(self) -> None:
        """Close the lazily-created HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def build_backend(settings: Settings) -> LLMBackend | None:
    """Build the configured backend, or disable LLM calls when no key is set."""
    if not settings.llm_api_key:
        return None
    return OpenAICompatibleLLM(
        settings.llm_base_url,
        settings.llm_api_key,
        settings.llm_model,
        settings.llm_timeout_seconds,
        settings.llm_json_mode,
        settings.llm_max_tokens,
        settings.llm_reasoning_effort,
    )


def _provider_error(
    response: httpx.Response, api_key: str = ""
) -> tuple[str | None, str | None]:
    """Extract a short, sanitized provider error without exposing headers or keys."""
    try:
        error = response.json().get("error", {})
        code = error.get("code") if isinstance(error, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
    except (ValueError, TypeError):
        return None, None
    if not isinstance(code, str):
        code = None
    if not isinstance(message, str):
        message = None
    if message:
        message = re.sub(r"\s+", " ", message).replace("Bearer ", "[redacted] ")
        if api_key:
            message = message.replace(api_key, "[redacted]")
        message = message[:200]
    return code, message
