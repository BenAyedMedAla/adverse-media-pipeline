"""Retry classification tests without network access."""

import asyncio

import httpx
import pytest

from adverse_media.config import Settings
from adverse_media.decision.cascade import DecisionRecord, Zone
from adverse_media.llm.base import LLMBackend, LLMError
from adverse_media.llm.openai_compat import OpenAICompatibleLLM
from adverse_media.llm.synthesizer import HitSynthesizer
from tests.test_llm import record, valid


class ErrorBackend(LLMBackend):
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        raise self.error


@pytest.mark.asyncio
async def test_retryable_llm_error_retries_and_nonretryable_does_not() -> None:
    retryable = ErrorBackend(LLMError("temporary", retryable=True))
    result = await HitSynthesizer(retryable, Settings()).synthesize(
        [record()], "Acme GmbH", ["fraud"], "en"
    )
    assert retryable.calls == 2
    assert result.llm_retries == 1

    permanent = ErrorBackend(LLMError("bad request", retryable=False))
    result = await HitSynthesizer(permanent, Settings()).synthesize(
        [record()], "Acme GmbH", ["fraud"], "en"
    )
    assert permanent.calls == 1
    assert result.llm_retries == 0


@pytest.mark.asyncio
async def test_http_status_retryability() -> None:
    for status, expected in ((429, True), (500, True), (400, False), (401, False)):
        backend = OpenAICompatibleLLM("https://llm.example/v1", "secret-key", "model", 1, False)
        backend._client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request, status=status: httpx.Response(status))
        )
        with pytest.raises(LLMError) as error:
            await backend.complete("system", "user")
        assert error.value.retryable is expected
        await backend.close()


@pytest.mark.asyncio
async def test_payload_limits_reasoning_effort_and_provider_error_details() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            400,
            json={"error": {"code": "model_error", "message": "invalid request"}},
        )

    backend = OpenAICompatibleLLM(
        "https://llm.example/v1",
        "secret-key",
        "model",
        1,
        True,
        max_tokens=1024,
        reasoning_effort="low",
    )
    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError) as error:
        await backend.complete("system", "user")
    payload = __import__("json").loads(requests[0].content)
    assert payload["max_tokens"] == 1024
    assert payload["reasoning_effort"] == "low"
    assert error.value.status == 400
    assert error.value.provider_code == "model_error"
    assert error.value.provider_message == "invalid request"
    assert "secret-key" not in str(error.value)
    await backend.close()


@pytest.mark.asyncio
async def test_length_finish_reason_is_retryable_truncation() -> None:
    backend = OpenAICompatibleLLM("https://llm.example/v1", "secret-key", "model", 1, True)
    backend._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "{\"confirmed\":"},
                        }
                    ]
                },
            )
        )
    )
    with pytest.raises(LLMError) as error:
        await backend.complete("system", "user")
    assert error.value.retryable
    assert "truncated: finish_reason=length" in str(error.value)
    await backend.close()


@pytest.mark.asyncio
async def test_429_retry_after_is_capped_and_defaulted(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr("adverse_media.llm.synthesizer.asyncio.sleep", fake_sleep)
    backend = OpenAICompatibleLLM("https://llm.example/v1", "secret-key", "model", 1, False)
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "25"}),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": valid()},
                        }
                    ]
                },
            ),
        ]
    )
    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: next(responses)))
    batch = await HitSynthesizer(backend, Settings()).synthesize(
        [record()], "Acme GmbH", ["fraud"], "en"
    )
    assert waits == [25.0]
    assert batch.llm_invocations == 2
    await backend.close()

    waits.clear()
    backend = OpenAICompatibleLLM("https://llm.example/v1", "secret-key", "model", 1, False)
    responses = iter(
        [
            httpx.Response(429),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": valid()},
                        }
                    ]
                },
            ),
        ]
    )
    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: next(responses)))
    await HitSynthesizer(backend, Settings()).synthesize(
        [record()], "Acme GmbH", ["fraud"], "en"
    )
    assert waits == [5.0]
    await backend.close()


@pytest.mark.asyncio
async def test_rate_limit_budget_exhaustion_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr("adverse_media.llm.synthesizer.asyncio.sleep", fake_sleep)
    backend = OpenAICompatibleLLM("https://llm.example/v1", "secret-key", "model", 1, False)
    backend._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(429, headers={"Retry-After": "31"})
        )
    )
    batch = await HitSynthesizer(backend, Settings()).synthesize(
        [record()], "Acme GmbH", ["fraud"], "en"
    )
    assert batch.outcomes[0].source == "fallback"
    assert batch.llm_invocations == 4
    assert batch.llm_retries == 3
    assert waits == [30.0, 30.0, 30.0]
    await backend.close()


@pytest.mark.asyncio
async def test_validation_retry_is_independent_of_rate_limit_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr("adverse_media.llm.synthesizer.asyncio.sleep", fake_sleep)
    backend = OpenAICompatibleLLM("https://llm.example/v1", "secret-key", "model", 1, False)
    responses = iter(
        [
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(200, json={"choices": [{"message": {"content": "bad"}}]}),
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": valid()}}],
                },
            ),
        ]
    )
    backend._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: next(responses))
    )
    batch = await HitSynthesizer(backend, Settings()).synthesize(
        [record()], "Acme GmbH", ["fraud"], "en"
    )
    assert batch.outcomes[0].source == "llm"
    assert batch.llm_invocations == 4
    assert batch.llm_retries == 3
    assert waits == [5.0, 5.0, 5.0]
    await backend.close()


@pytest.mark.asyncio
async def test_non_429_status_does_not_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr("adverse_media.llm.synthesizer.asyncio.sleep", fake_sleep)
    backend = OpenAICompatibleLLM("https://llm.example/v1", "secret-key", "model", 1, False)
    backend._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(500))
    )
    await HitSynthesizer(backend, Settings()).synthesize(
        [record()], "Acme GmbH", ["fraud"], "en"
    )
    assert waits == []
    await backend.close()
