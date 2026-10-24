"""Selective LLM tests using fake backends and HTTP mock transport."""

import asyncio
import json
from datetime import date

import httpx
import pytest

from adverse_media.config import Settings
from adverse_media.decision.cascade import DecisionRecord, Zone
from adverse_media.llm.base import LLMBackend, LLMError
from adverse_media.llm.openai_compat import OpenAICompatibleLLM
from adverse_media.llm.schemas import extract_json
from adverse_media.llm.synthesizer import HitSynthesizer
from adverse_media.models import Article
from adverse_media.metrics import tag_micro_metrics


def record(zone: Zone = Zone.HIT) -> DecisionRecord:
    return DecisionRecord(
        article=Article(
            id="a1",
            title="Acme report",
            snippet="Acme was investigated for fraud by regulators.",
            source_url="https://example.com/a1",
            published_at=date(2026, 1, 1),
            language="en",
            feed="test",
        ),
        zone=zone,
        p_gate=0.8,
        matched_categories=["fraud"],
    )


def valid(confirmed: bool = True, categories: list[str] | None = None) -> str:
    return json.dumps(
        {
            "confirmed": confirmed,
            "categories": ["fraud"] if categories is None and confirmed else categories or [],
            "sentiment": "negative",
            "reason": "The snippet reports a regulatory fraud investigation.",
        }
    )


class FakeBackend(LLMBackend):
    def __init__(self, replies: list[str], delay: float = 0.0) -> None:
        self.replies = iter(replies)
        self.delay = delay
        self.users: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete(self, system: str, user: str) -> str:
        self.users.append(user)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return next(self.replies)
        finally:
            self.in_flight -= 1


@pytest.mark.asyncio
async def test_valid_fenced_and_rejected_outputs() -> None:
    backend = FakeBackend(["```json\n" + valid() + "\n```", valid(False)])
    batch = await HitSynthesizer(backend, Settings()).synthesize(
        [record(), record()], "Acme GmbH", ["fraud"], "en"
    )
    assert batch.outcomes[0].source == "llm"
    assert batch.outcomes[1].dropped
    assert batch.llm_invocations == 2


@pytest.mark.asyncio
async def test_invalid_then_valid_retries_and_two_invalid_falls_back() -> None:
    backend = FakeBackend(["not json", valid(), "still bad", "also bad"])
    batch = await HitSynthesizer(backend, Settings()).synthesize(
        [record(), record()], "Acme GmbH", ["fraud"], "en"
    )
    assert batch.llm_invocations == 4
    assert batch.llm_retries == 2
    assert batch.llm_fallbacks == 1
    assert "previous reply was invalid" in backend.users[1]


@pytest.mark.asyncio
async def test_invalid_categories_and_empty_confirmed_retry() -> None:
    backend = FakeBackend([valid(True, ["bribery"]), valid()])
    first = await HitSynthesizer(backend, Settings()).synthesize(
        [record()], "Acme GmbH", ["fraud"], "en"
    )
    empty_backend = FakeBackend([valid(True, []), valid()])
    second = await HitSynthesizer(empty_backend, Settings()).synthesize(
        [record()], "Acme GmbH", ["fraud"], "en"
    )
    assert first.llm_invocations == 2
    assert second.llm_invocations == 2
    assert first.llm_fallbacks == 0
    assert second.llm_fallbacks == 0


@pytest.mark.asyncio
async def test_concurrency_timeout_and_none_backend() -> None:
    backend = FakeBackend([valid()] * 4, delay=0.02)
    settings = Settings(llm_max_concurrency=2, llm_timeout_seconds=1)
    batch = await HitSynthesizer(backend, settings).synthesize(
        [record() for _ in range(4)], "Acme GmbH", ["fraud"], "en"
    )
    assert backend.max_in_flight <= 2
    assert batch.llm_invocations == 4

    timeout_backend = FakeBackend([valid()], delay=0.1)
    timeout_batch = await HitSynthesizer(
        timeout_backend, Settings(llm_timeout_seconds=0.001)
    ).synthesize([record(Zone.EDGE)], "Acme GmbH", ["fraud"], "en")
    assert timeout_batch.llm_fallbacks == 1
    assert "NOT verified by an LLM" in timeout_batch.outcomes[0].verdict.reason
    assert "possible victim or cleared party" in timeout_batch.outcomes[0].verdict.reason

    none_batch = await HitSynthesizer(None, Settings()).synthesize(
        [record(), record()], "Acme GmbH", ["fraud"], "en"
    )
    assert none_batch.llm_invocations == 0
    assert none_batch.llm_fallbacks == 2


@pytest.mark.asyncio
async def test_openai_compatible_payload_and_errors() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hello"}}]},
        )

    backend = OpenAICompatibleLLM("https://llm.example/v1", "secret-key", "model-x", 2, True)
    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert await backend.complete("sys", "usr") == "hello"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "model-x"
    assert payload["temperature"] == 0
    assert len(payload["messages"]) == 2
    assert payload["response_format"] == {"type": "json_object"}
    assert requests[0].headers["Authorization"] == "Bearer secret-key"
    await backend.close()

    for status in (429, 500):
        failing = OpenAICompatibleLLM("https://llm.example/v1", "secret-key", "model-x", 2, False)
        failing._client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request, status=status: httpx.Response(status))
        )
        with pytest.raises(LLMError) as error:
            await failing.complete("sys", "usr")
        assert str(status) in str(error.value)
        assert "secret-key" not in str(error.value)
        await failing.close()

    malformed = OpenAICompatibleLLM("https://llm.example/v1", "secret-key", "model-x", 2, False)
    malformed._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    )
    with pytest.raises(LLMError):
        await malformed.complete("sys", "usr")
    await malformed.close()


def test_json_extraction() -> None:
    assert extract_json("prefix ```json\n{\"x\": 1}\n``` suffix") == {"x": 1}
