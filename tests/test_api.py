"""API contract tests run in-process against the ASGI app."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from adverse_media.api import app
from adverse_media.schemas import ArticleResult

SAMPLE_PAYLOAD = {
    "company_name": "Acme GmbH",
    "keywords": ["corruption", "fraud", "bribery", "money laundering"],
    "language": "en",
}


def client() -> AsyncClient:
    """Build a client that calls the app without starting a server."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_health() -> None:
    async with client() as c:
        response = await c.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_analyze_matches_contract() -> None:
    async with client() as c:
        response = await c.post("/analyze", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"job_id", "status", "company_name", "metrics", "results"}
    assert set(body["metrics"]) == {
        "total_candidates_fetched",
        "filtered_out_by_decision_models",
        "decision_model_passes",
        "llm_invocations",
        "total_latency_ms",
    }
    assert set(ArticleResult.model_fields) == {
        "title",
        "source_url",
        "published_at",
        "snippet",
        "relevancy_score",
        "sentiment",
        "matched_categories",
        "reason",
    }
    uuid.UUID(body["job_id"])
    assert body["status"] == "completed"
    assert body["company_name"] == "Acme GmbH"


@pytest.mark.asyncio
async def test_analyze_rejects_empty_keywords() -> None:
    async with client() as c:
        response = await c.post("/analyze", json={**SAMPLE_PAYLOAD, "keywords": []})
    assert response.status_code == 422
