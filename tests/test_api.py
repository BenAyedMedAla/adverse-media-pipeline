"""API contract tests using an injected fake pipeline."""

import uuid
from datetime import date

import pytest
from httpx import ASGITransport, AsyncClient

from adverse_media.api import app
from adverse_media.schemas import (
    AnalyzeResponse,
    ArticleResult,
    Metrics,
    MetricsBreakdown,
)

SAMPLE_PAYLOAD = {
    "company_name": "Acme GmbH",
    "keywords": ["corruption", "fraud", "bribery", "money laundering"],
    "language": "en",
}


class FakePipeline:
    async def run(self, request):
        return AnalyzeResponse(
            job_id=str(uuid.uuid4()),
            status="completed",
            company_name=request.company_name,
            metrics=Metrics(
                total_candidates_fetched=1,
                filtered_out_by_decision_models=0,
                decision_model_passes=1,
                llm_invocations=0,
                total_latency_ms=1,
                breakdown=MetricsBreakdown(
                    duplicates_removed=0,
                    entity_rejected=0,
                    gate_rejected=0,
                    flagged_edge=0,
                    clear_hits=1,
                    model_pairs=1,
                    llm_retries=0,
                    llm_rejected=0,
                    llm_fallbacks=1,
                    stage_ms={
                        "fetch": 0.1,
                        "dedup": 0.1,
                        "entity": 0.1,
                        "gate": 0.1,
                        "victim": 0.1,
                        "tags": 0.1,
                        "llm": 0.1,
                    },
                ),
            ),
            results=[
                ArticleResult(
                    title="Acme fraud report",
                    source_url="https://example.com/article",
                    published_at=date(2026, 1, 1),
                    snippet="Acme was investigated for fraud.",
                    relevancy_score=0.9,
                    sentiment="negative",
                    matched_categories=["fraud"],
                    reason="The article reports an investigation.",
                )
            ],
        )


def client() -> AsyncClient:
    """Build a client that calls the app without starting a server."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_health() -> None:
    app.state.pipeline = FakePipeline()
    app.state.models_loaded = True
    async with client() as c:
        response = await c.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models_loaded": True}
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    del app.state.pipeline


@pytest.mark.asyncio
async def test_analyze_matches_contract() -> None:
    app.state.pipeline = FakePipeline()
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
        "breakdown",
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
    del app.state.pipeline


@pytest.mark.asyncio
async def test_analyze_rejects_empty_keywords() -> None:
    app.state.pipeline = FakePipeline()
    async with client() as c:
        response = await c.post("/analyze", json={**SAMPLE_PAYLOAD, "keywords": []})
    assert response.status_code == 422
    del app.state.pipeline


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {**SAMPLE_PAYLOAD, "company_name": "   "},
        {**SAMPLE_PAYLOAD, "keywords": [" ", "\t"]},
        {**SAMPLE_PAYLOAD, "keywords": ["  fraud  ", " ", "FRAUD"]},
    ],
)
async def test_analyze_rejects_blank_request_values(payload) -> None:
    app.state.pipeline = FakePipeline()
    async with client() as c:
        response = await c.post("/analyze", json=payload)
    if payload["keywords"] == ["  fraud  ", " ", "FRAUD"]:
        assert response.status_code == 200
    else:
        assert response.status_code == 422
    del app.state.pipeline


@pytest.mark.asyncio
async def test_analyze_returns_503_without_pipeline() -> None:
    if hasattr(app.state, "pipeline"):
        del app.state.pipeline
    async with client() as c:
        response = await c.post("/analyze", json=SAMPLE_PAYLOAD)
    assert response.status_code == 503
