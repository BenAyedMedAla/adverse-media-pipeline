"""Pipeline orchestration tests using only fakes."""

import json
from datetime import date

import pytest

from adverse_media.config import Settings
from adverse_media.decision.cascade import Zone
from adverse_media.llm.base import LLMBackend
from adverse_media.llm.synthesizer import HitSynthesizer
from adverse_media.models import Article
from adverse_media.pipeline import AdverseMediaPipeline
from adverse_media.schemas import AnalyzeRequest


def article(article_id: str, title: str, url: str | None = None) -> Article:
    return Article(
        id=article_id,
        title=title,
        snippet=f"{title} was investigated for fraud.",
        source_url=url or f"https://example.com/{article_id}",
        published_at=date(2026, 1, 1),
        language="en",
        feed="test",
    )


class FakeSource:
    name = "fake"

    def __init__(self, articles: list[Article], fail: bool = False) -> None:
        self.articles = articles
        self.fail = fail

    async def fetch(self, company_name: str, keywords: list[str], language: str):
        if self.fail:
            raise RuntimeError("unavailable")
        return self.articles


class FakeClassifier:
    def __init__(self, scores: list[list[float]]) -> None:
        self.scores = iter(scores)

    async def aentailment_probabilities(self, premises, hypotheses):
        return next(self.scores)


class FakeBackend(LLMBackend):
    def __init__(self, confirmed: bool = True) -> None:
        self.confirmed = confirmed

    async def complete(self, system: str, user: str) -> str:
        return json.dumps(
            {
                "confirmed": self.confirmed,
                "categories": ["fraud"] if self.confirmed else [],
                "sentiment": "negative",
                "reason": "The article reports a fraud investigation.",
            }
        )


def pipeline(sources, classifier, backend, settings=None):
    settings = settings or Settings()
    return AdverseMediaPipeline(
        sources,
        classifier,
        HitSynthesizer(backend, settings),
        settings,
    )


@pytest.mark.asyncio
async def test_counts_duplicates_entity_rejection_and_llm_rejection() -> None:
    duplicate = article("duplicate", "Acme fraud", "https://example.com/same")
    other = article("other", "Other company fraud")
    pipe = pipeline(
        [FakeSource([duplicate, article("copy", "Acme fraud", "https://example.com/same"), other])],
        FakeClassifier([[0.8], [0.1], [0.9]]),
        FakeBackend(confirmed=False),
    )
    response = await pipe.run(
        AnalyzeRequest(company_name=" Acme GmbH ", keywords=["Fraud", "fraud"], language="en")
    )
    breakdown = response.metrics.breakdown
    assert response.status == "completed"
    assert response.company_name == "Acme GmbH"
    assert response.metrics.total_candidates_fetched == 3
    assert breakdown.duplicates_removed == 1
    assert breakdown.entity_rejected == 1
    assert breakdown.llm_rejected == 1
    assert response.results == []
    assert response.metrics.decision_model_passes == breakdown.entity_rejected + 1


@pytest.mark.asyncio
async def test_fallback_results_and_sorted_relevancy() -> None:
    pipe = pipeline(
        [
            FakeSource(
                [
                    article("low", "Acme low"),
                    article("high", "Acme high"),
                ]
            )
        ],
        FakeClassifier([[0.7, 0.6], [0.1, 0.1], [0.9, 0.8]]),
        None,
    )
    response = await pipe.run(
        AnalyzeRequest(company_name="Acme GmbH", keywords=["fraud"], language="de")
    )
    assert response.metrics.llm_invocations == 0
    assert response.metrics.breakdown.llm_fallbacks == 2
    assert "NOT verified by an LLM" in response.results[0].reason
    assert [item.relevancy_score for item in response.results] == [0.7, 0.6]


@pytest.mark.asyncio
async def test_all_sources_failed_and_partial_failure() -> None:
    failed = pipeline(
        [FakeSource([], fail=True)],
        FakeClassifier([]),
        None,
    )
    response = await failed.run(
        AnalyzeRequest(company_name="Acme GmbH", keywords=["fraud"])
    )
    assert response.status == "failed"
    assert response.results == []
    assert response.metrics.breakdown.model_pairs == 0

    partial = pipeline(
        [FakeSource([], fail=True), FakeSource([article("a", "Acme fraud")])],
        FakeClassifier([[0.8], [0.1], [0.8]]),
        None,
    )
    response = await partial.run(
        AnalyzeRequest(company_name="Acme GmbH", keywords=["fraud"])
    )
    assert response.status == "completed"
    assert len(response.results) == 1
