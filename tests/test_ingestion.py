"""Concurrent fetching, timeout and failure isolation."""

import asyncio
import time
from datetime import date
from pathlib import Path

import pytest

from adverse_media.ingestion.sources import ArticleSource, LocalDatasetSource, fetch_candidates
from adverse_media.models import Article

DATASET = Path(__file__).parents[1] / "data" / "articles.json"


class FakeSource(ArticleSource):
    """Returns one article after a delay, or raises."""

    def __init__(self, name: str, delay_seconds: float = 0.0, fail: bool = False) -> None:
        self.name = name
        self.delay_seconds = delay_seconds
        self.fail = fail

    async def fetch(self, company_name: str, keywords: list[str], language: str) -> list[Article]:
        await asyncio.sleep(self.delay_seconds)
        if self.fail:
            raise RuntimeError("feed unavailable")
        return [
            Article(
                id=f"{self.name}-1",
                title=f"Story from {self.name}",
                snippet="...",
                source_url=f"https://{self.name}.example/1",
                published_at=date(2026, 1, 1),
                language="en",
                feed=self.name,
            )
        ]


async def fetch(sources: list[ArticleSource], timeout_seconds: float = 5.0) -> list[Article]:
    return (await fetch_candidates(sources, "Acme GmbH", ["fraud"], "en", timeout_seconds)).articles


@pytest.mark.asyncio
async def test_sources_run_concurrently() -> None:
    start = time.perf_counter()
    articles = await fetch([FakeSource("a", 0.3), FakeSource("b", 0.3)])
    elapsed = time.perf_counter() - start
    assert len(articles) == 2
    assert elapsed < 0.55


@pytest.mark.asyncio
async def test_slow_source_is_skipped() -> None:
    articles = await fetch([FakeSource("fast"), FakeSource("slow", 1.0)], timeout_seconds=0.1)
    assert [a.feed for a in articles] == ["fast"]


@pytest.mark.asyncio
async def test_failing_source_is_skipped() -> None:
    articles = await fetch([FakeSource("ok"), FakeSource("broken", fail=True)])
    assert [a.feed for a in articles] == ["ok"]


@pytest.mark.asyncio
async def test_fetch_result_counts_failed_sources() -> None:
    result = await fetch_candidates(
        [FakeSource("ok"), FakeSource("broken", fail=True)],
        "Acme GmbH",
        ["fraud"],
        "en",
        5,
    )
    assert result.sources_total == 2
    assert result.sources_failed == 1
    assert len(result.articles) == 1


@pytest.mark.asyncio
async def test_local_source_returns_only_its_feed() -> None:
    articles = await LocalDatasetSource(DATASET, "newswire").fetch("Acme GmbH", ["fraud"], "en")
    assert articles
    assert all(a.feed == "newswire" for a in articles)
