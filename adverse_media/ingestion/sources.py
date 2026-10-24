"""Candidate article sources and concurrent, fault-tolerant fetching."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from adverse_media.models import Article

logger = logging.getLogger(__name__)


class ArticleSource(ABC):
    """A feed of candidate news articles."""

    name: str

    @abstractmethod
    async def fetch(self, company_name: str, keywords: list[str], language: str) -> list[Article]:
        """Return candidate articles for the company and keywords."""


class LocalDatasetSource(ArticleSource):
    """Serves one feed of the local JSON dataset with simulated network latency."""

    def __init__(self, path: Path | str, feed: str, delay_seconds: float = 0.0) -> None:
        self.path = Path(path)
        self.name = feed
        self.feed = feed
        self.delay_seconds = delay_seconds

    async def fetch(self, company_name: str, keywords: list[str], language: str) -> list[Article]:
        """Return every article in this feed.

        This simulates a broad candidate feed; a real source would query by company and keywords.
        `language` is ignored: the stub has no language-specific query, and the multilingual
        decision layer evaluates articles in any language.
        """
        records = await asyncio.to_thread(self._load)
        await asyncio.sleep(self.delay_seconds)
        return [Article.model_validate(r) for r in records if r["feed"] == self.feed]

    def _load(self) -> list[dict]:
        """Read and parse the dataset file (blocking; run in a worker thread)."""
        return json.loads(self.path.read_text(encoding="utf-8"))


async def fetch_candidates(
    sources: list[ArticleSource],
    company_name: str,
    keywords: list[str],
    language: str,
    timeout_seconds: float,
) -> list[Article]:
    """Fetch from all sources concurrently; a slow or failing source is skipped, never fatal."""
    batches = await asyncio.gather(
        *(_fetch_one(s, company_name, keywords, language, timeout_seconds) for s in sources)
    )
    return [article for batch in batches for article in batch]


async def _fetch_one(
    source: ArticleSource,
    company_name: str,
    keywords: list[str],
    language: str,
    timeout_seconds: float,
) -> list[Article]:
    """Fetch from one source, returning an empty list on timeout or error."""
    try:
        return await asyncio.wait_for(source.fetch(company_name, keywords, language), timeout_seconds)
    except TimeoutError:
        logger.warning("Source %r timed out after %ss; skipping", source.name, timeout_seconds)
    except Exception:
        logger.warning("Source %r failed; skipping", source.name, exc_info=True)
    return []
