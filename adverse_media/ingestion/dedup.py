"""Deduplication by canonical URL, then by near-identical title."""

from urllib.parse import parse_qsl, urlencode, urlsplit

from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process

from adverse_media.models import Article

# Scorer: token_sort_ratio on normalized titles. Measured on the curated dataset: min duplicate
# score 91.3 (syndicated copies with an outlet suffix), max distinct-story score 63.1.
# The threshold is biased toward the duplicate side on purpose: a false merge silently drops a
# possible adverse hit, while a missed duplicate only shows a story twice.
# token_set_ratio was rejected despite a wider margin on the dataset: it scores any word-subset
# as 100, so "Acme fined" and "Acme fined over bribery in Munich" would collapse into one story.
TITLE_SCORER = fuzz.token_sort_ratio
TITLE_SIMILARITY_THRESHOLD = 85

TRACKING_PARAMS = {"gclid", "fbclid", "mc_cid", "mc_eid", "ref", "ref_src"}


def canonical_url(url: str) -> str:
    """Normalize a URL so trivially different links to the same page compare equal."""
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/")
    query = sorted(
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_PARAMS
    )
    # Scheme and fragment are dropped on purpose: http/https and #anchors are the same article.
    return f"{host}{path}?{urlencode(query)}" if query else f"{host}{path}"


def deduplicate(articles: list[Article]) -> tuple[list[Article], int]:
    """Keep the first occurrence of each story, preserving order; return (kept, removed_count)."""
    kept: list[Article] = []
    seen_urls: set[str] = set()
    kept_titles: list[str] = []
    for article in articles:
        url_key = canonical_url(article.source_url)
        title = default_process(article.title)
        if url_key in seen_urls:
            continue
        if process.extractOne(
            title, kept_titles, scorer=TITLE_SCORER, score_cutoff=TITLE_SIMILARITY_THRESHOLD
        ):
            continue
        seen_urls.add(url_key)
        kept_titles.append(title)
        kept.append(article)
    return kept, len(articles) - len(kept)
