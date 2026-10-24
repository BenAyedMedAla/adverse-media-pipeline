"""Deduplication by canonical URL and syndicated title."""

from datetime import date

from adverse_media.ingestion.dedup import canonical_url, deduplicate
from adverse_media.models import Article


def make_article(id: str, title: str, url: str) -> Article:
    return Article(
        id=id,
        title=title,
        snippet="...",
        source_url=url,
        published_at=date(2026, 1, 1),
        language="en",
        feed="test",
    )


def test_url_variants_share_canonical_key() -> None:
    base = canonical_url("https://news.example/acme/raid")
    assert canonical_url("https://news.example/acme/raid?utm_source=x&utm_medium=y") == base
    assert canonical_url("http://news.example/acme/raid") == base
    assert canonical_url("https://www.news.example/acme/raid") == base
    assert canonical_url("https://news.example/acme/raid/") == base
    assert canonical_url("HTTPS://WWW.News.Example/acme/raid/#top") == base


def test_url_variants_collapse() -> None:
    articles = [
        make_article("1", "Acme raided", "https://news.example/acme/raid"),
        make_article("2", "Acme raided (update)", "http://www.news.example/acme/raid/?utm_source=x"),
    ]
    kept, removed = deduplicate(articles)
    assert [a.id for a in kept] == ["1"]
    assert removed == 1


def test_syndicated_copy_collapses() -> None:
    articles = [
        make_article("1", "Acme GmbH fined €12 million for rigging municipal tenders", "https://one.example/a"),
        make_article(
            "2",
            "Acme GmbH fined €12 million for rigging municipal tenders – Other Outlet",
            "https://two.example/b",
        ),
    ]
    kept, removed = deduplicate(articles)
    assert [a.id for a in kept] == ["1"]
    assert removed == 1


def test_short_headline_with_outlet_suffix_not_merged() -> None:
    # Known limitation, accepted: on short headlines the outlet suffix outweighs the shared text
    # and the score falls below the threshold. A missed duplicate only shows the story twice.
    # Stripping outlet suffixes in the normalizer was rejected: cutting at " – " would turn
    # "Acme GmbH – bribery probe" and "Acme GmbH – fraud probe" into the same title.
    articles = [
        make_article("1", "Acme GmbH fined for rigging tenders", "https://one.example/a"),
        make_article("2", "Acme GmbH fined for rigging tenders – Other Outlet", "https://two.example/b"),
    ]
    kept, removed = deduplicate(articles)
    assert [a.id for a in kept] == ["1", "2"]
    assert removed == 0


def test_title_subset_is_not_a_duplicate() -> None:
    articles = [
        make_article("1", "Acme fined", "https://one.example/a"),
        make_article("2", "Acme fined over bribery in Munich", "https://two.example/b"),
    ]
    kept, removed = deduplicate(articles)
    assert [a.id for a in kept] == ["1", "2"]
    assert removed == 0


def test_distinct_articles_kept_in_order() -> None:
    articles = [
        make_article("3", "Acme GmbH opens new plant in Saxony", "https://one.example/plant"),
        make_article("1", "Prosecutors raid Acme GmbH in bribery probe", "https://one.example/raid"),
        make_article("2", "Acme GmbH reports revenue growth", "https://two.example/revenue"),
        make_article("4", "Acme GmbH reports revenue growth", "https://three.example/copy"),
    ]
    kept, removed = deduplicate(articles)
    assert [a.id for a in kept] == ["3", "1", "2"]
    assert removed == 1
