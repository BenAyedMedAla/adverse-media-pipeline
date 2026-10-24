"""Integrity checks for the curated dataset and its gold labels."""

import json
from pathlib import Path

import pytest

from adverse_media.ingestion.dedup import deduplicate
from adverse_media.models import Article

DATA_DIR = Path(__file__).parents[1] / "data"
KINDS = {
    "true_positive",
    "fp_victim",
    "fp_training",
    "fp_denied_rumor",
    "fp_other_entity",
    "fp_mention_only",
    "noise",
    "duplicate",
}
CATEGORIES = {"corruption", "fraud", "bribery", "money laundering"}


@pytest.fixture(scope="module")
def articles() -> list[Article]:
    raw = json.loads((DATA_DIR / "articles.json").read_text(encoding="utf-8"))
    return [Article.model_validate(r) for r in raw]


@pytest.fixture(scope="module")
def labels() -> dict[str, dict]:
    raw = json.loads((DATA_DIR / "gold_labels.json").read_text(encoding="utf-8"))
    return {label["id"]: label for label in raw}


def test_ids_unique(articles: list[Article]) -> None:
    ids = [a.id for a in articles]
    assert len(ids) == len(set(ids))


def test_labels_match_articles_one_to_one(articles: list[Article], labels: dict[str, dict]) -> None:
    assert {a.id for a in articles} == set(labels)


def test_label_fields_valid(labels: dict[str, dict]) -> None:
    for label in labels.values():
        assert label["kind"] in KINDS
        assert label["split"] in {"dev", "test"}
        assert set(label["categories"]) <= CATEGORIES
        assert label["is_hit"] == (label["kind"] == "true_positive")
        assert bool(label["categories"]) == label["is_hit"]


def test_every_kind_in_both_splits(labels: dict[str, dict]) -> None:
    for split in ("dev", "test"):
        assert {l["kind"] for l in labels.values() if l["split"] == split} == KINDS


def test_false_positives_have_notes(labels: dict[str, dict]) -> None:
    for label in labels.values():
        if label["kind"].startswith("fp_"):
            assert label.get("note", "").strip()


def test_about_a_quarter_non_english(articles: list[Article]) -> None:
    share = sum(a.language != "en" for a in articles) / len(articles)
    assert 0.20 <= share <= 0.30


def test_duplicates_point_to_existing_ids(labels: dict[str, dict]) -> None:
    for label in labels.values():
        if label["kind"] == "duplicate":
            assert label["duplicate_of"] in labels
            assert labels[label["duplicate_of"]]["kind"] != "duplicate"


def test_dedup_removes_exactly_the_labelled_duplicates(
    articles: list[Article], labels: dict[str, dict]
) -> None:
    kept, removed = deduplicate(articles)
    removed_ids = {a.id for a in articles} - {a.id for a in kept}
    assert removed_ids == {i for i, l in labels.items() if l["kind"] == "duplicate"}
    assert removed == len(removed_ids)
