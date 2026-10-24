"""Decision cascade tests using scripted probabilities, never a real model."""

from datetime import date

import pytest

from adverse_media.config import Settings
from adverse_media.decision.cascade import DecisionCascade, Zone, select_tag_categories
from adverse_media.decision.entity_gate import EntityGate
from adverse_media.models import Article


def make_article(article_id: str, title: str) -> Article:
    return Article(
        id=article_id,
        title=title,
        snippet="The company was discussed in a report.",
        source_url=f"https://example.com/{article_id}",
        published_at=date(2026, 1, 1),
        language="en",
        feed="test",
    )


class FakeClassifier:
    def __init__(self, scores: list[list[float]]) -> None:
        self.scores = iter(scores)
        self.calls: list[tuple[list[str], list[str]]] = []

    async def aentailment_probabilities(
        self, premises: list[str], hypotheses: list[str]
    ) -> list[float]:
        self.calls.append((premises, hypotheses))
        return next(self.scores)


@pytest.mark.asyncio
async def test_entity_rejected_articles_never_reach_classifier() -> None:
    classifier = FakeClassifier([[0.6], [0.1], [0.2]])
    result = await DecisionCascade(
        EntityGate("Acme GmbH"), classifier, Settings()
    ).run([make_article("1", "Other company news"), make_article("2", "Acme report")], "Acme GmbH", ["fraud"])
    assert result.n_entity_rejected == 1
    assert len(classifier.calls[0][0]) == 1


@pytest.mark.asyncio
async def test_gate_boundary_and_stage_pair_counts() -> None:
    articles = [make_article("1", "Acme one"), make_article("2", "Acme two")]
    classifier = FakeClassifier([[0.50, 0.49], [0.30], [0.2, 0.8]])
    settings = Settings(gate_reject_threshold=0.50, victim_veto_threshold=0.30, tag_threshold=0.50)
    result = await DecisionCascade(EntityGate("Acme GmbH"), classifier, settings).run(
        articles, "Acme GmbH", ["fraud", "bribery"]
    )
    assert result.records[0].zone is Zone.EDGE
    assert result.records[1].rejected_stage == "gate"
    assert result.records[1].zone is Zone.NOISE
    assert result.model_passes == 2 + 1 + 2
    assert [len(call[0]) for call in classifier.calls] == [2, 1, 2]


@pytest.mark.asyncio
async def test_victim_below_threshold_is_hit_and_tag_fallback_selects_top() -> None:
    classifier = FakeClassifier([[0.9], [0.29], [0.1, 0.2]])
    settings = Settings(tag_threshold=0.5)
    result = await DecisionCascade(EntityGate("Acme GmbH"), classifier, settings).run(
        [make_article("1", "Acme report")], "Acme GmbH", ["fraud", "bribery"]
    )
    assert result.records[0].zone is Zone.HIT
    assert result.records[0].matched_categories == ["bribery"]


def test_tag_selection_caps_and_sorts_by_probability() -> None:
    assert select_tag_categories(
        {"fraud": 0.9, "bribery": 0.8, "corruption": 0.7},
        ["corruption", "fraud", "bribery"],
        0.5,
        2,
    ) == ["fraud", "bribery"]


def test_tag_selection_excludes_below_floor_and_breaks_ties_by_keyword_order() -> None:
    assert select_tag_categories(
        {"fraud": 0.49, "bribery": 0.5, "corruption": 0.5},
        ["corruption", "fraud", "bribery"],
        0.5,
        2,
    ) == ["corruption", "bribery"]


def test_tag_selection_falls_back_to_top_one_and_supports_max_one() -> None:
    probabilities = {"corruption": 0.4, "fraud": 0.6, "bribery": 0.5}
    assert select_tag_categories(probabilities, list(probabilities), 0.7, 2) == ["fraud"]
    assert select_tag_categories(probabilities, list(probabilities), 0.5, 1) == ["fraud"]


@pytest.mark.asyncio
async def test_noise_articles_do_not_run_victim_or_tags() -> None:
    classifier = FakeClassifier([[0.49]])
    result = await DecisionCascade(EntityGate("Acme GmbH"), classifier, Settings()).run(
        [make_article("1", "Acme report")], "Acme GmbH", ["fraud", "bribery"]
    )
    assert result.records[0].zone is Zone.NOISE
    assert [len(call[0]) for call in classifier.calls] == [1]
    assert result.model_passes == 1


@pytest.mark.asyncio
async def test_empty_input_returns_empty_result() -> None:
    classifier = FakeClassifier([])
    result = await DecisionCascade(EntityGate("Acme GmbH"), classifier, Settings()).run(
        [], "Acme GmbH", ["fraud"]
    )
    assert result.records == []
    assert result.n_input == 0
    assert result.model_passes == 0
