"""Shared tag-metric tests."""

import pytest

from adverse_media.metrics import tag_micro_metrics


def test_tag_metric_basic_cases() -> None:
    assert tag_micro_metrics([{"fraud"}], [{"fraud"}]) == (1.0, 1.0, 1.0)
    assert tag_micro_metrics([{"fraud"}], [{"fraud", "bribery"}]) == (1.0, 0.5, 2 / 3)
    assert tag_micro_metrics([set()], [{"fraud"}]) == (0.0, 0.0, 0.0)
    assert tag_micro_metrics([{"fraud", "bribery"}], [{"fraud"}]) == (0.5, 1.0, 2 / 3)
    assert tag_micro_metrics([{"a"}], [{"a", "b"}])[1] <= 1.0


def test_tag_metric_matches_dev_reference() -> None:
    predicted = [
        {"a", "b"}, {"a", "b"}, {"a", "b"}, {"a", "b"},
        {"a", "b"}, {"a", "c"}, {"d", "e"}, {"g"},
    ]
    gold = [
        {"a", "b"}, {"a", "b"}, {"a", "b"}, {"a", "b"},
        {"a", "b"}, {"a"}, {"f"}, {"h"},
    ]
    precision, recall, _ = tag_micro_metrics(predicted, gold)
    assert precision == pytest.approx(11 / 15)
    assert recall == pytest.approx(11 / 13)
    assert recall <= 1.0
