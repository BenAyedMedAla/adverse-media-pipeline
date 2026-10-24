"""Unit tests for classifier contracts and probe helpers."""

import math

import pytest
import torch

from adverse_media.decision.base import DecisionClassifier
from adverse_media.decision.hypotheses import join_risks
from adverse_media.decision.nli_classifier import two_way_entailment
from scripts.probe_hypotheses import auc


def test_two_way_entailment() -> None:
    expected = math.exp(2) / (math.exp(2) + math.exp(0))
    assert two_way_entailment(torch.tensor([9.0, 2.0, 0.0]), 1, 2) == pytest.approx(expected)


def test_fake_classifier_satisfies_abc_contract() -> None:
    class FakeClassifier(DecisionClassifier):
        def entailment_probabilities(
            self, premises: list[str], hypotheses: list[str]
        ) -> list[float]:
            return [0.5] * len(premises)

    assert FakeClassifier().entailment_probabilities(["a"], ["b"]) == [0.5]


def test_auc_perfect_reversed_and_tied() -> None:
    assert auc([1.0, 0.9, 0.1, 0.2], [True, True, False, False]) == 1.0
    assert auc([0.1, 0.2, 1.0, 0.9], [True, True, False, False]) == 0.0
    assert auc([0.5, 0.5], [True, False]) == 0.5


def test_join_risks() -> None:
    assert join_risks(["corruption", "fraud", "bribery", "money laundering"]) == (
        "corruption, fraud, bribery or money laundering"
    )


@pytest.mark.integration
def test_real_model_smoke() -> None:
    if not __import__("os").environ.get("RUN_MODEL_TESTS"):
        pytest.skip("set RUN_MODEL_TESTS=1 to download and run a model")
    from adverse_media.decision.nli_classifier import HFNLIClassifier

    classifier = HFNLIClassifier("MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli")
    classifier.load()
    assert len(classifier.entailment_probabilities(["Acme was fined."], ["Acme committed fraud."])) == 1
