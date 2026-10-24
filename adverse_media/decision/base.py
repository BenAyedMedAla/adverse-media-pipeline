"""Interfaces for zero-token decision classifiers."""

from abc import ABC, abstractmethod


class DecisionClassifier(ABC):
    """Classifier that scores premise/hypothesis pairs without generated tokens."""

    @abstractmethod
    def entailment_probabilities(
        self, premises: list[str], hypotheses: list[str]
    ) -> list[float]:
        """Return pairwise entailment probabilities using zero generated tokens."""

