"""CPU-friendly Hugging Face NLI classifier."""

import asyncio
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from adverse_media.decision.base import DecisionClassifier


def two_way_entailment(logits: Any, entail_idx: int, contra_idx: int) -> float:
    """Compute entailment probability after restricting softmax to two NLI labels."""
    values = torch.as_tensor(logits, dtype=torch.float32)
    probability = torch.softmax(values[[entail_idx, contra_idx]], dim=0)[0]
    return float(probability.item())


class HFNLIClassifier(DecisionClassifier):
    """Loads a Hugging Face sequence classifier and scores pairwise NLI inputs."""

    def __init__(
        self,
        model_name: str,
        max_length: int = 256,
        batch_size: int = 16,
        quantize: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.quantize = quantize
        self.tokenizer = None
        self.model = None
        self.entail_idx = None
        self.contra_idx = None
        self._async_lock = asyncio.Lock()
        # DEV probing showed int8 quantization collapsed AUC from 0.91 to 0.56
        # while saving only about 27% of runtime, so production uses float32.

    def load(self) -> None:
        """Load tokenizer and CPU float32 model, then resolve NLI label indices."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, dtype=torch.float32
        )
        self.model.to("cpu")
        if self.quantize:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        self.model.eval()
        labels = self.model.config.id2label
        normalized = {str(label).casefold(): int(index) for index, label in labels.items()}
        try:
            self.entail_idx = next(index for label, index in normalized.items() if "entail" in label)
            self.contra_idx = next(
                index for label, index in normalized.items() if "contrad" in label
            )
        except StopIteration as exc:
            raise ValueError(
                f"Model {self.model_name!r} must expose entailment and contradiction labels; "
                f"found {labels!r}"
            ) from exc

    def entailment_probabilities(
        self, premises: list[str], hypotheses: list[str]
    ) -> list[float]:
        """Return pairwise entailment probabilities for equal-length inputs."""
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have equal length")
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call load() before scoring")

        probabilities: list[float] = []
        for start in range(0, len(premises), self.batch_size):
            batch_premises = premises[start : start + self.batch_size]
            batch_hypotheses = hypotheses[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                output = self.model(**encoded)
            probabilities.extend(
                two_way_entailment(logits, self.entail_idx, self.contra_idx)
                for logits in output.logits
            )
        return probabilities

    async def aentailment_probabilities(
        self, premises: list[str], hypotheses: list[str]
    ) -> list[float]:
        """Run synchronous scoring in a worker thread, serialized by one lock."""
        async with self._async_lock:
            return await asyncio.to_thread(self.entailment_probabilities, premises, hypotheses)
