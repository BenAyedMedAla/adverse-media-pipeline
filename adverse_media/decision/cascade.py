"""The entity/NLI decision cascade used before the later LLM stage."""

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from adverse_media.config import Settings
from adverse_media.decision.entity_gate import EntityGate
from adverse_media.decision.hypotheses import (
    GATE_HYPOTHESIS,
    TAG_HYPOTHESIS,
    VICTIM_HYPOTHESIS,
    join_risks,
)
from adverse_media.models import Article


class Zone(str, Enum):
    NOISE = "noise"
    EDGE = "edge"
    HIT = "hit"


@dataclass
class DecisionRecord:
    article: Article
    zone: Zone
    p_gate: float | None = None
    p_victim: float | None = None
    tag_probabilities: dict[str, float] = field(default_factory=dict)
    matched_categories: list[str] = field(default_factory=list)
    rejected_stage: str | None = None


@dataclass
class CascadeResult:
    records: list[DecisionRecord]
    n_input: int
    n_entity_rejected: int
    n_gate_rejected: int
    n_edge: int
    n_hit: int
    model_passes: int
    stage_ms: dict[str, float]


def select_tag_categories(
    probabilities: dict[str, float],
    keywords: list[str],
    threshold: float,
    max_labels: int,
) -> list[str]:
    """Select threshold-passing tags by score, with a deterministic top-1 fallback."""
    ranked = sorted(
        keywords,
        key=lambda keyword: (-probabilities[keyword], keywords.index(keyword)),
    )
    selected = [keyword for keyword in ranked if probabilities[keyword] >= threshold]
    if not selected:
        selected = ranked[:1]
    return selected[:max_labels]


class DecisionCascade:
    """Run deterministic entity filtering followed by batched NLI decisions."""

    def __init__(self, gate: EntityGate, classifier: Any, settings: Settings) -> None:
        self.gate = gate
        self.classifier = classifier
        self.settings = settings

    async def _score(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        async_method = getattr(self.classifier, "aentailment_probabilities", None)
        if async_method is not None:
            result = async_method(premises, hypotheses)
            if inspect.isawaitable(result):
                return await result
        return await asyncio.to_thread(
            self.classifier.entailment_probabilities, premises, hypotheses
        )

    async def run(
        self, articles: list[Article], company_name: str, keywords: list[str]
    ) -> CascadeResult:
        """Return per-article decisions and stage telemetry."""
        del company_name
        # Stage timings are meaningful only when the injected classifier is real, not cached.
        stage_ms = {"entity": 0.0, "gate": 0.0, "victim": 0.0, "tags": 0.0}
        started = time.perf_counter()
        records = [
            DecisionRecord(
                article=article,
                zone=Zone.NOISE,
                rejected_stage=None if self.gate.matches(article) else "entity",
            )
            for article in articles
        ]
        stage_ms["entity"] = (time.perf_counter() - started) * 1000
        survivors = [record for record in records if record.rejected_stage is None]
        model_passes = 0
        if not survivors:
            return CascadeResult(
                records, len(articles), len(articles), 0, 0, 0, 0, stage_ms
            )

        premises = [f"{r.article.title}. {r.article.snippet}" for r in survivors]
        risks = join_risks(keywords)
        started = time.perf_counter()
        gate_scores = await self._score(
            premises,
            [
                GATE_HYPOTHESIS.format(company=self.gate.company_name, risks=risks)
            ]
            * len(survivors),
        )
        stage_ms["gate"] = (time.perf_counter() - started) * 1000
        model_passes += len(survivors)
        non_rejected: list[DecisionRecord] = []
        for record, score in zip(survivors, gate_scores):
            record.p_gate = score
            if score < self.settings.gate_reject_threshold:
                record.rejected_stage = "gate"
            else:
                non_rejected.append(record)
        if not non_rejected:
            return self._result(records, model_passes, stage_ms)

        premises = [f"{r.article.title}. {r.article.snippet}" for r in non_rejected]
        started = time.perf_counter()
        victim_scores = await self._score(
            premises,
            [
                VICTIM_HYPOTHESIS.format(company=self.gate.company_name)
            ]
            * len(non_rejected),
        )
        stage_ms["victim"] = (time.perf_counter() - started) * 1000
        model_passes += len(non_rejected)
        for record, score in zip(non_rejected, victim_scores):
            record.p_victim = score
            record.zone = Zone.EDGE if score >= self.settings.victim_veto_threshold else Zone.HIT

        tag_pairs = [(record, keyword) for record in non_rejected for keyword in keywords]
        if tag_pairs:
            started = time.perf_counter()
            tag_scores = await self._score(
                [
                    f"{record.article.title}. {record.article.snippet}"
                    for record, _ in tag_pairs
                ],
                [
                    TAG_HYPOTHESIS.format(keyword=keyword)
                    for _, keyword in tag_pairs
                ],
            )
            stage_ms["tags"] = (time.perf_counter() - started) * 1000
            model_passes += len(tag_pairs)
            for (record, keyword), score in zip(tag_pairs, tag_scores):
                record.tag_probabilities[keyword] = score
            for record in non_rejected:
                record.matched_categories = select_tag_categories(
                    record.tag_probabilities,
                    keywords,
                    self.settings.tag_threshold,
                    self.settings.tag_max_labels,
                )
        return self._result(records, model_passes, stage_ms)

    @staticmethod
    def _result(
        records: list[DecisionRecord], model_passes: int, stage_ms: dict[str, float]
    ) -> CascadeResult:
        return CascadeResult(
            records=records,
            n_input=len(records),
            n_entity_rejected=sum(r.rejected_stage == "entity" for r in records),
            n_gate_rejected=sum(r.rejected_stage == "gate" for r in records),
            n_edge=sum(r.zone is Zone.EDGE for r in records if r.rejected_stage is None),
            n_hit=sum(r.zone is Zone.HIT for r in records if r.rejected_stage is None),
            model_passes=model_passes,
            stage_ms=stage_ms,
        )
