"""Small orchestration layer for the adverse-media analysis flow."""

import logging
import time
import uuid
from collections.abc import Sequence

from adverse_media.config import Settings
from adverse_media.decision.cascade import DecisionCascade
from adverse_media.decision.entity_gate import EntityGate
from adverse_media.decision.nli_classifier import HFNLIClassifier
from adverse_media.ingestion.dedup import deduplicate
from adverse_media.ingestion.sources import ArticleSource, fetch_candidates
from adverse_media.llm.synthesizer import HitSynthesizer
from adverse_media.schemas import AnalyzeRequest, AnalyzeResponse, ArticleResult, Metrics, MetricsBreakdown

logger = logging.getLogger(__name__)
STAGE_NAMES = ("fetch", "dedup", "entity", "gate", "victim", "tags", "llm")


class AdverseMediaPipeline:
    """Run ingestion, decision filtering, and selective LLM synthesis."""

    def __init__(
        self,
        sources: Sequence[ArticleSource],
        classifier: HFNLIClassifier,
        synthesizer: HitSynthesizer,
        settings: Settings,
    ) -> None:
        self.sources = list(sources)
        self.classifier = classifier
        self.synthesizer = synthesizer
        self.settings = settings

    async def run(self, request: AnalyzeRequest) -> AnalyzeResponse:
        """Analyze one request and return its contract response."""
        company_name = request.company_name.strip()
        keywords = self._normalize_keywords(request.keywords)
        if not keywords:
            raise ValueError("at least one non-empty keyword is required")
        language = request.language.strip()
        job_id = str(uuid.uuid4())
        started = time.perf_counter()

        fetch_started = time.perf_counter()
        fetched = await fetch_candidates(
            self.sources,
            company_name,
            keywords,
            language,
            self.settings.source_timeout_seconds,
        )
        fetch_ms = (time.perf_counter() - fetch_started) * 1000

        dedup_started = time.perf_counter()
        articles, duplicates_removed = deduplicate(fetched.articles)
        dedup_ms = (time.perf_counter() - dedup_started) * 1000

        if fetched.sources_total > 0 and fetched.sources_failed == fetched.sources_total:
            response = self._failed_response(
                job_id,
                company_name,
                int((time.perf_counter() - started) * 1000),
            )
            logger.info(
                "job_id=%s company=%s candidates=0 filtered=0 passes=0 invocations=0 latency_ms=%s",
                job_id,
                company_name,
                response.metrics.total_latency_ms,
            )
            return response

        cascade = await DecisionCascade(
            EntityGate(company_name), self.classifier, self.settings
        ).run(articles, company_name, keywords)
        records = [record for record in cascade.records if record.rejected_stage is None]
        llm_started = time.perf_counter()
        synthesis = await self.synthesizer.synthesize(
            records, company_name, keywords, language
        )
        llm_ms = (time.perf_counter() - llm_started) * 1000
        outcomes = [outcome for outcome in synthesis.outcomes if not outcome.dropped]
        results = sorted(
            [
                ArticleResult(
                    title=outcome.record.article.title,
                    source_url=outcome.record.article.source_url,
                    published_at=outcome.record.article.published_at,
                    snippet=outcome.record.article.snippet,
                    relevancy_score=round(outcome.record.p_gate or 0.0, 4),
                    sentiment=outcome.verdict.sentiment,
                    matched_categories=outcome.verdict.categories
                    if outcome.source == "llm"
                    else outcome.record.matched_categories,
                    reason=outcome.verdict.reason,
                )
                for outcome in outcomes
            ],
            key=lambda result: result.relevancy_score,
            reverse=True,
        )
        stage_ms = {
            "fetch": fetch_ms,
            "dedup": dedup_ms,
            **{name: cascade.stage_ms.get(name, 0.0) for name in STAGE_NAMES[2:6]},
            "llm": llm_ms,
        }
        metrics = Metrics(
            total_candidates_fetched=len(fetched.articles),
            filtered_out_by_decision_models=(
                cascade.n_entity_rejected + cascade.n_gate_rejected
            ),
            decision_model_passes=len(articles),
            llm_invocations=synthesis.llm_invocations,
            total_latency_ms=int((time.perf_counter() - started) * 1000),
            breakdown=MetricsBreakdown(
                duplicates_removed=duplicates_removed,
                entity_rejected=cascade.n_entity_rejected,
                gate_rejected=cascade.n_gate_rejected,
                flagged_edge=cascade.n_edge,
                clear_hits=cascade.n_hit,
                model_pairs=cascade.model_passes,
                llm_retries=synthesis.llm_retries,
                llm_rejected=synthesis.llm_rejected,
                llm_fallbacks=synthesis.llm_fallbacks,
                stage_ms=stage_ms,
            ),
        )
        logger.info(
            "job_id=%s company=%s candidates=%s filtered=%s passes=%s invocations=%s latency_ms=%s",
            job_id,
            company_name,
            metrics.total_candidates_fetched,
            metrics.filtered_out_by_decision_models,
            metrics.decision_model_passes,
            metrics.llm_invocations,
            metrics.total_latency_ms,
        )
        return AnalyzeResponse(
            job_id=job_id,
            status="completed",
            company_name=company_name,
            metrics=metrics,
            results=results,
        )

    @staticmethod
    def _normalize_keywords(keywords: list[str]) -> list[str]:
        """Trim keywords and keep the first spelling of each case-insensitive value."""
        normalized: list[str] = []
        seen: set[str] = set()
        for keyword in keywords:
            value = keyword.strip()
            key = value.casefold()
            if value and key not in seen:
                normalized.append(value)
                seen.add(key)
        return normalized

    @staticmethod
    def _failed_response(job_id: str, company_name: str, latency_ms: int) -> AnalyzeResponse:
        """Return the intentionally minimal response for total source failure."""
        zero_stage_ms = {name: 0.0 for name in STAGE_NAMES}
        metrics = Metrics(
            total_candidates_fetched=0,
            filtered_out_by_decision_models=0,
            decision_model_passes=0,
            llm_invocations=0,
            total_latency_ms=latency_ms,
            breakdown=MetricsBreakdown(
                duplicates_removed=0,
                entity_rejected=0,
                gate_rejected=0,
                flagged_edge=0,
                clear_hits=0,
                model_pairs=0,
                llm_retries=0,
                llm_rejected=0,
                llm_fallbacks=0,
                stage_ms=zero_stage_ms,
            ),
        )
        return AnalyzeResponse(
            job_id=job_id,
            status="failed",
            company_name=company_name,
            metrics=metrics,
            results=[],
        )
