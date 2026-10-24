"""Validated, concurrent LLM verification with deterministic fallback."""

import asyncio
import logging
import time
from dataclasses import dataclass

from adverse_media.config import Settings
from adverse_media.decision.cascade import DecisionRecord, Zone
from adverse_media.llm.base import LLMBackend, LLMError
from adverse_media.llm.prompts import build_messages
from adverse_media.llm.schemas import LLMVerdict, extract_json

logger = logging.getLogger(__name__)

# Maximum extra requests after a rate-limit response.
RATE_LIMIT_MAX_RETRIES = 3
# Maximum seconds to wait before a rate-limit retry.
RATE_LIMIT_MAX_WAIT_SECONDS = 30


@dataclass
class SynthesisOutcome:
    record: DecisionRecord
    verdict: LLMVerdict
    source: str
    dropped: bool
    error: str | None = None
    raw_reply: str | None = None
    error_status: int | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass
class SynthesisBatch:
    outcomes: list[SynthesisOutcome]
    llm_invocations: int
    llm_retries: int
    llm_fallbacks: int
    llm_rejected: int
    mean_llm_latency_ms: float


def fallback_verdict(record: DecisionRecord) -> LLMVerdict:
    """Produce a transparent verdict when the LLM is unavailable or invalid."""
    categories = list(record.matched_categories)
    edge_note = " It may describe a possible victim or cleared party and needs review." if record.zone is Zone.EDGE else ""
    reason = (
        f"The article was flagged by the decision layer with relevancy score "
        f"{record.p_gate or 0.0:.3f} and categories {', '.join(categories) or 'none'}; "
        f"it was NOT verified by an LLM.{edge_note}"
    )
    return LLMVerdict.model_validate(
        {
            "confirmed": True,
            "categories": categories,
            "sentiment": "negative",
            "reason": reason,
        },
        context={"allowed": categories},
    )


class HitSynthesizer:
    """Validate selective LLM responses while isolating failures per article."""

    def __init__(self, backend: LLMBackend | None, settings: Settings) -> None:
        self.backend = backend
        self.settings = settings

    async def synthesize(
        self,
        records: list[DecisionRecord],
        company_name: str,
        keywords: list[str],
        language: str,
    ) -> SynthesisBatch:
        if self.backend is None:
            outcomes = [
                SynthesisOutcome(
                    r,
                    fallback_verdict(r),
                    "fallback",
                    False,
                    "LLMError: backend unavailable",
                    None,
                )
                for r in records
            ]
            for outcome in outcomes:
                logger.warning(
                    "LLM fallback article_id=%s attempts=0 error=%s",
                    outcome.record.article.id,
                    outcome.error,
                )
            return SynthesisBatch(
                outcomes,
                0,
                0,
                len(records),
                0,
                0.0,
            )
        semaphore = asyncio.Semaphore(self.settings.llm_max_concurrency)
        counters = {"invocations": 0, "retries": 0}

        async def one(record: DecisionRecord) -> tuple[SynthesisOutcome, float]:
            system, user = build_messages(record, company_name, keywords, language)
            last_error = ""
            last_raw: str | None = None
            last_status: int | None = None
            last_code: str | None = None
            last_message: str | None = None
            attempts = 0
            rate_limit_retries = 0
            general_retry_used = False
            started = time.perf_counter()
            while True:
                attempts += 1
                if attempts > 1:
                    user += (
                        f"\nYour previous reply was invalid: {last_error}. "
                        "Return only valid JSON with the required keys."
                    )
                try:
                    async with semaphore:
                        counters["invocations"] += 1
                        raw = await asyncio.wait_for(
                            self.backend.complete(system, user),
                            timeout=self.settings.llm_timeout_seconds,
                        )
                        last_raw = raw[:300]
                    verdict = LLMVerdict.model_validate(
                        extract_json(raw), context={"allowed": keywords}
                    )
                    latency = (time.perf_counter() - started) * 1000
                    return SynthesisOutcome(record, verdict, "llm", not verdict.confirmed), latency
                except (LLMError, asyncio.TimeoutError, ValueError) as exc:
                    if isinstance(exc, LLMError):
                        last_error = f"{type(exc).__name__}: {exc}"
                        retryable = exc.retryable
                        last_status = exc.status
                        last_code = exc.provider_code
                        last_message = exc.provider_message
                        retry_after = exc.retry_after_seconds
                    elif isinstance(exc, asyncio.TimeoutError):
                        last_error = "TimeoutError: LLM request timed out"
                        retryable = True
                        retry_after = None
                    else:
                        last_error = f"{type(exc).__name__}: {exc}"
                        retryable = True
                        retry_after = None
                    if last_status == 429 and rate_limit_retries < RATE_LIMIT_MAX_RETRIES:
                        rate_limit_retries += 1
                        counters["retries"] += 1
                        wait_seconds = min(
                            retry_after
                            if retry_after is not None
                            else 5.0,
                            RATE_LIMIT_MAX_WAIT_SECONDS,
                        )
                        await asyncio.sleep(wait_seconds)
                        continue
                    if last_status == 429:
                        break
                    if retryable and not general_retry_used:
                        general_retry_used = True
                        counters["retries"] += 1
                        continue
                    break
            latency = (time.perf_counter() - started) * 1000
            logger.warning(
                "LLM fallback article_id=%s attempts=%s error=%s status=%s code=%s message=%s",
                record.article.id,
                attempts,
                last_error,
                last_status,
                last_code,
                (last_message or "")[:200],
            )
            return (
                SynthesisOutcome(
                    record,
                    fallback_verdict(record),
                    "fallback",
                    False,
                    last_error,
                    last_raw,
                    last_status,
                    last_code,
                    last_message,
                ),
                latency,
            )

        results = await asyncio.gather(*(one(record) for record in records))
        outcomes = [result[0] for result in results]
        latencies = [result[1] for result in results]
        return SynthesisBatch(
            outcomes,
            counters["invocations"],
            counters["retries"],
            sum(outcome.source == "fallback" for outcome in outcomes),
            sum(outcome.dropped for outcome in outcomes),
            sum(latencies) / len(latencies) if latencies else 0.0,
        )
