"""Request and response models matching the assessment's API contract."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AnalyzeRequest(BaseModel):
    """Target company and risk keywords to screen for."""

    company_name: str = Field(min_length=1)
    keywords: list[str] = Field(min_length=1)
    language: str = "en"

    @field_validator("company_name", mode="before")
    @classmethod
    def strip_company_name(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value

    @field_validator("keywords", mode="before")
    @classmethod
    def strip_keywords(cls, value: list[str]) -> list[str]:
        return [
            keyword.strip()
            for keyword in value
            if isinstance(keyword, str) and keyword.strip()
        ]

    @model_validator(mode="after")
    def require_non_blank_values(self) -> "AnalyzeRequest":
        if not self.company_name:
            raise ValueError("company_name must not be blank")
        if not self.keywords:
            raise ValueError("at least one keyword must remain")
        return self


class MetricsBreakdown(BaseModel):
    """Per-stage telemetry for one analysis."""

    duplicates_removed: int = Field(ge=0)
    """Articles removed during deduplication."""
    entity_rejected: int = Field(ge=0)
    """Articles rejected by the deterministic entity gate."""
    gate_rejected: int = Field(ge=0)
    """Articles rejected by the NLI relevancy gate."""
    flagged_edge: int = Field(ge=0)
    """Articles flagged as possible victim or cleared-party cases."""
    clear_hits: int = Field(ge=0)
    """Articles that passed the victim check as clear hits."""
    model_pairs: int = Field(ge=0)
    """NLI premise-hypothesis pairs evaluated."""
    llm_retries: int = Field(ge=0)
    """LLM retry requests."""
    llm_rejected: int = Field(ge=0)
    """Articles rejected by the LLM verdict."""
    llm_fallbacks: int = Field(ge=0)
    """Articles using the deterministic fallback verdict."""
    stage_ms: dict[str, float]
    """Elapsed milliseconds for each pipeline stage."""


class Metrics(BaseModel):
    """Pipeline telemetry showing how the decision layer reduced noise and LLM usage."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    total_candidates_fetched: int = Field(ge=0)
    """Articles returned by all sources before deduplication."""
    filtered_out_by_decision_models: int = Field(ge=0)
    """Articles rejected by the decision layer and never sent to the LLM."""
    decision_model_passes: int = Field(ge=0)
    """Articles evaluated by the decision layer."""
    llm_invocations: int = Field(ge=0)
    """Calls made to the generative LLM."""
    total_latency_ms: int = Field(ge=0)
    """Wall-clock time for the whole job in milliseconds."""
    breakdown: MetricsBreakdown
    """Detailed counts and timings for the job."""


class ArticleResult(BaseModel):
    """A confirmed adverse media hit."""

    title: str
    source_url: str
    published_at: date
    snippet: str
    relevancy_score: float = Field(ge=0.0, le=1.0)
    sentiment: Literal["negative", "neutral", "positive"]
    matched_categories: list[str]
    reason: str


class AnalyzeResponse(BaseModel):
    """Adverse media report for one job."""

    job_id: str
    status: Literal["completed", "failed"]
    company_name: str
    metrics: Metrics
    results: list[ArticleResult]
