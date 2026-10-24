"""Request and response models matching the assessment's API contract."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AnalyzeRequest(BaseModel):
    """Target company and risk keywords to screen for."""

    company_name: str = Field(min_length=1)
    keywords: list[str] = Field(min_length=1)
    language: str = "en"


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
    total_latency_ms: int
    """Wall-clock time for the whole job in milliseconds."""


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
