"""FastAPI service exposing the adverse media pipeline."""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from adverse_media.schemas import AnalyzeRequest, AnalyzeResponse, Metrics


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Set up shared resources once per process."""
    # Decision models and the LLM client are loaded here in later steps.
    yield


app = FastAPI(title="Adverse Media Pipeline", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Screen candidate articles for adverse media about the requested company."""
    # TODO(step 5): wire in the pipeline; this returns an empty, contract-valid report.
    return AnalyzeResponse(
        job_id=str(uuid.uuid4()),
        status="completed",
        company_name=request.company_name,
        metrics=Metrics(
            total_candidates_fetched=0,
            filtered_out_by_decision_models=0,
            decision_model_passes=0,
            llm_invocations=0,
            total_latency_ms=0,
        ),
        results=[],
    )
