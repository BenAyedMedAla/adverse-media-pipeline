"""FastAPI service exposing the adverse media pipeline."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from adverse_media.config import get_settings
from adverse_media.decision.nli_classifier import HFNLIClassifier
from adverse_media.ingestion.sources import build_default_sources
from adverse_media.llm.openai_compat import build_backend
from adverse_media.llm.synthesizer import HitSynthesizer
from adverse_media.pipeline import AdverseMediaPipeline
from adverse_media.schemas import AnalyzeRequest, AnalyzeResponse

logger = logging.getLogger(__name__)
logging.getLogger("adverse_media").setLevel(logging.INFO)


class UTF8JSONResponse(JSONResponse):
    """JSON response whose charset is explicit for legacy clients."""

    media_type = "application/json; charset=utf-8"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load shared decision and LLM resources once per process."""
    if hasattr(app.state, "pipeline"):
        yield
        return
    settings = get_settings()
    classifier = HFNLIClassifier(settings.nli_model_name)
    await asyncio.to_thread(classifier.load)
    backend = build_backend(settings)
    app.state.pipeline = AdverseMediaPipeline(
        build_default_sources(settings.data_path),
        classifier,
        HitSynthesizer(backend, settings),
        settings,
    )
    app.state.backend = backend
    app.state.models_loaded = True
    try:
        yield
    finally:
        if backend is not None:
            await backend.close()


app = FastAPI(
    title="Adverse Media Pipeline",
    lifespan=lifespan,
    default_response_class=UTF8JSONResponse,
)


def get_pipeline(request: Request) -> AdverseMediaPipeline:
    """Return the configured pipeline or report unavailable service state."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline unavailable")
    return pipeline


@app.get("/health")
async def health(request: Request) -> dict[str, bool | str]:
    """Return liveness and model-loading state."""
    return {
        "status": "ok",
        "models_loaded": bool(
            getattr(
                request.app.state,
                "models_loaded",
                getattr(request.app.state, "pipeline", None) is not None,
            )
        ),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    pipeline: AdverseMediaPipeline = Depends(get_pipeline),
) -> AnalyzeResponse:
    """Screen candidate articles for adverse media about the requested company."""
    try:
        return await pipeline.run(request)
    except Exception:
        logger.exception("Analysis failed for company=%s", request.company_name)
        raise HTTPException(status_code=500, detail="analysis failed") from None
