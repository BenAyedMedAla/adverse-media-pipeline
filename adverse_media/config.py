"""Application settings loaded from environment variables and .env."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for thresholds, models and the LLM backend."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gate_reject_threshold: float = 0.50
    victim_veto_threshold: float = 0.30
    tag_threshold: float = 0.50
    tag_max_labels: int = Field(default=2, ge=1)

    nli_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "llama-3.1-8b-instant"
    llm_max_concurrency: int = 4
    llm_timeout_seconds: float = 20.0
    llm_json_mode: bool = True
    llm_max_tokens: int = Field(default=1024, ge=1)
    llm_reasoning_effort: Literal["low", "medium", "high"] | None = None

    source_timeout_seconds: float = 5
    data_path: str = "data/articles.json"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
