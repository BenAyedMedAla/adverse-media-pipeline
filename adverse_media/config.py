"""Application settings loaded from environment variables and .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for thresholds, models and the LLM backend."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    adverse_threshold_hit: float = 0.80
    adverse_threshold_reject: float = 0.40

    nli_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "llama-3.1-8b-instant"
    llm_max_concurrency: int = 4

    source_timeout_seconds: float = 5


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
