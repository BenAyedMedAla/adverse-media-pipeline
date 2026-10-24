"""Validated structured output from the selective LLM."""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class LLMVerdict(BaseModel):
    """The small structured verdict accepted from the LLM."""

    confirmed: bool
    categories: list[str]
    sentiment: Literal["negative", "neutral", "positive"]
    reason: str = Field(min_length=10, max_length=400)

    @field_validator("categories")
    @classmethod
    def normalize_categories(cls, value: list[str], info: ValidationInfo) -> list[str]:
        allowed = {
            str(item).casefold(): str(item)
            for item in (info.context or {}).get("allowed", [])
        }
        normalized: list[str] = []
        for category in value:
            key = category.casefold()
            if key not in allowed:
                raise ValueError(f"category is not requested: {category}")
            if allowed[key] not in normalized:
                normalized.append(allowed[key])
        return normalized

    @model_validator(mode="after")
    def require_category_for_confirmed(self) -> "LLMVerdict":
        if self.confirmed and not self.categories:
            raise ValueError("confirmed=true requires at least one category")
        return self


def extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from plain or fenced assistant text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("JSON response must be an object")
    return value
