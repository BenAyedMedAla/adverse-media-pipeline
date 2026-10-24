"""Short, explicit prompts for validating one decision-layer candidate."""

from datetime import date
from typing import Any

from adverse_media.decision.cascade import DecisionRecord, Zone

SYSTEM_PROMPT = (
    "You verify adverse-media screening hits. Judge ONLY from the title and snippet provided. "
    "Reply with one JSON object and nothing else."
)
# A short rules-first prompt avoids examples that can anchor a small verification task.
USER_TEMPLATE = """Target company: {company}
Risk categories to consider: {keywords}
Article language: {language_name}
Published date: {published_at}
Title: {title}
Snippet: {snippet}
Decision-layer signals: relevancy score p_gate={p_gate:.3f}; candidate categories={categories}

confirmed=true ONLY if this exact company (not a similarly named one) is accused of, investigated for, sanctioned for or found guilty of one of the categories. confirmed=false if it is the victim, was cleared or denies it, runs training or compliance programmes, is only mentioned in passing, or concerns a different entity with a similar name. categories must be the subset actually reported ([] if confirmed=false). sentiment is the article's tone toward the company. reason is 1-2 sentences in {language_name} citing a specific fact from the snippet, inventing nothing.
Return exactly these JSON keys: confirmed, categories, sentiment, reason."""


def language_name(language: str) -> str:
    """Return a readable language name for the output instruction."""
    return {"en": "English", "de": "German", "fr": "French"}.get(language, language)


def build_messages(
    record: DecisionRecord, company_name: str, keywords: list[str], language: str
) -> tuple[str, str]:
    """Build the system/user messages for one HIT or EDGE record."""
    user = USER_TEMPLATE.format(
        company=company_name,
        keywords=", ".join(keywords),
        language_name=language_name(language),
        published_at=record.article.published_at.isoformat(),
        title=record.article.title,
        snippet=record.article.snippet,
        p_gate=record.p_gate or 0.0,
        categories=", ".join(record.matched_categories),
    )
    if record.zone is Zone.EDGE:
        user += "\nDecision-layer note: this article may describe the company as a victim or cleared party; check carefully."
    return SYSTEM_PROMPT, user
