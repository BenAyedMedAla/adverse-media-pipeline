"""Tests for deterministic company matching."""

from datetime import date

from adverse_media.decision.entity_gate import EntityGate, normalize_company_name
from adverse_media.models import Article


def article(title: str, snippet: str = "") -> Article:
    return Article(
        id="x",
        title=title,
        snippet=snippet,
        source_url="https://example.com/x",
        published_at=date(2026, 1, 1),
        language="en",
        feed="test",
    )


def test_normalization_strips_legal_forms_and_accents() -> None:
    assert normalize_company_name("  Café Société GmbH AG  ") == "cafe societe"


def test_gate_matches_whole_tokens_and_apostrophes() -> None:
    gate = EntityGate("Acme GmbH")
    assert not gate.matches(article("Acmeville announces results"))
    assert gate.matches(article("Acme's board responds"))


def test_gate_rejects_absent_company() -> None:
    assert not EntityGate("Acme GmbH").matches(article("Other company investigated"))


def test_legal_form_limitation_is_explicit() -> None:
    # Known limitation: stripping legal forms makes "Acme Corp" match "Acme GmbH".
    assert EntityGate("Acme Corp").matches(article("Acme GmbH faces an inquiry"))

