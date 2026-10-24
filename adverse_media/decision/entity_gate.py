"""Deterministic company-name relevance gate."""

import re
import unicodedata

from adverse_media.models import Article

LEGAL_FORM_TOKENS = (
    "gmbh",
    "ag",
    "se",
    "kg",
    "ug",
    "inc",
    "corp",
    "corporation",
    "ltd",
    "llc",
    "plc",
    "sa",
    "sarl",
    "srl",
    "bv",
    "nv",
    "co",
)


def _normalize_text(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(char for char in folded if not unicodedata.combining(char))
    return re.sub(r"[^\w]+", " ", without_accents, flags=re.UNICODE).strip()


def normalize_company_name(name: str) -> str:
    """Normalize a company name and remove trailing legal-form tokens."""
    tokens = _normalize_text(name).split()
    while tokens and tokens[-1] in LEGAL_FORM_TOKENS:
        tokens.pop()
    return " ".join(tokens)


class EntityGate:
    """Deterministic rule that rejects articles without the requested company."""

    def __init__(self, company_name: str) -> None:
        self.company_name = company_name
        self.normalized_name = normalize_company_name(company_name)

    def matches(self, article: Article) -> bool:
        """Return whether the normalized company appears as a whole token sequence."""
        if not self.normalized_name:
            return False
        text = _normalize_text(f"{article.title} {article.snippet}")
        pattern = rf"(?<!\w){re.escape(self.normalized_name)}(?!\w)"
        # Fuzzy matching is intentionally avoided: short names create false matches.
        # Legal-form stripping means "Acme Corp" also matches "Acme GmbH"; NLI handles it.
        return re.search(pattern, text) is not None

