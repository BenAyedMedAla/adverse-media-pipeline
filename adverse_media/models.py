"""Internal domain models passed between pipeline stages."""

from datetime import date

from pydantic import BaseModel


class Article(BaseModel):
    """A candidate news article as returned by a source."""

    id: str
    title: str
    snippet: str
    source_url: str
    published_at: date
    language: str
    feed: str
