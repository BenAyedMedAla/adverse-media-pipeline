"""Interfaces for selective LLM backends."""

from abc import ABC, abstractmethod


class LLMError(Exception):
    """An LLM request or response failed."""

    def __init__(
        self,
        message: str,
        retryable: bool = False,
        status: int | None = None,
        provider_code: str | None = None,
        provider_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.provider_code = provider_code
        self.provider_message = provider_message


class LLMBackend(ABC):
    """Raw text backend that knows nothing about our schema; validation happens in the synthesizer."""

    @abstractmethod
    async def complete(self, system: str, user: str) -> str:
        """Return the raw assistant text."""
