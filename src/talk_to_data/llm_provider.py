"""
LLM provider abstraction. The rest of the app (nl_to_sql.py, the chatbot UI)
depends only on this interface, never on a concrete provider -- this is what
lets the platform run with zero API cost (mock provider) during development
and grading, while supporting real Anthropic-powered NL-to-SQL when a key is
configured, without changing any calling code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMProviderError(RuntimeError):
    """Raised on any provider failure; message is safe to show to end users."""


class LLMProvider(ABC):
    """Common interface every concrete provider (mock, anthropic, ...) must implement."""

    @abstractmethod
    def generate_sql(self, question: str, conversation_context: str = "") -> str:
        """Returns a SQL string (or the literal 'NO_VALID_QUERY') for the given question."""
        raise NotImplementedError

    @abstractmethod
    def generate_answer(self, question: str, sql_result_summary: str) -> str:
        """Returns a business-readable answer grounded in the given SQL result summary."""
        raise NotImplementedError


def get_llm_provider() -> LLMProvider:
    """
    Factory returning the configured provider. Falls back to the mock
    provider (with a logged warning) on any misconfiguration, so the app
    never crashes just because the LLM setup is incomplete.
    """
    provider_name = settings.LLM_PROVIDER

    if provider_name == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            logger.warning(
                "LLM_PROVIDER='anthropic' but ANTHROPIC_API_KEY is not set. "
                "Falling back to the mock provider."
            )
        else:
            from src.talk_to_data.providers.anthropic_provider import AnthropicProvider
            return AnthropicProvider()

    if provider_name not in ("mock", "anthropic"):
        logger.warning("Unrecognized LLM_PROVIDER='%s'. Falling back to the mock provider.", provider_name)

    from src.talk_to_data.providers.mock_provider import MockProvider
    return MockProvider()
