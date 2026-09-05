"""
Anthropic-powered LLM provider: the real NL-to-SQL implementation, used when
LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY is configured. All API errors
are caught and re-raised as LLMProviderError with a clean, user-safe
message -- the raw API exception is logged internally only.
"""

from __future__ import annotations

import anthropic

from src.talk_to_data.llm_provider import LLMProvider, LLMProviderError
from src.talk_to_data.prompt_templates import build_answer_generation_prompt, build_sql_generation_prompt
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnthropicProvider(LLMProvider):
    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    def _call(self, system_prompt: str, user_message: str) -> str:
        try:
            response = self._client.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=settings.ANTHROPIC_MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            text_blocks = [block.text for block in response.content if block.type == "text"]
            return "".join(text_blocks).strip()
        except anthropic.APIError as exc:
            logger.error("Anthropic API error: %s", exc)
            raise LLMProviderError(
                "The AI service is temporarily unavailable. Please try again later."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error calling Anthropic API: %s", exc)
            raise LLMProviderError(
                "The AI service encountered an unexpected error. Please try again later."
            ) from exc

    def generate_sql(self, question: str, conversation_context: str = "") -> str:
        system_prompt = build_sql_generation_prompt(conversation_context)
        raw_response = self._call(system_prompt, question)
        # Strip accidental markdown code fences, defensively.
        cleaned = raw_response.strip().strip("`")
        if cleaned.lower().startswith("sql"):
            cleaned = cleaned[3:].strip()
        return cleaned

    def generate_answer(self, question: str, sql_result_summary: str) -> str:
        prompt = build_answer_generation_prompt(question, sql_result_summary)
        return self._call("You are a precise, factual credit risk analytics assistant.", prompt)
