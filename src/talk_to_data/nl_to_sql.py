"""
Top-level orchestration for the Talk-to-Data chatbot:

User Question -> SQL Generation -> SQL Validator -> SQL Execution
-> Business-Readable Answer -> Response to User
"""

from __future__ import annotations

from dataclasses import dataclass

from src.database.database import DatabaseError
from src.talk_to_data.conversation_memory import ConversationMemory
from src.talk_to_data.llm_provider import LLMProviderError, get_llm_provider
from src.talk_to_data.query_runner import (
    execute_validated_sql,
    summarize_result_for_llm,
)
from src.talk_to_data.sql_validator import (
    SQLValidationError,
    validate_sql,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChatResponse:
    answer: str
    generated_sql: str | None
    validation_status: str
    result_preview: str | None = None


def _expand_follow_up_question(
    question: str,
    memory: ConversationMemory | None,
) -> str:
    """
    Converts supported contextual follow-up questions into standalone
    questions using the previous user question.

    Example:
        Previous: Which occupation groups have the highest default rate?
        Current: What about the second highest?

        Expanded:
        Which occupation group has the second highest default rate?
    """

    if memory is None:
        return question

    previous_question = memory.get_last_question()

    if not previous_question:
        return question

    q = question.lower().strip()
    previous = previous_question.lower().strip()

    follow_up_markers = (
        "second highest",
        "2nd highest",
        "second one",
        "what about the second",
    )

    is_second_highest_follow_up = any(
        marker in q for marker in follow_up_markers
    )

    if not is_second_highest_follow_up:
        return question

    # Follow-up to occupation ranking
    if "occupation" in previous:
        return (
            "Which occupation group has the second highest default rate?"
        )

    # Follow-up to income category ranking
    if "income" in previous and (
        "default rate" in previous
        or "category" in previous
    ):
        return (
            "Which income category has the second highest default rate?"
        )

    # If we cannot confidently resolve the context,
    # preserve the user's original question.
    return question


def ask_question(
    question: str,
    memory: ConversationMemory | None = None,
) -> ChatResponse:
    """
    Runs the full Talk-to-Data flow for one user question.

    Supported flow:

    User Question
        -> Context Resolution
        -> SQL Generation
        -> SQL Validation
        -> SQL Execution
        -> Answer Generation
        -> Response
    """

    provider = get_llm_provider()

    context = memory.get_context_text() if memory else ""

    # Resolve contextual follow-up into a standalone question
    effective_question = _expand_follow_up_question(question, memory)

    logger.info(
        "Original question: %s | Effective question: %s",
        question,
        effective_question,
    )

    # Step 1: Generate SQL
    try:
        generated_sql = provider.generate_sql(
            effective_question,
            conversation_context=context,
        )

    except LLMProviderError as exc:
        logger.error("LLM SQL generation failed: %s", exc)

        return ChatResponse(
            answer=str(exc),
            generated_sql=None,
            validation_status="error",
        )

    # No supported query generated
    if (
        not generated_sql
        or generated_sql.strip().upper() == "NO_VALID_QUERY"
    ):
        answer = (
            "I couldn't answer that using the available applicant data. "
            "You can ask about income, credit amounts, family situation, "
            "employment, occupation groups, or observed default rates."
        )

        return ChatResponse(
            answer=answer,
            generated_sql=None,
            validation_status="no_query",
        )

    # Step 2: Validate SQL
    try:
        validated_sql = validate_sql(generated_sql)

    except SQLValidationError as exc:
        logger.warning(
            "Generated SQL rejected by validator: %s | SQL: %s",
            exc,
            generated_sql,
        )

        return ChatResponse(
            answer=(
                "I couldn't safely process that question. "
                "Please try rephrasing it."
            ),
            generated_sql=generated_sql,
            validation_status="rejected",
        )

    # Step 3: Execute SQL
    try:
        result_df = execute_validated_sql(validated_sql)

    except DatabaseError as exc:
        logger.error("Query execution failed: %s", exc)

        return ChatResponse(
            answer=(
                "I couldn't retrieve the requested data at the moment. "
                "Please try again."
            ),
            generated_sql=validated_sql,
            validation_status="error",
        )

    result_summary = summarize_result_for_llm(result_df)

    # Step 4: Generate business-readable answer
    try:
        answer = provider.generate_answer(
            effective_question,
            result_summary,
        )

    except LLMProviderError as exc:
        logger.error("LLM answer generation failed: %s", exc)

        answer = (
            "The data was retrieved successfully, but I couldn't format "
            "the answer."
        )

    # Store the ORIGINAL question in memory, not the expanded version.
    if memory is not None:
        memory.add_turn(question, answer)

    return ChatResponse(
        answer=answer,
        generated_sql=validated_sql,
        validation_status="valid",
        result_preview=result_summary,
    )