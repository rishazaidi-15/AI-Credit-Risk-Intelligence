"""
Lightweight conversation memory for the Talk-to-Data chatbot.

Keeps only the last N turns so follow-up questions can use the context of
the most recent conversation without retaining unlimited chat history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.utils.config import settings


@dataclass
class ConversationMemory:
    max_turns: int = field(
        default_factory=lambda: settings.CONVERSATION_MEMORY_TURNS
    )
    _turns: list[dict] = field(default_factory=list)

    def add_turn(self, question: str, answer: str) -> None:
        self._turns.append(
            {
                "question": question,
                "answer": answer,
            }
        )

        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]

    def get_context_text(self) -> str:
        """Formats retained conversation turns as compact context."""
        if not self._turns:
            return ""

        lines = [
            f"Q: {turn['question']}\nA: {turn['answer']}"
            for turn in self._turns
        ]

        return "\n".join(lines)

    def get_last_question(self) -> str | None:
        """Returns the most recent user question, if available."""
        if not self._turns:
            return None

        return self._turns[-1]["question"]

    def clear(self) -> None:
        """Clears all stored conversation context."""
        self._turns = []

    def turns(self) -> list[dict]:
        """Returns a copy of the retained conversation turns."""
        return list(self._turns)