"""
Deterministic NL-to-SQL provider for the Talk-to-Data assistant.

Supports common questions about the applicant dataset, including:
- Aggregations
- Ranking
- Employment comparison
- Income and occupation analysis
- Children-related statistics
- Context-aware follow-up questions

This provider runs locally and does not require an API key.
"""

from __future__ import annotations

import ast
import re

from src.talk_to_data.llm_provider import LLMProvider
from src.utils.logger import get_logger


logger = get_logger(__name__)


class MockProvider(LLMProvider):
    """Deterministic provider for supported applicant-data questions."""

    # ==================================================
    # HELPERS
    # ==================================================

    @staticmethod
    def _get_last_turn(conversation_context: str) -> tuple[str, str]:
        """
        Extract the most recent question and answer from conversation memory.
        """
        if not conversation_context:
            return "", ""

        matches = re.findall(
            r"Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|\Z)",
            conversation_context,
            flags=re.DOTALL,
        )

        if not matches:
            return "", ""

        question, answer = matches[-1]

        return question.lower().strip(), answer.lower().strip()

    @staticmethod
    def _get_last_turn_text(conversation_context: str) -> str:
        """
        Return the complete text of the most recent conversation turn.
        """
        last_question, last_answer = MockProvider._get_last_turn(
            conversation_context
        )

        return f"{last_question} {last_answer}".lower().strip()

    @staticmethod
    def _get_rank_position(question: str) -> int | None:
        """
        Detect an explicitly requested ranking position.

        Examples:
        - second highest -> 2
        - 3rd highest -> 3
        - fourth highest -> 4
        """
        q = question.lower().strip()

        position_map = {
            "first": 1,
            "1st": 1,
            "second": 2,
            "2nd": 2,
            "third": 3,
            "3rd": 3,
            "fourth": 4,
            "4th": 4,
            "fifth": 5,
            "5th": 5,
        }

        for word, position in position_map.items():
            patterns = [
                f"{word} highest",
                f"{word} lowest",
                f"{word} place",
                f"{word} rank",
                f"about the {word}",
                f"and the {word}",
            ]

            if any(pattern in q for pattern in patterns):
                return position

        return None

    @staticmethod
    def _is_ranking_follow_up(
        question: str,
        conversation_context: str,
    ) -> bool:
        """
        Detect ranking follow-up questions.
        """
        if not conversation_context:
            return False

        q = question.lower().strip()

        ranking_terms = [
            "second highest",
            "2nd highest",
            "third highest",
            "3rd highest",
            "fourth highest",
            "4th highest",
            "fifth highest",
            "5th highest",
            "second lowest",
            "2nd lowest",
            "third lowest",
            "3rd lowest",
            "what about the second",
            "what about the third",
            "what about the fourth",
            "what about the fifth",
            "and the second",
            "and the third",
            "and the fourth",
            "and the fifth",
        ]

        return any(term in q for term in ranking_terms)

    @staticmethod
    def _get_ranking_topic(conversation_context: str) -> str | None:
        """
        Determine whether the most recent ranking was about income
        or occupation.
        """
        last_turn = MockProvider._get_last_turn_text(
            conversation_context
        )

        if (
            "occupation groups by observed default rate" in last_turn
            or "occupation groups with the highest observed default rates"
            in last_turn
        ):
            return "occupation"

        if (
            "income categories by observed default rate" in last_turn
            or "income categories with the highest observed default rates"
            in last_turn
        ):
            return "income"

        return None

    @staticmethod
    def _is_children_context(conversation_context: str) -> bool:
        """
        Check whether the most recent conversation turn was about children.
        """
        last_turn = MockProvider._get_last_turn_text(
            conversation_context
        )

        return (
            "children" in last_turn
            or "child" in last_turn
            or "cnt_children" in last_turn
        )

    @staticmethod
    def _parse_results(sql_result_summary: str) -> list[dict]:
        """
        Safely convert SQL results into a list of dictionaries.
        """
        if not sql_result_summary:
            return []

        try:
            parsed = ast.literal_eval(sql_result_summary)

            if isinstance(parsed, list):
                return [
                    row
                    for row in parsed
                    if isinstance(row, dict)
                ]

            if isinstance(parsed, dict):
                return [parsed]

        except (ValueError, SyntaxError):
            logger.warning(
                "Could not parse SQL result summary: %s",
                sql_result_summary,
            )

        return []

    @staticmethod
    def _get_answer_rank_position(question: str) -> int | None:
        """
        Determine which ranking position should be mentioned in the answer.
        """
        q = question.lower().strip()

        position_map = {
            "first": 1,
            "1st": 1,
            "second": 2,
            "2nd": 2,
            "third": 3,
            "3rd": 3,
            "fourth": 4,
            "4th": 4,
            "fifth": 5,
            "5th": 5,
        }

        for word, position in position_map.items():
            if (
                f"{word} highest" in q
                or f"{word} lowest" in q
                or f"about the {word}" in q
                or f"and the {word}" in q
            ):
                return position

        return None

    @staticmethod
    def _ordinal(position: int) -> str:
        """
        Convert a number into its ordinal form.
        """
        ordinal_map = {
            1: "1st",
            2: "2nd",
            3: "3rd",
            4: "4th",
            5: "5th",
        }

        return ordinal_map.get(position, f"{position}th")

    # ==================================================
    # SQL GENERATION
    # ==================================================

    def generate_sql(
        self,
        question: str,
        conversation_context: str = "",
    ) -> str:
        q = question.lower().strip()
        context = conversation_context.lower().strip()

        logger.info(
            "MockProvider generating SQL for question: %s",
            question,
        )

        # ==================================================
        # 1. RANKING FOLLOW-UPS
        # Must come before normal ranking patterns.
        # ==================================================

        if self._is_ranking_follow_up(question, conversation_context):
            position = self._get_rank_position(question)
            topic = self._get_ranking_topic(conversation_context)

            if position is not None and topic is not None:
                offset = position - 1

                # ------------------------------------------
                # Occupation ranking follow-up
                # ------------------------------------------
                if topic == "occupation":
                    return (
                        "SELECT OCCUPATION_TYPE, "
                        "AVG(TARGET) AS default_rate "
                        "FROM applicants "
                        "WHERE OCCUPATION_TYPE IS NOT NULL "
                        "GROUP BY OCCUPATION_TYPE "
                        "ORDER BY default_rate DESC "
                        f"LIMIT 1 OFFSET {offset}"
                    )

                # ------------------------------------------
                # Income ranking follow-up
                # ------------------------------------------
                if topic == "income":
                    return (
                        "SELECT NAME_INCOME_TYPE, "
                        "AVG(TARGET) AS default_rate "
                        "FROM applicants "
                        "GROUP BY NAME_INCOME_TYPE "
                        "ORDER BY default_rate DESC "
                        f"LIMIT 1 OFFSET {offset}"
                    )

        # ==================================================
        # 2. CHILDREN FOLLOW-UPS
        #
        # Examples after asking about children:
        # - What is the maximum?
        # - What about the minimum?
        # - And the average?
        # ==================================================

        if conversation_context and self._is_children_context(
            conversation_context
        ):
            # Maximum follow-up
            if (
                q in {"maximum", "max", "maximum number", "the maximum"}
                or "maximum" in q
                or "max" in q
                or "highest number" in q
            ):
                return (
                    "SELECT MAX(CNT_CHILDREN) AS max_children "
                    "FROM applicants"
                )

            # Minimum follow-up
            if (
                q in {"minimum", "min", "minimum number", "the minimum"}
                or "minimum" in q
                or "min" in q
                or "lowest number" in q
            ):
                return (
                    "SELECT MIN(CNT_CHILDREN) AS min_children "
                    "FROM applicants"
                )

            # Average follow-up
            if (
                q in {"average", "the average", "average number"}
                or "average" in q
                or "mean" in q
            ):
                return (
                    "SELECT AVG(CNT_CHILDREN) AS avg_children "
                    "FROM applicants"
                )

        # ==================================================
        # 3. EMPLOYMENT COMPARISON
        # ==================================================

        if "employ" in q and (
            "compare" in q
            or "comparison" in q
            or " vs " in q
            or "versus" in q
            or "difference" in q
        ):
            return (
                "SELECT "
                "CASE "
                "WHEN DAYS_EMPLOYED = 365243 THEN 'Not employed' "
                "ELSE 'Employed' "
                "END AS employment_status, "
                "AVG(TARGET) AS default_rate "
                "FROM applicants "
                "GROUP BY employment_status"
            )

        # ==================================================
        # 4. MAXIMUM NUMBER OF CHILDREN
        # ==================================================

        if (
            "maximum number of children" in q
            or "max number of children" in q
            or "most children" in q
            or "maximum children" in q
            or "highest number of children" in q
            or "max children" in q
        ):
            return (
                "SELECT MAX(CNT_CHILDREN) AS max_children "
                "FROM applicants"
            )

        # ==================================================
        # 5. MINIMUM NUMBER OF CHILDREN
        # ==================================================

        if (
            "minimum number of children" in q
            or "min number of children" in q
            or "least children" in q
            or "minimum children" in q
            or "lowest number of children" in q
            or "min children" in q
        ):
            return (
                "SELECT MIN(CNT_CHILDREN) AS min_children "
                "FROM applicants"
            )

        # ==================================================
        # 6. AVERAGE NUMBER OF CHILDREN
        # ==================================================

        if (
            "average number of children" in q
            or "average children" in q
            or "average number of child" in q
            or "mean number of children" in q
        ):
            return (
                "SELECT AVG(CNT_CHILDREN) AS avg_children "
                "FROM applicants"
            )

        # ==================================================
        # 7. COUNT APPLICANTS WITH CHILDREN
        # ==================================================

        if (
            "children" in q
            and (
                "how many" in q
                or "count" in q
                or "number of applicants" in q
            )
            and "maximum" not in q
            and "minimum" not in q
            and "average" not in q
            and "max" not in q
            and "min" not in q
            and "mean" not in q
        ):
            return (
                "SELECT COUNT(*) AS applicants_with_children "
                "FROM applicants "
                "WHERE CNT_CHILDREN > 0"
            )

        # ==================================================
        # 8. NUMBER OF OCCUPATION GROUPS
        # ==================================================

        if (
            "how many occupation groups" in q
            or "number of occupation groups" in q
            or "count occupation groups" in q
            or "how many occupations" in q
            or "number of occupations" in q
        ):
            return (
                "SELECT COUNT(DISTINCT OCCUPATION_TYPE) "
                "AS occupation_group_count "
                "FROM applicants "
                "WHERE OCCUPATION_TYPE IS NOT NULL"
            )

        # ==================================================
        # 9. INCOME CATEGORY RANKING
        # ==================================================

        if (
            "income category" in q
            or "income categories" in q
        ) and (
            "default rate" in q
            or "highest" in q
            or "top" in q
            or "rank" in q
        ):
            return (
                "SELECT NAME_INCOME_TYPE, "
                "AVG(TARGET) AS default_rate "
                "FROM applicants "
                "GROUP BY NAME_INCOME_TYPE "
                "ORDER BY default_rate DESC "
                "LIMIT 5"
            )

        # ==================================================
        # 10. OCCUPATION RANKING
        # ==================================================

        if "occupation" in q and (
            "highest" in q
            or "top" in q
            or "rank" in q
            or "default rate" in q
        ):
            return (
                "SELECT OCCUPATION_TYPE, "
                "AVG(TARGET) AS default_rate "
                "FROM applicants "
                "WHERE OCCUPATION_TYPE IS NOT NULL "
                "GROUP BY OCCUPATION_TYPE "
                "ORDER BY default_rate DESC "
                "LIMIT 5"
            )

        # ==================================================
        # 11. AVERAGE INCOME
        # ==================================================

        if "average" in q and "income" in q:
            return (
                "SELECT AVG(AMT_INCOME_TOTAL) AS avg_income "
                "FROM applicants"
            )

        # ==================================================
        # 12. AVERAGE CREDIT
        # ==================================================

        if "average" in q and "credit" in q:
            return (
                "SELECT AVG(AMT_CREDIT) AS avg_credit "
                "FROM applicants"
            )

        # ==================================================
        # 13. OVERALL DEFAULT RATE
        # ==================================================

        if (
            "overall default rate" in q
            or (
                "default rate" in q
                and "overall" in q
            )
        ):
            return (
                "SELECT AVG(TARGET) AS overall_default_rate "
                "FROM applicants"
            )

        # ==================================================
        # 14. TOTAL APPLICANTS
        # ==================================================

        if (
            "how many applicants" in q
            or "total applicants" in q
            or "number of applicants" in q
        ):
            return (
                "SELECT COUNT(*) AS total_applicants "
                "FROM applicants"
            )

        logger.warning(
            "MockProvider could not match a known pattern for: %s",
            question,
        )

        return "NO_VALID_QUERY"

    # ==================================================
    # ANSWER GENERATION
    # ==================================================

    def generate_answer(
        self,
        question: str,
        sql_result_summary: str,
    ) -> str:
        q = question.lower().strip()
        results = self._parse_results(sql_result_summary)

        if not results:
            return (
                "The available data doesn't contain enough information "
                "to answer that question."
            )

        first_result = results[0]

        # ==================================================
        # 1. MAXIMUM NUMBER OF CHILDREN
        # Uses SQL alias, so follow-ups also work.
        # ==================================================

        if "max_children" in first_result:
            value = first_result.get("max_children")

            if value is not None:
                return (
                    "The maximum number of children recorded for an "
                    f"applicant is **{int(value)}**."
                )

        # ==================================================
        # 2. MINIMUM NUMBER OF CHILDREN
        # ==================================================

        if "min_children" in first_result:
            value = first_result.get("min_children")

            if value is not None:
                return (
                    "The minimum number of children recorded for an "
                    f"applicant is **{int(value)}**."
                )

        # ==================================================
        # 3. AVERAGE NUMBER OF CHILDREN
        # ==================================================

        if "avg_children" in first_result:
            value = first_result.get("avg_children")

            if value is not None:
                return (
                    "The average number of children per applicant is "
                    f"**{float(value):.2f}**."
                )

        # ==================================================
        # 4. APPLICANTS WITH CHILDREN
        # ==================================================

        if "applicants_with_children" in first_result:
            value = first_result.get("applicants_with_children")

            if value is not None:
                return (
                    f"**{int(value):,} applicants** in the current dataset "
                    "have at least one child."
                )

        # ==================================================
        # 5. AVERAGE INCOME
        # ==================================================

        if "avg_income" in first_result:
            value = first_result.get("avg_income")

            if value is not None:
                return (
                    "The average applicant income is "
                    f"**{float(value):,.2f}**."
                )

        # ==================================================
        # 6. AVERAGE CREDIT
        # ==================================================

        if "avg_credit" in first_result:
            value = first_result.get("avg_credit")

            if value is not None:
                return (
                    "The average credit amount is "
                    f"**{float(value):,.2f}**."
                )

        # ==================================================
        # 7. NUMBER OF OCCUPATION GROUPS
        # ==================================================

        if "occupation_group_count" in first_result:
            value = first_result.get("occupation_group_count")

            if value is not None:
                return (
                    f"There are **{int(value)} distinct occupation groups** "
                    "among applicants with recorded occupation information."
                )

        # ==================================================
        # 8. TOTAL APPLICANTS
        # ==================================================

        if "total_applicants" in first_result:
            value = first_result.get("total_applicants")

            if value is not None:
                return (
                    f"The dataset contains **{int(value):,} applicants**."
                )

        # ==================================================
        # 9. OVERALL DEFAULT RATE
        # ==================================================

        if "overall_default_rate" in first_result:
            value = first_result.get("overall_default_rate")

            if value is not None:
                return (
                    "The overall observed default rate is "
                    f"**{float(value) * 100:.2f}%**."
                )

        # ==================================================
        # 10. EMPLOYMENT COMPARISON
        # ==================================================

        if "employment_status" in first_result:
            lines = []

            employed_rate = None
            not_employed_rate = None

            for row in results:
                status = row.get("employment_status")
                rate = row.get("default_rate")

                if status is not None and rate is not None:
                    lines.append(
                        f"- **{status}:** "
                        f"{float(rate) * 100:.2f}%"
                    )

                    if status == "Employed":
                        employed_rate = float(rate)
                    elif status == "Not employed":
                        not_employed_rate = float(rate)

            if lines:
                answer = "### Observed default-rate comparison\n\n"
                answer += "\n".join(lines)

                if (
                    employed_rate is not None
                    and not_employed_rate is not None
                ):
                    difference = abs(
                        employed_rate - not_employed_rate
                    ) * 100

                    if employed_rate > not_employed_rate:
                        higher_group = "Employed"
                        lower_group = "Not employed"
                    else:
                        higher_group = "Not employed"
                        lower_group = "Employed"

                    answer += (
                        f"\n\n**{higher_group} applicants** have the "
                        "higher observed default rate, by approximately "
                        f"**{difference:.2f} percentage points** compared "
                        f"with **{lower_group} applicants**."
                    )

                return answer

        # ==================================================
        # 11. INCOME CATEGORY SINGLE-RANK FOLLOW-UP
        # ==================================================

        if (
            "NAME_INCOME_TYPE" in first_result
            and len(results) == 1
        ):
            income_type = first_result.get("NAME_INCOME_TYPE")
            rate = first_result.get("default_rate")

            if income_type is not None and rate is not None:
                position = self._get_answer_rank_position(question)

                if position is not None:
                    ordinal = self._ordinal(position)

                    return (
                        f"**{income_type}** has the **{ordinal} highest "
                        "observed default rate**, at "
                        f"**{float(rate) * 100:.2f}%**."
                    )

        # ==================================================
        # 12. OCCUPATION SINGLE-RANK FOLLOW-UP
        # ==================================================

        if (
            "OCCUPATION_TYPE" in first_result
            and len(results) == 1
        ):
            occupation = first_result.get("OCCUPATION_TYPE")
            rate = first_result.get("default_rate")

            if occupation is not None and rate is not None:
                position = self._get_answer_rank_position(question)

                if position is not None:
                    ordinal = self._ordinal(position)

                    return (
                        f"**{occupation}** has the **{ordinal} highest "
                        "observed default rate**, at "
                        f"**{float(rate) * 100:.2f}%**."
                    )

        # ==================================================
        # 13. INCOME CATEGORY RANKING
        # ==================================================

        if "NAME_INCOME_TYPE" in first_result:
            lines = []

            for index, row in enumerate(results, start=1):
                income_type = row.get("NAME_INCOME_TYPE")
                rate = row.get("default_rate")

                if income_type is not None and rate is not None:
                    lines.append(
                        f"{index}. **{income_type}** — "
                        f"{float(rate) * 100:.2f}%"
                    )

            if lines:
                answer = (
                    "### Income categories by observed default rate\n\n"
                )
                answer += "\n".join(lines)

                top_category = first_result.get("NAME_INCOME_TYPE")
                top_rate = first_result.get("default_rate")

                if (
                    top_category is not None
                    and top_rate is not None
                ):
                    answer += (
                        "\n\nThe highest observed default rate is for "
                        f"**{top_category}**, at "
                        f"**{float(top_rate) * 100:.2f}%**."
                    )

                return answer

        # ==================================================
        # 14. OCCUPATION RANKING
        # ==================================================

        if "OCCUPATION_TYPE" in first_result:
            lines = []

            for index, row in enumerate(results, start=1):
                occupation = row.get("OCCUPATION_TYPE")
                rate = row.get("default_rate")

                if occupation is not None and rate is not None:
                    lines.append(
                        f"{index}. **{occupation}** — "
                        f"{float(rate) * 100:.2f}%"
                    )

            if lines:
                answer = (
                    "### Occupation groups by observed default rate\n\n"
                )
                answer += "\n".join(lines)

                top_occupation = first_result.get("OCCUPATION_TYPE")
                top_rate = first_result.get("default_rate")

                if (
                    top_occupation is not None
                    and top_rate is not None
                ):
                    answer += (
                        "\n\nThe highest observed default rate is for "
                        f"**{top_occupation}**, at "
                        f"**{float(top_rate) * 100:.2f}%**."
                    )

                return answer

        # ==================================================
        # 15. SAFE FALLBACK
        # ==================================================

        return (
            "The query was completed successfully, but I couldn't format "
            "this result into a more specific response."
        )