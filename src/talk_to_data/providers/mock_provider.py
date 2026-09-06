"""
Mock LLM provider: deterministic, keyword/regex-based NL-to-SQL for offline
development and testing (no API key, no cost, no network calls). Covers
counts (including distinct-category counts), numeric aggregates (max/min/avg),
overall and grouped default-rate rankings, ordinal ranking follow-ups,
top-N/bottom-N, and comparisons -- plus conversation-context-aware follow-up
resolution.

This is NOT a real LLM. It never guesses: any question it cannot map safely
to a supported query pattern returns the literal string "NO_VALID_QUERY".
When LLM_PROVIDER=anthropic, this provider is not used at all -- see
anthropic_provider.py for the real LLM-powered implementation.

CONDITION-ORDER NOTE (important -- read before modifying):
generate_sql() resolves intents in this strict priority order:
    1. Explicit comparison queries
    2. Ordinal follow-up ranking ("second highest", "3rd lowest", ...)
    3. Top-N / Bottom-N ranking ("top 5", "bottom 3", ...)
    4. Numeric aggregate operations (MAX / MIN / AVG) on a metric column
    5. Count questions (plain counts AND distinct-category counts)
    6. Grouped / overall default-rate ranking questions
    7. NO_VALID_QUERY fallback
Aggregate detection (step 4) MUST stay before count detection (step 5),
because phrases like "number of children" contain the words "number of"
(a count trigger) even when the actual question is "maximum number of
children" (an aggregate). Reordering these two steps reintroduces that bug.
"""

from __future__ import annotations

import ast
import re

from src.talk_to_data.llm_provider import LLMProvider
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MockProvider(LLMProvider):
    """Deterministic pattern-matcher covering the full set of supported query patterns."""

    # -------------------------------------------------------------------
    # Vocabulary tables
    # -------------------------------------------------------------------

    EMPLOYMENT_CASE_EXPR = "CASE WHEN DAYS_EMPLOYED = 365243 THEN 'Not employed' ELSE 'Employed' END"

    # (phrase, target_column) -- checked longest-phrase-first so specific
    # phrases ("income category") are matched before generic ones.
    GROUP_PHRASES: list[tuple[str, str]] = [
        ("income type", "NAME_INCOME_TYPE"),
        ("income types", "NAME_INCOME_TYPE"),
        ("income category", "NAME_INCOME_TYPE"),
        ("income categories", "NAME_INCOME_TYPE"),
        ("education group", "NAME_EDUCATION_TYPE"),
        ("education groups", "NAME_EDUCATION_TYPE"),
        ("education level", "NAME_EDUCATION_TYPE"),
        ("education", "NAME_EDUCATION_TYPE"),
        ("family status", "NAME_FAMILY_STATUS"),
        ("housing type", "NAME_HOUSING_TYPE"),
        ("housing types", "NAME_HOUSING_TYPE"),
        ("housing", "NAME_HOUSING_TYPE"),
        ("occupation group", "OCCUPATION_TYPE"),
        ("occupation groups", "OCCUPATION_TYPE"),
        ("occupations", "OCCUPATION_TYPE"),
        ("occupation", "OCCUPATION_TYPE"),
        ("organization type", "ORGANIZATION_TYPE"),
        ("organization types", "ORGANIZATION_TYPE"),
        ("organization", "ORGANIZATION_TYPE"),
        ("employment status", "__EMPLOYMENT__"),
        ("genders", "CODE_GENDER"),
        ("gender", "CODE_GENDER"),
    ]

    # (phrase, (column, label)) -- checked longest-phrase-first. "age" is
    # matched with a word-boundary check separately because it's a substring
    # of "average" and must not false-match.
    METRIC_PHRASES: list[tuple[str, tuple[str, str]]] = [
        ("number of children", ("CNT_CHILDREN", "children")),
        ("family size", ("CNT_FAM_MEMBERS", "family_size")),
        ("family members", ("CNT_FAM_MEMBERS", "family_size")),
        ("credit amount", ("AMT_CREDIT", "credit")),
        ("loan amount", ("AMT_CREDIT", "credit")),
        ("annuity", ("AMT_ANNUITY", "annuity")),
        ("income", ("AMT_INCOME_TOTAL", "income")),
        ("credit", ("AMT_CREDIT", "credit")),
        ("loan", ("AMT_CREDIT", "credit")),
        ("children", ("CNT_CHILDREN", "children")),
        ("age", ("__AGE__", "age_years")),
    ]

    MAX_WORDS = ["maximum", "max", "highest", "largest", "most"]
    MIN_WORDS = ["minimum", "min", "lowest", "smallest", "least"]
    AVG_WORDS = ["average", "avg", "mean"]

    ORDINAL_WORDS = {
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    }
    NUMBER_WORDS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }

    FRIENDLY_GROUP_LABELS = {
        "OCCUPATION_TYPE": "Occupation groups",
        "NAME_INCOME_TYPE": "Income categories",
        "NAME_EDUCATION_TYPE": "Education groups",
        "NAME_FAMILY_STATUS": "Family status groups",
        "NAME_HOUSING_TYPE": "Housing types",
        "CODE_GENDER": "Genders",
        "ORGANIZATION_TYPE": "Organization types",
        "employment_status": "Employment status groups",
    }

    # column -> (SQL alias for COUNT(DISTINCT ...), human-readable plural label)
    DISTINCT_COUNT_TARGETS = {
        "OCCUPATION_TYPE": ("distinct_occupation_groups", "occupation groups"),
        "NAME_INCOME_TYPE": ("distinct_income_categories", "income categories"),
        "NAME_EDUCATION_TYPE": ("distinct_education_groups", "education groups"),
        "NAME_HOUSING_TYPE": ("distinct_housing_types", "housing types"),
        "NAME_FAMILY_STATUS": ("distinct_family_statuses", "family status categories"),
        "CODE_GENDER": ("distinct_genders", "genders"),
        "ORGANIZATION_TYPE": ("distinct_organization_types", "organization types"),
    }
    # reverse lookup: alias -> human label, used during answer formatting
    DISTINCT_COUNT_LABELS = {alias: label for alias, label in DISTINCT_COUNT_TARGETS.values()}

    COUNT_KEY_PHRASES = {
        "applicants_with_children": "applicants with at least one child",
        "unemployed_applicants": "applicants who are not currently employed",
        "employed_applicants": "applicants who are currently employed",
        "female_applicants": "female applicants",
        "male_applicants": "male applicants",
        "applicants_owning_car": "applicants who own a car",
        "applicants_owning_realty": "applicants who own real estate",
        "total_applicants": "applicants in total",
    }

    METRIC_LABELS = {
        "income": "applicant income",
        "credit": "credit amount",
        "annuity": "loan annuity",
        "children": "number of children",
        "family_size": "family size",
        "age_years": "applicant age",
    }

    # Custom sentence templates per (metric_suffix, agg_prefix), overriding
    # the generic phrasing for a more natural result. Falls back to the
    # generic template in _format_scalar_answer when not listed here.
    METRIC_SENTENCE_TEMPLATES = {
        ("children", "avg"): "The average number of children per applicant is {value}.",
        ("children", "max"): "The maximum number of children recorded for an applicant is {value}.",
        ("children", "min"): "The minimum number of children recorded for an applicant is {value}.",
        ("income", "avg"): "The average applicant income in the dataset is {value}.",
        ("income", "max"): "The highest recorded applicant income is {value}.",
        ("income", "min"): "The lowest recorded applicant income is {value}.",
        ("credit", "avg"): "The average credit amount is {value}.",
        ("credit", "max"): "The highest recorded credit amount is {value}.",
        ("credit", "min"): "The lowest recorded credit amount is {value}.",
        ("annuity", "avg"): "The average loan annuity is {value}.",
        ("annuity", "max"): "The highest recorded loan annuity is {value}.",
        ("annuity", "min"): "The lowest recorded loan annuity is {value}.",
        ("family_size", "avg"): "The average family size is {value}.",
        ("family_size", "max"): "The largest family size recorded is {value}.",
        ("family_size", "min"): "The smallest family size recorded is {value}.",
        ("age_years", "avg"): "The average applicant age is {value}.",
        ("age_years", "max"): "The oldest applicant on record is {value}.",
        ("age_years", "min"): "The youngest applicant on record is {value}.",
    }

    def __init__(self) -> None:
        self._group_phrases_sorted = sorted(self.GROUP_PHRASES, key=lambda p: len(p[0]), reverse=True)
        self._metric_phrases_sorted = sorted(self.METRIC_PHRASES, key=lambda p: len(p[0]), reverse=True)

    # -------------------------------------------------------------------
    # Text normalization helpers
    # -------------------------------------------------------------------

    def _normalize(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _word_present(self, text: str, word: str) -> bool:
        return re.search(r"\b" + re.escape(word) + r"\b", text) is not None

    def _any_word_present(self, text: str, words: list[str]) -> bool:
        return any(self._word_present(text, w) for w in words)

    # -------------------------------------------------------------------
    # Entity / intent detectors
    # -------------------------------------------------------------------

    def _detect_group_column(self, q: str) -> str | None:
        for phrase, col in self._group_phrases_sorted:
            if phrase in q:
                return col
        return None

    def _detect_metric_column(self, q: str) -> tuple[str, str] | None:
        for phrase, metric in self._metric_phrases_sorted:
            if phrase == "age":
                if self._word_present(q, "age"):
                    return metric
            elif phrase in q:
                return metric
        return None

    def _detect_agg_func(self, q: str) -> str | None:
        if self._any_word_present(q, self.MAX_WORDS):
            return "MAX"
        if self._any_word_present(q, self.MIN_WORDS):
            return "MIN"
        if self._any_word_present(q, self.AVG_WORDS):
            return "AVG"
        return None

    def _detect_order_direction(self, q: str) -> str | None:
        agg = self._detect_agg_func(q)
        if agg == "MAX":
            return "DESC"
        if agg == "MIN":
            return "ASC"
        return None

    def _mentions_default_rate(self, q: str) -> bool:
        if "default rate" in q:
            return True
        if "payment difficult" in q:
            return True
        if ("percentage" in q or "percent" in q or "proportion" in q) and (
            "default" in q or "payment difficult" in q
        ):
            return True
        return False

    def _extract_ordinal_rank(self, q: str) -> tuple[int | None, str | None]:
        rank = None
        for word, value in self.ORDINAL_WORDS.items():
            if self._word_present(q, word):
                rank = value
                break
        if rank is None:
            m = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", q)
            if m:
                rank = int(m.group(1))
        if rank is None:
            return None, None
        direction = self._detect_order_direction(q)
        return rank, direction

    def _extract_top_n(self, q: str) -> tuple[int | None, str | None]:
        number_alt = "|".join(self.NUMBER_WORDS.keys())

        m = re.search(r"\btop\s+(\d+|" + number_alt + r")\b", q)
        if m:
            token = m.group(1)
            n = int(token) if token.isdigit() else self.NUMBER_WORDS[token]
            return n, "DESC"

        m = re.search(r"\bbottom\s+(\d+|" + number_alt + r")\b", q)
        if m:
            token = m.group(1)
            n = int(token) if token.isdigit() else self.NUMBER_WORDS[token]
            return n, "ASC"

        m = re.search(r"\bshow\s+(\d+)\b", q)
        if m and self._detect_group_column(q):
            n = int(m.group(1))
            direction = self._detect_order_direction(q) or "DESC"
            return n, direction

        return None, None

    # -------------------------------------------------------------------
    # Conversation-context (follow-up) resolution
    # -------------------------------------------------------------------

    def _get_previous_questions(self, conversation_context: str) -> list[str]:
        if not conversation_context or not isinstance(conversation_context, str):
            return []
        try:
            return re.findall(r"^Q:\s*(.+)$", conversation_context, flags=re.MULTILINE)
        except Exception:  # noqa: BLE001 -- malformed context must never crash SQL generation
            return []

    def _get_context_topic(self, conversation_context: str) -> tuple[str | None, object]:
        """
        Scans previous questions from most recent to oldest and returns the
        first topic found, tagged by type: ("group", column) or
        ("metric", (column, label)). Returns (None, None) if no topic can
        be resolved from the available context.
        """
        for prev_q in reversed(self._get_previous_questions(conversation_context)):
            norm = self._normalize(prev_q)
            group_col = self._detect_group_column(norm)
            if group_col:
                return "group", group_col
            if self._word_present(norm, "oldest") or self._word_present(norm, "youngest"):
                return "metric", ("__AGE__", "age_years")
            metric = self._detect_metric_column(norm)
            if metric:
                return "metric", metric
        return None, None

    # -------------------------------------------------------------------
    # SQL builders
    # -------------------------------------------------------------------

    def _build_ranking_sql(self, group_col: str, direction: str, limit: int | None = None, offset: int = 0) -> str:
        if group_col == "__EMPLOYMENT__":
            select_expr = f"{self.EMPLOYMENT_CASE_EXPR} AS employment_status"
            group_expr = self.EMPLOYMENT_CASE_EXPR
            where_clause = None
        else:
            select_expr = group_col
            group_expr = group_col
            where_clause = f"WHERE {group_col} IS NOT NULL" if group_col == "OCCUPATION_TYPE" else None

        parts = [f"SELECT {select_expr}, AVG(TARGET) AS default_rate FROM applicants"]
        if where_clause:
            parts.append(where_clause)
        parts.append(f"GROUP BY {group_expr}")
        parts.append(f"ORDER BY default_rate {direction}")
        if limit is not None:
            parts.append(f"LIMIT {limit}")
            if offset:
                parts.append(f"OFFSET {offset}")
        return " ".join(parts)

    def _build_aggregate_sql(self, metric: tuple[str, str], agg_func: str) -> str:
        col, label = metric
        if col == "__AGE__":
            expr = "-DAYS_BIRTH / 365.25"
            alias = f"{agg_func.lower()}_age_years"
            return f"SELECT {agg_func}({expr}) AS {alias} FROM applicants"
        alias = f"{agg_func.lower()}_{label}"
        return f"SELECT {agg_func}({col}) AS {alias} FROM applicants"

    def _build_distinct_count_sql(self, group_col: str) -> str | None:
        target = self.DISTINCT_COUNT_TARGETS.get(group_col)
        if not target:
            return None
        alias, _ = target
        return f"SELECT COUNT(DISTINCT {group_col}) AS {alias} FROM applicants"

    # -------------------------------------------------------------------
    # Comparison query builder
    # -------------------------------------------------------------------

    def _try_comparison(self, q: str) -> str | None:
        if ("employed" in q and "unemployed" in q) or "employment status" in q:
            return self._build_ranking_sql("__EMPLOYMENT__", "DESC")

        if self._word_present(q, "male") and "female" in q:
            return self._build_ranking_sql("CODE_GENDER", "DESC")

        if "compare" in q or self._word_present(q, "vs") or self._word_present(q, "versus"):
            group_col = self._detect_group_column(q)
            if group_col:
                direction = self._detect_order_direction(q) or "DESC"
                limit = 5 if self._detect_order_direction(q) is not None else None
                return self._build_ranking_sql(group_col, direction, limit=limit)

        return None

    # -------------------------------------------------------------------
    # Count query builder (plain counts + distinct-category counts)
    # -------------------------------------------------------------------

    def _try_count(self, q: str) -> str | None:
        has_count_intent = bool(re.search(r"\b(how many|count|total|number of)\b", q))
        if not has_count_intent:
            return None

        # Distinct-category counts, e.g. "how many occupation groups are there?"
        # Only treated as a distinct-category count when the question is NOT
        # actually asking about applicants (that phrasing is handled below).
        group_col = self._detect_group_column(q)
        if group_col and group_col in self.DISTINCT_COUNT_TARGETS and "applicant" not in q:
            distinct_sql = self._build_distinct_count_sql(group_col)
            if distinct_sql:
                return distinct_sql

        if "child" in q:
            return "SELECT COUNT(*) AS applicants_with_children FROM applicants WHERE CNT_CHILDREN > 0"
        if "unemployed" in q or "not employed" in q:
            return "SELECT COUNT(*) AS unemployed_applicants FROM applicants WHERE DAYS_EMPLOYED = 365243"
        if "employed" in q:
            return "SELECT COUNT(*) AS employed_applicants FROM applicants WHERE DAYS_EMPLOYED != 365243"
        if "female" in q:
            return "SELECT COUNT(*) AS female_applicants FROM applicants WHERE CODE_GENDER = 'F'"
        if self._word_present(q, "male"):
            return "SELECT COUNT(*) AS male_applicants FROM applicants WHERE CODE_GENDER = 'M'"
        if "own" in q and "car" in q:
            return "SELECT COUNT(*) AS applicants_owning_car FROM applicants WHERE FLAG_OWN_CAR = 'Y'"
        if "own" in q and ("real estate" in q or "realty" in q or "property" in q):
            return "SELECT COUNT(*) AS applicants_owning_realty FROM applicants WHERE FLAG_OWN_REALTY = 'Y'"
        if "applicant" in q:
            return "SELECT COUNT(*) AS total_applicants FROM applicants"
        return None

    # -------------------------------------------------------------------
    # generate_sql -- required interface method
    # -------------------------------------------------------------------

    def generate_sql(self, question: str, conversation_context: str = "") -> str:
        logger.info("MockProvider generating SQL for question: %s", question)
        q = self._normalize(question)

        if not q:
            logger.warning("MockProvider received an empty question.")
            return "NO_VALID_QUERY"

        # 1. Explicit comparison queries
        comparison_sql = self._try_comparison(q)
        if comparison_sql:
            return comparison_sql

        group_col = self._detect_group_column(q)
        is_rate_phrase = self._mentions_default_rate(q)

        if self._word_present(q, "oldest"):
            current_metric, current_agg = ("__AGE__", "age_years"), "MAX"
        elif self._word_present(q, "youngest"):
            current_metric, current_agg = ("__AGE__", "age_years"), "MIN"
        else:
            current_metric = self._detect_metric_column(q)
            current_agg = self._detect_agg_func(q)

        # 2. Ordinal ranking follow-ups ("second highest", "3rd lowest", ...)
        rank, rank_direction = self._extract_ordinal_rank(q)
        if rank is not None:
            effective_group = group_col
            if effective_group is None:
                topic_type, topic_val = self._get_context_topic(conversation_context)
                if topic_type == "group":
                    effective_group = topic_val
            if effective_group:
                direction = rank_direction or "DESC"
                return self._build_ranking_sql(effective_group, direction, limit=1, offset=rank - 1)
            logger.warning("MockProvider: ordinal rank found but no topic could be resolved for: %s", question)
            return "NO_VALID_QUERY"

        # 3. Top N / Bottom N ranking queries
        top_n, top_n_direction = self._extract_top_n(q)
        if top_n is not None:
            effective_group = group_col
            if effective_group is None:
                topic_type, topic_val = self._get_context_topic(conversation_context)
                if topic_type == "group":
                    effective_group = topic_val
            if effective_group:
                direction = top_n_direction or "DESC"
                return self._build_ranking_sql(effective_group, direction, limit=top_n, offset=0)
            logger.warning("MockProvider: top-N request found but no topic could be resolved for: %s", question)
            return "NO_VALID_QUERY"

        # 4. Numeric MAX/MIN/AVG aggregate queries -- MUST run before count
        #    detection (see module docstring for why this order matters).
        #    Skipped when the question is really about default-rate ranking
        #    (e.g. "highest default rate by income category") or explicitly
        #    names a group column (that belongs to the ranking branch below).
        if current_agg and current_metric and not is_rate_phrase and not group_col:
            return self._build_aggregate_sql(current_metric, current_agg)

        # Aggregate follow-up with no metric in THIS question -- resolve from context.
        if current_agg and not current_metric and not is_rate_phrase and not group_col:
            topic_type, topic_val = self._get_context_topic(conversation_context)
            if topic_type == "metric":
                return self._build_aggregate_sql(topic_val, current_agg)

        # 5. Count questions (plain counts and distinct-category counts)
        count_sql = self._try_count(q)
        if count_sql:
            return count_sql

        # 6. Grouped / overall default-rate ranking questions
        order_direction = self._detect_order_direction(q)
        if group_col and (is_rate_phrase or order_direction is not None):
            direction = order_direction or "DESC"
            return self._build_ranking_sql(group_col, direction, limit=5, offset=0)

        if group_col is None and order_direction is not None and current_metric is None:
            topic_type, topic_val = self._get_context_topic(conversation_context)
            if topic_type == "group":
                return self._build_ranking_sql(topic_val, order_direction, limit=5, offset=0)

        if is_rate_phrase and not group_col:
            return "SELECT AVG(TARGET) AS overall_default_rate FROM applicants"

        logger.warning("MockProvider could not match a supported pattern for: %s", question)
        return "NO_VALID_QUERY"

    # -------------------------------------------------------------------
    # Answer parsing helpers
    # -------------------------------------------------------------------

    def _sanitize_result_string(self, s: str) -> str:
        """Strips numpy scalar reprs (e.g. np.float64(0.13) -> 0.13) that pandas
        can produce, so ast.literal_eval can parse the result reliably."""
        s = re.sub(r"(?:np|numpy)\.(?:float64|float32|int64|int32)\(([^()]+)\)", r"\1", s)
        return s

    def _parse_result_summary(self, sql_result_summary: str) -> list | None:
        if sql_result_summary is None:
            return None
        if not isinstance(sql_result_summary, str):
            # Defensive: caller might pass an already-parsed list/dict.
            if isinstance(sql_result_summary, dict):
                return [sql_result_summary]
            if isinstance(sql_result_summary, list):
                return sql_result_summary
            return None

        s = sql_result_summary.strip()
        if not s:
            return None

        start = s.find("[")
        end = s.rfind("]")
        if start == -1 or end == -1 or end < start:
            # Might be a bare dict string like "{'avg_income': 168000.0}"
            start_d, end_d = s.find("{"), s.rfind("}")
            if start_d == -1 or end_d == -1 or end_d < start_d:
                return None
            dict_str = self._sanitize_result_string(s[start_d:end_d + 1])
            try:
                data = ast.literal_eval(dict_str)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
                logger.warning("MockProvider could not parse SQL result summary: %s", exc)
                return None
            return [data] if isinstance(data, dict) else None

        list_str = self._sanitize_result_string(s[start:end + 1])
        try:
            data = ast.literal_eval(list_str)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
            logger.warning("MockProvider could not parse SQL result summary: %s", exc)
            return None

        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        return None

    def _format_number(self, value: object, style: str) -> str:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return str(value)

        if style == "currency":
            return f"\u20b9{num:,.0f}"
        if style == "integer":
            return f"{num:,.0f}"
        if style == "decimal2":
            return f"{num:,.2f}"
        if style == "years":
            return f"{num:,.1f} years"
        if style == "percent":
            return f"{num * 100:.2f}%"
        return f"{num:,.2f}"

    def _ordinal_suffix(self, n: int) -> str:
        if 10 <= (n % 100) <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    def _format_scalar_answer(self, key: str, value: object) -> str:
        if key in self.DISTINCT_COUNT_LABELS:
            label = self.DISTINCT_COUNT_LABELS[key]
            return f"There are {self._format_number(value, 'integer')} distinct {label} in the dataset."

        if key in self.COUNT_KEY_PHRASES:
            return f"There are {self._format_number(value, 'integer')} {self.COUNT_KEY_PHRASES[key]}."

        if key == "overall_default_rate":
            return f"The overall observed default rate is {self._format_number(value, 'percent')}."

        m = re.match(r"^(avg|max|min)_(.+)$", key)
        if not m:
            return f"The result is {self._format_number(value, 'decimal2')}."

        prefix, suffix = m.group(1), m.group(2)
        label = self.METRIC_LABELS.get(suffix, suffix.replace("_", " "))

        if suffix in ("income", "credit", "annuity"):
            style = "currency"
        elif suffix == "age_years":
            style = "years"
        else:
            style = "decimal2" if prefix == "avg" else "integer"

        formatted = self._format_number(value, style)

        template = self.METRIC_SENTENCE_TEMPLATES.get((suffix, prefix))
        if template:
            return template.format(value=formatted)

        if prefix == "avg":
            return f"The average {label} is {formatted}."
        if prefix == "max":
            return f"The highest recorded {label} is {formatted}."
        return f"The lowest recorded {label} is {formatted}."

    def _format_single_group_rate_answer(self, question: str, row: dict) -> str:
        rate_key = "default_rate"
        category_key = next((k for k in row if k != rate_key), None)
        if category_key is None or rate_key not in row:
            return "The query completed successfully, but the result could not be formatted clearly."

        category_value = row[category_key]
        rate_pct = self._format_number(row[rate_key], "percent")

        norm_q = self._normalize(question)
        rank, direction = self._extract_ordinal_rank(norm_q)
        word = "lowest" if (direction or "DESC") == "ASC" else "highest"

        if rank is not None and rank > 1:
            ordinal = self._ordinal_suffix(rank)
            return f"{category_value} has the {ordinal} {word} observed default rate, at {rate_pct}."

        return f"{category_value} has the {word} observed default rate, at {rate_pct}."

    def _format_ranking_table_answer(self, data: list[dict], rate_key: str) -> str:
        category_key = next((k for k in data[0] if k != rate_key), None)
        if category_key is None:
            return self._format_generic_answer(data)

        label = self.FRIENDLY_GROUP_LABELS.get(category_key, category_key.replace("_", " ").title())
        lines = [f"### {label} by observed default rate", ""]
        for i, row in enumerate(data, start=1):
            cat_val = row.get(category_key, "Unknown")
            rate_pct = self._format_number(row.get(rate_key), "percent")
            lines.append(f"{i}. **{cat_val}** \u2014 {rate_pct}")

        try:
            ascending = float(data[0][rate_key]) <= float(data[-1][rate_key])
        except (TypeError, ValueError, KeyError):
            ascending = False
        conclusion_word = "lowest" if ascending and len(data) > 1 else "highest"

        top_val = data[0].get(category_key)
        top_rate = self._format_number(data[0].get(rate_key), "percent")
        lines.append("")
        lines.append(
            f"The {conclusion_word} observed default rate in this ranking is for "
            f"**{top_val}**, at **{top_rate}**."
        )
        return "\n".join(lines)

    def _format_generic_answer(self, data: list) -> str:
        try:
            first = data[0]
            if isinstance(first, dict):
                parts = [f"{k}: {v}" for k, v in first.items()]
                return "Here's what the data shows: " + ", ".join(parts) + "."
            return f"Here's what the data shows: {data}."
        except Exception:  # noqa: BLE001 -- must never crash answer generation
            return "The query completed successfully, but the result could not be formatted clearly."

    # -------------------------------------------------------------------
    # generate_answer -- required interface method
    # -------------------------------------------------------------------

    def generate_answer(self, question: str, sql_result_summary: str) -> str:
        logger.info("MockProvider generating answer for question: %s", question)

        data = self._parse_result_summary(sql_result_summary)
        if data is None:
            return "The query completed successfully, but the result could not be formatted clearly."

        if len(data) == 0:
            return "No matching records were found in the dataset for that question."

        first_row = data[0]
        if not isinstance(first_row, dict) or len(first_row) == 0:
            return "The query completed successfully, but the result could not be formatted clearly."

        keys = list(first_row.keys())

        if len(data) == 1 and len(keys) == 1:
            return self._format_scalar_answer(keys[0], first_row[keys[0]])

        rate_key = "default_rate" if "default_rate" in keys else None
        if rate_key and len(keys) == 2:
            if len(data) == 1:
                return self._format_single_group_rate_answer(question, first_row)
            return self._format_ranking_table_answer(data, rate_key)

        return self._format_generic_answer(data)d