"""Explainability page: visual dashboard for global and local SHAP explanations."""

from __future__ import annotations

import html

import streamlit as st

from app.components.ui_components import error_banner, info_banner, risk_badge
from src.data.feature_engineering import get_feature_lists
from src.data.loader import DatasetNotFoundError, DatasetValidationError, load_raw_data
from src.explainability.shap_explainer import (
    ExplanationUnavailableError,
    compute_full_global_importance,
    compute_numeric_context,
    format_feature_value,
    get_engineered_background,
    local_explanation,
    prediction_confidence,
)
from src.ml.predict import is_model_available


# -------------------------------------------------------------------
# VISUAL CONSTANTS
# -------------------------------------------------------------------

_RISK_INCREASE_COLOR = "#EF5350"
_RISK_INCREASE_DARK = "#B71C1C"

_RISK_DECREASE_COLOR = "#2ECC71"
_RISK_DECREASE_DARK = "#176B3A"

_GLOBAL_PRIMARY = "#4F8EF7"
_GLOBAL_SECONDARY = "#7C5CFC"

_CARD_BG = "rgba(255,255,255,0.025)"
_CARD_BORDER = "rgba(255,255,255,0.10)"
_TRACK_BG = "rgba(255,255,255,0.07)"
_TEXT_MUTED = "rgba(255,255,255,0.62)"


# -------------------------------------------------------------------
# DATA
# -------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset for global explanation...")
def _load_data_cached():
    return load_raw_data()


# -------------------------------------------------------------------
# PAGE STYLING
# -------------------------------------------------------------------

def _inject_explainability_styles() -> None:
    """Inject page-specific CSS for a more polished visual dashboard."""

    st.markdown(
        """
        <style>

        /* Section headings */
        .xai-section-title {
            font-size: 1.65rem;
            font-weight: 700;
            margin-top: 0.4rem;
            margin-bottom: 0.15rem;
            letter-spacing: -0.02em;
        }

        .xai-section-subtitle {
            color: rgba(255,255,255,0.62);
            font-size: 0.92rem;
            margin-bottom: 1.2rem;
        }

        /* General chart card */
        .xai-chart-card {
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 14px;
            padding: 20px 20px 16px 20px;
            margin-top: 8px;
            margin-bottom: 10px;
        }

        /* Global chart header */
        .global-chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(255,255,255,0.07);
        }

        .global-chart-title {
            font-size: 0.95rem;
            font-weight: 650;
        }

        .global-chart-note {
            font-size: 0.75rem;
            color: rgba(255,255,255,0.50);
        }

        /* Global rows */
        .global-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }

        .rank-pill {
            width: 27px;
            height: 27px;
            min-width: 27px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.73rem;
            font-weight: 700;
            background: rgba(79,142,247,0.13);
            color: #8BB8FF;
            border: 1px solid rgba(79,142,247,0.18);
        }

        .global-feature-name {
            width: 230px;
            min-width: 230px;
            font-size: 0.84rem;
            line-height: 1.25;
            color: rgba(255,255,255,0.88);
            overflow-wrap: anywhere;
        }

        .global-bar-track {
            flex-grow: 1;
            height: 25px;
            background: rgba(255,255,255,0.055);
            border-radius: 8px;
            overflow: hidden;
            position: relative;
            border: 1px solid rgba(255,255,255,0.035);
        }

        .global-bar-fill {
            height: 100%;
            border-radius: 7px;
            position: relative;
            min-width: 4px;
            box-shadow: 0 2px 10px rgba(79,142,247,0.16);
        }

        .global-value {
            width: 72px;
            min-width: 72px;
            text-align: right;
            font-size: 0.78rem;
            font-weight: 650;
            color: rgba(255,255,255,0.82);
        }

        .global-percent {
            width: 52px;
            min-width: 52px;
            text-align: right;
            font-size: 0.72rem;
            color: rgba(255,255,255,0.45);
        }

        /* Local chart legend */
        .local-legend {
            display: flex;
            justify-content: center;
            gap: 24px;
            align-items: center;
            margin-bottom: 18px;
            font-size: 0.78rem;
            color: rgba(255,255,255,0.68);
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 7px;
        }

        .legend-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
        }

        /* Local rows */
        .local-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 13px;
        }

        .local-label {
            width: 235px;
            min-width: 235px;
            overflow-wrap: anywhere;
        }

        .local-feature {
            font-size: 0.83rem;
            font-weight: 600;
            color: rgba(255,255,255,0.90);
            line-height: 1.2;
        }

        .local-applicant-value {
            font-size: 0.70rem;
            color: rgba(255,255,255,0.47);
            margin-top: 4px;
        }

        .local-chart-area {
            flex-grow: 1;
            height: 30px;
            display: flex;
            align-items: stretch;
            position: relative;
        }

        .local-left,
        .local-right {
            width: 50%;
            height: 100%;
            background: rgba(255,255,255,0.035);
            position: relative;
        }

        .local-left {
            display: flex;
            justify-content: flex-end;
            border-radius: 7px 0 0 7px;
        }

        .local-right {
            display: flex;
            justify-content: flex-start;
            border-radius: 0 7px 7px 0;
        }

        .local-center-line {
            width: 2px;
            min-width: 2px;
            background: rgba(255,255,255,0.38);
            position: relative;
            z-index: 3;
        }

        .local-negative-bar {
            height: 100%;
            align-self: stretch;
            border-radius: 7px 0 0 7px;
            background: linear-gradient(
                90deg,
                rgba(46,204,113,0.98),
                rgba(46,204,113,0.58)
            );
            box-shadow: 0 2px 10px rgba(46,204,113,0.13);
        }

        .local-positive-bar {
            height: 100%;
            align-self: stretch;
            border-radius: 0 7px 7px 0;
            background: linear-gradient(
                90deg,
                rgba(239,83,80,0.58),
                rgba(239,83,80,0.98)
            );
            box-shadow: 0 2px 10px rgba(239,83,80,0.13);
        }

        .local-shap-value {
            width: 74px;
            min-width: 74px;
            text-align: right;
            font-size: 0.76rem;
            font-weight: 700;
        }

        /* Insight cards */
        .insight-card {
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }

        .insight-title {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: rgba(255,255,255,0.48);
            margin-bottom: 6px;
        }

        .insight-value {
            font-size: 0.93rem;
            font-weight: 650;
            color: rgba(255,255,255,0.90);
        }

        .context-row {
            display: flex;
            justify-content: space-between;
            gap: 15px;
            padding: 11px 0;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            font-size: 0.82rem;
        }

        .context-row:last-child {
            border-bottom: none;
        }

        .context-label {
            color: rgba(255,255,255,0.75);
            font-weight: 600;
        }

        .context-details {
            color: rgba(255,255,255,0.55);
            text-align: right;
        }

        @media (max-width: 850px) {

            .global-feature-name {
                width: 150px;
                min-width: 150px;
            }

            .local-label {
                width: 160px;
                min-width: 160px;
            }

            .global-percent {
                display: none;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# PREDICTION SUMMARY
# -------------------------------------------------------------------

def _render_prediction_summary_card(prediction: dict) -> None:
    probability = float(prediction["probability"])
    risk_band = prediction["risk_band"]

    with st.container(border=True):

        col1, col2 = st.columns([1, 2])

        with col1:
            risk_badge(probability, risk_band)

        with col2:
            st.metric(
                "Probability of Default",
                f"{probability * 100:.1f}%"
            )

        st.progress(min(max(probability, 0.0), 1.0))

        confidence = prediction_confidence(probability)

        st.caption(
            f"Model confidence (distance from the 50% decision threshold): "
            f"{confidence * 100:.0f}%. "
            f"This measures how far the prediction is from the point of maximum "
            f"uncertainty and is not a calibrated statistical confidence interval."
        )

        st.write(
            f"The model estimates a **{probability * 100:.1f}% probability of default**, "
            f"placing this applicant in the **{risk_band} RISK** category."
        )


# -------------------------------------------------------------------
# GLOBAL FEATURE IMPORTANCE CHART
# -------------------------------------------------------------------

def _render_horizontal_importance_chart(items: list[dict]) -> None:
    """
    Premium ranked horizontal feature-importance chart.

    Importance uses mean absolute SHAP values, therefore the chart measures
    magnitude of influence rather than whether a feature increases or decreases
    risk overall.
    """

    if not items:
        st.info("No importance data available.")
        return

    max_val = max(
        float(item["mean_abs_shap_value"])
        for item in items
    ) or 1.0

    rows_html = []

    for idx, item in enumerate(items, start=1):

        value = float(item["mean_abs_shap_value"])
        label = html.escape(str(item["business_label"]))

        width_pct = max(3, (value / max_val) * 100)
        relative_pct = (value / max_val) * 100

        # Alternate the gradient slightly by ranking.
        if idx == 1:
            gradient = (
                "linear-gradient(90deg, "
                "#4F8EF7 0%, #7C5CFC 100%)"
            )
        else:
            gradient = (
                "linear-gradient(90deg, "
                "#3E78D4 0%, #5C86E8 100%)"
            )

        row = (
            '<div class="global-row">'
            f'<div class="rank-pill">{idx}</div>'
            f'<div class="global-feature-name">{label}</div>'
            '<div class="global-bar-track">'
            f'<div class="global-bar-fill" '
            f'style="width:{width_pct:.2f}%;background:{gradient};"></div>'
            '</div>'
            f'<div class="global-value">{value:.4f}</div>'
            f'<div class="global-percent">{relative_pct:.0f}%</div>'
            '</div>'
        )

        rows_html.append(row)

    chart_html = (
        '<div class="xai-chart-card">'
        '<div class="global-chart-header">'
        '<div class="global-chart-title">'
        'Top 10 features by average model influence'
        '</div>'
        '<div class="global-chart-note">'
        'Relative to strongest feature'
        '</div>'
        '</div>'
        + "".join(rows_html)
        + '</div>'
    )

    st.markdown(chart_html, unsafe_allow_html=True)


# -------------------------------------------------------------------
# LOCAL DIVERGING SHAP CONTRIBUTION CHART
# -------------------------------------------------------------------

def _render_diverging_contribution_chart(
    top_increasing: list[dict],
    top_decreasing: list[dict],
) -> None:
    """
    Professional diverging SHAP chart.

    Left side  = negative SHAP = reduces predicted default risk.
    Right side = positive SHAP = increases predicted default risk.
    """

    combined = list(top_increasing) + list(top_decreasing)

    if not combined:
        st.info("No SHAP contributions available for this applicant.")
        return

    combined_sorted = sorted(
        combined,
        key=lambda item: abs(float(item["shap_value"])),
        reverse=True,
    )

    max_abs = max(
        abs(float(item["shap_value"]))
        for item in combined_sorted
    ) or 1.0

    rows_html = []

    for item in combined_sorted:

        value = float(item["shap_value"])

        label = html.escape(str(item["business_label"]))
        applicant_value = html.escape(
            str(item.get("applicant_value_display", ""))
        )

        bar_width = max(
            4,
            (abs(value) / max_abs) * 100,
        )

        if value < 0:
            negative_bar = (
                f'<div class="local-negative-bar" '
                f'style="width:{bar_width:.2f}%;"></div>'
            )
            positive_bar = ""

            value_color = _RISK_DECREASE_COLOR

        else:
            negative_bar = ""
            positive_bar = (
                f'<div class="local-positive-bar" '
                f'style="width:{bar_width:.2f}%;"></div>'
            )

            value_color = _RISK_INCREASE_COLOR

        row = (
            '<div class="local-row">'
            '<div class="local-label">'
            f'<div class="local-feature">{label}</div>'
            f'<div class="local-applicant-value">'
            f'Applicant value: {applicant_value}'
            '</div>'
            '</div>'

            '<div class="local-chart-area">'
            '<div class="local-left">'
            f'{negative_bar}'
            '</div>'

            '<div class="local-center-line"></div>'

            '<div class="local-right">'
            f'{positive_bar}'
            '</div>'
            '</div>'

            f'<div class="local-shap-value" '
            f'style="color:{value_color};">'
            f'{value:+.4f}'
            '</div>'
            '</div>'
        )

        rows_html.append(row)

    chart_html = (
        '<div class="xai-chart-card">'
        '<div class="local-legend">'
        '<div class="legend-item">'
        f'<span class="legend-dot" style="background:{_RISK_DECREASE_COLOR};"></span>'
        '<span>Reduces predicted default risk</span>'
        '</div>'
        '<div class="legend-item">'
        f'<span class="legend-dot" style="background:{_RISK_INCREASE_COLOR};"></span>'
        '<span>Increases predicted default risk</span>'
        '</div>'
        '</div>'

        '<div style="display:flex;justify-content:center;gap:18px;'
        'font-size:0.68rem;color:rgba(255,255,255,0.35);'
        'margin-top:-10px;margin-bottom:14px;">'
        '<span>← LOWER RISK</span>'
        '<span>| BASELINE |</span>'
        '<span>HIGHER RISK →</span>'
        '</div>'

        + "".join(rows_html)
        + '</div>'
    )

    st.markdown(chart_html, unsafe_allow_html=True)


# -------------------------------------------------------------------
# GLOBAL EXPLAINABILITY
# -------------------------------------------------------------------

def _render_global_section() -> None:

    st.markdown(
        '<div class="xai-section-title">Global Model Explainability</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="xai-section-subtitle">'
        'See which variables have the strongest average influence on the '
        'model across the evaluation sample.'
        '</div>',
        unsafe_allow_html=True,
    )

    try:

        df = _load_data_cached()

        sample_size = min(500, len(df))
        sample = df.sample(
            sample_size,
            random_state=1,
        )

        all_importance = compute_full_global_importance(sample)

        if not all_importance:
            info_banner("No global importance could be computed.")
            return

        total_importance = sum(
            float(item["mean_abs_shap_value"])
            for item in all_importance
        ) or 1.0

        top5_importance = sum(
            float(item["mean_abs_shap_value"])
            for item in all_importance[:5]
        )

        cumulative_pct = (
            top5_importance / total_importance
        ) * 100

        # Metrics
        m1, m2, m3, m4, m5 = st.columns(5)

        m1.metric(
            "Features Used",
            len(all_importance),
        )

        m2.metric(
            "Top Feature",
            all_importance[0]["business_label"],
        )

        m3.metric(
            "Top Mean |SHAP|",
            f"{all_importance[0]['mean_abs_shap_value']:.4f}",
        )

        m4.metric(
            "Top-5 Influence",
            f"{cumulative_pct:.1f}%",
        )

        m5.metric(
            "Samples Used",
            sample_size,
        )

        st.write("")

        _render_horizontal_importance_chart(
            all_importance[:10]
        )

        st.caption(
            "Global importance is calculated using mean absolute SHAP values. "
            "A larger bar means the feature has a stronger average influence on "
            "the model's predictions. Because absolute SHAP values are used here, "
            "this chart measures influence magnitude, not whether a feature "
            "generally increases or decreases default risk."
        )

    except (DatasetNotFoundError, DatasetValidationError) as exc:
        error_banner(str(exc))

    except ExplanationUnavailableError as exc:
        error_banner(str(exc))


# -------------------------------------------------------------------
# LOCAL EXPLAINABILITY
# -------------------------------------------------------------------

def _render_local_section() -> None:

    st.markdown(
        '<div class="xai-section-title">Local Explanation — Last Prediction</div>',
        unsafe_allow_html=True,
    )

    applicant = st.session_state["last_applicant"]
    prediction = st.session_state["last_prediction"]

    probability = float(prediction["probability"])
    risk_band = prediction["risk_band"]

    st.markdown(
        '<div class="xai-section-subtitle">'
        f'This applicant was classified as <b>{risk_band} RISK</b> with a '
        f'<b>{probability * 100:.1f}% predicted probability of default</b>.'
        '</div>',
        unsafe_allow_html=True,
    )

    try:
        local = local_explanation(applicant)

    except ExplanationUnavailableError as exc:
        error_banner(str(exc))
        return

    # ---------------------------------------------------------------
    # LOCAL CHART
    # ---------------------------------------------------------------

    st.markdown(
        "**Risk Contribution Breakdown**"
    )

    st.caption(
        "Each bar shows how strongly an individual feature pushes this "
        "specific prediction toward lower or higher predicted default risk."
    )

    _render_diverging_contribution_chart(
        local["top_risk_increasing"],
        local["top_risk_decreasing"],
    )

    # ---------------------------------------------------------------
    # KEY FACTORS SUMMARY
    # ---------------------------------------------------------------

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="insight-card">'
            '<div class="insight-title">Risk-increasing signals</div>',
            unsafe_allow_html=True,
        )

        if local["top_risk_increasing"]:

            for factor in local["top_risk_increasing"][:5]:

                label = html.escape(
                    str(factor["business_label"])
                )

                value = html.escape(
                    str(factor["applicant_value_display"])
                )

                shap_value = float(
                    factor["shap_value"]
                )

                st.markdown(
                    f'<div style="margin-bottom:10px;">'
                    f'<span style="color:{_RISK_INCREASE_COLOR};font-weight:700;">●</span> '
                    f'<b>{label}</b><br>'
                    f'<span style="font-size:0.78rem;color:rgba(255,255,255,0.58);">'
                    f'Applicant value: {value} &nbsp;•&nbsp; '
                    f'SHAP: {shap_value:+.4f}'
                    f'</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        else:

            st.caption(
                "No strong risk-increasing factors identified."
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:

        st.markdown(
            '<div class="insight-card">'
            '<div class="insight-title">Risk-reducing signals</div>',
            unsafe_allow_html=True,
        )

        if local["top_risk_decreasing"]:

            for factor in local["top_risk_decreasing"][:5]:

                label = html.escape(
                    str(factor["business_label"])
                )

                value = html.escape(
                    str(factor["applicant_value_display"])
                )

                shap_value = float(
                    factor["shap_value"]
                )

                st.markdown(
                    f'<div style="margin-bottom:10px;">'
                    f'<span style="color:{_RISK_DECREASE_COLOR};font-weight:700;">●</span> '
                    f'<b>{label}</b><br>'
                    f'<span style="font-size:0.78rem;color:rgba(255,255,255,0.58);">'
                    f'Applicant value: {value} &nbsp;•&nbsp; '
                    f'SHAP: {shap_value:+.4f}'
                    f'</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        else:

            st.caption(
                "No strong risk-reducing factors identified."
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # STATISTICAL CONTEXT
    # ---------------------------------------------------------------

    try:

        df_for_stats = _load_data_cached()

        background_sample = df_for_stats.sample(
            min(500, len(df_for_stats)),
            random_state=1,
        )

        engineered_background = get_engineered_background(
            background_sample
        )

    except (
        DatasetNotFoundError,
        DatasetValidationError,
        ExplanationUnavailableError,
    ):
        engineered_background = None

    except Exception:
        # Statistical context is supplementary and should never break
        # the explainability page.
        engineered_background = None

    if engineered_background is not None:

        numeric_cols_set = set(
            get_feature_lists()[0]
        )

        combined_sorted = sorted(
            local["top_risk_increasing"]
            + local["top_risk_decreasing"],
            key=lambda item: abs(
                float(item["shap_value"])
            ),
            reverse=True,
        )

        seen_columns: set[str] = set()
        context_rows = []

        for item in combined_sorted:

            raw_col = item.get("raw_column")

            if (
                not raw_col
                or raw_col in seen_columns
                or raw_col not in numeric_cols_set
            ):
                continue

            seen_columns.add(raw_col)

            ctx = compute_numeric_context(
                engineered_background,
                raw_col,
                item.get("applicant_value_raw"),
            )

            if ctx:
                context_rows.append(
                    (item, ctx)
                )

            if len(context_rows) >= 5:
                break

        if context_rows:

            st.write("")
            st.markdown(
                "### Statistical Context"
            )

            st.caption(
                "How the applicant's important numerical values compare with "
                "the background dataset."
            )

            context_html = (
                '<div class="xai-chart-card">'
            )

            for item, ctx in context_rows:

                label = html.escape(
                    str(item["business_label"])
                )

                applicant_display = html.escape(
                    str(item["applicant_value_display"])
                )

                median_display = html.escape(
                    str(
                        format_feature_value(
                            ctx["column"],
                            ctx["median"],
                        )
                    )
                )

                percentile = float(
                    ctx["percentile"]
                )

                n_samples = int(
                    ctx["n_samples"]
                )

                context_html += (
                    '<div class="context-row">'
                    '<div>'
                    f'<div class="context-label">{label}</div>'
                    '</div>'
                    '<div class="context-details">'
                    f'Applicant: <b>{applicant_display}</b><br>'
                    f'Median: {median_display} &nbsp;•&nbsp; '
                    f'{percentile:.0f}th percentile '
                    f'({n_samples} applicants)'
                    '</div>'
                    '</div>'
                )

            context_html += "</div>"

            st.markdown(
                context_html,
                unsafe_allow_html=True,
            )

    st.caption(
        "These local explanations are computed from the actual trained model's "
        "SHAP values for this specific applicant. They are not fixed or "
        "hardcoded business rules."
    )


# -------------------------------------------------------------------
# HOW TO INTERPRET
# -------------------------------------------------------------------

def _render_how_to_interpret() -> None:

    with st.expander("How to interpret this explanation"):

        st.markdown(
            """
            **Global explainability**
            - Shows which variables influence the model most across many applicants.
            - Uses mean absolute SHAP values, so it measures the average magnitude
              of influence.

            **Local explainability**
            - Shows why the model produced this particular prediction.
            - **Positive SHAP values** push the prediction toward higher predicted
              default risk.
            - **Negative SHAP values** push the prediction toward lower predicted
              default risk.

            **Statistical context**
            - Compares important numerical applicant values with the evaluation
              background data using medians and percentiles.

            **Important limitation**
            - Feature importance and SHAP contributions explain model behavior.
            - They do **not** prove that a feature causes loan default.
            - SHAP contribution values are model units, not direct probability
              percentage-point changes.
            """
        )


# -------------------------------------------------------------------
# MAIN PAGE
# -------------------------------------------------------------------

def render() -> None:

    _inject_explainability_styles()

    st.header("Prediction Explainability")

    st.caption(
        "Understand which factors influence the model globally and why it "
        "produced the latest applicant's risk prediction."
    )

    if not is_model_available():

        info_banner(
            "No trained model found yet. Train one first: "
            "`python -m src.ml.train`."
        )

        return

    has_prediction = (
        "last_prediction" in st.session_state
        and "last_applicant" in st.session_state
    )

    # ---------------------------------------------------------------
    # PREDICTION OVERVIEW
    # ---------------------------------------------------------------

    if has_prediction:

        _render_prediction_summary_card(
            st.session_state["last_prediction"]
        )

    else:

        info_banner(
            "No prediction available yet. Go to Risk Prediction and assess an "
            "applicant to see a personalized local explanation here."
        )

    st.divider()

    # ---------------------------------------------------------------
    # GLOBAL EXPLAINABILITY
    # ---------------------------------------------------------------

    _render_global_section()

    st.divider()

    # ---------------------------------------------------------------
    # LOCAL EXPLAINABILITY
    # ---------------------------------------------------------------

    if has_prediction:
        _render_local_section()

    st.divider()

    # ---------------------------------------------------------------
    # INTERPRETATION GUIDE
    # ---------------------------------------------------------------

    _render_how_to_interpret()