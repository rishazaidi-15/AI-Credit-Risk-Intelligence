"""Risk Prediction page."""

from __future__ import annotations

import streamlit as st

from app.components.ui_components import error_banner, info_banner, risk_badge
from src.database.database import DatabaseError, save_prediction
from src.ml.predict import ModelNotTrainedError, is_model_available, predict_single


def render() -> None:
    st.header("Credit Risk Prediction")
    st.caption("Enter applicant details to get a default-risk assessment.")

    if not is_model_available():
        info_banner(
            "No trained model found yet. Train one first: `python -m src.ml.train` "
            "(requires the real dataset -- see README 'Dataset Setup')."
        )
        return

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            code_gender = st.selectbox("Gender", ["F", "M"])
            cnt_children = st.number_input("Number of children", min_value=0, max_value=15, value=0)
            cnt_fam_members = st.number_input("Family members", min_value=1, max_value=15, value=2)
            name_family_status = st.selectbox(
                "Family status", ["Married", "Single / not married", "Civil marriage", "Widow", "Separated"]
            )
            age_years = st.number_input("Age (years)", min_value=18, max_value=100, value=35)

        with col2:
            st.markdown("**Financial**")
            amt_income_total = st.number_input("Annual income", min_value=0, value=180000, step=5000)
            amt_credit = st.number_input("Requested credit amount", min_value=0, value=600000, step=10000)
            amt_annuity = st.number_input("Loan annuity", min_value=0, value=27000, step=1000)
            amt_goods_price = st.number_input("Goods price", min_value=0, value=540000, step=10000)
            flag_own_car = st.selectbox("Owns a car", ["N", "Y"])
            flag_own_realty = st.selectbox("Owns real estate", ["N", "Y"])

        with col3:
            st.markdown("**Employment & Credit Signal**")
            name_income_type = st.selectbox(
                "Income type", ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"]
            )
            occupation_type = st.selectbox(
                "Occupation", ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers", "Other"]
            )
            employed_years = st.number_input("Years employed (0 if not employed)", min_value=0, max_value=60, value=5)
            ext_source_1 = st.slider("External credit score #1", 0.0, 1.0, 0.5)
            ext_source_2 = st.slider("External credit score #2", 0.0, 1.0, 0.5)
            ext_source_3 = st.slider("External credit score #3", 0.0, 1.0, 0.5)

        submitted = st.form_submit_button("Assess Risk")

    if not submitted:
        return

    applicant = {
        "CODE_GENDER": code_gender,
        "CNT_CHILDREN": cnt_children,
        "CNT_FAM_MEMBERS": cnt_fam_members,
        "NAME_FAMILY_STATUS": name_family_status,
        "DAYS_BIRTH": -int(age_years * 365.25),
        "AMT_INCOME_TOTAL": amt_income_total,
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": amt_annuity,
        "AMT_GOODS_PRICE": amt_goods_price,
        "FLAG_OWN_CAR": flag_own_car,
        "FLAG_OWN_REALTY": flag_own_realty,
        "NAME_INCOME_TYPE": name_income_type,
        "OCCUPATION_TYPE": occupation_type,
        "DAYS_EMPLOYED": -int(employed_years * 365.25) if employed_years > 0 else 365243,
        "EXT_SOURCE_1": ext_source_1,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
    }

    try:
        result = predict_single(applicant)
    except ModelNotTrainedError as exc:
        error_banner(str(exc))
        return
    except Exception:  # noqa: BLE001 -- never expose raw errors to the user
        error_banner("The prediction could not be completed. Please check your inputs and try again.")
        return

    st.session_state["last_applicant"] = applicant
    st.session_state["last_prediction"] = result

    st.divider()
    st.subheader("Result")
    risk_badge(result["probability"], result["risk_band"])

    try:
        save_prediction(
            sk_id_curr=None,
            probability=result["probability"],
            risk_band=result["risk_band"],
            model_name="active_pipeline",
        )
    except DatabaseError:
        pass  # non-fatal -- prediction already shown to the user

    st.caption("See the Explainability and Rules pages for why this applicant received this classification.")
