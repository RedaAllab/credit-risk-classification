"""Interactive demo for the credit risk model: fill in a loan application,
get the champion XGBoost model's default probability, decision, and a
per-applicant SHAP explanation.

Run locally with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import GRADE_MAP
from src.style import BLUE, RED, set_style

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "xgboost_final.joblib"

HOME_OWNERSHIP_OPTIONS = ["MORTGAGE", "OTHER", "OWN", "RENT"]
LOAN_INTENT_OPTIONS = [
    "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE",
]


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


def build_feature_row(inputs: dict, feature_columns: list[str]) -> pd.DataFrame:
    """Turn form inputs into the exact one-row, one-hot-encoded layout the
    model was trained on (see src/preprocessing.py encode_features)."""
    row = {
        "person_age": inputs["person_age"],
        "person_income": inputs["person_income"],
        "person_emp_length": inputs["person_emp_length"],
        "loan_grade": GRADE_MAP[inputs["loan_grade"]],
        "loan_amnt": inputs["loan_amnt"],
        "loan_int_rate": inputs["loan_int_rate"],
        "loan_percent_income": round(inputs["loan_amnt"] / inputs["person_income"], 2),
        "cb_person_default_on_file": 1 if inputs["cb_person_default_on_file"] == "Y" else 0,
        "cb_person_cred_hist_length": inputs["cb_person_cred_hist_length"],
        "is_missing_rate": 0,
        "is_missing_emplength": 0,
    }
    # One-hot columns: drop_first=True at training time dropped the alphabetically
    # first category (MORTGAGE, DEBTCONSOLIDATION), so those stay all-zero here too.
    for col in feature_columns:
        if col.startswith("person_home_ownership_"):
            row[col] = int(col == f"person_home_ownership_{inputs['person_home_ownership']}")
        elif col.startswith("loan_intent_"):
            row[col] = int(col == f"loan_intent_{inputs['loan_intent']}")

    return pd.DataFrame([row])[feature_columns]


def main():
    st.set_page_config(page_title="Credit Risk Demo", page_icon="\U0001f4b3", layout="centered")
    set_style()

    st.title("Credit Risk Model: Live Demo")
    st.markdown(
        "Fill in a loan application below to get the champion **XGBoost** model's "
        "default probability, decision, and a per-applicant SHAP explanation. "
        "See the [full analysis](https://github.com/RedaAllab/credit-risk-classification) "
        "for the model comparison, statistical validation, and explainability behind this."
    )

    if not MODEL_PATH.exists():
        st.error(
            "Model artifact not found. Run `python scripts/train_final_model.py` "
            "from the repo root first."
        )
        return

    artifact = load_artifact()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    threshold = artifact["threshold"]

    with st.form("application_form"):
        col1, col2 = st.columns(2)
        with col1:
            person_age = st.number_input("Applicant age", 18, 100, 30)
            person_income = st.number_input("Annual income ($)", 1000, 2_000_000, 60000, step=1000)
            person_emp_length = st.number_input("Employment length (years)", 0, 60, 5)
            person_home_ownership = st.selectbox("Home ownership", HOME_OWNERSHIP_OPTIONS, index=3)
            cb_person_cred_hist_length = st.number_input("Credit history length (years)", 0, 60, 5)
        with col2:
            loan_amnt = st.number_input("Loan amount ($)", 500, 50000, 10000, step=500)
            loan_int_rate = st.number_input("Interest rate (%)", 5.0, 25.0, 11.0, step=0.1)
            loan_grade = st.selectbox("Loan grade", list(GRADE_MAP.keys()), index=1)
            loan_intent = st.selectbox("Loan purpose", LOAN_INTENT_OPTIONS, index=1)
            cb_person_default_on_file = st.selectbox("Prior default on file?", ["N", "Y"])

        submitted = st.form_submit_button("Assess risk", use_container_width=True)

    if not submitted:
        return

    inputs = {
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "person_home_ownership": person_home_ownership,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_grade": loan_grade,
        "loan_intent": loan_intent,
        "cb_person_default_on_file": cb_person_default_on_file,
    }
    X_row = build_feature_row(inputs, feature_columns)
    proba = model.predict_proba(X_row)[0, 1]
    is_default = proba >= threshold

    st.divider()
    result_col, metric_col = st.columns([2, 1])
    with metric_col:
        st.metric("Default probability", f"{proba:.1%}")
    with result_col:
        if is_default:
            st.error(f"**Flagged for review** (probability above the {threshold:.0%} decision threshold)")
        else:
            st.success(f"**Approved** (probability below the {threshold:.0%} decision threshold)")

    st.subheader("Why this decision?")
    st.caption("SHAP contribution of each feature to this specific prediction (red pushes toward default, blue pushes away).")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_row)

    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
