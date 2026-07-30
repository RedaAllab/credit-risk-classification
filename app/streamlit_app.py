"""Interactive demo for the credit risk model: fill in a loan application,
get the champion XGBoost model's default probability, decision, and a
per-applicant SHAP explanation, plus how the 4 challenger models would have
scored the same applicant.

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
from src.style import MODEL_COLORS, set_style

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "all_models.joblib"

HOME_OWNERSHIP_OPTIONS = ["MORTGAGE", "OTHER", "OWN", "RENT"]
LOAN_INTENT_OPTIONS = [
    "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE",
]

# Validated notebook results (see README "Results" table), shown for context
# in the sidebar rather than recomputed at runtime. AutoML (FLAML) is the
# exception: it isn't in the notebook, so its numbers come from the test-set
# evaluation printed by scripts/train_automl.py instead - re-run that script
# and update this row if the model is retrained.
MODEL_METRICS = {
    "Logistic Regression (L1)": {"ROC-AUC": 0.857, "F2": 0.709, "Recall": 0.577},
    "Random Forest": {"ROC-AUC": 0.934, "F2": 0.790, "Recall": 0.702},
    "XGBoost": {"ROC-AUC": 0.950, "F2": 0.820, "Recall": 0.727},
    "MLP": {"ROC-AUC": 0.906, "F2": 0.744, "Recall": 0.729},
    "AutoML (FLAML)": {"ROC-AUC": 0.946, "F2": 0.816, "Recall": 0.877},
}

DEFAULT_PROFILE = {
    "person_age": 30,
    "person_income": 60000,
    "person_emp_length": 5,
    "person_home_ownership": "RENT",
    "cb_person_cred_hist_length": 5,
    "loan_amnt": 10000,
    "loan_int_rate": 11.0,
    "loan_grade": "B",
    "loan_intent": "EDUCATION",
    "cb_person_default_on_file": "N",
}
LOW_RISK_PROFILE = {
    "person_age": 35,
    "person_income": 95000,
    "person_emp_length": 10,
    "person_home_ownership": "MORTGAGE",
    "cb_person_cred_hist_length": 12,
    "loan_amnt": 5000,
    "loan_int_rate": 6.5,
    "loan_grade": "A",
    "loan_intent": "DEBTCONSOLIDATION",
    "cb_person_default_on_file": "N",
}
HIGH_RISK_PROFILE = {
    "person_age": 22,
    "person_income": 22000,
    "person_emp_length": 0,
    "person_home_ownership": "RENT",
    "cb_person_cred_hist_length": 2,
    "loan_amnt": 18000,
    "loan_int_rate": 19.5,
    "loan_grade": "F",
    "loan_intent": "MEDICAL",
    "cb_person_default_on_file": "Y",
}


def _apply_profile(profile: dict):
    for key, value in profile.items():
        st.session_state[key] = value


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


def build_feature_row(inputs: dict, feature_columns: list[str]) -> pd.DataFrame:
    """Turn form inputs into the exact one-row, one-hot-encoded layout the
    models were trained on (see src/preprocessing.py encode_features)."""
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


def render_sidebar():
    st.sidebar.header("Model performance")
    st.sidebar.caption("Validated on a held-out test set, see the full notebook for methodology.")
    metrics_df = pd.DataFrame(MODEL_METRICS).T
    st.sidebar.table(metrics_df)
    st.sidebar.markdown(
        "[Full analysis on GitHub](https://github.com/RedaAllab/credit-risk-classification)"
    )


def render_model_comparison(models: dict, X_row: pd.DataFrame):
    st.subheader("Champion vs. challengers")
    st.caption("Predicted default probability from all 5 models for this applicant.")

    names = [name for name in MODEL_COLORS if name in models]
    probs = [models[name]["model"].predict_proba(X_row)[0, 1] for name in names]
    colors = [MODEL_COLORS[name] for name in names]

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(names, probs, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted probability of default")
    for bar, p in zip(bars, probs):
        ax.text(min(p + 0.02, 0.95), bar.get_y() + bar.get_height() / 2, f"{p:.1%}", va="center")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def main():
    st.set_page_config(page_title="Credit Risk Demo", page_icon="\U0001f4b3", layout="centered")
    set_style()

    st.title("Credit Risk Model: Live Demo")
    st.markdown(
        "Fill in a loan application below to get the champion **XGBoost** model's "
        "default probability, decision, and a per-applicant SHAP explanation, "
        "then compare it against the 4 challenger models. "
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
    models = artifact["models"]
    champion_name = artifact["champion"]
    feature_columns = artifact["feature_columns"]
    champion = models[champion_name]["model"]
    threshold = models[champion_name]["threshold"]

    render_sidebar()

    st.write("Try an example applicant, or fill in your own below:")
    example_col1, example_col2 = st.columns(2)
    with example_col1:
        st.button(
            "Low-risk profile", use_container_width=True,
            on_click=_apply_profile, args=(LOW_RISK_PROFILE,),
        )
    with example_col2:
        st.button(
            "High-risk profile", use_container_width=True,
            on_click=_apply_profile, args=(HIGH_RISK_PROFILE,),
        )

    # Defaults set once via session_state (not the widgets' `value`/`index`
    # args), so the example-profile buttons above can override them on a
    # rerun without Streamlit's "value set both ways" warning.
    for key, value in DEFAULT_PROFILE.items():
        st.session_state.setdefault(key, value)

    with st.form("application_form"):
        col1, col2 = st.columns(2)
        with col1:
            person_age = st.number_input("Applicant age", 18, 100, key="person_age")
            person_income = st.number_input(
                "Annual income ($)", 1000, 2_000_000, step=1000, key="person_income"
            )
            person_emp_length = st.number_input(
                "Employment length (years)", 0, 60, key="person_emp_length"
            )
            person_home_ownership = st.selectbox(
                "Home ownership", HOME_OWNERSHIP_OPTIONS, key="person_home_ownership"
            )
            cb_person_cred_hist_length = st.number_input(
                "Credit history length (years)", 0, 60, key="cb_person_cred_hist_length"
            )
        with col2:
            loan_amnt = st.number_input(
                "Loan amount ($)", 500, 50000, step=500, key="loan_amnt"
            )
            loan_int_rate = st.number_input(
                "Interest rate (%)", 5.0, 25.0, step=0.1, key="loan_int_rate"
            )
            loan_grade = st.selectbox(
                "Loan grade", list(GRADE_MAP.keys()), key="loan_grade"
            )
            loan_intent = st.selectbox("Loan purpose", LOAN_INTENT_OPTIONS, key="loan_intent")
            cb_person_default_on_file = st.selectbox(
                "Prior default on file?", ["N", "Y"], key="cb_person_default_on_file"
            )

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
    proba = champion.predict_proba(X_row)[0, 1]
    is_default = proba >= threshold
    loan_percent_income = X_row["loan_percent_income"].iloc[0]

    st.divider()
    result_col, metric_col, ratio_col = st.columns([2, 1, 1])
    with metric_col:
        st.metric("Default probability", f"{proba:.1%}")
    with ratio_col:
        st.metric("Loan / income ratio", f"{loan_percent_income:.0%}")
    with result_col:
        if is_default:
            st.error(f"**Flagged for review** (probability above the {threshold:.0%} decision threshold)")
        else:
            st.success(f"**Approved** (probability below the {threshold:.0%} decision threshold)")

    st.subheader("Why this decision?")
    st.caption("SHAP contribution of each feature to this specific prediction (red pushes toward default, blue pushes away).")

    explainer = shap.TreeExplainer(champion)
    shap_values = explainer(X_row)

    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()
    render_model_comparison(models, X_row)


if __name__ == "__main__":
    main()
