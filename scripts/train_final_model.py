"""Train all 4 models compared in the notebook and persist them for the
Streamlit demo (app/streamlit_app.py): the XGBoost champion for the
Approved/Flagged decision and SHAP explanation, plus the 3 challengers
(Logistic L1, Random Forest, MLP) so the app can show a champion-vs-
challengers probability comparison for the same applicant.

Hyperparameters are the ones GridSearchCV selected in
notebooks/credit_risk_modeling_en.ipynb (sections 3, 4, 5, 6) rather than
re-run here, since re-running the full search is slow and this script's only
job is to reproduce those already-validated models as loadable artifacts.

Usage: python scripts/train_final_model.py
"""

import sys
from pathlib import Path

import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import find_best_f2_threshold
from src.preprocessing import clean_dataset, encode_features, load_dataset, split_features_target

RANDOM_STATE = 42

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
}
LOGISTIC_L1_C = 0.03
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_split": 2,
    "max_features": "sqrt",
}
MLP_PARAMS = {
    "hidden_layer_sizes": (128, 64, 32),
    "activation": "tanh",
    "alpha": 0.01,
}


def train_all(X_train, y_train):
    scale_pos_weight = int((y_train == 0).sum() / (y_train == 1).sum())

    xgboost_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        **XGBOOST_PARAMS,
    )
    xgboost_model.fit(X_train, y_train)

    logistic_l1 = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            penalty="l1", solver="liblinear", class_weight="balanced",
            C=LOGISTIC_L1_C, max_iter=1000, random_state=RANDOM_STATE,
        )),
    ])
    logistic_l1.fit(X_train, y_train)

    random_forest = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, **RF_PARAMS,
    )
    random_forest.fit(X_train, y_train)

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            max_iter=1000, early_stopping=True, validation_fraction=0.1,
            random_state=RANDOM_STATE, **MLP_PARAMS,
        )),
    ])
    mlp.fit(X_train, y_train)

    return {
        "Logistic (L1)": logistic_l1,
        "Random Forest": random_forest,
        "XGBoost": xgboost_model,
        "MLP": mlp,
    }


def main():
    root = Path(__file__).resolve().parent.parent
    df = load_dataset(root / "data" / "credit_risk_dataset.csv")
    df = clean_dataset(df)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_features_target(df, random_state=RANDOM_STATE)

    fitted_models = train_all(X_train, y_train)

    models_artifact = {}
    for name, model in fitted_models.items():
        y_proba_test = model.predict_proba(X_test)[:, 1]
        threshold, f2 = find_best_f2_threshold(y_test, y_proba_test)
        print(f"{name}: test F2-score {f2:.3f} at threshold {threshold:.2f}")
        models_artifact[name] = {"model": model, "threshold": float(threshold)}

    artifact = {
        "champion": "XGBoost",
        "feature_columns": list(X_train.columns),
        "models": models_artifact,
    }

    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)
    out_path = models_dir / "all_models.joblib"
    # compress=3: the Random Forest alone is >100MB uncompressed (GitHub's hard
    # file size limit), compresses down to ~27MB with negligible load-time cost.
    joblib.dump(artifact, out_path, compress=3)
    print(f"Saved model artifact to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
