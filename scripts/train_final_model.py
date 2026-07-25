"""Train the champion XGBoost model on the full dataset and persist it for
the Streamlit demo (app/streamlit_app.py), so the app loads a ready model
instead of retraining on every run.

Hyperparameters are the ones GridSearchCV selected in
notebooks/credit_risk_modeling_en.ipynb (section 5) rather than re-run here,
since re-running the full search is slow and this script's only job is to
reproduce that already-validated model as a loadable artifact.

Usage: python scripts/train_final_model.py
"""

import sys
from pathlib import Path

import joblib
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import find_best_f2_threshold
from src.preprocessing import clean_dataset, encode_features, load_dataset, split_features_target

RANDOM_STATE = 42
BEST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
}


def main():
    root = Path(__file__).resolve().parent.parent
    df = load_dataset(root / "data" / "credit_risk_dataset.csv")
    df = clean_dataset(df)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_features_target(df, random_state=RANDOM_STATE)

    scale_pos_weight = int((y_train == 0).sum() / (y_train == 1).sum())

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        **BEST_PARAMS,
    )
    model.fit(X_train, y_train)

    y_proba_test = model.predict_proba(X_test)[:, 1]
    threshold, f2 = find_best_f2_threshold(y_test, y_proba_test)
    print(f"Test F2-score at threshold {threshold:.2f}: {f2:.3f}")

    artifact = {
        "model": model,
        "feature_columns": list(X_train.columns),
        "threshold": float(threshold),
    }

    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)
    out_path = models_dir / "xgboost_final.joblib"
    joblib.dump(artifact, out_path)
    print(f"Saved model artifact to {out_path}")


if __name__ == "__main__":
    main()
