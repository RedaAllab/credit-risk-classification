"""Train a FLAML AutoML challenger for the credit risk model and save it as a
standalone artifact (models/automl_challenger.joblib). Run
scripts/merge_automl_challenger.py afterwards to fold it into the shared
models/all_models.joblib consumed by app/streamlit_app.py, alongside the 4
notebook models (see train_final_model.py).

FLAML replaces the manual GridSearchCV search from the notebook with a
time-budgeted search across model families already used in this project
(Random Forest, L1-regularized logistic regression) plus several this project
never tried: LightGBM, Extra Trees, L2-regularized logistic regression, SGD
linear, and SVC - so it can find hyperparameters (or a different estimator
family entirely) that the notebook's grid never tried.

"kneighbor" (KNN) is deliberately NOT in ESTIMATOR_LIST: this script passes
sample_weight to handle the ~22% default rate the same way as every other
model in this project (see train()), and scikit-learn's KNeighborsClassifier
doesn't accept sample_weight at all - it crashes the whole search the moment
FLAML tries it, not just that one trial. It also wasn't expected to be
competitive on this data (32K rows, mostly one-hot features), so it isn't
worth a separate imbalance-handling path just to include it.

SVC in particular scales poorly (O(n^2)-O(n^3)) on this dataset's ~26K
training rows; if it dominates the time budget without ever beating the
tree-based estimators, drop "svc" from ESTIMATOR_LIST rather than raising
--time-budget to compensate.

Two things this script deliberately does NOT do, both for the same reason:

- It does not include "xgboost" in the search space: flaml[automl] pins
  xgboost<3.0, which conflicts with this project's xgboost==3.2.0 (used by
  the XGBoost champion in train_final_model.py). FLAML is installed from
  requirements-automl.txt in its own throwaway environment, never alongside
  the pinned xgboost.
- It does not load or write models/all_models.joblib. That artifact already
  contains an XGBoost model pickled under xgboost==3.2.0; unpickling it in
  this script's xgboost<3.0 environment and re-saving would silently
  round-trip that model's booster through a different xgboost version.
  merge_automl_challenger.py does that merge instead, run under the main
  requirements.txt environment where the xgboost version matches.

The winning estimator is unwrapped to a plain scikit-learn/LightGBM object
before being saved, so neither the shared artifact nor the Streamlit app that
loads it ever need flaml installed at inference time. For most estimators
that's just FLAML's automl.model.estimator, but two need special handling
(see _unwrap): "svc" (LinearSVC has no native predict_proba) and "sgd"
(FLAML silently runs it through a per-row Normalizer() that isn't part of
the raw estimator) would otherwise produce broken or silently wrong
predictions once unwrapped naively - verified by comparing raw vs FLAML-
wrapped predict_proba output on this dataset.

Usage:
    python -m venv .venv-automl && source .venv-automl/bin/activate
    pip install -r requirements-automl.txt
    python scripts/train_automl.py [--time-budget SECONDS]
"""

import argparse
import sys
from pathlib import Path

import joblib
from flaml import AutoML
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import find_best_f2_threshold
from src.model_wrappers import LinearSVCProbaWrapper
from src.preprocessing import clean_dataset, encode_features, load_dataset, split_features_target

RANDOM_STATE = 42
ESTIMATOR_LIST = ["lgbm", "rf", "extra_tree", "lrl1", "lrl2", "sgd", "svc"]
MODEL_NAME = "AutoML (FLAML)"


def train(X_train, y_train, time_budget: int) -> AutoML:
    # Same class-imbalance handling as train_final_model.py's scale_pos_weight,
    # expressed as sample_weight since it must apply across both the boosted-tree
    # and linear estimators in ESTIMATOR_LIST.
    positive_weight = (y_train == 0).sum() / (y_train == 1).sum()
    sample_weight = y_train.map({0: 1.0, 1: positive_weight})

    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        sample_weight=sample_weight,
        task="classification",
        metric="f1",
        estimator_list=ESTIMATOR_LIST,
        time_budget=time_budget,
        eval_method="cv",
        n_splits=5,
        seed=RANDOM_STATE,
        verbose=0,
    )
    return automl


def _unwrap(automl: AutoML, X_train):
    """Unwrap FLAML's estimator wrapper to a plain scikit-learn-compatible
    object with a real predict_proba (see module docstring for why "svc" and
    "sgd" need special-casing instead of automl.model.estimator directly)."""
    raw = automl.model.estimator
    if automl.best_estimator == "svc":
        return LinearSVCProbaWrapper(raw)
    if automl.best_estimator == "sgd":
        pipeline = Pipeline([("normalizer", Normalizer()), ("sgd", raw)])
        pipeline.named_steps["normalizer"].fit(X_train)
        return pipeline
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time-budget", type=int, default=300, help="AutoML search budget in seconds"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    df = load_dataset(root / "data" / "credit_risk_dataset.csv")
    df = clean_dataset(df)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_features_target(df, random_state=RANDOM_STATE)

    automl = train(X_train, y_train, args.time_budget)
    print(f"Best estimator: {automl.best_estimator}")
    print(f"Best config: {automl.best_config}")

    fitted_model = _unwrap(automl, X_train)

    y_proba_test = fitted_model.predict_proba(X_test)[:, 1]
    threshold, f2 = find_best_f2_threshold(y_test, y_proba_test)
    print(f"{MODEL_NAME}: test F2-score {f2:.3f} at threshold {threshold:.2f}")

    out_path = root / "models" / "automl_challenger.joblib"
    joblib.dump({"model": fitted_model, "threshold": float(threshold)}, out_path)
    print(f"Saved AutoML challenger to {out_path}")
    print("Run scripts/merge_automl_challenger.py (in the main environment) to fold it into models/all_models.joblib")


if __name__ == "__main__":
    main()
