import numpy as np
import pandas as pd
import pytest

flaml = pytest.importorskip(
    "flaml", reason="flaml is only installed via requirements-automl.txt, not requirements.txt"
)

from flaml import AutoML  # noqa: E402

from scripts.train_automl import _unwrap, train  # noqa: E402


@pytest.fixture
def X_y_train():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({
        "person_age": rng.integers(20, 60, n),
        "person_income": rng.integers(20000, 100000, n),
        "loan_amnt": rng.integers(500, 20000, n),
        "loan_percent_income": rng.uniform(0.01, 0.6, n),
    })
    y = pd.Series(rng.integers(0, 2, n))
    return X, y


def test_train_runs_within_budget_and_produces_valid_probabilities(X_y_train):
    X, y = X_y_train
    # a few seconds is enough to confirm the pipeline runs end to end; real runs
    # use scripts/train_automl.py's much larger default --time-budget
    automl = train(X, y, time_budget=5)

    fitted_model = _unwrap(automl, X)
    proba = fitted_model.predict_proba(X)[:, 1]

    assert ((proba >= 0) & (proba <= 1)).all()
    assert len(proba) == len(y)


@pytest.mark.parametrize("estimator", ["svc", "sgd"])
def test_unwrap_matches_flaml_wrapper_for_estimators_needing_special_casing(X_y_train, estimator):
    # Regression test: automl.model.estimator alone is broken for these two -
    # "svc" (LinearSVC) has no native predict_proba at all, and "sgd" silently
    # drops FLAML's per-row Normalizer() preprocessing, changing predictions.
    # _unwrap must reproduce what FLAML's own wrapper predicts.
    X, y = X_y_train
    automl = AutoML()
    automl.fit(
        X_train=X, y_train=y, task="classification", metric="f1",
        estimator_list=[estimator], time_budget=5, eval_method="cv",
        n_splits=3, seed=42, verbose=0,
    )

    fitted_model = _unwrap(automl, X)
    proba = fitted_model.predict_proba(X)
    expected_proba = automl.model.predict_proba(X)

    np.testing.assert_allclose(proba, expected_proba)
