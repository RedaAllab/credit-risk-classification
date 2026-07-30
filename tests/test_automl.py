import numpy as np
import pandas as pd
import pytest

flaml = pytest.importorskip(
    "flaml", reason="flaml is only installed via requirements-automl.txt, not requirements.txt"
)

from scripts.train_automl import train  # noqa: E402


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

    fitted_model = automl.model.estimator
    proba = fitted_model.predict_proba(X)[:, 1]

    assert ((proba >= 0) & (proba <= 1)).all()
    assert len(proba) == len(y)
