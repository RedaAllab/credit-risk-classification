"""Small scikit-learn-compatible wrapper needed to unwrap FLAML's "svc"
AutoML estimator (see scripts/train_automl.py) to a plain, flaml-free object
that models/all_models.joblib and app/streamlit_app.py can use without flaml
installed at inference time.
"""


class LinearSVCProbaWrapper:
    """Adds predict_proba to a fitted sklearn.svm.LinearSVC.

    LinearSVC has no native predict_proba. FLAML's own SVCEstimator falls
    back to the Platt-scaling-style approximation scikit-learn exposes
    internally as LinearSVC._predict_proba_lr; this wrapper calls the same
    method so unwrapped predictions match what FLAML evaluated during search.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model._predict_proba_lr(X)
