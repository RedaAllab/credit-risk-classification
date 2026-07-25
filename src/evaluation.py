"""Model evaluation helpers shared across the four models compared in
notebooks/credit_risk_modeling_en.ipynb (threshold search, confusion
matrix + ROC curve plotting), extracted to avoid repeating the same
few lines of code once per model.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, fbeta_score


def find_best_f2_threshold(y_true, y_probs, n_thresholds: int = 200):
    """Find the classification threshold that maximizes the F2-score."""
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    scores = [fbeta_score(y_true, (y_probs >= t).astype(int), beta=2) for t in thresholds]
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]


def plot_confusion_and_roc(y_true, y_pred, y_proba, model_name: str, cmap="Blues", line_color=None):
    """Side-by-side confusion matrix and ROC curve for one model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=cmap, ax=axes[0])
    axes[0].set_title(f"Confusion matrix: {model_name}")

    RocCurveDisplay.from_predictions(y_true, y_proba, ax=axes[1], name=model_name, color=line_color)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#c3c2b7")
    axes[1].set_title(f"ROC curve: {model_name}")

    fig.tight_layout()
    return fig, axes
