import matplotlib
import numpy as np

matplotlib.use("Agg")

from src.evaluation import find_best_f2_threshold, plot_confusion_and_roc


def test_find_best_f2_threshold_recovers_a_clean_separation():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 500 + [1] * 500)
    # scores cleanly separated around 0.5, so the optimal threshold should sit near there
    y_probs = np.concatenate([
        rng.uniform(0.0, 0.4, 500),
        rng.uniform(0.6, 1.0, 500),
    ])

    threshold, f2 = find_best_f2_threshold(y_true, y_probs)

    assert 0.3 < threshold < 0.7
    assert f2 > 0.95


def test_find_best_f2_threshold_favors_recall_over_precision():
    # an imperfect classifier where lowering the threshold trades precision for recall;
    # F2 (beta=2) weights recall more, so the optimum should sit below 0.5
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_probs = np.array([0.9, 0.6, 0.4, 0.2, 0.8, 0.55, 0.3, 0.2, 0.1, 0.05])

    threshold, _ = find_best_f2_threshold(y_true, y_probs, n_thresholds=100)

    assert threshold <= 0.5


def test_plot_confusion_and_roc_returns_two_axes():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 100)
    y_proba = rng.uniform(0, 1, 100)
    y_pred = (y_proba >= 0.5).astype(int)

    fig, axes = plot_confusion_and_roc(y_true, y_pred, y_proba, "Test Model")

    assert len(axes) == 2
    assert "Test Model" in axes[0].get_title()
    assert "Test Model" in axes[1].get_title()
