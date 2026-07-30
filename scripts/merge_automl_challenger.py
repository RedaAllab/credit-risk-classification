"""Fold the AutoML challenger (models/automl_challenger.joblib, produced by
scripts/train_automl.py) into the shared model artifact consumed by
app/streamlit_app.py (models/all_models.joblib), under the key
"AutoML (FLAML)" so it appears in the champion-vs-challengers comparison.

Run this in the main requirements.txt environment, not the requirements-automl.txt
one: models/all_models.joblib already holds an XGBoost model pickled under this
project's pinned xgboost==3.2.0, and re-saving it from an environment with a
different xgboost version (as scripts/train_automl.py's environment
necessarily has - see that script's docstring) risks silently corrupting it.

Usage: python scripts/merge_automl_challenger.py
"""

from pathlib import Path

import joblib

# Kept in sync with scripts/train_automl.py's MODEL_NAME and src/style.py's
# MODEL_COLORS key; not imported from train_automl.py directly since that
# module imports flaml, which this script's (main) environment doesn't have.
MODEL_NAME = "AutoML (FLAML)"


def main():
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    challenger_path = models_dir / "automl_challenger.joblib"
    artifact_path = models_dir / "all_models.joblib"

    if not challenger_path.exists():
        raise SystemExit(
            f"{challenger_path} not found - run scripts/train_automl.py first "
            "(in the requirements-automl.txt environment)."
        )

    challenger = joblib.load(challenger_path)
    artifact = joblib.load(artifact_path)
    artifact["models"][MODEL_NAME] = challenger
    joblib.dump(artifact, artifact_path, compress=3)
    print(f"Updated {artifact_path} with '{MODEL_NAME}' ({artifact_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
