"""Wrap the notebook's trained model in a single preprocessing+model pipeline.

Input : the joblib file exported from the notebook (dict with keys
        'model', 'label_encoder', 'feature_names').
Output: models/intrusion_pipeline.joblib

The RandomForest is NOT retrained - the fitted model object is reused as-is.

Usage (from the project root, with scikit-learn==1.2.2 installed):
    python scripts/export_pipeline.py path/to/rf_model.joblib
"""
import sys
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from intrusion.preprocessing import FEATURE_NAMES, FeatureCleaner  # noqa: E402


def main(src: str) -> None:
    raw = joblib.load(src)
    model, encoder = raw["model"], raw["label_encoder"]

    if list(raw["feature_names"]) != FEATURE_NAMES:
        raise SystemExit("Feature names in the model file differ from the expected training columns.")
    if list(model.feature_names_in_) != FEATURE_NAMES:
        raise SystemExit("Model feature_names_in_ differ from the expected training columns.")

    cleaner = FeatureCleaner(feature_names=FEATURE_NAMES).fit(None)
    pipeline = Pipeline([("cleaner", cleaner), ("model", model)])

    out = ROOT / "models" / "intrusion_pipeline.joblib"
    out.parent.mkdir(exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "label_encoder": encoder,
            "feature_names": FEATURE_NAMES,
        },
        out,
        compress=3,
    )
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB); classes = {list(encoder.classes_)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
