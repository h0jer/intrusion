"""Load the exported pipeline once and run predictions on raw data."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .preprocessing import FEATURE_NAMES

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "intrusion_pipeline.joblib"


class IntrusionDetector:
    """Wraps the saved (preprocessing + RandomForest) pipeline and label encoder."""

    def __init__(self, path: Path | str = MODEL_PATH):
        bundle = joblib.load(path)  # only load files you trust (pickle-based)
        self.pipeline = bundle["pipeline"]
        self.label_encoder = bundle["label_encoder"]
        self.feature_names: list[str] = list(bundle["feature_names"])
        self.classes: list[str] = list(self.label_encoder.classes_)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a frame with ``prediction`` and one ``proba_<class>`` column per class.

        ``df`` is raw data (same columns as the training CSV, ``label`` optional).
        Preprocessing is applied by the pipeline itself.
        """
        codes = self.pipeline.predict(df)
        proba = self.pipeline.predict_proba(df)
        out = pd.DataFrame(
            proba, columns=[f"proba_{c}" for c in self.classes], index=df.index
        )
        out.insert(0, "prediction", self.label_encoder.inverse_transform(codes.astype(int)))
        return out

    def predict_one(self, row: dict) -> dict:
        res = self.predict(pd.DataFrame([row])).iloc[0]
        return {
            "prediction": str(res["prediction"]),
            "probabilities": {c: float(res[f"proba_{c}"]) for c in self.classes},
        }


# First row of the LuFlow 2020.10.01 file as displayed in the notebook (label: benign)
EXAMPLE_ROW = {
    "avg_ipt": 0.0,
    "bytes_in": 0,
    "bytes_out": 14480,
    "dest_ip": 786,
    "dest_port": 9200.0,
    "entropy": 2.116894,
    "num_pkts_out": 10,
    "num_pkts_in": 0,
    "proto": 6,
    "src_ip": 786,
    "src_port": 43432.0,
    "time_end": 1601521393115058,
    "time_start": 1601521393114933,
    "total_entropy": 30652.621,
    "duration": 0.000125,
}
assert list(EXAMPLE_ROW) == FEATURE_NAMES
