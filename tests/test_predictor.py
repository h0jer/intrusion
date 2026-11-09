import numpy as np
import pandas as pd
import pytest

from intrusion.predictor import EXAMPLE_ROW, IntrusionDetector
from intrusion.preprocessing import FEATURE_NAMES


@pytest.fixture(scope="module")
def detector():
    return IntrusionDetector()


def test_model_loads_with_expected_schema(detector):
    assert detector.classes == ["benign", "malicious", "outlier"]
    assert detector.feature_names == FEATURE_NAMES
    assert len(detector.pipeline.named_steps["model"].estimators_) == 100


def test_valid_prediction(detector):
    # Row 0 of the dataset; the notebook shows its label as "benign".
    res = detector.predict_one(EXAMPLE_ROW)
    assert res["prediction"] == "benign"
    assert set(res["probabilities"]) == set(detector.classes)
    assert sum(res["probabilities"].values()) == pytest.approx(1.0)


def test_missing_ports_are_imputed_with_minus_one(detector):
    with_nan = dict(EXAMPLE_ROW, dest_port=np.nan, src_port=np.nan)
    with_minus_one = dict(EXAMPLE_ROW, dest_port=-1, src_port=-1)
    a = detector.predict(pd.DataFrame([with_nan]))
    b = detector.predict(pd.DataFrame([with_minus_one]))
    pd.testing.assert_frame_equal(a, b)


def test_column_order_and_extra_label_column_are_handled(detector):
    df = pd.DataFrame([EXAMPLE_ROW])
    shuffled = df[list(reversed(FEATURE_NAMES))].assign(label="benign")
    pd.testing.assert_frame_equal(detector.predict(df), detector.predict(shuffled))


def test_batch_prediction_shape(detector):
    df = pd.DataFrame([EXAMPLE_ROW] * 5)
    out = detector.predict(df)
    assert len(out) == 5
    assert list(out.columns) == ["prediction"] + [f"proba_{c}" for c in detector.classes]


def test_missing_column_raises(detector):
    row = dict(EXAMPLE_ROW)
    row.pop("entropy")
    with pytest.raises(ValueError, match="Missing required feature"):
        detector.predict(pd.DataFrame([row]))


def test_null_in_non_port_column_raises(detector):
    with pytest.raises(ValueError, match="Null values"):
        detector.predict(pd.DataFrame([dict(EXAMPLE_ROW, entropy=np.nan)]))


def test_non_numeric_value_raises(detector):
    with pytest.raises(ValueError, match="numeric"):
        detector.predict(pd.DataFrame([dict(EXAMPLE_ROW, bytes_in="abc")]))


def test_model_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        IntrusionDetector(tmp_path / "nope.joblib")
