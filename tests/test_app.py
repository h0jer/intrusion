"""Smoke tests that run the real Streamlit script headlessly."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 120


def _run():
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def test_app_starts_without_errors():
    at = _run()
    assert not at.exception
    assert "Network Intrusion Detection" in at.title[0].value


def test_single_flow_prediction_shows_benign():
    at = _run()
    at.button[0].click().run()
    assert not at.exception
    assert any("benign" in s.value for s in at.success)


def test_changed_input_still_predicts():
    at = _run()
    at.number_input[0].set_value(5.0)  # avg_ipt
    at.button[0].click().run()
    assert not at.exception
    assert len(at.success) + len(at.warning) == 1
