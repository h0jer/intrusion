"""Streamlit app: network intrusion detection on LuFlow flow records."""
import pandas as pd
import streamlit as st

from intrusion.predictor import EXAMPLE_ROW, IntrusionDetector
from intrusion.preprocessing import FEATURE_NAMES, PORT_COLUMNS

st.set_page_config(page_title="Network Intrusion Detection", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_detector() -> IntrusionDetector:
    """Load the pipeline once per server process (not on every interaction)."""
    return IntrusionDetector()


try:
    detector = get_detector()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load the model: {exc}")
    st.stop()

st.title("🛡️ Network Intrusion Detection")
st.caption(
    "Random Forest trained on the LuFlow dataset (2020-10-01). "
    f"Classes: {', '.join(detector.classes)}. "
    "Preprocessing (missing ports → -1) is applied automatically."
)

tab_single, tab_batch = st.tabs(["Single flow", "Batch (CSV)"])

# ----------------------------------------------------------------------------
# Single flow
# ----------------------------------------------------------------------------
with tab_single:
    st.write("Enter the flow features (defaults: first row of the dataset, labelled *benign* in the notebook).")
    with st.form("single_flow"):
        cols = st.columns(3)
        values = {}
        for i, name in enumerate(FEATURE_NAMES):
            default = EXAMPLE_ROW[name]
            with cols[i % 3]:
                if name in PORT_COLUMNS:
                    values[name] = st.number_input(
                        name, value=float(default), min_value=-1.0, max_value=65535.0, step=1.0,
                        help="-1 means missing port (same imputation as in training).",
                    )
                elif isinstance(default, int):
                    values[name] = st.number_input(name, value=int(default), min_value=0, step=1)
                else:
                    values[name] = st.number_input(name, value=float(default), min_value=0.0, format="%.6f")
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        try:
            result = detector.predict_one(values)
        except ValueError as exc:
            st.error(f"Invalid input: {exc}")
        else:
            if result["prediction"] == "benign":
                st.success(f"Prediction: **{result['prediction']}**")
            else:
                st.warning(f"Prediction: **{result['prediction']}**")
            st.bar_chart(pd.Series(result["probabilities"], name="probability"))

# ----------------------------------------------------------------------------
# Batch
# ----------------------------------------------------------------------------
with tab_batch:
    st.write(
        "Upload a CSV with the same columns as the training data "
        f"(`{'`, `'.join(FEATURE_NAMES)}`). A `label` column is optional; "
        f"empty `{'`/`'.join(PORT_COLUMNS)}` values are allowed."
    )
    upload = st.file_uploader("CSV file", type="csv")
    if upload is not None:
        try:
            data = pd.read_csv(upload)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the CSV: {exc}")
        else:
            try:
                with st.spinner(f"Predicting {len(data):,} rows..."):
                    preds = detector.predict(data)
            except ValueError as exc:
                st.error(f"Invalid data: {exc}")
            else:
                st.subheader("Predicted class counts")
                st.bar_chart(preds["prediction"].value_counts())

                if "label" in data.columns:
                    acc = (data["label"].astype(str).to_numpy() == preds["prediction"].to_numpy()).mean()
                    st.metric("Accuracy vs. provided `label` column", f"{acc:.4%}")

                result = pd.concat([data, preds], axis=1)
                st.dataframe(result.head(1000), use_container_width=True)
                st.caption("Showing the first 1,000 rows." if len(result) > 1000 else f"{len(result):,} rows.")
                st.download_button(
                    "Download predictions (CSV)",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
