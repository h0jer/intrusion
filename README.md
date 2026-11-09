# 🛡️ Network Intrusion Detection — Streamlit App

A small Streamlit app that classifies network flow records as **benign**, **malicious** or **outlier**
using a Random Forest trained on the [LuFlow](https://www.kaggle.com/datasets/mryanm/luflow-network-intrusion-detection-data-set)
network intrusion dataset (file `2020/10/2020.10.01/2020.10.01.csv`).

## Problem statement

Given the statistics of a network flow (bytes, packets, ports, entropy, timing, …), decide whether the flow is
normal traffic or part of an attack. The app supports a single-flow form and batch prediction from a CSV file.

## Model and approach

All details below come from the training notebook (`notebooks/intrusion-detection-acc99.ipynb`):

- **Data:** 1,081,952 flows × 16 columns. Stratified 80/20 train/test split (`random_state=42`).
- **Features (15):** `avg_ipt, bytes_in, bytes_out, dest_ip, dest_port, entropy, num_pkts_out, num_pkts_in, proto, src_ip, src_port, time_end, time_start, total_entropy, duration` (all numeric; IPs are already anonymized integers).
- **Preprocessing:** missing `src_port` / `dest_port` values are filled with `-1`. Nothing else (no scaling, no encoding).
- **Model:** `RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')`.
- **Labels:** `LabelEncoder` → `0 = benign`, `1 = malicious`, `2 = outlier`.
- **Reported results in the notebook:** test accuracy **0.99895** (train accuracy 0.99998).

**Preprocessing is bundled with the model.** `models/intrusion_pipeline.joblib` contains a scikit-learn `Pipeline`
(`FeatureCleaner` → `RandomForestClassifier`) plus the label encoder, so raw data (with empty ports, any column order,
an optional `label` column) is handled automatically. The model itself was **not retrained**: the fitted model exported
from the notebook is reused as-is.

> ⚠️ **Caveat:** the model uses `time_start`, `time_end` and the IP columns as features. With a random split this
> can make the 99.9% accuracy optimistic (the model can memorize time windows/addresses), so expect lower
> performance on traffic from other days or networks. The model was kept unchanged on purpose.

## Technologies

Python 3.10 · Streamlit · scikit-learn 1.2.2 · pandas · NumPy · pytest

## Project structure

```text
intrusion-detection-app/
├── app.py                          # Streamlit UI
├── intrusion/
│   ├── preprocessing.py            # FeatureCleaner (notebook preprocessing)
│   └── predictor.py                # loads the pipeline, runs predictions
├── models/
│   └── intrusion_pipeline.joblib   # preprocessing + trained model + label encoder (~18 MB)
├── notebooks/
│   └── intrusion-detection-acc99.ipynb
├── scripts/
│   └── export_pipeline.py          # wraps the notebook's model into the pipeline (no retraining)
├── tests/
│   ├── test_predictor.py
│   └── test_app.py
├── .streamlit/config.toml
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Installation

The model was pickled with **scikit-learn 1.2.2**, which supports **Python 3.8–3.11 only** (not 3.12+) and NumPy < 2.
Use Python 3.10 or 3.11.

```bash
git clone <your-repo-url>
cd intrusion-detection-app

python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 (health check: http://localhost:8501/_stcore/health).

## Using the app

**Single flow tab** — the form is pre-filled with the first row of the dataset (labelled `benign` in the notebook).
Click **Predict** to see the class and the per-class probabilities. Port `-1` means "missing port".

**Batch (CSV) tab** — upload a CSV containing the 15 feature columns. Optional extras:
- a `label` column (strings `benign` / `malicious` / `outlier`) → the app shows accuracy against it;
- empty `src_port` / `dest_port` cells → imputed with `-1` automatically.

You can try it with the original dataset file `2020.10.01.csv` (not included in this repo because of its size).
Results can be downloaded as CSV.

Example row (CSV header + one flow):

```csv
avg_ipt,bytes_in,bytes_out,dest_ip,dest_port,entropy,num_pkts_out,num_pkts_in,proto,src_ip,src_port,time_end,time_start,total_entropy,duration
0.0,0,14480,786,9200.0,2.116894,10,0,6,786,43432.0,1601521393115058,1601521393114933,30652.621,0.000125
```

Expected prediction: `benign`.

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

## Re-exporting the model (optional)

If you re-train in the notebook and save `{"model": rf_model, "label_encoder": encoder, "feature_names": list(x_train.columns)}`
with `joblib.dump(...)`, rebuild the bundled pipeline with:

```bash
python scripts/export_pipeline.py path/to/rf_model.joblib
```

## Security note

`.joblib` files are pickle-based: only load model files you trust.
