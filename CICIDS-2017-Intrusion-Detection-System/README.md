# CICIDS-2017 Intrusion Detection System

A solo-developed machine-learning application for analysing network-flow CSV files and detecting benign or malicious traffic. The repository combines an initial Streamlit application scaffold with the original CICIDS-2017 research scripts and EDA outputs.

## Current application status

**Milestone 1: CSV upload and data-quality validation.** The Streamlit app currently inspects an uploaded CSV and reports dimensions, numeric features, duplicate rows, missing values, infinite values, and non-numeric columns. **It does not make attack predictions yet.**

Prediction will be added only after the exact training-time feature schema and preprocessing artifacts are recovered and tested.

## Run locally (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Repository layout

```text
app.py                         # Streamlit UI
src/ids_app/                   # Reusable application logic
tests/                         # Unit tests
data/raw/                      # Local CICIDS source files (not committed)
data/processed/                # Generated data (not committed)
models/                        # Selected model artifacts (not committed by default)
docs/eda/                      # Selected EDA figures and reports
research/legacy_scripts/       # Original research scripts, preserved for reference
research/results/               # Previously saved model-comparison summaries
```

## Research scripts

The scripts under `research/legacy_scripts/` are preserved from the supplied research archive. They are not yet refactored into the application pipeline; some may contain machine-specific paths or require additional dependencies. Treat their outputs as historical until experiments are reproduced.

## Data and model policy

- Do not commit raw CICIDS-2017 CSV datasets or large generated datasets to ordinary Git history.
- Do not commit `.pkl`/joblib model artifacts until the chosen artifact is reviewed and its provenance, size, dependencies, feature order, and matching preprocessing are documented.
- Pickle/joblib files can execute code when loaded. Only load artifacts you created and trust.
- Keep training and test data separate. Fit transformations and resampling on training data only.

## Next milestones

1. Recover the exact feature schema and preprocessing used by one selected trained model.
2. Build and test a single inference pipeline that applies transformations in the same order as training.
3. Validate predictions on held-out data and reproduce evaluation metrics.
4. Add prediction summaries, per-class metrics, and downloadable results to the dashboard.
5. Consider live packet capture only after offline CSV detection is reliable.
