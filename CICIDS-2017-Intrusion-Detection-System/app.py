from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ids_app.validation import inspect_dataframe  # noqa: E402

st.set_page_config(page_title="CICIDS Intrusion Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ CICIDS-2017 Intrusion Detection System")
st.caption("Milestone 1 · Dataset upload and validation")

st.info(
    "This first milestone validates uploaded network-flow CSV files. "
    "It does not yet classify traffic: the existing trained models are missing "
    "their matching preprocessing artifacts and feature schema."
)

uploaded = st.file_uploader("Upload a CICIDS-style network-flow CSV", type=["csv"])
if uploaded is None:
    st.markdown("### What this application will do")
    st.markdown(
        "1. Validate network-flow records.\n"
        "2. Apply the exact preprocessing used during training.\n"
        "3. Predict benign traffic or an attack category.\n"
        "4. Display alerts, confidence, and traffic statistics."
    )
    st.stop()

try:
    frame = pd.read_csv(uploaded, low_memory=False)
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

report = inspect_dataframe(frame)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{report['rows']:,}")
c2.metric("Columns", f"{report['columns']:,}")
c3.metric("Numeric features", f"{report['numeric_features']:,}")
c4.metric("Duplicate rows", f"{report['duplicate_rows']:,}")

st.subheader("Validation")
checks = [
    ("Column names", "Passed" if report["blank_column_names"] == 0 else f"{report['blank_column_names']} blank names"),
    ("Non-numeric columns", ", ".join(report["non_numeric_columns"]) if report["non_numeric_columns"] else "None"),
    ("Missing values", f"{report['missing_values']:,}"),
    ("Infinite values", f"{report['infinite_values']:,}"),
    ("Target column", "Label column found" if report["has_label"] else "No Label column (prediction input may omit it)"),
]
for label, value in checks:
    st.write(f"**{label}:** {value}")

if report["missing_values"] or report["infinite_values"] or report["non_numeric_columns"]:
    st.warning("The file needs preprocessing or correction before model inference.")
else:
    st.success("Basic data checks passed. Model compatibility has not yet been verified.")

st.subheader("Preview")
st.dataframe(frame.head(20), use_container_width=True)

with st.expander("Column names"):
    st.code("\n".join(map(str, frame.columns)))

st.download_button(
    "Download validation report",
    data=pd.Series(report).to_json(indent=2),
    file_name="validation_report.json",
    mime="application/json",
)
