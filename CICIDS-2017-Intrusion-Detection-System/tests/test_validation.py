import pandas as pd

from ids_app.validation import inspect_dataframe


def test_inspect_dataframe_reports_basic_quality():
    frame = pd.DataFrame({
        " Flow Duration ": [1, 2, 1],
        "Total Fwd Packets": [3.0, float("inf"), 3.0],
        "Label": ["BENIGN", "DoS", "BENIGN"],
    })
    report = inspect_dataframe(frame)
    assert report["rows"] == 3
    assert report["columns"] == 3
    assert report["duplicate_rows"] == 1
    assert report["infinite_values"] == 1
    assert report["has_label"] is True
    assert report["non_numeric_columns"] == []
