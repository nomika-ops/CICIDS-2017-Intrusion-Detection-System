from __future__ import annotations

import numpy as np
import pandas as pd


def inspect_dataframe(data: pd.DataFrame) -> dict:
    """Return basic, non-destructive quality checks for an uploaded flow CSV."""
    frame = data.copy()
    frame.columns = [str(column).strip() for column in frame.columns]

    numeric = frame.select_dtypes(include=[np.number])
    infinite_values = int(np.isinf(numeric.to_numpy()).sum()) if not numeric.empty else 0
    non_numeric = [
        str(column)
        for column in frame.columns
        if column != "Label" and not pd.api.types.is_numeric_dtype(frame[column])
    ]

    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "numeric_features": int(len(numeric.columns) - (1 if "Label" in numeric.columns else 0)),
        "duplicate_rows": int(frame.duplicated().sum()),
        "blank_column_names": int(sum(not name for name in frame.columns)),
        "non_numeric_columns": non_numeric,
        "missing_values": int(frame.isna().sum().sum()),
        "infinite_values": infinite_values,
        "has_label": "Label" in frame.columns,
    }
