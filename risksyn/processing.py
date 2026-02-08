import warnings
from typing import Optional

import numpy as np
import pandas as pd

# Numeric columns with at most this many unique values are auto-treated as categorical
_AUTO_CATEGORICAL_THRESHOLD = 10


def _auto_categorize(
    data: pd.DataFrame, domain: Optional[dict]
) -> tuple[pd.DataFrame, dict]:
    """Auto-detect low-cardinality numeric columns and treat as categorical.

    Numeric columns with <= _AUTO_CATEGORICAL_THRESHOLD unique values are
    cast to string dtype and given categorical domains to avoid private
    bounds estimation.

    Returns a (possibly modified) copy of data and the augmented domain.
    """
    domain = dict(domain) if domain else {}
    cols_to_cast = []
    for col, series in data.items():
        if col in domain:
            continue
        if series.dtype.kind in "ui":  # uint, int only
            if series.nunique() <= _AUTO_CATEGORICAL_THRESHOLD:
                values = sorted(str(v) for v in series.unique())
                domain[col] = values
                cols_to_cast.append(col)
                warnings.warn(
                    f"PrivacyLeakage: No categorical domain provided for "
                    f"numeric column '{col}' — treating as categorical with "
                    f"values {values}. Provide explicit domain to avoid this.",
                    UserWarning,
                    stacklevel=3,
                )
    if cols_to_cast:
        data = data.copy()
        data[cols_to_cast] = data[cols_to_cast].astype(str)
    return data, domain


def _numeric_cols_without_bounds(data: pd.DataFrame, domain: Optional[dict]) -> list[str]:
    """Return names of numeric columns that lack explicit domain bounds."""
    cols = []
    for col, series in data.items():
        if series.dtype.kind not in "Mmfui":
            continue
        if domain is None:
            cols.append(col)
            continue
        col_domain = domain.get(col)
        if isinstance(col_domain, list):
            continue  # categorical domain
        if isinstance(col_domain, dict) and col_domain.get("lower") is not None and col_domain.get("upper") is not None:
            continue  # bounds provided
        cols.append(col)
    return cols


def _detect_float_precision(series: pd.Series) -> int:
    """Detect the number of decimal places used in a float series."""
    vals = series.dropna()
    if len(vals) == 0:
        return 6
    max_decimals = 0
    for v in vals:
        s = f"{v:g}"  # compact repr, strips trailing zeros
        if "." in s:
            max_decimals = max(max_decimals, len(s.split(".")[1]))
    return max_decimals


def _detect_column_schema(data: pd.DataFrame) -> dict:
    """Record original dtype and float precision for each column."""
    schema = {}
    for col, series in data.items():
        entry = {"dtype": series.dtype}
        if series.dtype.kind == "f":
            entry["precision"] = _detect_float_precision(series)
        schema[col] = entry
    return schema


def _apply_column_schema(synth: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Cast synthetic data back to original types and precision."""
    for col in synth.columns:
        if col not in schema:
            continue
        entry = schema[col]
        dtype = entry["dtype"]

        if "precision" in entry:
            synth[col] = np.round(synth[col].astype(float), entry["precision"])

        if dtype.kind in "ui":
            synth[col] = synth[col].astype(dtype)
        elif dtype.kind == "f":
            synth[col] = synth[col].astype(dtype)
    return synth
