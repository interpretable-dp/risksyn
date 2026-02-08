import numpy as np
import pandas as pd
import pytest

from risksyn.processing import (
    _apply_column_schema,
    _auto_categorize,
    _detect_column_schema,
    _detect_float_precision,
    _numeric_cols_without_bounds,
)


# === _detect_float_precision ===


def test_detect_float_precision_two_decimals():
    s = pd.Series([1.23, 4.56, 7.89])
    assert _detect_float_precision(s) == 2


def test_detect_float_precision_integers_as_float():
    s = pd.Series([1.0, 2.0, 3.0])
    assert _detect_float_precision(s) == 0


def test_detect_float_precision_mixed():
    s = pd.Series([1.1, 2.34, 3.0])
    assert _detect_float_precision(s) == 2


def test_detect_float_precision_high_precision():
    # :g uses 6 significant digits, so 1.123456 -> "1.12346" -> 5 decimals
    s = pd.Series([1.123456])
    assert _detect_float_precision(s) == 5


def test_detect_float_precision_empty():
    s = pd.Series([], dtype=float)
    assert _detect_float_precision(s) == 6


def test_detect_float_precision_with_nans():
    s = pd.Series([1.23, np.nan, 4.5])
    assert _detect_float_precision(s) == 2


# === _detect_column_schema ===


def test_detect_column_schema_records_dtypes():
    df = pd.DataFrame({"a": [1, 2], "b": [1.5, 2.5], "c": ["x", "y"]})
    schema = _detect_column_schema(df)
    assert schema["a"]["dtype"] == df["a"].dtype
    assert schema["b"]["dtype"] == df["b"].dtype
    assert schema["c"]["dtype"] == df["c"].dtype


def test_detect_column_schema_records_float_precision():
    df = pd.DataFrame({"a": [1.23, 4.56], "b": [1, 2]})
    schema = _detect_column_schema(df)
    assert schema["a"]["precision"] == 2
    assert "precision" not in schema["b"]


# === _apply_column_schema ===


def test_apply_column_schema_restores_int_dtype():
    original = pd.DataFrame({"a": [1, 2, 3]})
    schema = _detect_column_schema(original)
    # Simulate synthetic output as float (common from generation)
    synth = pd.DataFrame({"a": [1.7, 2.3, 3.9]})
    result = _apply_column_schema(synth, schema)
    assert result["a"].dtype == original["a"].dtype
    assert list(result["a"]) == [1, 2, 3]  # truncated to int


def test_apply_column_schema_rounds_float_precision():
    original = pd.DataFrame({"a": [1.23, 4.56]})
    schema = _detect_column_schema(original)
    synth = pd.DataFrame({"a": [1.23456, 4.56789]})
    result = _apply_column_schema(synth, schema)
    assert list(result["a"]) == [1.23, 4.57]


def test_apply_column_schema_preserves_string_columns():
    original = pd.DataFrame({"a": [1.5, 2.5], "b": ["x", "y"]})
    schema = _detect_column_schema(original)
    synth = pd.DataFrame({"a": [1.555, 2.444], "b": ["x", "y"]})
    result = _apply_column_schema(synth, schema)
    assert result["b"].dtype == object
    assert list(result["b"]) == ["x", "y"]


def test_apply_column_schema_handles_unknown_columns():
    """Columns in synth not in schema should be left alone."""
    schema = _detect_column_schema(pd.DataFrame({"a": [1.0]}))
    synth = pd.DataFrame({"a": [1.555], "extra": [99]})
    result = _apply_column_schema(synth, schema)
    assert "extra" in result.columns


# === _auto_categorize ===


def test_auto_categorize_binary_int_columns():
    df = pd.DataFrame({
        "binary": [0, 1, 0, 1, 1],
        "ternary": [0, 1, 2, 0, 1],
        "continuous": np.random.uniform(0, 100, 5),
        "cat": ["a", "b", "c", "a", "b"],
    })
    out_data, domain = _auto_categorize(df, None)
    assert domain["binary"] == ["0", "1"]
    assert domain["ternary"] == ["0", "1", "2"]
    assert out_data["binary"].dtype == object
    assert out_data["ternary"].dtype == object
    assert "continuous" not in domain
    assert "cat" not in domain


def test_auto_categorize_respects_existing_domain():
    df = pd.DataFrame({"x": [0, 1, 0, 1]})
    user_domain = {"x": {"lower": 0, "upper": 1}}
    _, domain = _auto_categorize(df, user_domain)
    assert domain["x"] == {"lower": 0, "upper": 1}


def test_auto_categorize_skips_high_cardinality():
    df = pd.DataFrame({"x": list(range(11))})
    _, domain = _auto_categorize(df, None)
    assert "x" not in domain


def test_auto_categorize_does_not_modify_original():
    df = pd.DataFrame({"x": [0, 1, 0, 1]})
    out_data, _ = _auto_categorize(df, None)
    assert df["x"].dtype.kind == "i"  # original unchanged
    assert out_data["x"].dtype == object


# === _numeric_cols_without_bounds ===


def test_numeric_cols_without_bounds_no_domain():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3, 4]})
    assert set(_numeric_cols_without_bounds(df, None)) == {"x", "y"}


def test_numeric_cols_without_bounds_with_bounds():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    assert _numeric_cols_without_bounds(df, {"x": {"lower": 0, "upper": 10}}) == []


def test_numeric_cols_without_bounds_skips_list_domain():
    df = pd.DataFrame({"x": [0, 1, 0, 1]})
    assert _numeric_cols_without_bounds(df, {"x": [0, 1]}) == []


def test_numeric_cols_without_bounds_skips_string_columns():
    df = pd.DataFrame({"x": ["a", "b"], "y": [1.0, 2.0]})
    assert _numeric_cols_without_bounds(df, None) == ["y"]
