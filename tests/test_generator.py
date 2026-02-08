import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_wine

from risksyn import AIMGenerator, Risk
from risksyn.generator import _auto_categorize, _requires_private_preprocessing


# Simple dataset for fast tests
@pytest.fixture
def simple_df():
    """Small numeric dataset for fast tests."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "a": np.random.uniform(0, 100, 50),
            "b": np.random.uniform(-50, 50, 50),
        }
    )


@pytest.fixture
def simple_domain():
    """Domain for simple_df."""
    return {
        "a": {"lower": 0.0, "upper": 100.0},
        "b": {"lower": -50.0, "upper": 50.0},
    }


# Wine dataset for integration test
@pytest.fixture
def wine_df():
    """Wine dataset."""
    return load_wine(as_frame=True).frame


@pytest.fixture
def wine_domain(wine_df):
    """Domain specification for wine dataset."""
    return {
        col: {"lower": float(wine_df[col].min()), "upper": float(wine_df[col].max())}
        for col in wine_df.columns
    }


# === Fast unit tests ===


def test_generate_basic(simple_df, simple_domain):
    """Basic generation test with simple data."""
    risk = Risk.from_zcdp(0.5)
    gen = AIMGenerator(risk=risk)
    gen.fit(simple_df, domain=simple_domain)
    synth = gen.generate(count=10)

    assert len(synth) == 10
    assert list(synth.columns) == list(simple_df.columns)


@pytest.mark.parametrize("proc_epsilon", [None, 1.0])
def test_raises_when_rho_less_than_proc_rho(simple_df, proc_epsilon):
    """Should raise in fit() when risk.zcdp <= proc_rho and preprocessing is needed."""
    risk = Risk.from_zcdp(0.1)
    gen = AIMGenerator(risk=risk, proc_epsilon=proc_epsilon)
    with pytest.raises(ValueError, match="Insufficient privacy budget"):
        gen.fit(simple_df)  # No domain -> needs preprocessing


def test_warns_when_gen_rho_less_than_proc_rho():
    """Should warn in fit() when gen_rho < proc_rho and preprocessing is needed."""
    # proc_rho for proc_epsilon=1.0 is ~0.368
    # risk.zcdp=0.5 gives gen_rho=0.132 < proc_rho -> warning
    np.random.seed(42)
    df = pd.DataFrame({
        "x": np.random.uniform(-1000, 1000, 500),
        "y": np.random.uniform(-1000, 1000, 500),
    })
    risk = Risk.from_zcdp(0.5)
    gen = AIMGenerator(risk=risk, proc_epsilon=1.0)
    with pytest.warns(
        UserWarning,
        match="Privacy budget for generation .* is smaller than for processing",
    ):
        gen.fit(df)


def test_no_warning_when_domain_provided(simple_df, simple_domain):
    """Should not warn when full domain is provided (no preprocessing needed)."""
    risk = Risk.from_zcdp(0.008)  # Would warn without domain
    gen = AIMGenerator(risk=risk)
    with pytest.warns() as record:
        gen.fit(simple_df, domain=simple_domain)
    user_warnings = [w for w in record if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 0


def test_no_error_with_low_budget_when_domain_provided(simple_df, simple_domain):
    """Should not error with very low budget when full domain is provided."""
    risk = Risk.from_zcdp(0.001)  # Would error without domain
    gen = AIMGenerator(risk=risk)
    gen.fit(simple_df, domain=simple_domain)
    synth = gen.generate(count=5)
    assert len(synth) == 5


def test_generate_before_fit_raises():
    """Should raise if generate() called before fit()."""
    risk = Risk.from_zcdp(0.5)
    gen = AIMGenerator(risk=risk)
    with pytest.raises(RuntimeError, match="Must call fit"):
        gen.generate(count=10)


def test_auto_categorize_binary_int_columns():
    """Low-cardinality int columns should be auto-categorized and cast to str."""
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
    assert "continuous" not in domain  # float, not int
    assert "cat" not in domain  # not numeric


def test_auto_categorize_respects_existing_domain():
    """Auto-categorization should not override user-provided domain."""
    df = pd.DataFrame({"x": [0, 1, 0, 1]})
    user_domain = {"x": {"lower": 0, "upper": 1}}
    _, domain = _auto_categorize(df, user_domain)
    assert domain["x"] == {"lower": 0, "upper": 1}


def test_auto_categorize_skips_high_cardinality():
    """Int columns with >10 unique values should not be auto-categorized."""
    df = pd.DataFrame({"x": list(range(11))})
    _, domain = _auto_categorize(df, None)
    assert "x" not in domain


def test_requires_private_preprocessing_false_for_list_domain():
    """List domain on a numeric column means no preprocessing needed."""
    df = pd.DataFrame({"x": [0, 1, 0, 1]})
    assert not _requires_private_preprocessing(df, {"x": [0, 1]})


def test_binary_int_columns_fit_without_domain():
    """Binary int columns should fit without explicit domain via auto-categorization."""
    np.random.seed(42)
    df = pd.DataFrame({
        "a": np.random.choice([0, 1], 100),
        "b": np.random.choice([0, 1], 100),
        "cat": np.random.choice(["x", "y"], 100),
    })
    risk = Risk.from_advantage(0.25)
    gen = AIMGenerator(risk=risk)
    gen.fit(df)
    synth = gen.generate(count=10)
    assert len(synth) == 10
    assert list(synth.columns) == list(df.columns)


def test_bounds_estimation_failure_raises_value_error():
    """Should raise ValueError (not TypeError) when private bounds estimation fails."""
    np.random.seed(42)
    # High-cardinality float column with tiny budget -> approx_bounds will fail
    df = pd.DataFrame({"x": np.random.uniform(0, 1, 50)})
    risk = Risk.from_zcdp(0.001)
    gen = AIMGenerator(risk=risk, proc_epsilon=0.001)
    with pytest.raises(ValueError, match="Private bounds estimation failed"):
        gen.fit(df)


def test_categorical_only_no_preprocessing():
    """Categorical-only data should not require preprocessing."""
    df = pd.DataFrame({
        "cat1": ["a", "b", "c"] * 10,
        "cat2": ["x", "y", "z"] * 10,
    })
    risk = Risk.from_zcdp(0.001)  # Very low budget
    gen = AIMGenerator(risk=risk)
    gen.fit(df)
    synth = gen.generate(count=5)
    assert len(synth) == 5


# === Serialization tests ===


def test_store_and_load(simple_df, simple_domain, tmp_path):
    """Test storing and loading a fitted generator."""
    risk = Risk.from_zcdp(0.5)
    gen = AIMGenerator(risk=risk)
    gen.fit(simple_df, domain=simple_domain)

    # Store
    gen.store(tmp_path / "generator")

    # Load
    loaded = AIMGenerator.load(tmp_path / "generator")

    # Generate from loaded
    synth = loaded.generate(count=10)
    assert len(synth) == 10
    assert list(synth.columns) == list(simple_df.columns)


def test_store_before_fit_raises(tmp_path):
    """Should raise if store() called before fit()."""
    risk = Risk.from_zcdp(0.5)
    gen = AIMGenerator(risk=risk)
    with pytest.raises(RuntimeError, match="Must call fit"):
        gen.store(tmp_path / "generator")


# === Slower integration test with wine ===


@pytest.mark.parametrize("advantage", [0.1])
def test_wine_integration(advantage: float, wine_df, wine_domain):
    """Integration test with wine dataset."""
    risk = Risk.from_advantage(advantage)
    gen = AIMGenerator(risk=risk)
    gen.fit(wine_df, domain=wine_domain)
    synth = gen.generate(count=10)

    assert len(synth) == 10
    assert list(synth.columns) == list(wine_df.columns)
