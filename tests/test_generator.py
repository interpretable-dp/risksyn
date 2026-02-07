import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_wine

from risksyn import Generator, Risk


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
    risk = Risk.from_advantage(0.2)
    gen = Generator(risk=risk, model="mst")
    gen.fit(simple_df, domain=simple_domain)
    synth = gen.generate(count=10)

    assert len(synth) == 10
    assert list(synth.columns) == list(simple_df.columns)


@pytest.mark.parametrize("proc_epsilon", [None, 1.0])
def test_raises_when_rho_less_than_proc_rho(simple_df, proc_epsilon):
    """Should raise in fit() when risk.zcdp <= proc_rho and preprocessing is needed."""
    risk = Risk.from_zcdp(0.1)
    gen = Generator(risk=risk, model="mst", proc_epsilon=proc_epsilon)
    with pytest.raises(ValueError, match="Insufficient privacy budget"):
        gen.fit(simple_df)  # No domain -> needs preprocessing


def test_warns_when_gen_rho_less_than_proc_rho():
    """Should warn in fit() when gen_rho < proc_rho and preprocessing is needed."""
    # proc_rho for proc_epsilon=1.0 is ~0.368
    # risk.zcdp=0.5 gives gen_rho=0.132 < proc_rho -> warning
    np.random.seed(42)
    df = pd.DataFrame({"x": np.random.uniform(-1000, 1000, 500)})
    risk = Risk.from_zcdp(0.5)
    gen = Generator(risk=risk, model="mst", proc_epsilon=1.0)
    with pytest.warns(
        UserWarning,
        match="Privacy budget for generation .* is smaller than for processing",
    ):
        gen.fit(df)


def test_no_warning_when_domain_provided(simple_df, simple_domain):
    """Should not warn when full domain is provided (no preprocessing needed)."""
    risk = Risk.from_zcdp(0.008)  # Would warn without domain
    gen = Generator(risk=risk, model="mst")
    with pytest.warns() as record:
        gen.fit(simple_df, domain=simple_domain)
    user_warnings = [w for w in record if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 0


def test_no_error_with_low_budget_when_domain_provided(simple_df, simple_domain):
    """Should not error with very low budget when full domain is provided."""
    risk = Risk.from_zcdp(0.001)  # Would error without domain
    gen = Generator(risk=risk, model="mst")
    gen.fit(simple_df, domain=simple_domain)
    synth = gen.generate(count=5)
    assert len(synth) == 5


def test_generate_before_fit_raises():
    """Should raise if generate() called before fit()."""
    risk = Risk.from_advantage(0.2)
    gen = Generator(risk=risk, model="mst")
    with pytest.raises(RuntimeError, match="Must call fit"):
        gen.generate(count=10)


def test_unknown_model():
    """Should raise for unknown model."""
    risk = Risk.from_advantage(0.2)
    with pytest.raises(ValueError, match="Unknown model"):
        Generator(risk=risk, model="unknown")


def test_categorical_only_no_preprocessing():
    """Categorical-only data should not require preprocessing."""
    df = pd.DataFrame({"cat": ["a", "b", "c"] * 10})
    risk = Risk.from_zcdp(0.001)  # Very low budget
    gen = Generator(risk=risk, model="mst")
    gen.fit(df)
    synth = gen.generate(count=5)
    assert len(synth) == 5


# === Serialization tests ===


def test_store_and_load(simple_df, simple_domain, tmp_path):
    """Test storing and loading a fitted generator."""
    risk = Risk.from_advantage(0.2)
    gen = Generator(risk=risk, model="mst")
    gen.fit(simple_df, domain=simple_domain)

    # Store
    gen.store(tmp_path / "generator")

    # Load
    loaded = Generator.load(tmp_path / "generator")

    # Generate from loaded
    synth = loaded.generate(count=10)
    assert len(synth) == 10
    assert list(synth.columns) == list(simple_df.columns)


def test_store_before_fit_raises(tmp_path):
    """Should raise if store() called before fit()."""
    risk = Risk.from_advantage(0.2)
    gen = Generator(risk=risk, model="mst")
    with pytest.raises(RuntimeError, match="Must call fit"):
        gen.store(tmp_path / "generator")


# === Slower integration test with wine ===


@pytest.mark.parametrize("advantage", [0.1])
def test_wine_integration(advantage: float, wine_df, wine_domain):
    """Integration test with wine dataset."""
    risk = Risk.from_advantage(advantage)
    gen = Generator(risk=risk, model="mst")
    gen.fit(wine_df, domain=wine_domain)
    synth = gen.generate(count=10)

    assert len(synth) == 10
    assert list(synth.columns) == list(wine_df.columns)
