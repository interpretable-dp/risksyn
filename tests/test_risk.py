import warnings

import pytest
from riskcal.analysis import get_advantage_from_zcdp, get_beta_from_zcdp

from risksyn.risk import ADVANTAGE_WARN_MULTIPLIER, Risk


@pytest.mark.parametrize("rho", [0.1, 1.0, 10.0])
def test_from_zcdp(rho: float):
    risk = Risk.from_zcdp(rho)
    assert risk.zcdp == rho


@pytest.mark.parametrize("advantage", [0.1, 0.5, 0.95])
def test_from_advantage(advantage: float):
    risk = Risk.from_advantage(advantage)
    recovered = get_advantage_from_zcdp(risk.zcdp)
    assert recovered == pytest.approx(advantage, abs=1e-3)


@pytest.mark.parametrize(
    "tpr,fpr",
    [
        (0.9, 0.1),
        (0.95, 0.05),
    ],
)
def test_from_err_rates(tpr: float, fpr: float):
    risk = Risk.from_err_rates(tpr=tpr, fpr=fpr)
    recovered_beta = get_beta_from_zcdp(risk.zcdp, alpha=fpr)
    expected_beta = 1 - tpr
    assert recovered_beta == pytest.approx(expected_beta, abs=1e-3)


@pytest.mark.parametrize(
    "success,baseline",
    [
        (0.9, 0.1),
        (0.95, 0.05),
    ],
)
def test_from_success_at_baseline(success: float, baseline: float):
    risk = Risk.from_success_at_baseline(success=success, baseline=baseline)
    # Should be equivalent to from_err_rates(tpr=success, fpr=baseline)
    expected = Risk.from_err_rates(tpr=success, fpr=baseline)
    assert risk.zcdp == pytest.approx(expected.zcdp, abs=1e-6)


@pytest.mark.parametrize(
    "advantage,baseline",
    [
        (0.2, 0.1),
        (0.5, 0.1),
    ],
)
def test_from_advantage_at_baseline(advantage: float, baseline: float):
    risk = Risk.from_advantage_at_baseline(advantage, baseline)
    expected_beta = 1 - advantage - baseline
    recovered_beta = get_beta_from_zcdp(risk.zcdp, alpha=baseline)
    assert recovered_beta == pytest.approx(expected_beta, abs=1e-3)


def test_from_advantage_at_baseline_invalid():
    with pytest.raises(ValueError, match="advantage \\+ baseline <= 1"):
        Risk.from_advantage_at_baseline(advantage=0.9, baseline=0.5)


# === | operator tests ===


def test_or_takes_min_rho():
    r1 = Risk.from_zcdp(1.0)
    r2 = Risk.from_zcdp(2.0)
    combined = r1 | r2
    assert combined.zcdp == 1.0


def test_or_commutative():
    r1 = Risk.from_zcdp(1.0)
    r2 = Risk.from_zcdp(2.0)
    assert (r1 | r2).zcdp == (r2 | r1).zcdp


def test_or_with_factory_methods():
    r1 = Risk.from_advantage(0.01)
    r2 = Risk.from_advantage_at_baseline(0.05, 0.1)
    combined = r1 | r2
    assert combined.zcdp == min(r1.zcdp, r2.zcdp)


# === Advantage warning tests ===


def test_advantage_at_baseline_warns_large_multiplier():
    with pytest.warns(UserWarning, match="51.0x"):
        Risk.from_advantage_at_baseline(advantage=0.05, baseline=0.001)


def test_advantage_at_baseline_warns_at_threshold():
    # Exactly at threshold — should warn
    baseline = 0.01
    advantage = baseline * (ADVANTAGE_WARN_MULTIPLIER - 1)
    with pytest.warns(UserWarning, match=f"{ADVANTAGE_WARN_MULTIPLIER}.0x"):
        Risk.from_advantage_at_baseline(advantage=advantage, baseline=baseline)


def test_advantage_at_baseline_no_warn_below_threshold():
    # Below threshold — no warning
    baseline = 0.01
    advantage = baseline * (ADVANTAGE_WARN_MULTIPLIER - 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Risk.from_advantage_at_baseline(advantage=advantage, baseline=baseline)


# === Calibrator tests ===


from risksyn.risk import calibrate_parameters_to_risk


def test_calibrate_happy_path():
    """Basic calibration returns epsilon and delta."""
    risk = Risk.from_zcdp(1.0)
    params = calibrate_parameters_to_risk(risk)
    assert "epsilon" in params
    assert "delta" in params
    assert params["epsilon"] == 1.0
    assert params["delta"] > 0
    assert "proc_epsilon" not in params


def test_calibrate_with_proc_epsilon():
    """Calibration with proc_epsilon deducts budget and includes it in output."""
    risk = Risk.from_zcdp(1.0)
    params = calibrate_parameters_to_risk(risk, proc_epsilon=0.1)
    assert "epsilon" in params
    assert "delta" in params
    assert params["proc_epsilon"] == 0.1


def test_calibrate_insufficient_budget():
    """Should raise when proc_epsilon exhausts the budget."""
    risk = Risk.from_zcdp(0.01)
    with pytest.raises(ValueError, match="Insufficient privacy budget"):
        calibrate_parameters_to_risk(risk, proc_epsilon=1.0)
