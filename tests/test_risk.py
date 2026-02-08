import pytest
from riskcal.analysis import get_advantage_from_zcdp, get_beta_from_zcdp

from risksyn.risk import Risk


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
