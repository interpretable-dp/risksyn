import math
import warnings

from dpmm.models.base.mechanisms.cdp2adp import cdp_delta
from riskcal.analysis import get_advantage_from_zcdp, get_beta_from_zcdp

# Bisection defaults
_DEFAULT_TOL = 1e-4
_DEFAULT_MAX_ITER = 100

# Warn when multiplicative increase in success rate exceeds this factor
ADVANTAGE_WARN_MULTIPLIER = 3


class Risk:
    """Specification on the target level of risk.

    Internally converts to zCDP.

    Use factory methods to create:
    - Risk.from_zcdp(rho)
    - Risk.from_advantage(advantage)
    - Risk.from_err_rates(tpr, fpr)
    - Risk.from_advantage_at_baseline(advantage, baseline)
    """

    def __init__(self, rho: float):
        self._rho = rho

    @staticmethod
    def from_zcdp(rho: float) -> "Risk":
        return Risk(rho)

    @staticmethod
    def from_advantage(
        advantage: float,
        tol: float = _DEFAULT_TOL,
        max_iter: int = _DEFAULT_MAX_ITER,
    ) -> "Risk":
        """
        From worst-case advantage, maximum difference between success rate - baseline or TPR - FPR
        in the case of membership inference attacks.
        """
        rho = _find_rho_for_advantage(advantage, tol, max_iter)
        return Risk(rho)

    @staticmethod
    def from_err_rates(
        tpr: float,
        fpr: float,
        tol: float = _DEFAULT_TOL,
        max_iter: int = _DEFAULT_MAX_ITER,
    ) -> "Risk":
        """
        From membership inference TPR and FPR.
        """
        alpha = fpr
        beta = 1 - tpr
        rho = _find_rho_for_alpha_beta(alpha, beta, tol, max_iter)
        return Risk(rho)

    @staticmethod
    def from_success_at_baseline(
        success: float,
        baseline: float,
        tol: float = _DEFAULT_TOL,
        max_iter: int = _DEFAULT_MAX_ITER,
    ) -> "Risk":
        """
        From success rate (TPR) at a given baseline (FPR).
        """
        return Risk.from_err_rates(tpr=success, fpr=baseline, tol=tol, max_iter=max_iter)

    @staticmethod
    def from_advantage_at_baseline(
        advantage: float,
        baseline: float,
        tol: float = _DEFAULT_TOL,
        max_iter: int = _DEFAULT_MAX_ITER,
    ) -> "Risk":
        """
        From advantage (success rate - baseline) and baseline success rate.
        """
        # advantage = success - baseline = (1 - beta) - alpha
        beta = 1 - advantage - baseline
        if not 0 <= beta <= 1:
            raise ValueError(f"Invalid: We must have advantage + baseline <= 1.")
        times = (advantage + baseline) / baseline
        if times >= ADVANTAGE_WARN_MULTIPLIER:
            warnings.warn(
                f"The current advantage corresponds to the adversary's success "
                f"increasing {times:.1f}x. Consider setting success and baseline "
                f"requirements exactly.",
                UserWarning,
                stacklevel=2,
            )
        rho = _find_rho_for_alpha_beta(baseline, beta, tol, max_iter)
        return Risk(rho)

    @property
    def zcdp(self) -> float:
        """Get the converted zCDP parameter."""
        return self._rho

    def __or__(self, other: "Risk") -> "Risk":
        """Combine risk specs — takes minimum rho (most restrictive)."""
        if not isinstance(other, Risk):
            return NotImplemented
        return Risk(min(self._rho, other._rho))

    def __repr__(self) -> str:
        return f"Risk(rho={self._rho:.6f})"


def _find_rho_for_advantage(
    target_advantage: float,
    tol: float = _DEFAULT_TOL,
    max_iter: int = _DEFAULT_MAX_ITER,
) -> float:
    """Binary search for rho that yields target advantage."""
    lo, hi = 1e-6, 100.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        adv = get_advantage_from_zcdp(mid)

        if abs(adv - target_advantage) < tol:
            return mid

        if adv > target_advantage:
            hi = mid
        else:
            lo = mid

    return (lo + hi) / 2


def _find_rho_for_alpha_beta(
    alpha: float,
    target_beta: float,
    tol: float = _DEFAULT_TOL,
    max_iter: int = _DEFAULT_MAX_ITER,
) -> float:
    """Binary search for rho that yields target beta at given alpha."""
    lo, hi = 1e-6, 100.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        beta = get_beta_from_zcdp(mid, alpha)

        if abs(beta - target_beta) < tol:
            return mid

        if beta > target_beta:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2


# Low-level implementation detail; not the actual privacy guarantee.
_CALIBRATION_EPSILON = 1.0


def _epsilon_to_rho(epsilon: float) -> float:
    """Convert pure-DP epsilon to zCDP rho: rho = epsilon^2 / 2."""
    return epsilon**2 / 2


def calibrate_parameters_to_risk(
    risk: Risk, proc_epsilon: float | None = None
) -> dict:
    """Calibrate (epsilon, delta) parameters from a Risk specification.

    Converts a Risk object into the (epsilon, delta) parameters needed by
    differential privacy pipelines. Optionally accounts for preprocessing
    budget.

    Parameters
    ----------
    risk : Risk
        Risk specification.
    proc_epsilon : float, optional
        Epsilon budget to reserve for preprocessing. If provided, deducted
        from the total privacy budget and included in the output.

    Returns
    -------
    dict
        ``{"epsilon": ..., "delta": ...}`` and optionally ``"proc_epsilon"``.

    Raises
    ------
    ValueError
        If the privacy budget is insufficient.
    """
    total_rho = risk.zcdp

    if proc_epsilon is not None:
        proc_rho = _epsilon_to_rho(proc_epsilon)
        gen_rho = total_rho - proc_rho
        if gen_rho <= 0:
            raise ValueError(
                f"Insufficient privacy budget: risk.zcdp={total_rho:.6f} <= "
                f"proc_rho={proc_rho:.6f}. Provide domain bounds for numeric "
                "columns, relax the risk requirement, or decrease proc_epsilon."
            )
    else:
        gen_rho = total_rho

    delta = cdp_delta(gen_rho, _CALIBRATION_EPSILON)

    result = {"epsilon": _CALIBRATION_EPSILON, "delta": delta}
    if proc_epsilon is not None:
        result["proc_epsilon"] = proc_epsilon
    return result
