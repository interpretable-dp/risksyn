from riskcal.analysis import get_advantage_from_zcdp, get_beta_from_zcdp

# Bisection defaults
_DEFAULT_TOL = 1e-4
_DEFAULT_MAX_ITER = 100


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
        rho = _find_rho_for_alpha_beta(baseline, beta, tol, max_iter)
        return Risk(rho)

    @property
    def zcdp(self) -> float:
        """Get the converted zCDP parameter."""
        return self._rho

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
