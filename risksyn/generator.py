import json
import math
import warnings
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from dpmm.models.base.mechanisms.cdp2adp import cdp_delta
from dpmm.pipelines import AIMPipeline, MSTPipeline, PrivBayesPipeline
from dpmm.pipelines.base import GenerativePipeline

from risksyn.risk import Risk

# Empirically tested default from dpmm library
_DEFAULT_PROC_EPSILON = 0.1

# Low-level implementation detail; not the actual privacy guarantee.
_CALIBRATION_EPSILON = 1.0

MODELS = {
    "mst": MSTPipeline,
    "aim": AIMPipeline,
    "privbayes": PrivBayesPipeline,
}


def _epsilon_to_rho_laplace(epsilon: float) -> float:
    """Convert epsilon to rho for Laplace mechanism.

    Optimal conversion from Harrison & Manurangsi, 2025 (https://arxiv.org/abs/2510.25746)
    """
    return epsilon + math.exp(-epsilon) - 1


def _requires_private_preprocessing(data: pd.DataFrame, domain: Optional[dict]) -> bool:
    """Check if any numeric column lacks bounds in domain.

    Private preprocessing is required when numeric columns don't have
    explicit lower/upper bounds, forcing private bounds estimation.
    """
    for col, series in data.items():
        if series.dtype.kind in "Mmfui":  # datetime, timedelta, float, uint, int
            if domain is None:
                return True
            col_domain = domain.get(col, {})
            if not isinstance(col_domain, dict):
                return True  # categorical-style domain for numeric column
            if col_domain.get("lower") is None or col_domain.get("upper") is None:
                return True
    return False


class Generator:
    """Generate synthetic data with interpretable privacy risk guarantees.

    Parameters
    ----------
    risk : Risk
        Risk specification defining the privacy guarantee.
    model : str, default "aim"
        Synthesis model to use. One of "mst", "aim", or "privbayes".
    proc_epsilon : float, default 0.1
        Epsilon budget allocated to data preprocessing (domain estimation).
        Only used if domain bounds are not provided for numeric columns.

    Raises
    ------
    ValueError
        If model is unknown.

    Examples
    --------
    >>> from risksyn import Risk, Generator
    >>> risk = Risk.from_advantage(0.2)
    >>> gen = Generator(risk=risk, model="mst")
    >>> gen.fit(df, domain={"age": {"lower": 0, "upper": 100}})
    >>> synthetic_df = gen.generate(count=1000)
    """

    def __init__(
        self,
        risk: Risk,
        model: str = "aim",
        proc_epsilon: float = _DEFAULT_PROC_EPSILON,
    ):
        if model not in MODELS:
            raise ValueError(
                f"Unknown model: {model}. Choose from {list(MODELS.keys())}"
            )

        self._risk = risk
        self._model = model
        self._proc_epsilon = proc_epsilon
        self._pipeline = None

    def fit(self, data: pd.DataFrame, domain: Optional[dict] = None) -> "Generator":
        """Fit the generator to the data.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame to fit the model on.
        domain : dict, optional
            Domain specification for columns. For numeric columns, use
            ``{"col": {"lower": min, "upper": max}}``. For categorical columns,
            use ``{"col": ["val1", "val2", ...]}``. Required when privacy budget
            is low to avoid private domain estimation failures.

        Returns
        -------
        Generator
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If privacy budget is insufficient for the required preprocessing
            (when numeric columns lack domain bounds).

        Warns
        -----
        UserWarning
            If privacy budget for generation is smaller than for processing.
        """
        needs_preprocessing = _requires_private_preprocessing(data, domain)

        if needs_preprocessing:
            if self._proc_epsilon is None:
                raise ValueError(
                    f"Insufficient privacy budget for pre-processing. Provide domain bounds for numeric columns, "
                    "or provide nonzero proc_epsilon."
                )
            proc_rho = _epsilon_to_rho_laplace(self._proc_epsilon)
            gen_rho = self._risk.zcdp - proc_rho
            if gen_rho <= 0:
                raise ValueError(
                    f"Insufficient privacy budget: risk.zcdp={self._risk.zcdp:.6f} <= "
                    f"proc_rho={proc_rho:.6f}. Provide domain bounds for numeric columns, "
                    "relax the risk requirement, or decrease proc_epsilon."
                )
            if gen_rho < proc_rho:
                warnings.warn(
                    f"Privacy budget for generation ({gen_rho:.6f}) is smaller than for "
                    f"processing ({proc_rho:.6f}). Consider providing domain bounds for "
                    "numeric columns, relaxing the risk requirement, or decreasing proc_epsilon.",
                    UserWarning,
                    stacklevel=2,
                )
            proc_epsilon = self._proc_epsilon
        else:
            gen_rho = self._risk.zcdp  # Full budget for generation
            proc_epsilon = None

        # Convert generation zCDP to (epsilon, delta)-DP
        delta = cdp_delta(gen_rho, _CALIBRATION_EPSILON)

        pipeline_cls = MODELS[self._model]
        self._pipeline = pipeline_cls(
            epsilon=_CALIBRATION_EPSILON,
            delta=delta,
            proc_epsilon=proc_epsilon,
        )
        self._pipeline.fit(data, domain)
        return self

    def generate(self, count: int) -> pd.DataFrame:
        """Generate synthetic records.

        Parameters
        ----------
        count : int
            Number of records to generate.

        Returns
        -------
        pd.DataFrame
            DataFrame with synthetic data matching the schema of the fitted data.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        if self._pipeline is None:
            raise RuntimeError("Must call fit() before generate()")
        return self._pipeline.generate(n_records=count)

    def store(self, path: Union[str, Path]) -> None:
        """Store the fitted generator to disk.

        Parameters
        ----------
        path : str or Path
            Directory path to store the generator.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        if self._pipeline is None:
            raise RuntimeError("Must call fit() before store()")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Store metadata
        metadata = {
            "risk_zcdp": self._risk.zcdp,
            "model": self._model,
            "proc_epsilon": self._proc_epsilon,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Store pipeline
        pipeline_path = path / "pipeline"
        pipeline_path.mkdir(parents=True, exist_ok=True)
        self._pipeline.store(pipeline_path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Generator":
        """Load a fitted generator from disk.

        Parameters
        ----------
        path : str or Path
            Directory path containing the stored generator.

        Returns
        -------
        Generator
            The loaded generator, ready for generation.
        """
        path = Path(path)

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        # Create generator instance
        risk = Risk.from_zcdp(metadata["risk_zcdp"])
        gen = cls(
            risk=risk,
            model=metadata["model"],
            proc_epsilon=metadata["proc_epsilon"],
        )

        # Load pipeline
        gen._pipeline = GenerativePipeline.load(path / "pipeline")

        return gen
