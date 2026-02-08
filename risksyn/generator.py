import json
import warnings
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from dpmm.pipelines import AIMPipeline
from dpmm.pipelines.base import GenerativePipeline

from risksyn.risk import Risk, calibrate_parameters_to_risk, _epsilon_to_rho_laplace

# Empirically tested default from dpmm library
_DEFAULT_PROC_EPSILON = 0.1

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
                domain[col] = sorted(str(v) for v in series.unique())
                cols_to_cast.append(col)
    if cols_to_cast:
        data = data.copy()
        data[cols_to_cast] = data[cols_to_cast].astype(str)
    return data, domain


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
            if isinstance(col_domain, list):
                continue  # categorical domain provided, no bounds needed
            if not isinstance(col_domain, dict):
                return True
            if col_domain.get("lower") is None or col_domain.get("upper") is None:
                return True
    return False


class AIMGenerator:
    """Generate synthetic data with interpretable privacy risk guarantees.

    Uses the AIM (Adaptive and Iterative Mechanism) pipeline for synthesis.

    Parameters
    ----------
    risk : Risk
        Risk specification defining the privacy guarantee.
    degree : int, default 2
        Maximum degree of marginals used by AIM.
    max_model_size : int, default 80
        Maximum model size parameter for AIM.
    compress : bool, default True
        Whether to use compression in AIM.
    proc_epsilon : float, default 0.1
        Epsilon budget allocated to data preprocessing (domain estimation).
        Only used if domain bounds are not provided for numeric columns.

    Examples
    --------
    >>> from risksyn import Risk, AIMGenerator
    >>> risk = Risk.from_advantage(0.2)
    >>> gen = AIMGenerator(risk=risk, degree=3)
    >>> gen.fit(df, domain={"age": {"lower": 0, "upper": 100}})
    >>> synthetic_df = gen.generate(count=1000)
    """

    def __init__(
        self,
        risk: Risk,
        degree: int = 2,
        max_model_size: int = 80,
        compress: bool = True,
        proc_epsilon: float = _DEFAULT_PROC_EPSILON,
    ):
        self._risk = risk
        self._degree = degree
        self._max_model_size = max_model_size
        self._compress = compress
        self._proc_epsilon = proc_epsilon
        self._pipeline = None

    def fit(self, data: pd.DataFrame, domain: Optional[dict] = None) -> "AIMGenerator":
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
        AIMGenerator
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
        data, domain = _auto_categorize(data, domain)

        needs_preprocessing = _requires_private_preprocessing(data, domain)

        if needs_preprocessing:
            proc_epsilon = self._proc_epsilon
            if proc_epsilon is None:
                raise ValueError(
                    "Insufficient privacy budget for pre-processing. Provide domain bounds for numeric columns, "
                    "or provide nonzero proc_epsilon."
                )
            params = calibrate_parameters_to_risk(self._risk, proc_epsilon=proc_epsilon)

            proc_rho = _epsilon_to_rho_laplace(proc_epsilon)
            gen_rho = self._risk.zcdp - proc_rho
            if gen_rho < proc_rho:
                warnings.warn(
                    f"Privacy budget for generation ({gen_rho:.6f}) is smaller than for "
                    f"processing ({proc_rho:.6f}). Consider providing domain bounds for "
                    "numeric columns, relaxing the risk requirement, or decreasing proc_epsilon.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            params = calibrate_parameters_to_risk(self._risk)

        self._pipeline = AIMPipeline(
            epsilon=params["epsilon"],
            delta=params["delta"],
            compress=self._compress,
            max_model_size=self._max_model_size,
            proc_epsilon=params.get("proc_epsilon"),
            gen_kwargs={"degree": self._degree},
        )
        _BOUNDS_ERROR_MSG = (
            "Private bounds estimation failed for one or more numeric columns. "
            "This typically happens when the privacy budget is too small to detect "
            "data bounds. Remedies: (1) provide explicit domain bounds for numeric "
            "columns via the domain parameter, e.g. domain={'col': {'lower': 0, "
            "'upper': 100}}, (2) increase proc_epsilon, or (3) relax the risk "
            "requirement."
        )
        try:
            self._pipeline.fit(data, domain)
        except (TypeError, KeyError) as e:
            raise ValueError(_BOUNDS_ERROR_MSG) from e
        except ValueError as e:
            if "Private bounds estimation failed" not in str(e):
                raise ValueError(_BOUNDS_ERROR_MSG) from e
            raise
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

        metadata = {
            "risk_zcdp": self._risk.zcdp,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        pipeline_path = path / "pipeline"
        pipeline_path.mkdir(parents=True, exist_ok=True)
        self._pipeline.store(pipeline_path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AIMGenerator":
        """Load a fitted generator from disk.

        Parameters
        ----------
        path : str or Path
            Directory path containing the stored generator.

        Returns
        -------
        AIMGenerator
            The loaded generator, ready for generation.
        """
        path = Path(path)

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        risk = Risk.from_zcdp(metadata["risk_zcdp"])
        gen = cls(risk=risk)

        gen._pipeline = GenerativePipeline.load(path / "pipeline")

        return gen
