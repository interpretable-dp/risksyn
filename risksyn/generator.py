import json
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from dpmm.pipelines import AIMPipeline
from dpmm.pipelines.base import GenerativePipeline

from risksyn.processing import (
    _apply_column_schema,
    _auto_categorize,
    _detect_column_schema,
    _numeric_cols_without_bounds,
)
from risksyn.risk import Risk, calibrate_parameters_to_risk, _epsilon_to_rho_laplace

# Empirically tested default from dpmm library
_DEFAULT_PROC_EPSILON = 0.1


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
        self._column_schema = None

    def fit(
        self,
        data: pd.DataFrame,
        domain: Optional[dict] = None,
        unsafe_infer_bounds: bool = False,
    ) -> "AIMGenerator":
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
        unsafe_infer_bounds : bool, default False
            If True, infer domain bounds from data min/max for numeric columns
            that lack explicit bounds. This leaks information about the data
            and weakens the privacy guarantee.

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
            If privacy budget for generation is smaller than for processing,
            or if ``unsafe_infer_bounds`` is used.
        """
        self._column_schema = _detect_column_schema(data)

        data, domain = _auto_categorize(data, domain)
        unbounded_cols = _numeric_cols_without_bounds(data, domain)

        if unbounded_cols and unsafe_infer_bounds:
            domain = dict(domain) if domain else {}
            for col in unbounded_cols:
                series = data[col]
                lower, upper = float(series.min()), float(series.max())
                domain[col] = {"lower": lower, "upper": upper}
                warnings.warn(
                    f"PrivacyLeakage: Inferring bounds for numeric column "
                    f"'{col}' from data (lower={lower}, upper={upper}). "
                    f"This leaks information and weakens the privacy guarantee. "
                    f"Provide explicit bounds via the domain parameter to avoid this.",
                    UserWarning,
                    stacklevel=2,
                )
            unbounded_cols = []

        if unbounded_cols:
            proc_epsilon = self._proc_epsilon
            if proc_epsilon is None:
                raise ValueError(
                    f"No domain bounds for numeric columns {unbounded_cols}. "
                    "Provide domain bounds, set proc_epsilon > 0, or use "
                    "unsafe_infer_bounds=True."
                )
            params = calibrate_parameters_to_risk(self._risk, proc_epsilon=proc_epsilon)

            n_cols = len(unbounded_cols)
            eps_per_col = proc_epsilon / n_cols
            # Each column further splits its budget in half for bounds estimation
            eps_for_bounds = eps_per_col / 2

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

            if eps_for_bounds < 0.05:
                warnings.warn(
                    f"Epsilon budget for private bounds estimation is too small "
                    f"(proc_epsilon={proc_epsilon} / {n_cols} columns / 2 = "
                    f"{eps_for_bounds:.4f} per column). Private bounds estimation "
                    f"will fail at this level. Provide explicit domain bounds for "
                    f"columns {unbounded_cols}, e.g.:\n"
                    f"  domain={{'{unbounded_cols[0]}': {{'lower': <min>, 'upper': <max>}}}}",
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
        synth = self._pipeline.generate(n_records=count)
        if self._column_schema:
            synth = _apply_column_schema(synth, self._column_schema)
        return synth

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

        column_schema = {}
        if self._column_schema:
            for col, entry in self._column_schema.items():
                col_entry = {"dtype": str(entry["dtype"])}
                if "precision" in entry:
                    col_entry["precision"] = entry["precision"]
                column_schema[col] = col_entry

        metadata = {
            "risk_zcdp": self._risk.zcdp,
            "column_schema": column_schema,
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

        schema_raw = metadata.get("column_schema")
        if schema_raw:
            schema = {}
            for col, entry in schema_raw.items():
                col_entry = {"dtype": np.dtype(entry["dtype"])}
                if "precision" in entry:
                    col_entry["precision"] = entry["precision"]
                schema[col] = col_entry
            gen._column_schema = schema

        gen._pipeline = GenerativePipeline.load(path / "pipeline")

        return gen
