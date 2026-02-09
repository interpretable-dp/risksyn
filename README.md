# risksyn

[![docs](https://app.readthedocs.org/projects/risksyn/badge/?version=latest)](https://risksyn.readthedocs.io/latest/)
[![CI](https://github.com/interpretable-dp/risksyn/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/interpretable-dp/risksyn/actions/workflows/ci.yml)

---

⚠️  This is a research prototype. Avoid or be extra careful when using in production.

---

Synthetic data generation with interpretable privacy guarantees in terms of attack risk.

This library provides privacy-preserving synthetic tabular data generation using differential
privacy, with privacy specified in terms of interpretable attack risk (advantage, TPR/FPR) rather
than the standard epsilon-delta parameters.

### Intended Uses

The library is designed to work in the following settings:

- **Low-dimensional tabular data up to ≈32 features.** This library is designed to work with the
  state-of-the-art [Select-Measure-Generate](https://differentialprivacy.org/synth-data-1/) paradigm
  for privacy-preserving generation of synthetic data, which is competitive with non-private
  generation but loses utility and speed to, e.g., privacy-preserving GANs, in high-dimensional
  domains[^1].
- **Validity for low-degree marginals.** Replacing real data with any kind of synthetic data in
  general does not preserve the validity of statistical analyses performed on the substituted data,
  i.e., the users of synthetic data cannot know how close or far are the results of the analysis to
  those on the actual real data[^2]. The Select-Measure-Generate architecture, however, does
  preserve the validity of _marginals_ with a given degree—queries like "how many men over 60 with
  diabetes are in the dataset"—which are guaranteed to be close to the original, depending on the
  privacy level. At the moment, the library does not support the derivation of confidence intervals
  for these queries.
- **Each record corresponds to one individual.** At the moment, each record must correspond to one
  individual for the privacy guarantee to be meaningful at the level of individuals.

### Installation

Currently, you need to get a copy of the repo, e.g., with `git clone`, and install locally:
```
pip install -e .
```

For local development:
<!--pytest.mark.skip-->
```bash
uv sync --dev
```

### Quickstart

There are two ways to use the library:
- _Basic._ Generate synthetic data using AIM[^3], one of the state-of-the-art algorithms for
  generating privacy-preserving synthetic tabular data. Unlike other implementations of this
  algorithm, this library enables to specify the privacy requirement in terms of interpretable
  attack risk.
- _Customized._ Calibrate the noise parameters to ensure a given level of attack risk that can then
  be passed as input to the [dpmm](https://github.com/sassoftware/dpmm) library calls for precise
  control over the generation pipeline.

#### Specifying Target Risk

The main feature of this library is the ability to specify the target level of privacy risk in terms
of interpretable attack success rates instead of the classical approach of using epsilon-delta
parameters. This is done using the `Risk` class. We detail and explain all of the options in the [risk
specification and modeling guide](https://risksyn.readthedocs.io/latest/risk-modeling.html).

```python
from risksyn import Risk

# Membership inference attack error rates
risk = Risk.from_err_rates(tpr=0.6, fpr=0.1)  # Max 60% TPR at 10% FPR

# Inference attack advantage
risk = Risk.from_advantage(0.2)  # 20% max advantage

# Inference attack advantage at a given baseline
risk = Risk.from_advantage_at_baseline(advantage=0.2, baseline=0.5)

# Inference attack success rate at a given baseline
risk = Risk.from_success_at_baseline(success=0.7, baseline=0.5)

# Combine multiple risk requirements (takes the most restrictive)
risk = Risk.from_advantage(0.01) | Risk.from_advantage_at_baseline(0.05, 0.001)

# For advanced users: Check the converted zCDP value
print(f"zCDP rho: {risk.zcdp:.4f}")
```

#### Basic Usage

Here is an example of the basic usage. We generate synthetic data to preserve all 3-way marginals
well and ensure that the maximum additive advantage of any inference attack aimed to learn
information about the real records based on the synthetic data is at most 20 percentage points:

```python
import pandas as pd
from risksyn import Risk, AIMGenerator

# Sample data
df = pd.DataFrame({
    "age": [25, 30, 35, 40, 45],
    "income": [50000, 60000, 70000, 80000, 90000],
    "city": ["NYC", "LA", "NYC", "SF", "LA"],
})

# Specify domain bounds for numeric columns
domain = {
    "age": {"lower": 18, "upper": 100},
    "income": {"lower": 0, "upper": 500000},
}

# Create generator with risk specification
risk = Risk.from_advantage(0.2)  # Ensure max 20p.p. advantage
gen = AIMGenerator(risk=risk, degree=3)
gen.fit(df, domain=domain)

# Generate synthetic records
synthetic_df = gen.generate(count=100)
print(synthetic_df.head())
```

We detail the options and parameters in the [generation
guide](https://risksyn.readthedocs.io/latest/generation.html).

#### Customized Generation Pipelines with Calibration Utilities

For direct control over the [dpmm](https://github.com/sassoftware/dpmm/) generation pipeline, we
provide `calibrate_parameters_to_risk` that converts a `Risk` object into intermediate calibrated
noise parameters.  If numeric columns need private domain estimation, pass `proc_epsilon` to reserve
part of the budget for preprocessing—otherwise the privacy guarantee may not hold. See the
[generation guide](https://risksyn.readthedocs.io/latest/generation.html) for details.

<!--pytest.mark.skip-->
```python
from risksyn import Risk, calibrate_parameters_to_risk
from dpmm.pipelines import AIMPipeline

risk = Risk.from_advantage(0.2)
params = calibrate_parameters_to_risk(risk, proc_epsilon=0.1)

# Verbose for illustrative purposes:
pipeline = AIMPipeline(
    epsilon=params["epsilon"],
    delta=params["delta"],
    gen_kwargs={"degree": 3},
)
# Simpler usage:
pipeline = AIMPipeline(
    gen_kwargs={"degree": 3},
    **calibrate_parameters_to_risk(risk)
)

pipeline.fit(df, domain)
synthetic_df = pipeline.generate(n_records=10)
```


[^1]: [Graphical vs. Deep Generative Models: Measuring the Impact of DP on Utility](https://arxiv.org/abs/2305.10994). ACM CCS 2024.
[^2]: [Should I use Synthetic Data for That?](https://arxiv.org/abs/2602.03791) 2025.
[^3]: [AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data](https://arxiv.org/abs/2201.12677). VLDB 2022.
