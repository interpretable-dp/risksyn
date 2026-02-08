# risksyn

Synthetic data generation with interpretable privacy guarantees in terms of attack risk.

---

This library provides privacy-preserving synthetic tabular data generation using differential
privacy, with privacy specified in terms of concrete attack risk (advantage, TPR/FPR) rather
than abstract epsilon-delta parameters. We provide a detailed explanation on specifying target risk
in the documentation.

### Intended Uses

This library implements state-of-the-art methods for privacy-preserving synthetic tabular data
generation—MST[^1] and AIM[^2]—using the [dpmm library](https://github.com/sassoftware/dpmm/)[^3] as a backend.
These models use the [Select-Measure-Generate](https://differentialprivacy.org/synth-data-1/) approach,
which is competitive with non-private generation but loses utility and speed to, e.g., GANs in high-dimensional
domains[^4]. Currently, each record must correspond to one individual to ensure privacy.

Note that replacing real data with synthetic data loses the validity of statistical analyses performed
on the substituted data[^5]. The Select-Measure-Generate architecture preserves the validity of low-degree
_marginals_—queries like "how many men over 60 with diabetes are in the dataset"—which are guaranteed
to be close to the original, depending on the privacy level.

In a glance:

| ---------------- | --------------------------------- |
| Domain           | Tabular data                      |
| Dimensionality   | Up to 32 features                 |
| Privacy unit     | One data record = one individual  |
| Validity         | Low-degree marginals only         |
|------------------|-----------------------------------|

### Installation

Install with pip:
<!--pytest.mark.skip-->
```bash
pip install risksyn
```

For local development:
<!--pytest.mark.skip-->
```bash
uv sync --dev
```

### Quickstart

#### Basic Usage

Generate synthetic data which is designed to preserve well all 3-way marginals, with a maximum attack advantage of 20%:

```python
import pandas as pd
from risksyn import Risk, Generator

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
risk = Risk.from_advantage(0.2)  # Max 20% attack advantage
gen = Generator(risk=risk, model="aim", gen_kwargs={"degree": 3})
gen.fit(df, domain=domain)

# Generate synthetic records
synthetic_df = gen.generate(count=100)
print(synthetic_df.head())
```

#### Specifying Risk

The `Risk` class provides multiple ways to specify privacy requirements in the form of maximum
allowed attack risk. We detail and explain all of the options in the documentation.

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

# For advanced users: Check the converted zCDP value
print(f"zCDP rho: {risk.zcdp:.4f}")
```

#### Choosing a Model

Three synthesis models are available:

```python
from risksyn import Risk, Generator

risk = Risk.from_advantage(0.2)

# AIM (default) - Adaptive and Iterative Mechanism
gen = Generator(risk=risk, model="aim")

# MST - Maximum Spanning Tree
gen = Generator(risk=risk, model="mst")

# PrivBayes - Bayesian network approach
gen = Generator(risk=risk, model="privbayes")
```

#### Model Parameters

Pass model-specific parameters via `gen_kwargs`. For AIM and PrivBayes, the `degree` parameter
controls the maximum degree of preserved marginals (MST is inherently degree-2):

```python
from risksyn import Risk, Generator

risk = Risk.from_advantage(0.3)

# Capture pairwise correlations (default)
gen = Generator(risk=risk, model="aim", gen_kwargs={"degree": 2})

# Capture 3-way correlations (requires more privacy budget)
gen = Generator(risk=risk, model="aim", gen_kwargs={"degree": 3})
```

#### Saving and Loading Models

<!--pytest.mark.skip-->
```python
from risksyn import Generator

# After fitting...
gen.store("my_generator")

# Later, load and generate
loaded = Generator.load("my_generator")
synthetic_df = loaded.generate(count=1000)
```


### References

[^1]: [Winning the NIST Contest: A scalable and general approach to differentially private synthetic data](https://arxiv.org/abs/2108.04978). JPC 2021.
[^2]: [AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data](https://arxiv.org/abs/2201.12677). VLDB 2022.
[^3]: [dpmm: Differentially Private Marginal Models](https://arxiv.org/abs/2506.00322). TPDP 2025.
[^4]: [Graphical vs. Deep Generative Models: Measuring the Impact of DP on Utility](https://arxiv.org/abs/2305.10994). ACM CCS 2024.
[^5]: [Should I use Synthetic Data for That?](https://arxiv.org/abs/2602.03791). 2025.
