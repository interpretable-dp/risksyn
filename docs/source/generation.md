# Generation

This guide explains how to generate privacy-preserving synthetic data, covering both the streamlined
`AIMGenerator` interface and the lower-level calibration function.

## How Marginal-Based Generation Works

### Main Generation Algorithm

This library uses the AIM (Adaptive and Iterative Mechanism)[^1] algorithm for privacy-preserving
synthetic data generation. AIM belongs to the
[Select-Measure-Generate](https://differentialprivacy.org/synth-data-1/) family of algorithms, which
work in three steps:

1. **Select** which statistics of the original data to preserve.
2. **Measure** those statistics with calibrated noise added—this is how we get the privacy
   guarantees using differential privacy.
3. **Generate** synthetic data that matches the noisy statistics.

The statistics that AIM preserves are called **marginals**. A marginal is a frequency table over a
subset of columns. For example, suppose a dataset has columns for city, income bracket, and age
group:

- A **1-way marginal** of "city" counts how many people live in each city.
- A **2-way marginal** of ("city", "income bracket") counts how many people in each city fall into
  each income bracket—capturing the *relationship* between city and income.
- A **3-way marginal** of ("city", "income bracket", "age group") captures the three-way
  interaction—for example, that high-income young people concentrate in certain cities.

The number of columns in a marginal is its **degree**. Higher-degree marginals capture more complex
relationships, but there are exponentially more of them. AIM adaptively selects which marginals to
measure, focusing the privacy budget on the marginals that matter most for the data distribution.

### Pre-Processing

To provide end-to-end privacy guarantees, we need to ensure there is no leakage privacy in the
entire synthetic data generation pipeline, and that includes pre-processing. The AIM algorithm
expects to take as input the following data domain information:
- _Bounds._  The minimum and maximum values of numeric features
- _Binning._ Marginals are inherently discrete, thus we have to discretize numeric features by
  clustering the domain

Both of these pre-processing procedures can come with privacy leakage, as in the worst-case
adversaries could learn information about outliers with extreme values of their features. Unless the
bounds and bins are specified from domain knowledge, we need to allocate a part of the privacy
budget, i.e., a part of all of the randomness used to ensure the target level of risk, to
pre-processing.

## AIMGenerator

### Parameters

#### `risk`

The `Risk` specification defining the privacy guarantee. See the
[Risk Modeling](risk-modeling.md) guide for how to choose this.

#### `degree` (default: 2)

The maximum degree of marginals that AIM will try to preserve. This is the most important parameter
for controlling the utility-privacy trade-off:

- **`degree=1`**: Only preserves individual column distributions. Each column's frequencies will be
  close to the original, but correlations between columns (e.g., "older people tend to have higher
  incomes") will not be captured. Very private, since few statistics need measuring—but the
  synthetic data may be misleading for any analysis that depends on relationships between columns.

- **`degree=2`** (default): Preserves pairwise relationships between columns. Captures correlations
  like "city and income are related" or "age and health status are correlated." A good starting
  point for most datasets.

- **`degree=3`**: Preserves three-way interactions. Captures patterns like "the relationship between
  income and health status differs by city." Better for datasets where multi-column dependencies
  matter, but requires more privacy budget since there are many more three-way marginals to measure.

**The trade-off:** Increasing degree captures more complex patterns but spreads the privacy budget
across more marginals, adding more noise to each one. For a given privacy budget, there is a sweet
spot: going too high can hurt utility because of the added noise.

#### `max_model_size` (default: 80)

Controls the maximum size (in MB) of the internal graphical model that AIM builds. A larger value
enables AIM to include more marginals in the model, potentially improving fidelity. The default of 80
works well in most cases. Increasing this value may improve utility for datasets with many columns
but will increase computation time and memory usage.

#### `compress` (default: True)

Enables compression in AIM, which reduces the number of marginals that need to be measured by
combining related ones. This generally improves utility for a given privacy budget and should be left
enabled unless you have a specific reason to disable it.

#### `proc_epsilon` (default: 0.1)

Controls the amount of randomness via the classical epsilon parameter in differential privacy that
is allocated to pre-processing procedures: estimation of domain bounds and clustering of numeric features.
It is highly recommended to at least provide domain bounds whenever possible, as detailed next.

### Domain Specification

The `domain` parameter in `fit()` tells the generator the range of possible values for each column:

```python
domain = {
    # Numeric columns: specify lower and upper bounds
    "age": {"lower": 18, "upper": 100},
    "income": {"lower": 0, "upper": 500000},

    # Categorical columns: list all possible values (inferred from data if omitted)
    "city": ["NYC", "LA", "SF", "Chicago"],
}
```

For numeric columns, providing bounds avoids private preprocessing and preserves more privacy budget
for generation. For categorical columns, the domain is inferred from the data if not specified.

### Full Example

```python
import pandas as pd
from risksyn import Risk, AIMGenerator

df = pd.DataFrame({
    "age": [25, 30, 35, 40, 45],
    "income": [50000, 60000, 70000, 80000, 90000],
    "city": ["NYC", "LA", "NYC", "SF", "LA"],
})

domain = {
    "age": {"lower": 18, "upper": 100},
    "income": {"lower": 0, "upper": 500000},
}

risk = Risk.from_advantage(0.2)
gen = AIMGenerator(risk=risk, degree=3, max_model_size=80)
gen.fit(df, domain=domain)

synthetic_df = gen.generate(count=1000)
```

### Saving and Loading

Fitted generators can be saved to disk and loaded later:

```python
# Save
gen.store("my_generator")

# Load and generate
loaded = AIMGenerator.load("my_generator")
synthetic_df = loaded.generate(count=1000)
```

## Using Calibration Utilities with dpmm

For users who need direct control over the [dpmm](https://github.com/sassoftware/dpmm/)
pipeline—for example, to use a different pipeline type, customize the generation process, or
integrate into an existing workflow—`calibrate_parameters_to_risk` converts a `Risk` specification
into the
`(epsilon, delta)` parameters that dpmm expects. Importantly, these `(epsilon, delta)` values
are *not* supposed to be used for interpretation of the privacy guarantees—the risk level is
input to the calibration procedure—but rather an intermediate technical crutch to set the
noise levels in the backend.

### Without Private Preprocessing

When you provide complete domain bounds for all numeric columns, no private preprocessing is needed.
The full privacy budget goes to generation:

```python
from risksyn import Risk, calibrate_parameters_to_risk
from dpmm.pipelines import AIMPipeline

risk = Risk.from_advantage(0.2)
params = calibrate_parameters_to_risk(risk)

pipeline = AIMPipeline(
    epsilon=params["epsilon"],
    delta=params["delta"],
    gen_kwargs={"degree": 3},
)
pipeline.fit(df, domain)
synthetic_df = pipeline.generate(n_records=1000)
```

### With Private Preprocessing

When numeric columns lack explicit bounds, dpmm can estimate them privately. You must reserve part of
the privacy budget for this by passing `proc_epsilon`:

```python
params = calibrate_parameters_to_risk(risk, proc_epsilon=0.1)

pipeline = AIMPipeline(
    epsilon=params["epsilon"],
    delta=params["delta"],
    proc_epsilon=params["proc_epsilon"],
    gen_kwargs={"degree": 3},
)
pipeline.fit(df)  # bounds estimated privately
synthetic_df = pipeline.generate(n_records=1000)
```

### Preprocessing Responsibility

When using `AIMGenerator`, preprocessing is handled automatically—it detects whether numeric
columns need private domain estimation and allocates the budget accordingly.

When using the calibration utilities directly, **you are responsible** for getting this right. The
`proc_epsilon` values passed to `calibrate_parameters_to_risk` and to `AIMPipeline` must be
consistent. The calibration function deducts the preprocessing budget from the total privacy budget
before computing the generation parameters. If dpmm performs private preprocessing without the budget
being accounted for in the calibration step, the overall privacy guarantee may not hold at the
specified risk level.

## References

[^1]: [AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data](https://arxiv.org/abs/2201.12677). VLDB 2022.
