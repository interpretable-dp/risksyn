# Risk Modeling

There are two persistent myths about the privacy of synthetic data. First, that synthetization alone
ensures privacy of the original data by default. This is false[^1]. Second, that the privacy risk
can be measured in terms of similarity of synthetic records to the real data. Although such
similarity has some relation to privacy, the standard similarity measurements do not tell anything
about privacy[^2][^3]. Instead, the appropriate way to reason about privacy of synthetic data or
other data-driven releases is by analyzing the risk of relevant threats: re-identification,
inference, and reconstruction attacks—any of which could enable adversaries to learn sensitive
information about the target data by observing the synthetic dataset or the generative algorithm.
This library provides an interface which ensures privacy-by-design using the standard approach known
as differential privacy, with a tunable level of privacy risk. Differential privacy is a
mathematical framework for quantifying the privacy guarantees that one obtains from noise and
randomness, that are deliberately added to the synthetic data generation procedure. The stronger is
the privacy guarantee that we require, the more noise we have to add. In practice, we need to strike
a balance between the privacy and utility requirements.

## Attacks

Next, we describe the standard threats relevant to synthetic data releases.

### Membership Inference Attacks

An adversary aims to infer whether a given record has been part of the original dataset by
observing the synthetic dataset or the generative model. We quantify the risk of this attack
using two notions:

- **TPR** (true positive rate, sensitivity). The probability of the adversary correctly guessing
  that the record is in the original dataset when it is actually in the dataset.
- **FPR** (false positive rate, inverse specificity). The probability of the adversary incorrectly
  guessing that the record is in when it is actually not.

Any membership inference attack has both a TPR and FPR. The most dangerous attacks are those that
have very low FPR, and at least somewhat high TPR. An additional way to present the success rate
of the attack, is through the _advantage_:

- **Advantage**. The difference TPR - FPR.

Advantage is useful, because it is a single number that represents the trade-off between TPR and
FPR. Its drawback, however, is that it discards the information on whether the attack has low FPR,
which is the most dangerous setting.

### Re-identification and Singling Out

This threat has been initially conceptualized in the context of classical anonymization methods,
but naturally extends to other aggregate data-driven releases of models and synthetic data[^4].
In this setting, an adversary aims to find a simple rule, a _predicate_, e.g., a Colombian woman
over 60 that speaks Mandarin, which matches only one record in the original real dataset, after
observing the synthetic dataset or the generative model.

We quantify the risk of this attack using two notions:

- **Success rate**. The probability of the adversary successfully finding the rule that only
  matches one person.
- **Baseline**. The probability of the adversary finding the rule that only matches one person
  by chance.

To present the risk using a single number, we can use the _advantage_:

- **Advantage**. The additive difference Success rate - Baseline.

### Attribute Inference Attacks

An adversary aims to infer an attribute value of a given partial record where they lack some
parts of the record by observing the synthetic dataset or the generative model.

We quantify the risk of this attack using two notions:

- **Success rate**. The probability of the adversary successfully inferring the value of the
  attribute.
- **Baseline**. The probability of the adversary successfully inferring the value of the
  attribute by chance. For instance, if the adversary aims to infer HIV status, what is the
  chance of them guessing the HIV status correctly given only the knowledge of the partial
  record, before observing the synthetic dataset.

To present the risk using a single number, we can use the _advantage_:

- **Advantage**. The additive difference Success rate - Baseline.

### Reconstruction Attacks

An adversary aims to reconstruct a record either given a partial record or even no record to
start with by observing the synthetic dataset or the generative model.

We quantify the risk of this attack using two notions:

- **Success rate**. The probability of the adversary successfully reconstructing the relevant
  parts of the record. What constitutes the relevant part, and what is successful enough for
  the adversary—is anything. For instance, the adversary might decide it is a success if they
  reconstruct age within 3 years error and zip code within a city.
- **Baseline**. The probability of the adversary successfully reconstructing a relevant part
  of the record by chance.

To present the risk using a single number, we can use the _advantage_:

- **Advantage**. The additive difference Success rate - Baseline.

## Threat Model

All the risk measures that the library supports are within the strong-adversary threat model,
which is standard in differential privacy. In this threat model, the adversary's success is
computed based on the assumption that they could observe all of the dataset but the target
record they aim to infer some information about, and that they have access to the original
generative model and the algorithm.

A common reaction is to discard this threat model as overly strong and pessimistic to the point
of not being relevant in practice. Although this is somewhat justified, there are some crucial reasons
to stick to such strong model:

1. The mathematical framework of differential privacy enables us to provide _provable privacy
   guarantees,_ unlike many other approaches, and the reliable standard approaches to do so only
   work within this threat model. In other words, this threat model is what makes it possible to
   provide provably privacy guarantees.

2. This threat model represents the strong threat model in principle, therefore the guarantees
   we can derive must hold within arbitrary weaker, more realistic threat models. This makes the
   risk measures quite useful as they remain the same regardless of the any specific attack strategy.

## Specifying Maximum Risk

This library uses the unifying framework of reasoning about all of the threats above
simultaneously[^5]. In this framework, we need:

- **TPR/Success rate.** A single number to limit the TPR of membership inference attacks and the
  success rate of the other attacks, and
- **FPR/Baseline.** A single number to set the FPR of membership inference attacks and the baseline
  of the other attacks.

In other words, we need to specify maximum TPR/Success rate at any given FPR/Baseline, and the
library will ensure that that all of the threats above must provably satisfy these constraints.

_Note:_ The unifying approach is convenient because specifying only these two quantities enables us to limit
the success rates of many different kinds of attack. It is only, however, precise for membership
inference attacks. For the other attacks, the risk specification is pessimistic. In other words, if
we specify a certain maximum risk of reconstruction attacks, we correctly limit the risk of these
attacks, but there might not be any attack that actually reaches that risk. Thus, the protection
will be stronger than strictly necessary to limit that threat; therefore, we will have to add more
noise than strictly necessary.

### Success and Baseline vs. Advantage

A common way to quantify risk is additive advantage, often used for membership inference attacks as
TPR - FPR. It also naturally works with other attacks, where advantage is Success rate - Baseline.
We support specifying the risk as either setting TPR/Success Rate and FPR/Baseline with `Risk.from_success_at_baseline`, or Advantage and
Baseline with `Risk.from_advantage_at_baseline`, which are equivalent but different ways to specify risk.

### Additive vs. Multiplicative Advantage

When specifying risk using additive advantage at a given baseline, the same additive advantage can
correspond to very different multiplicative increases in the adversary's success rate depending on the
baseline. For example, an advantage of 0.05 at a baseline of 0.001 sounds small, but it means the
adversary's success rate increases from 0.1% to 5.1%—a 51x multiplicative increase.

The library warns when the multiplicative factor is large, as this
suggests the additive advantage may give a false sense of security. In such cases, consider using
`Risk.from_success_at_baseline` to specify the exact success rate and baseline directly, which makes
the multiplicative increase explicit.

### Combining Risk Requirements

When multiple threats are relevant—for example, both membership inference and re-identification—it
may not be obvious which one imposes the strictest privacy requirement. The `|` operator combines
multiple `Risk` specifications by selecting the minimum rho (most restrictive), automatically
choosing whichever constraint is tightest:

```python
from risksyn import Risk

# Whichever constraint is more restrictive wins
risk = Risk.from_advantage(0.01) | Risk.from_advantage_at_baseline(0.05, 0.001)
```

This ensures that all specified threat constraints are satisfied simultaneously.

### Estimating Baselines vs. Baseline-Independent Specification

For re-identification and singling out, attribute inference, and data reconstruction, we need to
estimate reasonable baseline success rates of attacks that we want to limit. Next, we provide
examples of baseline estimates:
- **Singling out.** Suppose that the adversary aims to single out a record out of a population of
  1000 potential real records. To model the fact that the adversary has no preference across the
  1000 candidate records, we can set the baseline to 1 in 1000.
- **Attribute inference.** Suppose that the adversary aims to infer an attribute which
  corresponds to a gene mutation with 10 possible value. We can model the baseline success
  rate of an attacker that guesses based on population statistics by finding the probability of the
  most common mutation, say, 1/4.  Then, we can set the baseline to 1/4.
- **Reconstruction attacks.** Suppose that the adversary aims to reconstruct an individual's yearly
  income to within 5\% error. To model the adversary that uses publicly available data and knows the
  individual's job type, we can get a distribution of income for this job type from publicly
  available sources, and compute the probability of guessing the income within 5\% error by chance
  in this distribution, say 1/10.  Then, we set the baseline to 1/10.

**Baseline-independent specification.** Sometimes, it might be challenging to faithfully estimate
baseline success rates of relevant attacks. In this case, it is possible to limit the risk in terms
of the worst-case additive advantage across all possible attacks with `Risk.from_advantage`,
corresponding to the highest possible Success Rate - Baseline across all baseline values. This
should be thought as a measure of last resort, as this is quite pessimistic, thus we will need to
add a lot of noise to limit even relatively small values of such advantage.

### Example Usage

The `Risk` class provides several factory methods for specifying privacy risk based on different
threat scenarios. Here are practical examples for each threat type.

#### Membership Inference

For membership inference attacks, you can specify risk in terms of advantage or error rates:

```python
from risksyn import Risk

# Limit the adversary's advantage (TPR - FPR) to at most 10%
risk = Risk.from_advantage(0.10)

# Alternatively, specify exact error rates:
# Limit TPR to 5% when FPR is 1% (very low false positive rate)
risk = Risk.from_err_rates(tpr=0.05, fpr=0.01)

# This is exactly equivalent to:
risk = Risk.from_success_at_baseline(success=0.05, baseline=0.01)
```

#### Re-identification, Attribute Inference, Reconstruction Attacks

For re-identification attacks, specify either the specific values of the maximum success rate at a
given baseline or additive advantage at a given baseline:

```python
from risksyn import Risk

# Baseline: 1 in 100 chance of attacker succeeding by random guessing.
# Limit success rate to at most 1 in 10 (10x the baseline)
risk = Risk.from_success_at_baseline(success=0.1, baseline=0.01)

# Same risk: at most 10x the baseline is 0.09 additive advantage.
risk = Risk.from_advantage_at_baseline(advantage=0.09, baseline=0.01)

# Baseline-independent specification of maximum advantage.
risk = Risk.from_advantage(advantage=0.09)
```

To use the `Risk` specification for synthetic data generation, see the
[Generation](generation.md) guide.

## Advanced Usage

**Under the hood.** The `Risk` class maps the given risk notions to a zCDP parameter.  During the
`fit()` call, the library then maps zCDP back to epsilon/delta as a crutch, as the backend library
`dpmm` does not support a direct zCDP parameterization. All the described risk notions correspond to
a single point on the f-DP trade-off curve. As zCDP is a single-parameter privacy notion, a single
point on the curve uniquely determines the zCDP parameter.

**Additional notions of risk.** The guide above employs the unifying framework[^5] that uses the
f-DP trade-off curve to loosely upper bound success rates at given baselines for various attacks.
For specific threat models, e.g., binary attribute inference, it is possible to obtain more precise
bounds. In particular, it is possible to map f-DP or privacy profile to Bayes risk under binary
attribute inference (see Appendix D in the unifying framework paper[^5]). The current library's
interface does not support directly specifying risk in these terms, but you can map any notion of
risk to f-DP and supply that, ignoring the suggested interpretation of the points on the curve as
success/baseline.

## References

[^1]: [Synthetic Data – Anonymisation Groundhog Day](https://arxiv.org/abs/2011.07018). Usenix 2022.
[^2]: [On the Inadequacy of Similarity-based Privacy Metrics](https://arxiv.org/abs/2312.05114). IEEE S&P 2025.
[^3]: [The DCR Delusion: Measuring the Privacy Risk of Synthetic Data](https://arxiv.org/abs/2505.01524). ESORICS 2025.
[^4]: [Towards Formalizing the GDPR's Notion of Singling Out](https://arxiv.org/abs/1904.06009). PNAS 2020.
[^5]: [Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks](https://arxiv.org/pdf/2507.06969). NeurIPS 2025.
