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
  the adversary–is anything. For instance, the adversary might decide it is a success if they
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

- (a) a single number to limit the TPR of membership inference attacks and the success rate of
  the other attacks, and
- (b) a single number to set the FPR of membership inference attacks and the baseline of the
  other attacks.

In other words, we need to specify maximum TPR/Success Rate at any given FPR/Baseline, and the
library will ensure that that all of the threats above must provably satisfy these constraints.

In this unifying approach, we can precisely characterize the TPR and FPR of membership inference
attacks. For the other attacks, however, the cost of the simple unifying approach is losing the
precision. In other words, if we specify a certain maximum risk of reconstruction attacks, we
will indeed limit the risk of these attacks, but there might not be any attack that actually
reaches that risk, thus the protection will be stronger than strictly necessary to limit that
threat. Therefore, we will have to add more noise than strictly necessary.

### Examples

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
```

#### Re-identification and Singling Out

For re-identification attacks, specify the maximum success rate at a given baseline:

```python
from risksyn import Risk

# Baseline: 1 in 10,000 chance of singling out by random guessing
# Limit success rate to at most 1 in 1,000 (10x the baseline)
risk = Risk.from_success_at_baseline(success=0.001, baseline=0.0001)
```

#### Attribute Inference and Reconstruction

For attribute inference, specify the advantage over random guessing:

```python
from risksyn import Risk

# Baseline: 50% chance of guessing a binary attribute (e.g., HIV status)
# Allow at most 5% advantage over baseline
risk = Risk.from_advantage_at_baseline(advantage=0.05, baseline=0.50)
```

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
