# Bayesian FDR: Mathematical Derivation

This document explains the probability theory behind the `fdr_alpha` and `fdr_beta`
columns produced by `save_trans_summary`.

---

## 1. The Hypothesis Testing Setup

After running `fit_trans` with `function_type='additive_hill'`, every trans gene $i$
has two latent binary indicators:

$$\alpha_i \in \{0, 1\}, \quad \beta_i \in \{0, 1\}$$

- $\alpha_i = 1$: the positive Hill component (component A) is active for gene $i$
- $\beta_i = 1$: the negative Hill component (component B) is active for gene $i$

These define four null hypotheses per gene:

| Null | Meaning |
|------|---------|
| $H_0^{(i,A)}: \alpha_i = 0$ | Gene $i$ has no positive regulation |
| $H_0^{(i,B)}: \beta_i = 0$ | Gene $i$ has no negative regulation |

For `single_hill`, only $H_0^{(i,A)}$ is tested.

---

## 2. The Prior on $\alpha_i$ and $\beta_i$

The model places a **sparse prior** on whether each component is active:

$$\alpha_i \sim \text{Bernoulli}(p_0), \quad p_0 = 10^{-6}$$

In log-odds form, this is $\text{logit}(p_0) \approx -13.8$.  The identical prior is placed on $\beta_i$.

The prior encodes the belief that *very few genes are truly regulated* by the cis
perturbation.  With $n = 10{,}000$ trans genes, the prior predicts approximately 0.01
truly regulated genes — an extremely conservative assumption, intended to impose
strong sparsity.

During variational inference, the AutoNormal guide fits a Normal distribution on the
logit of each component:

$$\text{logit}(\alpha_i) \sim \mathcal{N}(\mu_i, \sigma_i^2)$$

where $\mu_i$ and $\sigma_i$ are learned variational parameters.  The posterior mean is
approximated from samples:

$$E[\alpha_i \mid \text{data}] \approx \frac{1}{S} \sum_{s=1}^{S} \alpha_i^{(s)}$$

where $\alpha_i^{(s)}$ are draws from the guide.

---

## 3. Local False Discovery Rate

The **local false discovery rate** (lfdr) for component $A$ of gene $i$ is the posterior
probability that the null hypothesis is true:

$$\text{lfdr}_{i,A} = P(H_0^{(i,A)} \mid \text{data}) = P(\alpha_i = 0 \mid \text{data}) = 1 - E[\alpha_i \mid \text{data}]$$

Intuitively:
- $\text{lfdr}_{i,A} \approx 1$: the data gives almost no evidence that component A is active; this gene is almost certainly a false positive if called
- $\text{lfdr}_{i,A} \approx 0$: the posterior is concentrated at $\alpha_i = 1$; this gene is almost certainly a true positive

This is the Bayesian analogue of a frequentist p-value.  Unlike a p-value, it is a
**direct probability statement about the hypothesis** given the data.

---

## 4. Bayesian FDR for a Called Set

Given any set $S$ of called genes/components, define the **false discovery proportion** (FDP):

$$\text{FDP}(S) = \frac{\#\{(i,c) \in S : H_0^{(i,c)} \text{ true}\}}{|S|}$$

The (frequentist) FDR is $E[\text{FDP}(S)]$.  In the Bayesian framework, after
observing the data, we have a **posterior FDR**:

$$\text{BayesFDR}(S) = E\left[\text{FDP}(S) \mid \text{data}\right]$$

Expanding:

$$\text{BayesFDR}(S)
= E\left[\frac{1}{|S|}\sum_{(i,c) \in S} \mathbf{1}[H_0^{(i,c)} \text{ true}] \;\middle|\; \text{data}\right]
= \frac{1}{|S|} \sum_{(i,c) \in S} P(H_0^{(i,c)} \mid \text{data})
= \frac{1}{|S|} \sum_{(i,c) \in S} \text{lfdr}_{i,c}$$

> **Key result**: the Bayesian FDR of a called set is simply the *mean of the local FDRs*
> over all called tests.  No distributional assumptions, no permutations needed.

---

## 5. The Optimal Calling Rule

**Goal**: find the largest set $S$ such that $\text{BayesFDR}(S) \leq q$ (a target level,
e.g., $q = 0.05$).

**Claim**: the optimal set is a **top-$k$ set** obtained by sorting tests in ascending
order of local FDR and calling the $k$ most significant, where $k$ is the largest
integer satisfying $\overline{\ell}(k) \leq q$.

*Proof sketch*: Suppose $S^*$ is an optimal set of size $k$.  Any test
$(i,c) \notin S^*$ with $\text{lfdr}_{i,c} > \overline{\ell}(k)$ would increase the
mean lfdr if added.  Any test in $S^*$ with $\text{lfdr}_{i,c} > \overline{\ell}(k)$
could be replaced by a test outside $S^*$ with lower lfdr, reducing the mean.  So at
optimality, $S^*$ is exactly the set of tests with the $k$ smallest local FDRs.

Define the **cumulative mean** at rank $k$ (sorting by lfdr ascending):

$$\overline{\ell}(k) = \frac{1}{k} \sum_{j=1}^{k} \ell_{(j)}$$

where $\ell_{(1)} \leq \ell_{(2)} \leq \cdots$ are the order statistics.  Then the
optimal threshold is:

$$k^* = \max\left\{k : \overline{\ell}(k) \leq q\right\}$$

---

## 6. Q-Values and the Running Minimum

The **q-value** $q_{i,c}$ is the minimum Bayesian FDR at which test $(i,c)$ would be
included in the called set:

$$q_{i,c} = \min_{k \geq \text{rank}(i,c)} \overline{\ell}(k)$$

where $\text{rank}(i,c)$ is the position of test $(i,c)$ in the sorted list.

**Interpretation**: if you want to call all tests with $q_{i,c} \leq q$, the expected
FDR among those calls is at most $q$.  This is directly analogous to a
Benjamini-Hochberg adjusted p-value.

**Why not just use $\overline{\ell}(\text{rank}(i,c))$ directly?**

The cumulative mean $\overline{\ell}(k)$ is not necessarily monotone.  It satisfies:

$$\overline{\ell}(k+1) = \frac{k \cdot \overline{\ell}(k) + \ell_{(k+1)}}{k+1}$$

so $\overline{\ell}$ *decreases* when $\ell_{(k+1)} < \overline{\ell}(k)$ (the $(k+1)$th
gene has lower lfdr than the current average) and *increases* otherwise.

If $\overline{\ell}(k_1) > \overline{\ell}(k_2)$ for some $k_2 > k_1$, then the test at
rank $k_1$ can be included in the larger called set $\{1,\ldots,k_2\}$ at the lower FDR
$\overline{\ell}(k_2)$.  So the minimum FDR to include rank-$k_1$ is
$\min_{k \geq k_1} \overline{\ell}(k)$, which may be strictly less than $\overline{\ell}(k_1)$.

The running minimum from the right:

$$q_{(k)} = \min\left(\overline{\ell}(k),\, q_{(k+1)}\right) \quad \text{(evaluated right to left)}$$

ensures **monotonicity**: $q_{(1)} \leq q_{(2)} \leq \cdots$.  In code:

```python
cumfdr       = np.cumsum(lfdr_sorted) / (np.arange(n) + 1)
qvals_sorted = np.minimum.accumulate(cumfdr[::-1])[::-1]
```

---

## 7. Multiple Testing: Pooling Alpha and Beta for `additive_hill`

For `additive_hill` there are $2n$ tests (one alpha and one beta per gene).  The question
is: should we compute q-values separately for the $n$ alpha tests and the $n$ beta tests,
or pool them into one family?

**Pooling is correct** when you want to control FDR over all effect calls jointly.
Here is why this matters in practice:

### Separate families (wrong for joint control)

If you compute q-values separately:
- Alpha q-values are computed over $n$ tests
- Beta q-values are computed over $n$ tests

Calling all genes with `fdr_alpha <= 0.05` controls FDR at 5% among positive-component
calls.  Calling all genes with `fdr_beta <= 0.05` controls FDR at 5% among
negative-component calls.

But the *combined* list of called effects has an FDR that could approach 10%: you are
making two independent 5%-FDR calls per gene, so a null gene can be falsely called
on either component.

### Pooled family (implemented in bayesDREAM)

Stack the $2n$ local FDRs into one array:

$$\boldsymbol{\ell} = \left(\underbrace{\ell_{1,A}, \ldots, \ell_{n,A}}_{n \text{ alpha tests}},\; \underbrace{\ell_{1,B}, \ldots, \ell_{n,B}}_{n \text{ beta tests}}\right)$$

Sort all $2n$ values, compute the cumulative mean and running minimum as before.
The q-value for component $(i,c)$ is its position in this joint ranking.

**Effect on stringency**: a test with local FDR $\ell$ gets a higher (more conservative)
q-value when pooled than when ranked against only $n$ tests, because there are more
competing tests with small lfdr.  Concretely, if exactly half the genes have active
alpha and half have active beta (both low lfdr), pooling effectively doubles the
number of strong tests, which pushes the cumulative mean at every rank downward —
making q-values *smaller* (less conservative).  Conversely, if only a handful of
genes have very low lfdr in either component, each low-lfdr test ranks first in the
$2n$ pool just as it would in the $n$ pool.

The practical effect is approximately a 2× increase in multiple-testing stringency
for null genes, matching the intuition that `additive_hill` has twice the number of
testing opportunities.

### Formal statement

**Proposition**: Calling all tests $(i,c)$ with $q_{i,c}^{\text{pool}} \leq q$ controls
the Bayesian FDR over the joint family of $2n$ hypotheses at level $q$:

$$\text{BayesFDR}\left(\{(i,c) : q_{i,c}^{\text{pool}} \leq q\}\right) \leq q$$

*Proof*: Let $S_q = \{(i,c) : q_{i,c}^{\text{pool}} \leq q\}$.  By definition of the
q-value and the running minimum construction, $S_q$ is a top-$k$ set for some $k$
satisfying $\overline{\ell}(k) \leq q$.  Therefore:

$$\text{BayesFDR}(S_q) = \overline{\ell}(k) \leq q \qquad \square$$

---

## 8. Connection to the KL Cost in the ELBO

The ELBO for a feature with $\text{logit}(\alpha_i) \sim \mathcal{N}(\mu_i, \sigma_i^2)$
in the guide and $\text{logit}(\alpha_i) = \text{logit}(p_0)$ in the prior includes:

$$-\text{KL}\left(\mathcal{N}(\mu_i, \sigma_i^2) \;\|\; \delta_{\text{logit}(p_0)}\right)$$

For a feature where the posterior has concentrated to $\alpha_i \approx 1$, the KL
from the prior logit $\approx -13.8$ is approximately:

$$\text{KL} \approx \log\frac{1 - p_0}{p_0} \approx \log\left(10^6\right) \approx 13.8 \text{ nats}$$

The model will only pay this KL cost if the data likelihood improvement for gene $i$
exceeds 13.8 nats.  In terms of posterior probability:

$$E[\alpha_i] = P(\alpha_i = 1 \mid \text{data}) = \frac{p_0 \cdot P(\text{data} \mid \alpha_i=1)}{p_0 \cdot P(\text{data} \mid \alpha_i=1) + (1-p_0) \cdot P(\text{data} \mid \alpha_i=0)}$$

Rearranging in terms of the Bayes factor $\text{BF}_{10} = P(\text{data} \mid \alpha_i=1) / P(\text{data} \mid \alpha_i=0)$:

$$E[\alpha_i] = \frac{p_0 \cdot \text{BF}_{10}}{p_0 \cdot \text{BF}_{10} + (1 - p_0)}$$

To reach $E[\alpha_i] = 0.5$ (equally probable active/inactive), the data must provide:

$$\text{BF}_{10} = \frac{1 - p_0}{p_0} \approx 10^6$$

The $p_0 = 10^{-6}$ prior is therefore a **strong regularizer**: only features with
overwhelming data support will have $E[\alpha_i]$ near 1 and hence low lfdr.

---

## 9. Calibration Considerations

The Bayesian FDR is exactly calibrated *if* the model is correctly specified.  In
practice, three sources of miscalibration can arise:

### 9.1 Misspecified prior $p_0$

If more than 1-in-$10^6$ genes are truly regulated, the posterior $E[\alpha_i]$ will
be systematically underestimated (because the model "charges" too high a KL penalty
for activation).  The resulting local FDRs will be too large, and the Bayesian FDR
will be **conservative** — you will call fewer genes than you should at any given $q$.

If very few genes are truly regulated (fewer than 1-in-$10^6$), the reverse applies,
but this is an extreme scenario.

### 9.2 RelaxedBernoulli temperature approximation

The model samples $\alpha_i$ from a **RelaxedBernoulli** (BinConcrete) distribution
with temperature $T$, not a true Bernoulli.  The FDR calculation treats

$$E[\alpha_i] \approx P(\alpha_i = 1 \mid \text{data})$$

but this is only exact as $T \to 0$.  At finite $T$, the posterior samples lie in
$(0, 1)$ rather than $\{0, 1\}$, and the posterior mean is:

$$E[\text{RelaxedBernoulli}(T, \sigma(\mu_i))] = \sigma(\mu_i) \cdot f(T)$$

where $f(T) \to 1$ as $T \to 0$.  At the default final temperature $T = 0.1$, the
relaxation is highly concentrated near 0 and 1, so the approximation is accurate.
At higher temperatures (e.g., the warmup phases where $T \in [0.5, 1.0]$), the
posterior means are "smeared" and would give unreliable local FDRs — but the summary
is only ever called on the final converged posterior, which uses $T_\text{final} = 0.1$.

### 9.3 VI approximation error

AutoNormal uses a mean-field Normal guide, which may underestimate posterior variance
(a known property of mean-field VI).  Underestimated variance means the posterior mean
$E[\alpha_i]$ can be pulled toward 0.5 rather than the true value, making the lfdr
less discriminating.

### 9.3 Model misspecification

If the true dose-response is not well-approximated by an additive Hill function, the
likelihood $P(\text{data} \mid \alpha_i)$ is mis-evaluated for both $\alpha_i = 0$ and
$\alpha_i = 1$, making the Bayes factors and hence local FDRs unreliable.

---

## 10. Summary of the Algorithm

Given posterior samples $\{\alpha_i^{(s)}\}_{s=1}^{S}$ for each gene $i$:

**Step 1: Compute posterior means**
$$E_i \leftarrow \frac{1}{S}\sum_{s=1}^{S} \alpha_i^{(s)} \in [0,1]$$

**Step 2: Compute local FDRs**
$$\ell_i \leftarrow 1 - E_i$$

**Step 3: Sort ascending**
$$\ell_{(1)} \leq \ell_{(2)} \leq \cdots \leq \ell_{(n)}$$

**Step 4: Compute cumulative mean**
$$\overline{\ell}(k) = \frac{1}{k}\sum_{j=1}^{k} \ell_{(j)} = \text{BayesFDR if top-}k \text{ called}$$

**Step 5: Running minimum (ensure monotonicity)**
$$q_{(k)} = \min_{k' \geq k}\, \overline{\ell}(k')$$

**Step 6: Map back to original order**

The q-value for gene $i$ at its rank $r_i$ is $q_{(r_i)}$.

**For `additive_hill`**: concatenate the $n$ alpha local FDRs and $n$ beta local FDRs
into a single vector of length $2n$, apply Steps 3–6 to this pooled vector, then split
the output back into `fdr_alpha` (first $n$) and `fdr_beta` (last $n$).

**Calling rule at target FDR $q$**: call all tests with q-value $\leq q$.  The
expected fraction of false positives among the called set will be at most $q$.

---

## 11. Relationship to Your CI-Based Criterion

The existing `classification` column determines whether a component is active based on
whether 0 is inside the 95% credible interval of $n_a$ (or $n_b$).  Formally:

$$\text{dep\_a}_i = \mathbf{1}[0 \notin \text{CI}_{95}(\alpha_i)] \;\wedge\; \mathbf{1}[0 \notin \text{CI}_{95}(n_{a,i})]$$

This is a **hard threshold** applied to two parameters jointly.  By contrast, `fdr_alpha`
is a **soft continuous ranking** based solely on the posterior probability that
$\alpha_i > 0$.

| Property | CI criterion | `fdr_alpha` |
|----------|-------------|-------------|
| Input | Posterior of $\alpha$ and $n_a$ | Posterior of $\alpha$ only |
| Output | Binary (active/inactive) | Continuous probability |
| Threshold | Fixed (0 in 95% CI) | User-chosen FDR level |
| Multiple testing | Not corrected | Corrected via q-value |
| Interpretable as | "Both indicators non-zero" | "Expected FDR at this threshold" |

A gene can pass the CI criterion but have high `fdr_alpha` if most other genes also
show evidence (shifting the cumulative mean up), or fail the CI criterion but have
low `fdr_alpha` if the data strongly support $\alpha > 0$ even though the CI marginally
crosses 0.  Plotting `fdr_alpha` against `dep_a` (from the classification) shows how
these two views of evidence correlate.
