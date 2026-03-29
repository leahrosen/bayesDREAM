# Bayesian FDR: Mathematical Derivation

This document explains the probability theory behind the `fdr_alpha` and `fdr_beta`
columns produced by `save_trans_summary`, and the FDR-based dependency criterion
used for classification and `full_log2fc` computation.

---

## 1. The Hypothesis Testing Setup

After running `fit_trans` with `function_type='additive_hill'`, every trans gene $i$
has two latent sparsity indicators:

$$\alpha_i \in [0, 1], \quad \beta_i \in [0, 1]$$

- $\alpha_i \approx 1$: the positive Hill component (component A) is active for gene $i$
- $\beta_i \approx 1$: the negative Hill component (component B) is active for gene $i$

These define two null hypotheses per gene:

| Null | Meaning |
|------|---------|
| $H_0^{(i,A)}$: component A inactive | Gene $i$ has no positive regulation |
| $H_0^{(i,B)}$: component B inactive | Gene $i$ has no negative regulation |

For `single_hill`, only $H_0^{(i,A)}$ is tested.

---

## 2. The Prior on $\alpha_i$ and $\beta_i$

The model places a **sparse prior** on whether each component is active.
In log-odds form:

$$\text{logit}(\alpha_i) \sim \mathcal{N}(\text{logit}(p_0),\; \sigma_\alpha^2)$$

with $p_0 = 10^{-6}$, so $\text{logit}(p_0) \approx -13.8$.  The identical prior is
placed on $\beta_i$.

This encodes the belief that *very few trans genes are truly regulated* by the cis
perturbation.  Only genes for which the data provides sufficient likelihood improvement
will shift away from this prior.

During variational inference, the AutoNormal guide fits mean and variance parameters
$(\mu_i, \sigma_i)$ for each component's logit, from which posterior samples
$\alpha_i^{(s)}$ are drawn.

---

## 3. The Alpha–Vmax Non-Identifiability Problem

A naive activity measure would be $E[\alpha_i \mid \text{data}]$.  This fails in
practice because $\alpha_i$ and $\text{Vmax}_{a,i}$ are **non-identified jointly**:
only their product is constrained by the data.

The Hill model for gene $i$ is:

$$y_i = A_i + \alpha_i \cdot \text{Vmax}_{a,i} \cdot h_+(x; K_{a,i}, n_{a,i})
       + \beta_i \cdot \text{Vmax}_{b,i} \cdot h_-(x; K_{b,i}, n_{b,i})$$

The data directly constrains $\alpha_i \cdot \text{Vmax}_{a,i}$ (the effective
amplitude of component A), but not $\alpha_i$ or $\text{Vmax}_{a,i}$ separately.
As a result:

- For a **true positive** gene: $\alpha_i \cdot \text{Vmax}_{a,i}$ is pulled to the
  true amplitude, but the posterior for $\alpha_i$ alone can remain at $0.6$–$0.8$
  rather than approaching 1.
- $E[\alpha_i]$ therefore stays depressed for true positives, making
  $\text{lfdr} = 1 - E[\alpha_i]$ inflated (too conservative).

---

## 4. The Product-Based Activity Probability

To avoid the non-identifiability, the activity probability is computed from the
**normalized effect size**:

$$r_i^{(s)} = \frac{\alpha_i^{(s)} \cdot \text{Vmax}_{a,i}^{(s)}}{A_i^{(s)}}$$

Dividing by $A_i$ makes the quantity expression-scale invariant (a 10% perturbation
on a lowly expressed gene has the same scale as a 10% perturbation on a highly
expressed gene).

The **local FDR** for component A of gene $i$ is then:

$$\text{lfdr}_{i,A} = 1 - P\!\left(r_i > \varepsilon \mid \text{data}\right)
= 1 - \frac{1}{S}\sum_{s=1}^{S} \mathbf{1}\!\left[r_i^{(s)} > \varepsilon\right]$$

where $\varepsilon = 0.01$ (1% of baseline) is a threshold below which the component
is considered negligibly active.

### Why epsilon is insensitive

Empirically, there is a 50+ log$_2$-unit gap between true negatives and true positives
in the posterior mean of $r_i$:

| Gene class | $\log_2 E[r_i]$ |
|------------|-----------------|
| True negatives | −30 to −54 |
| True positives | −2.7 to +7.7 |

Any $\varepsilon$ in the range $[10^{-8},\, 0.05]$ gives identical rankings and hence
identical q-values.  The default $\varepsilon = 0.01$ lies comfortably in this plateau.

Analogously for component B: $r_{i,B}^{(s)} = \beta_i^{(s)} \cdot \text{Vmax}_{b,i}^{(s)} / A_i^{(s)}$.

---

## 5. Bayesian FDR for a Called Set

Given any set $S$ of called gene/component pairs, the **posterior FDR** is:

$$\text{BayesFDR}(S) = E\!\left[\text{FDP}(S) \mid \text{data}\right]
= \frac{1}{|S|} \sum_{(i,c) \in S} \text{lfdr}_{i,c}$$

> **Key result**: the Bayesian FDR of a called set equals the *mean local FDR* over
> all called tests.  No distributional assumptions, no permutations required.

---

## 6. The Optimal Calling Rule and Q-Values

The **q-value** $q_{i,c}$ is the minimum Bayesian FDR at which test $(i,c)$ would be
included in the called set:

$$q_{i,c} = \min_{k \geq \text{rank}(i,c)} \overline{\ell}(k)$$

where $\overline{\ell}(k)$ is the cumulative mean of the $k$ smallest local FDRs.

**Calling rule**: call all tests with $q_{i,c} \leq q$ to control the expected FDR
at level $q$ (default $q = 0.05$).

**Monotonicity**: because $\overline{\ell}(k)$ is not necessarily monotone, the
running minimum from the right is applied:

$$q_{(k)} = \min\!\left(\overline{\ell}(k),\; q_{(k+1)}\right)$$

ensuring $q_{(1)} \leq q_{(2)} \leq \cdots$.  In code:

```python
cumfdr       = np.cumsum(lfdr_sorted) / (np.arange(n) + 1)
qvals_sorted = np.minimum.accumulate(cumfdr[::-1])[::-1]
```

---

## 7. Multiple Testing: Pooled Family for `additive_hill`

For `additive_hill`, there are $2n$ tests (one per component per gene).  Both
components are pooled into a **single family** of $2n$ tests:

$$\boldsymbol{\ell} = \left(
  \underbrace{\ell_{1,A}, \ldots, \ell_{n,A}}_{n \text{ alpha tests}},\;
  \underbrace{\ell_{1,B}, \ldots, \ell_{n,B}}_{n \text{ beta tests}}
\right)$$

Steps 3–6 are applied to this pooled vector.  The output is split back into
`fdr_alpha` (first $n$ entries) and `fdr_beta` (last $n$ entries).

**Why pool?** Calling `fdr_alpha ≤ 0.05` and `fdr_beta ≤ 0.05` separately controls
FDR at 5% per component family, but the combined list of calls has FDR up to ~10%.
Pooling ensures that the joint FDR over all effect calls is controlled at 5%.

**Formal guarantee**: Calling all tests with $q_{i,c}^{\text{pool}} \leq q$ satisfies:

$$\text{BayesFDR}\!\left(\{(i,c) : q_{i,c}^{\text{pool}} \leq q\}\right) \leq q \qquad \square$$

---

## 8. Full Algorithm

Given posterior samples $\{\alpha_i^{(s)}, \text{Vmax}_{a,i}^{(s)}, A_i^{(s)}\}_{s=1}^{S}$
for each gene $i$ (and analogously for component B):

**Step 1: Compute normalized effect size per sample**
$$r_{i,A}^{(s)} \leftarrow \frac{\alpha_i^{(s)} \cdot \text{Vmax}_{a,i}^{(s)}}{\max(A_i^{(s)},\, 10^{-12})}$$

**Step 2: Compute activity probability**
$$p_{i,A} \leftarrow \frac{1}{S}\sum_{s=1}^{S} \mathbf{1}[r_{i,A}^{(s)} > \varepsilon]$$

**Step 3: Compute local FDR**
$$\ell_{i,A} \leftarrow 1 - p_{i,A}$$

**Step 4: Pool and sort** (concatenate alpha and beta local FDRs for `additive_hill`)

**Step 5: Cumulative mean and running minimum** → q-values

**Step 6: Map back** to per-gene `fdr_alpha` and `fdr_beta`

**Calling rule at target FDR $q$**: call component $(i,c)$ if $q_{i,c} \leq q$.

---

## 9. FDR-Based Dependency in Classification

The q-values are not only stored as output columns — they **drive the per-feature
classification** and `full_log2fc` computation.

### Dependency masks

A component is considered **active** (dep_mask = True) if its q-value is below the
FDR threshold:

$$\text{dep\_mask}_{i,A} = q_{i,A}^{\text{pool}} < 0.05$$
$$\text{dep\_mask}_{i,B} = q_{i,B}^{\text{pool}} < 0.05$$

These are used by `_classify_additive_hill` to assign each gene to:
`flat`, `single_positive`, `single_negative`, `additive_positive`, `additive_negative`,
`non_monotonic_min`, or `non_monotonic_max`.

### `full_log2fc` / `full_delta_p`

A gene is classified as **flat** (full dynamic range = 0) if both components fail the
FDR threshold:

$$\text{is\_flat}_i = (q_{i,A}^{\text{pool}} \geq 0.05) \;\wedge\; (q_{i,B}^{\text{pool}} \geq 0.05)$$

For non-flat genes the dynamic range is computed over the theoretical x-range by
evaluating the full fitted additive Hill function.

### Consistency between output columns and classification

The q-values used internally for dep_mask are computed by the same pooled formula as
the `fdr_alpha`/`fdr_beta` output columns.  Consequently:

- A gene classified as `flat` will have `fdr_alpha >= 0.05` **and** `fdr_beta >= 0.05`
- A gene classified as `single_positive` or `single_negative` will have exactly one of
  `fdr_alpha < 0.05` or `fdr_beta < 0.05`
- A gene classified as `additive_*` or `non_monotonic_*` will have both
  `fdr_alpha < 0.05` and `fdr_beta < 0.05`

The `fdr_threshold` parameter (default 0.05) can be passed to `_add_additive_hill_params`
to change this consistently for both classification and output.

---

## 10. Calibration Considerations

The Bayesian FDR is exactly calibrated if the model is correctly specified.  In
practice, three sources of miscalibration can arise:

### 10.1 Alpha–Vmax posterior correlation

If $\alpha_i$ and $\text{Vmax}_{a,i}$ are positively correlated in the posterior
(which they will be for true positives), the mean of the product exceeds the product
of means:

$$E[\alpha_i \cdot \text{Vmax}_{a,i}] \geq E[\alpha_i] \cdot E[\text{Vmax}_{a,i}]$$

This inflates $p_{i,A}$ for true positives (good) and also slightly for true negatives
if Vmax_a has a heavy prior tail.  In practice the prior on Vmax_a is log-normal with
moderate variance, so this effect is small for null genes.

### 10.2 Misspecified prior $p_0$

The $p_0 = 10^{-6}$ prior imposes a very high KL cost for activation.  If more genes
are truly regulated than this prior predicts, posteriors will be systematically pulled
toward 0, making $p_{i,A}$ smaller for true positives and the FDR **conservative**.
This is the preferred direction of error (fewer false positives at the cost of some
power).

### 10.3 RelaxedBernoulli temperature approximation

The model samples $\alpha_i$ from a RelaxedBernoulli (BinConcrete) with temperature
$T_{\text{final}} = 0.1$, not a true Bernoulli.  Posterior samples lie in $(0, 1)$.
At $T = 0.1$ the distribution is highly concentrated near 0 and 1, so the per-sample
product $\alpha_i^{(s)} \cdot \text{Vmax}_{a,i}^{(s)}$ approximates the true
Bernoulli-gated product well.

### 10.4 VI approximation error

Mean-field VI tends to underestimate posterior variance.  For most parameters this
means the posterior mean is reliable but the tails are underrepresented.  The product
$\alpha_i \cdot \text{Vmax}_{a,i}$ is relatively robust because VI minimizes
reverse-KL, which pins the mode; the product at the mode is well-estimated.

---

## 11. Relationship to the Old CI-Based Criterion

Prior to the current implementation, dependency was assessed by checking whether 0
lies inside the 95% credible interval of $\alpha_i$ **and** $n_{a,i}$:

$$\text{dep\_mask}_{i,A}^{\text{old}} =
  \mathbf{1}[0 \notin \text{CI}_{95}(\alpha_i)] \;\wedge\;
  \mathbf{1}[0 \notin \text{CI}_{95}(n_{a,i})]$$

This approach has two fundamental problems:

1. **Alpha CI never contains 0**: the RelaxedBernoulli CI is bounded away from 0,
   so $\mathbf{1}[0 \notin \text{CI}_{95}(\alpha_i)] = 1$ for every gene.  The alpha
   check is vacuous.
2. **$n_a$ CI too wide with loose nmin/nmax**: once the Hill coefficient bounds were
   widened from ±20 to physically-derived values (~±38), the posterior CI for $n_a$
   spans 0 for essentially all genes, making the $n_a$ check also vacuous.

The current product-based FDR approach bypasses both issues entirely.

| Property | Old CI criterion | Current FDR criterion |
|----------|-----------------|----------------------|
| Basis | Posterior CI of $\alpha$ and $n_a$ | $P(\alpha \cdot \text{Vmax}_a / A > \varepsilon)$ |
| Alpha–Vmax identifiability | Ignores | Avoided by using product |
| Multiple testing | Not corrected | Pooled q-value corrected |
| Threshold | Hard (0 in CI) | Calibrated FDR = 0.05 |
| Consistent with output FDR columns | No | Yes |
