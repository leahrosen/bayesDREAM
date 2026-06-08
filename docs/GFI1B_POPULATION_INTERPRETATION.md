# GFI1B Trans-Effect Gene Population Interpretation

**Context**: Transcriptome-wide bayesDREAM run with GFI1B as cis gene (high MOI, CRISPRi/a
dataset). All parameters are from the `additive_hill` trans fit. EC50 values are in log2FC
space (log2 of x_true relative to NTC mean). The additive Hill model has two arms: arm a
(typically capturing positive effects) and arm b (typically capturing negative effects), each
with an independent coefficient and EC50.

This document characterises each visually distinct gene population, assesses whether it
represents true GFI1B-mediated biology or a methodological false positive, and identifies
the mechanism responsible where applicable.

---

## Summary table

| ID | Significance | log2 NTC expr | EC50\_a (log2FC) | Hill\_a coeff | EC50\_b (log2FC) | Hill\_b coeff | True signal? |
|----|-------------|---------------|-----------------|--------------|-----------------|--------------|-------------|
| A1 | a only | −3 to −10 | −3 to −5 | moderate +, distributed | — | — | No |
| A2 | a only | < −5 | 0 to +1 | ≈ 0 | — | — | No |
| A3 | a only | −5 to 0 | bimodal: −2 to −1 or ≈ 0 | ≈ 0, slightly − | — | — | Unlikely |
| B1 | both | 0 to +5 | −2 to −5 | very large + | spike near 0 | very large − | Partly |
| B2 | both | > 0 | < −4 | slightly + | diffuse | ≈ 0 | Unlikely |
| B3 | both | > 0 | spike ≈ −1 | 5–20 (very large +) | spike near 0 | very large − | No |
| B4 | both | −5 to 0 | diffuse | ≈ 0 | diffuse | ≈ 0 | No |
| C1 | b only | < −10 | — | — | diffuse | ≈ 0 | No |

---

## Only Hill a significant

### A1 — Lowly expressed, moderate positive Hill_a, very negative EC50_a

| Parameter | Value |
|-----------|-------|
| log2 NTC expression | −3 to −10 |
| EC50\_a | −3 to −5 (log2FC) |
| Hill\_a coefficient | Moderate positive; distributed (not spiked) |
| EC50\_b | — |
| Hill\_b coefficient | — |

**Interpretation: False positive.**

These are genes barely detected in NTC cells. The positive Hill_a coefficient indicates apparent
upregulation as GFI1B decreases (if CRISPRi) or a response centred well below the NTC level.
The very negative EC50_a places the response midpoint deep in the low-x_true / zero-count
regime — at x_true values corresponding to 0–1 raw GFI1B counts.

The mechanism is the **residual sum_factor step at the 0→1 count boundary**. After
`refit_sumfactor`, a smooth spline is subtracted from the sum_factor–x_true relationship,
but the abrupt jump between the 0-count cluster and the 1-count cluster cannot be removed
by a smooth function. For a gene with very low NTC expression (mean count ≈ 0.01–0.1),
even a 5–10% residual sum_factor error translates to a large apparent fold-change, and the
Hill model fits this as a wide positive lobe with its midpoint at the count boundary. Because
the residual is present for every gene, and always at the same x_true location, the EC50 is
broadly distributed around the same negative log2FC values rather than spiked (the exact
boundary position varies slightly across cells depending on sum_factor magnitude).

The distributed (non-spiked) EC50 distinguishes A1 from B3: these genes are fitting
**diffuse sum_factor residual variance**, not a sharp step.

---

### A2 — Very lowly expressed, near-zero Hill_a, EC50_a near NTC level

| Parameter | Value |
|-----------|-------|
| log2 NTC expression | < −5 |
| EC50\_a | 0 to +1 (log2FC) |
| Hill\_a coefficient | ≈ 0 (very small) |
| EC50\_b | — |
| Hill\_b coefficient | — |

**Interpretation: False positive (noise floor).**

These genes are extremely lowly expressed. The Hill_a coefficient is near zero, meaning
the model has fit a dose-response that is statistically credible but biologically negligible
in magnitude. EC50 near zero means the midpoint is at or above the NTC level — the model
is fitting variation in the high-x_true / well-expressed GFI1B targeting cells rather than
in the count-boundary regime.

The likely mechanism is that for extremely lowly expressed genes, the NB likelihood is so
weakly informative that even tiny systematic variation in sum_factor or guide effects (from
high MOI contamination or NTC heterogeneity) produces a credible posterior for a small
non-zero Hill_a. The significance threshold is met not because the effect is large but
because the posterior interval for a small positive Hill_a excludes zero. This is a
**signal-to-noise false positive**: the posterior reflects genuine data variation but that
variation is not GFI1B-mediated.

---

### A3 — Moderate expression, Hill_a ≈ 0 (slightly negative), bimodal EC50_a

| Parameter | Value |
|-----------|-------|
| log2 NTC expression | −5 to 0 |
| EC50\_a | Bimodal: mode at −2 to −1 AND mode at ≈ 0 |
| Hill\_a coefficient | ≈ 0, slightly negative; distributed |
| EC50\_b | — |
| Hill\_b coefficient | — |

**Interpretation: Unlikely to be true signal; two overlapping false-positive sub-mechanisms.**

This is the largest population and appears bimodal in EC50_a, suggesting two overlapping
sub-populations with different fitting behaviours:

- **EC50_a mode at −2 to −1**: The Hill model is partially fitting the 0→1 GFI1B count
  boundary step, which in these moderately expressed trans genes produces a small but
  detectable change in expected counts. The slightly negative Hill_a suggests the model
  is fitting a **negative** response to decreasing GFI1B — i.e., these genes go down as
  GFI1B is lost, which is consistent with genes that are co-regulated with the general
  differentiation state (sum_factor confound) rather than specific GFI1B targets.

- **EC50_a mode at ≈ 0**: Fitting variation near the NTC level, similar to A2 but for more
  highly expressed genes. The wider x_true range for these genes means the Hill_a arm can
  find a midpoint at the NTC mean with a near-zero coefficient.

The bimodal EC50 with a distributed Hill_a near zero is the hallmark of a population where
the model is tracking **general differentiation-state variation** rather than a specific
GFI1B dose-response. Some genes in this population may have genuine GFI1B-correlated
expression, but they cannot be distinguished from the confound without a permutation null.

---

## Both Hill a and Hill b significant

### B1 — Well-expressed, very large positive Hill_a, very large negative Hill_b

| Parameter | Value |
|-----------|-------|
| log2 NTC expression | 0 to +5 |
| EC50\_a | −2 to −5 (log2FC) |
| Hill\_a coefficient | Very large positive (≫ 0) |
| EC50\_b | Spike near 0 (log2FC) |
| Hill\_b coefficient | Very large negative |

**Interpretation: Partly true signal, amplified by artifact.**

These well-expressed genes show the largest Hill coefficients in the dataset — large positive
Hill_a at low x_true and large negative Hill_b near the NTC level. The large positive Hill_a
is consistent with genuine GFI1B biology: as GFI1B is reduced (CRISPRi) genes involved in
alternative lineage programs (e.g., myeloid differentiation) are upregulated. The EC50 in the
−2 to −5 range places this midpoint in the regime where GFI1B is substantially reduced below
NTC — consistent with a threshold effect in differentiation.

However, **two artifacts amplify these effects**:

1. The very large negative Hill_b coefficient paired with a spike in EC50_b near 0 is the
   model fitting the **within-bin negative slope** (see below, and Image 5 in the diagnosis
   doc). Cells with the same raw GFI1B count but higher differentiation score (higher sum_factor)
   have higher x_true — but the trans gene is genuinely co-regulated with differentiation state,
   so the trans model sees this as a sharp downward response near the NTC level. This component
   is **confounded biology** (real correlation, wrong attribution).

2. The sum_factor step at the 0→1 boundary inflates the apparent magnitude of Hill_a because
   it creates a large jump in trans gene expression at low x_true — the model cannot distinguish
   between "these cells have low GFI1B and high trans gene expression due to differentiation
   state" and "GFI1B drives this trans gene."

**B1 likely contains real GFI1B trans targets, but the Hill_b arm and the magnitude of Hill_a
are inflated by confound.** A permutation null would be needed to separate true targets from
the confounded population.

---

### B2 — Well-expressed, slightly positive Hill_a, very negative EC50_a

| Parameter | Value |
|-----------|-------|
| log2 NTC expression | > 0 |
| EC50\_a | < −4 (log2FC) |
| Hill\_a coefficient | Slightly positive |
| EC50\_b | Diffuse |
| Hill\_b coefficient | ≈ 0 |

**Interpretation: Unlikely true signal; weak diffuse response in the zero-count regime.**

The very negative EC50_a places the midpoint in the extreme low-x_true / 0-count regime.
For well-expressed trans genes, the residual sum_factor step at the 0→1 boundary creates
a small but detectable systematic change that the Hill model fits with a modest positive
coefficient. The near-zero Hill_b with diffuse EC50_b suggests the model has fit only
the ascending component (at very low x_true) with a small slope.

This population is likely a **weaker version of B1**: the same confound, but where the
trans gene's expression is less tightly coupled to differentiation state, so the fitted
Hill_a is small and the Hill_b arm barely qualifies for significance. The diffuse EC50_b
argues against a specific dose-response midpoint.

---

### B3 — Well-expressed, EC50_a spike at ≈ −1, very large positive Hill_a, very large negative Hill_b

| Parameter | Value |
|-----------|-------|
| log2 NTC expression | > 0 |
| EC50\_a | Spike at ≈ −1 (log2FC) |
| Hill\_a coefficient | 5–20 (very large positive) |
| EC50\_b | Spike near 0 (log2FC) |
| Hill\_b coefficient | Very large negative (down to ≈ −40) |

**Interpretation: False positive — the clearest artifact in the dataset.**

The discrete EC50_a spike at a single log2FC value (≈ −1) is the definitive signature of
the **0→1 GFI1B raw count boundary**. In x_true space, cells with 0 raw GFI1B counts are
clustered at a value that translates to approximately −1 log2FC below NTC. The abrupt jump
from the 0-count cluster to the 1-count cluster is exactly what a Hill sigmoid is optimised
to fit: it finds this step, places EC50_a at the midpoint, and assigns a very large Hill_a
coefficient to capture the sharpness of the step.

The simultaneously very large negative Hill_b (EC50_b near 0) is the model fitting the
**within-bin negative slope** that persists after refit: within the 1-count cluster, cells
with higher differentiation score (higher sum_factor / x_true) show lower trans gene
expression. The additive Hill model accounts for this by adding a sharp downward Hill_b
component centred at the NTC level.

Together these two arms are fitting a single step function as a biphasic Hill curve:
sharp rise at the 0→1 boundary (Hill_a), then partial suppression within the 1-count
bin (Hill_b). This is entirely artifactual. The spiked EC50 (not distributed) is the
conclusive evidence — genuine biology would produce a spread of EC50 values across the
x_true range, not a spike at the count boundary for every gene in this population.

The extremely negative Hill_b values (≤ −10, sometimes ≤ −40) occur when the within-bin
slope is steep relative to the trans gene's expression level. These extreme values are
essentially fitting a near-vertical step on the sub-integer scale, which is numerically
possible but biologically incoherent.

---

### B4 — Moderate expression, both Hill_a and Hill_b ≈ 0, EC50_b diffuse

| Parameter | Value |
|-----------|-------|
| log2 NTC expression | −5 to 0 |
| EC50\_a | Diffuse |
| Hill\_a coefficient | ≈ 0 |
| EC50\_b | Diffuse |
| Hill\_b coefficient | ≈ 0 |

**Interpretation: False positive — both arms near zero, both pass the credibility threshold.**

These genes occupy the middle of the expression range and show essentially no dose-response
in either arm, yet both pass the significance threshold. The likely mechanism is that for
moderately expressed genes, the posterior for both Hill_a and Hill_b has sufficient mass
away from zero (driven by the diffuse sum_factor residual variation across the x_true range)
to be called significant, even though the point estimates are near zero and the effect sizes
are negligible.

This population is the **background inflation** of the significance test: when every gene
has a small, non-specific dose-response-like signature due to the sum_factor confound,
even near-zero Hill coefficients can be statistically credible. The diffuse EC50_b confirms
there is no specific dose-response structure — the b arm midpoint wanders across the x_true
range without clustering at any biologically interpretable value.

---

## Only Hill b significant

### C1 — Extremely lowly expressed, Hill_b ≈ 0

| Parameter | Value |
|-----------|-------|
| log2 NTC expression | < −10 |
| EC50\_a | — |
| Hill\_b coefficient | ≈ 0 |
| EC50\_b | Diffuse |

**Interpretation: False positive (noise floor, negative arm).**

The mirror image of A2 but for the b arm. These genes are so lowly expressed that they
fall below the reliable detection threshold of the NB likelihood. The near-zero Hill_b
coefficient and diffuse EC50_b indicate no structured dose-response. Significance is
passed because the posterior for a small negative Hill_b narrowly excludes zero, driven by
stochastic variation in the very low counts rather than any GFI1B signal.

These genes are essentially **not detected** in this experiment. The b-arm significance
is a consequence of the model fitting noise in the count distribution of near-zero-expressed
genes.

---

## Cross-cutting notes

### The EC50_b spike and extreme negative Hill_b coefficients

The very large negative Hill_b coefficients (down to −40) visible in the bottom-left scatter
appear in populations B1 and B3, and appear to require Hill_a > 2–3 to co-occur. The
mechanism is consistent across both: the **within-bin negative slope** (Image 5 in the
diagnosis document). Within each raw GFI1B count bin, cells with higher x_true (higher
differentiation score) tend to have lower sum_factor — this is real biology (differentiation
→ changed total RNA). The trans model sees this as a downward response near the NTC/1-count
level, fits it with a sharp negative Hill_b centred near EC50_b ≈ 0, and assigns a large
negative coefficient to capture its steepness.

This explains why extreme Hill_b requires Hill_a > 2–3: both large effects arise from
the same step-function structure, where the a arm fits the between-bin (0→1 count) step
and the b arm fits the within-bin slope. Genes where the between-bin step is large (hence
large Hill_a) are typically well-expressed genes strongly coupled to differentiation state
— these are the same genes where the within-bin slope (hence large negative Hill_b) is
also most visible.

### What distinguishes true signal from artifact

No single parameter reliably separates true biology from confound in this run. However,
the following are informative:

| Feature | More likely artifact | More likely true signal |
|---------|---------------------|------------------------|
| EC50_a distribution | Spiked at a discrete value | Broadly distributed |
| log2 NTC expression | < −3 | > 0 |
| Hill_b coefficient | Very large negative (< −5) | Moderate or zero |
| EC50_a value | Matches a count boundary | In the biologically plausible range |
| Hill_a directionality | Positive for all genes equally | Directionally consistent with known biology |

The **permutation null** (`--permtype All`) is the definitive test. If the same populations
and parameter distributions appear in the null run, the FDR calibration is still technically
valid (comparing a confounded real run to an equally confounded null), and the significant
genes are those whose confounded signal is specifically stronger than the average. If the
null run does not show these populations, the real run contains genuine structured signal
that is being misattributed.

### EC50 prior bias and its contribution to false positives

The K_a (EC50) prior is LogNormal. Before a recent fix, `K_log_mu` was set as
`log(x_ntc_mean) − 0.5 × sigma²`, which centres the **mean** of K_a in linear space
at x_ntc_mean but shifts the **log2FC distribution** 2.2 log2FC *below* NTC:

| Statistic | Before fix | After fix |
|-----------|-----------|-----------|
| Mean of K_a (linear) | x_ntc_mean | 4.5 × x_ntc_mean |
| Median of K_a (log2FC) | −2.2 | 0 |
| Mode of K_a (log2FC) | −6.5 | −4.3 |
| 95% CI (log2FC) | [−7.2, +2.8] | [−5.0, +5.0] |

The pre-fix prior placed most of its density at EC50 values below the NTC level. For a
CRISPRi dataset, the targeting cells span roughly −6 to 0 log2FC, so the median at −2.2
was within the range — the bias was non-catastrophic. For a CRISPRa dataset (targeting
cells > 0 log2FC), the entire prior mode would have been below the observed range, making
a shallow-gradient fit (EC50 << observed x_true) the prior-preferred explanation for any
upward trend, strongly amplifying false positives.

The **shallow-gradient mechanism**: when EC50 << the observed x_true range, the Hill
function is evaluated only on its plateau (≈ Vmax everywhere). Any mean shift in the
trans gene — including confound-driven variation — produces a credible non-zero Hill
coefficient because the model fits it as a constant offset rather than a dose-response
curve. A prior that places EC50 within the observed range forces the model to fit an
actual sigmoid, which is harder to satisfy with diffuse confound variance.

**Effect on the populations in this run**: populations A1, A2, B2, and B4 are the most
likely to have been inflated by the prior bias, since they all involve very negative
EC50_a with near-zero Hill coefficients — exactly the shallow-gradient regime. B3 has
EC50_a spiked at −1 log2FC (the count boundary), which the data constrains tightly
regardless of the prior. B1 has genuine large Hill_a, also data-constrained.

The prior fix (now `K_log_mu = log(x_ntc_mean)`, centring the median at log2FC = 0 with
±5 log2FC 95% CI) should reduce inflation of A1, A2, B2, B4 in future runs, but will
not eliminate the artifacts rooted in the discrete-count structure (see below).

---

### Discrete GFI1B counts as the primary structural confound

The root cause of the B3 artifact — and a significant contributor to B1 and A1 — is that
GFI1B has very low raw counts in most cells. The distribution is approximately:

```
0 counts: ~54% of cells
1 count:  ~27% of cells
2 counts: ~11% of cells
3+ counts: ~8% of cells
```

This discretisation means x_true (posterior cis expression) is **not continuous**: it
forms tight clusters at values corresponding to integer raw GFI1B counts, with gaps
between them. In log2FC space, the 0-count cluster sits at roughly −1 log2FC below the
NTC mean (because cells with 0 raw counts have x_true pulled by the prior toward a small
positive value, not exactly 0).

The **0→1 count boundary** is the critical step:

- The jump in x_true between the 0-count cluster and the 1-count cluster is large and
  abrupt — it cannot be smoothed by `refit_sumfactor` because a spline cannot remove a
  discontinuity at a single x_true value.
- For any gene whose expression changes between cells with 0 vs 1 GFI1B count — whether
  due to genuine GFI1B biology, differentiation-state confound, or stochastic variation —
  the Hill model sees an apparent step function and fits it with a large Hill_a coefficient
  and EC50 at the boundary.
- The spiked EC50_a distribution at ≈ −1 log2FC (population B3) is the definitive
  signature of this: every gene in B3 has its EC50 at the same location because the step
  is at the same x_true value for all genes.

Higher-count experiments (e.g., higher MOI or a more highly expressed cis gene) would
spread x_true more continuously and reduce the severity of this artifact. A model that
explicitly accounts for the discrete count structure of x_true (e.g., by marginalising
over raw count values rather than using a point estimate of x_true) would be more robust.

The **within-bin slope** (populations B1 and B3, Hill_b arm) is a secondary consequence:
within each integer-count bin, cells with higher sum_factor have higher x_true (same
integer count, larger normalised value), but the trans gene is anti-correlated with
differentiation state, so the model sees a downward slope inside the 1-count bin and
fits it as a sharp negative Hill_b near EC50_b ≈ 0.

The o_x underestimation fix (replacing the Gamma(9,3) containment prior with the
empirical Bayes o_y estimate from fit_ntc) reduces the confidence of individual x_true
estimates, slightly widening the within-bin clusters. This marginally softens the
sharpness of the step but does not remove it.

---

### Populations with unclear or uncertain mechanisms

Three populations have proposed mechanisms but remain uncertain:

**A3** (moderate expression, bimodal EC50_a, Hill_a ≈ 0, slightly negative): The bimodal
EC50 distribution suggests two overlapping sub-populations, one fitting the 0→1 boundary
step and one fitting variation near the NTC level. The slightly negative Hill_a (genes
going *down* as GFI1B decreases) is consistent with co-regulated differentiation genes
rather than direct GFI1B targets, but cannot be confirmed without a permutation null. The
mechanism for the near-zero coefficient in both sub-populations is unclear: it could be a
genuine weak dose-response or a shallow-gradient EC50 prior artefact.

**B2** (well-expressed, slightly positive Hill_a, EC50_a < −4, diffuse Hill_b ≈ 0): The
proposed mechanism is that these are a weaker version of B1 — the same 0→1 boundary
step and within-bin confound, but for trans genes less tightly coupled to differentiation
state. However, it is also possible that these genes have genuine but weak GFI1B
regulation that the Hill_a arm is detecting at the threshold of significance. The diffuse
Hill_b and near-zero coefficient are consistent with both interpretations.

**B4** (moderate expression, both arms ≈ 0, both significant): Proposed as background
inflation from the sum_factor confound, where diffuse non-specific variation across the
x_true range produces statistically credible but negligibly sized Hill coefficients. This
is plausible but the mechanism is the least well-characterised in the dataset. It is also
possible that some B4 genes have genuine dose-responses too weak to be interpreted as a
specific population — they would be indistinguishable from the confound at this
resolution.

---

### Population B1 as the most likely reservoir of true biology

Based on the expression level (log2NTC > 0, so reliably detected), the direction of Hill_a
(positive, consistent with genes upregulated when GFI1B is reduced → myeloid program), and
the EC50_a placement (in the range where substantial GFI1B knockdown has occurred, not
pinned to a count boundary), B1 is the population most likely to contain real GFI1B trans
effects. However, the Hill_b arm and the magnitude of Hill_a within this population are
inflated by the within-bin slope confound. Any downstream analysis should focus on Hill_a
from B1 rather than Hill_b.
