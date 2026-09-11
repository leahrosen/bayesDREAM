# False Positive Diagnosis: GFI1B Transcriptome-Wide Run

**Context**: Transcriptome-wide bayesDREAM run with GFI1B as the cis gene (high MOI dataset).
Every gene called significant — `fdr_alpha` and `fdr_beta` are near zero for the vast majority
of the ~21,000 genes tested. This document records the diagnostic investigation.

```
          fdr_alpha      fdr_beta
count  21242.000000  21242.000000
mean       0.019174      0.341691
std        0.126260      0.382622
min        0.000000      0.000000
25%        0.000000      0.048602
50%        0.000000      0.196239
75%        0.000000      0.318511
max        1.000000      1.000000
```

---

## Observed Patterns

### Significance and parameter distributions

![Significance vs. log2 NTC expression, and Hill parameter scatter plots](figures/false_positives_fig3_significance_vs_ntc_expression.png)

Four visually distinct false-positive populations:

1. **Broad "a significant" population** (green, log2 NTC expression ≈ −5 to −10): genes with
   very low NTC expression. EC50_a clusters near zero in log2FC space, Hill_a coefficient is
   moderate (not extreme). These respond to the sum_factor–x_true correlation (see below),
   not to true dose-response signal. **Also heavily driven by the A parameter collapse
   described below** — see Root Cause #4.

2. **EC50_a spike** (cluster at a single log2FC value near zero): Hill fits locking onto the
   boundary between 0-count and 1-count GFI1B cells. The step function at the count boundary
   is exactly what a Hill function is optimised to fit.

3. **EC50_b spike(s)** (one or two discrete log2FC values in the bottom row): analogous to #2
   but for the negative Hill arm; likely correspond to the 1→2 and/or 2→3 count transitions.

4. **A parameter collapse** (subset of population #1): For a subset of "a significant" genes,
   A is estimated at 2^−42 to 2^−48 while NTC expression sits at 2^−2 — a gap of ~40 log2
   units. These are not mild false positives; the model has driven A to machine-epsilon zero.
   See Root Cause #4 for the mechanism and the fix that was applied.

---

### o_x discrepancy between technical and cis fits

![Technical vs. cis posterior for o_x](figures/false_positives_fig4_ox_discrepancy.png)

- **Technical posterior**: o_x ≈ 0.78–0.83
- **Cis posterior**: o_x ≈ 0.70–0.75

The cis model estimates GFI1B overdispersion ~0.07–0.08 units lower than the NTC-derived
technical estimate. The cis model explains some cell-to-cell variance via guide-level effects
(x_eff_g), so the residual variance left for o_x is smaller. This is a form of variance
partitioning, but the consequence is that x_true becomes overconfident: a cell with 0 GFI1B
counts vs. 1 count is pushed to very different x_true values, sharpening the discrete banding.

---

### Sum factor vs. x_true, stratified by raw cis count

![Sum factor vs. x_true coloured by raw GFI1B count](figures/false_positives_fig5_sumfactor_vs_xtrue_by_rawcount.png)

This is the central diagnostic image. Key observations:

- **Discrete count banding is clear and persists across all sum factor variants** (sizeFactor,
  sum_factor, sum_factor_adj, sum_factor_refit). The refit does not remove the banding.

- **Within each raw count bin, there is a genuine negative slope**: cells with the same integer
  count but different x_true values (because x_true is estimated from guide effects + cis model
  uncertainty) show lower sum_factor at higher x_true. This slope is biologically real —
  it reflects the differentiation-state correlation described below — and **should not be
  corrected away**. However, it still confounds trans fitting because the trans model sees it
  as a dose-response signal.

- **The between-bin jumps are the dominant artifact**: sum_factor differs substantially between
  count bins (especially 0-count vs. 1-count). `refit_sumfactor` fits a smooth spline through
  the aggregate x_true–sum_factor relationship, but a step function cannot be well-approximated
  by a smooth spline. The residual between-bin structure leaks into the trans model.

- **NTC cells** (shown in the dashed blue/grey lines in the top panels): their sum_factor vs.
  x_true trajectory is distinct from GFI1B-targeting cells at the same x_true values, suggesting
  that NTC x_true variation is not equivalent to GFI1B perturbation variation.

---

### Cell-type scores vs. x_true: GFI1B vs. NTC

![Cell type program scores vs. log2(x_true)](figures/false_positives_fig6_celltype_scores_vs_xtrue.png)

- Strong expected biology: myeloid score drops sharply at high x_true; erythroid scores rise.
  This confirms that x_true is tracking real differentiation state.

- **NTC curve shows a bump near log2(x_true) ≈ −0.5**: NTC cells should not have a structured
  response to x_true (they are not GFI1B-perturbed). The bump corresponds to the 0→1 count
  transition: NTC cells with 0 GFI1B counts happen to be in a different cell state than NTC
  cells with ≥1 count, and the model is picking this up via x_true.

---

## Root Cause Hierarchy

### 1. Discrete cis gene counts create a step function that `refit_sumfactor` cannot smooth (primary cause)

GFI1B is lowly expressed in this dataset. Most cells have 0, 1, 2, or 3 raw GFI1B counts.
The cis model maps these discrete counts to distinct x_true clusters. The sum factor varies
across these bins partly for biological reasons (differentiation state) and partly as a
count-discretization artifact (0-count cells are biologically heterogeneous — some are truly
GFI1B-low, some are capture failures).

`refit_sumfactor` removes the smooth x_true-correlated component of the sum factor via spline
regression. But the residual between-bin jumps remain. Every trans gene then shows a
step-function response at count boundaries, which a Hill equation fits with high confidence.

### 2. o_x underestimation in the cis model amplifies the discretization

Because the cis model partitions variance into guide effects vs. residual overdispersion,
it arrives at a lower o_x than the technical fit. This makes x_true more tightly estimated
around integer-count values, sharpening the discrete steps and making the Hill fits more
confident.

### 3. The sum factor genuinely correlates with differentiation state (real biology, unavoidable)

GFI1B drives erythropoiesis. Cells with low GFI1B → myeloid fate → systematically different
total RNA → different sum factor. This is real biology that the sum_factor is designed to
capture, but it creates a genuine x_true–sum_factor dependency that any normalization strategy
must grapple with. Importantly, the **within-bin negative slope in Image #5 is this real
biological signal and should not be removed** — but it does add noise to trans fits even after
refit, because the model cannot distinguish it from dose-response signal in the absence of
between-bin variation.

### 4. A parameter collapse for lowly-expressed genes (prior design + initialisation trap) ✅ FIXED

#### What was observed

For "a significant" genes with EC50_a < −3 in log2FC space and high effect size, A was
estimated at 2^−42 to 2^−48 while NTC expression sat at 2^−2. Vmax_a tracked NTC expression
nearly 1:1 (Image #16 top-left). The A/Vmax correlation was 1:1 — Vmax had absorbed the entire
NTC expression level because A was at zero.

#### What A represents geometrically

In the `additive_hill` model:
```
y = A + alpha * Hill_a(x; Vmax_a, K_a, n_a) + beta * Hill_b(x; Vmax_b, K_b, n_b)
```
Both Hill functions go from 0 → Vmax as x increases when n > 0, and Vmax → 0 when n < 0.
Therefore:
- **Positive association (n_a > 0)**: A = y at x → 0 = global minimum
- **Negative association (n_a < 0)**: A = y at x → ∞ = global minimum
- **Non-monotonic**: A < min(y at x→0, y at x→∞) — A is below both endpoints

A is always the **global floor** of the function, not a value that changes direction with x_true.

#### The prior and why it caused the collapse

The negbinom A prior was `Exponential(rate_A)` where:
```python
rate_A = (2 - w) / Amean    # Amean = Q05 of guide means
w = o_y / (o_y + E[o_y])    # o_y weight ∈ (0, 1)
```

Weight behaviour:
| w | regime | rate_A | mean(A) | P(A ≥ Q05) |
|---|--------|--------|---------|-------------|
| →0 | quiet gene (low o_y) | 2/Q05 | Q05/2 | exp(−2) ≈ 13.5% |
| 0.5 | average | 1.5/Q05 | Q05/1.5 | ≈ 22% |
| →1 | noisy gene (high o_y) | 1/Q05 | Q05 | exp(−1) ≈ 37% |

In all regimes, mean(A) ≤ Q05. For lowly-expressed genes, Q05 is itself near zero.

**The mode of Exponential is always 0**, regardless of rate. The variational guide (AutoNormal)
parameterises log(A) and initialises it near the prior's log-mean ≈ log(Q05). For a gene with
Q05 ≈ 10⁻⁵, the guide starts with log(A) ≈ −12 → A ≈ 6 × 10⁻⁶. The model then finds a
locally stable solution:

```
A ≈ 0   →  Vmax_a absorbs NTC expression
         →  EC50 pushed far below the observable x_true range (Hill is flat everywhere)
         →  function is constant in the observable range  (good local fit!)
         →  no gradient signal from the likelihood to pull A back up
         →  prior gradient always points toward smaller A (Exponential mode = 0)
```

This is a **local minimum trap**: once in it, neither the likelihood nor the prior provide
an escaping force. The resulting fit looks like a strong positive Hill response
(large Vmax, EC50 << NTC, A ≈ 0), which the hypothesis test then flags as significant.

The low-expression enrichment (population #1 of the false positives) follows directly:
- Low expression → Q05 near zero → A initialises near zero → trap engaged
- High o_y (overdispersion) for low-expression genes → w → 1 → mean(A) = Q05 ≈ 0 anyway

#### The fix (implemented in this commit)

Two changes in `fit_trans` (prior computation) and `_model_y` (prior sampling):

**1. Floor Q05 at q01(mu_ntc)** — prevents Amean from being near-zero:
```python
q01_ntc = torch.quantile(posterior_samples_ntc['mu_ntc'], 0.01, dim=0)  # [T]
y_ntc_tensor = posterior_samples_ntc['mu_ntc'].mean(dim=0)               # [T]
Amean_tensor = Amean_tensor.clamp_min(q01_ntc)
```
`mu_ntc` is the technical-posterior NTC mean per gene. Using q01 rather than the mean is
conservative: it only raises the floor for genes where Q05 is below the 1st percentile of
NTC expression — genuinely pathological cases.

**2. Weight interpolation to y_ntc for noisy genes** — fixes the initialisation anchor:
```python
# OLD: mean(A) ∈ [Q05/2, Q05] — all near zero for lowly-expressed genes
rate_A = (2.0 - w) / Amean_tensor

# NEW: mean(A) interpolates from Q05/2 (quiet) to y_ntc (noisy)
mean_A = (1.0 - w) * Amean_tensor / 2.0  +  w * y_ntc_tensor
rate_A = 1.0 / mean_A
```

With the new formula:
- **w → 0 (quiet gene, informative likelihood)**: mean(A) = Q05/2 — same as before; data
  can identify the true floor, allow aggressive shrinkage below Q05.
- **w → 1 (noisy gene, uninformative likelihood)**: mean(A) = y_ntc (NTC posterior mean).
  The guide initialises log(A) ≈ log(y_ntc), starting near the NTC level, not near zero.
  P(A ≥ y_ntc) = exp(−1) ≈ 37% — NTC expression is well within the prior.

The Exponential distribution is deliberately kept (mode = 0) because simulations show it
improves extrapolation beyond the observed x_true range. The fix is in the *mean* (and
therefore the initialisation point), not the distributional form.

#### Interaction with EC50 prior bias

The A collapse is a consequence, not a cause, of the EC50 being driven below the observable
range. The chain is:
```
EC50 prior too low (pre-existing bug, fixed separately)
  → EC50 << NTC level
  → Hill function flat in observable range
  → no likelihood gradient for A
  → prior alone determines A → Exponential pulls to zero
```
Both fixes are complementary and necessary:
- EC50 fix prevents the function from going flat in the first place
- A prior fix means that even if EC50 drifts low, A doesn't collapse

### 5. NTC cells carry other perturbations (high MOI confounding)

NTC cells in this experiment do not carry GFI1B guides, but in high MOI they carry other
(non-GFI1B) guides. Some of these other guides perturb cell state, causing systematic shifts
in gene expression that are independent of GFI1B but correlated across genes. When the NTC
cells' gene expression profiles are used to establish the baseline in the trans model, this
other-guide perturbation adds uncontrolled variation. It does not create a GFI1B-specific
dose-response signal by itself, but it widens the NTC distribution and potentially biases
EC50 estimates centred on the NTC mean.

---

## Why Essentially Every Gene Is Called Significant

Two parallel causal chains converge on the same result:

**Chain A: sum_factor / count discretisation (affects all expression levels)**
```
discrete GFI1B counts → discrete x_true clusters
    → step-function structure in sum_factor vs. x_true
    → refit_sumfactor removes smooth component but not between-bin steps
    → residual sum_factor–x_true correlation
    → every trans gene appears to have systematic over/under-expression
      across x_true bins (relative to normalised expectation)
    → Hill model fits the step function at count boundaries with high confidence
    → very low p-values for essentially all genes
    → FDR near zero across the board
```

**Chain B: A parameter collapse (concentrated in lowly-expressed genes)**
```
lowly-expressed gene → Q05 of guide means near zero
    → Exponential A prior centred near zero
    → guide initialises log(A) at log(Q05) ≈ −∞
    → A starts near zero → Vmax_a absorbs NTC expression
    → EC50 (already biased low by separate prior bug) pushed below observable range
    → Hill function flat in observable range → no likelihood gradient on A
    → prior pulls A further toward zero (Exponential mode = 0)
    → stable false solution: A ≈ 0, large Vmax, ultra-low EC50
    → hypothesis test: large effect size, "a significant"
```

Chain B explains the enrichment of "a significant" calls specifically among lowly-expressed
genes (log2 NTC expression < −5), and the extreme A values (2^−40 to 2^−48) seen in Image #21.
Chain A is the dominant driver for the transcriptome-wide inflation.

**Chain B has been fixed** (Q05 floor + weight interpolation to y_ntc; see Root Cause #4).
Chain A requires model-level changes (see Potential Mitigations).

---

## Outstanding Questions / Hypotheses to Test

1. **Is the permutation null also inflated?** If running `--permtype All` also gives near-zero
   FDR for all genes, the FDR calibration itself is broken (both real and null have the same
   artifact). If only the real run is inflated, the discrete x_true signal is real but arises
   from confounding, not GFI1B effects.

2. **Does the sum_factor_refit residual correlate with raw count bin?** A plot of
   `sum_factor_refit` vs. `GFI1B_raw_count` (as integer, not x_true) would quantify how much
   between-bin signal remains after refit.

3. **How wide is the NTC x_true distribution?** If NTC cells span 2+ log2 units of x_true,
   they are being used as a biologically heterogeneous reference, inflating apparent signal.

---

## Implemented Fixes

**A prior: Q05 floor + NTC weight interpolation** ✅ (this commit)  
See Root Cause #4 for full description. Addresses Chain B (A collapse for lowly-expressed genes).
- `Amean_tensor.clamp_min(q01_ntc)` — floors Q05 at 1st percentile of NTC posterior expression
- `mean_A = (1-w)*Q05/2 + w*y_ntc` — noisy genes now anchor at NTC level, not at Q05 ≈ 0
- Files changed: `bayesDREAM/fitting/trans.py` (`_model_y` and `fit_trans`)

---

## Potential Mitigations (Not Yet Implemented)

**EC50 prior centred at NTC level** *(high priority, partially addressed)*  
The EC50 prior was biased toward values below the NTC expression level. This was identified
separately and is in progress. Without this fix, the Hill can still saturate within the
observable range, re-engaging Chain B even with the new A prior.

**Anchor o_x in the cis model to the technical fit** *(most targeted for Chain A)*  
Use a tighter prior for o_x in `fit_cis`, centred on the technical estimate. This prevents
variance partitioning from artificially sharpening x_true and would smooth the discrete banding.
Risk: may underfit the cis model if true residual overdispersion is genuinely lower.

**Robust sum factor using stable genes**  
Recompute the sum factor using only genes confirmed insensitive to GFI1B perturbation (e.g.,
from permutation nulls or known housekeeping genes), then rerun `refit_sumfactor`. This would
break the biological coupling between sum_factor and differentiation state — but defining
"stable genes" is circular without an unbiased null.

**Guide-level aggregation for trans fitting**  
Aggregate cells to guide-level means before trans fitting. This averages out single-cell count
discretization and reduces high-MOI contamination noise. Loses single-cell resolution but may
be necessary for a gene with very low counts per cell.

**Separate treatment of 0-count cells**  
Explicitly model 0-count cells as a mixture of "true zero" and "dropout", rather than placing
them on the same x_true axis as cells with observed counts. This is a substantial model change.
