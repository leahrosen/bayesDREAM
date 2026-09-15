# o_x Prior Diagnosis: Why Gamma(9, 3) Is Wrong for fit_cis

**Context**: The overdispersion prior structure in bayesDREAM was copied from cell2location. It
works correctly for `fit_ntc` and `fit_trans` but is systematically wrong for `fit_cis`,
causing x_true to be discretized by raw integer counts. This document records the diagnosis.

---

## The Prior Structure

Both `fit_ntc` and `fit_cis` use the same hierarchical prior:

```python
beta_o ~ Gamma(alpha=9, beta=3)       # mean = 3
o_x    ~ Exponential(rate=beta_o)     # conditional mean = 1/beta_o
phi_x   = 1 / o_x**2
```

This structure comes directly from cell2location:

![Cell2location Gamma(9,3) containment prior](figures/ox_prior_cell2location_gamma93.png)

Cell2location calls this a **containment prior**: it places most probability mass on small `o_x`
(large `phi`, close to Poisson), which is the right inductive bias for a transcriptome-wide model
where most genes have low overdispersion.

The implied prior statistics for `o_x`:

| Statistic | Value |
|-----------|-------|
| Prior mean o_x | 0.375 |
| Prior median o_x | 0.24 |
| Prior 5th–95th percentile | 0.017 – 1.19 |
| Implied phi at prior median | ~17 (nearly Poisson) |

---

## Where the Prior Is Appropriate

### fit_ntc and fit_trans

The technical fit estimates confirm the prior is reasonable for typical trans genes.

![Technical posterior: o_y vs mu_ntc](figures/ox_prior_technical_oy_vs_muntc.png)

For moderately expressed genes (log2 mu_ntc ≈ −3 to 2), the fitted o_y is concentrated in the
0.3–1.0 range. The prior median of 0.24 is somewhat low but within the plausible range, and with
21,000 genes being fit the data dominates the prior comfortably.

The cell2location intent holds: this is a containment prior keeping most genes close to Poisson,
with the collective data across all genes tightly constraining the shared `beta_o` hyperparameter.

In `fit_trans`, x_true is treated as **fixed** (a point estimate passed in, not a latent
variable), so there is no variance-absorption problem. The prior behaves as intended.

### Lowly-expressed genes in fit_ntc

The horizontal band at log2(o_y) ≈ 0 (o_y ≈ 1) visible in the 2D density plot for very low
expression genes (log2 mu_ntc < −7) is the Gamma(9,3) prior dominating when counts are too
sparse for the data to speak. This is acceptable behaviour for trans genes — we don't expect to
learn reliable overdispersion from genes with mean counts near zero, and the containment prior
prevents wild estimates.

---

## Where the Prior Fails: fit_cis

The cis model has a structural feature that cell2location does not: a **per-cell latent
variable** `x_true`.

```python
x_true ~ LogNormal(log2(x_mean_from_guides), sigma_from_guides)   # [N] latent
x_obs  ~ NB(phi=1/o_x², mu = alpha_x · x_true · sum_factor)       # likelihood
```

This creates two compounding problems.

### Problem 1: x_true absorbs variance that would otherwise inform o_x

In cell2location the NB likelihood must explain all cell-to-cell variance in counts via
overdispersion. In the cis model, `x_true` is a free latent variable that the ELBO rewards for
tracking each observed integer count as closely as possible. Moving `x_true` toward
`x_obs / sum_factor` improves the likelihood term; the cost is a KL penalty for departing from
the guide prior, which is modest when guides have well-separated means (GFI1B-targeting vs NTC).

The optimizer finds a solution where x_true closely tracks discrete counts and o_x drifts low,
because both moves improve the ELBO. Cell2location's containment prior was never designed to
resist this gradient — the gradient doesn't exist in their model.

### Problem 2: One gene, not a shared hyperprior across thousands

In cell2location, `beta_o` is a single scalar estimated jointly from all genes. Even weak
per-gene data is overwhelmed by the collective posterior on `beta_o`. The prior on `beta_o`
matters little.

In `fit_cis` there is one gene and one `beta_o`. The Gamma(9,3) prior **is** the effective prior
on `o_x`. Its median of 0.24 is already below the technical estimate of ~0.80, and the
x_true absorption gradient pulls in the same direction.

### The result

| | o_x | phi = 1/o_x² |
|---|---|---|
| Gamma(9,3) prior median | 0.24 | 17.4 (near Poisson) |
| Cis posterior (GFI1B run) | 0.72 | 1.93 |
| Technical posterior (GFI1B) | 0.80 | 1.56 |

The cis posterior falls 0.08 o_x units short of the technical estimate. That gap raises phi from
1.56 to 1.93 — a tighter NB likelihood — which sharpens the boundary between integer count bins
in x_true space. See `FALSE_POSITIVES_GFI1B_DIAGNOSIS.md` for how this drives genome-wide false
positives.

### Why this is general to all lowly-expressed cis genes

Any gene with mean count per cell below ~5 will have most cells in the `{0, 1, 2, 3}` count
bin regime. For these genes:

1. x_true can perfectly track the integer count with little KL cost (the guide prior is wide
   relative to the count-bin spacing)
2. The residual variance left for o_x is minimal
3. The Gamma(9,3) prior pulls o_x toward its median of 0.24, compounding the problem
4. The result is x_true ≈ discrete{0, 1/sf, 2/sf, 3/sf}, which the Hill trans model
   interprets as a dose-response curve

For highly-expressed cis genes (mean counts ≥ 20), the integer steps cover a small fraction of
the x_true range, and within-guide variance (from `sigma_eff`) spans multiple integer steps, so
the effect is negligible.

---

## The Fix

The technical fit already produces the correct answer: it estimates `o_y` for every gene
(including the cis gene) from NTC cells, without any latent x_true interfering. That value is
the natural **empirical Bayes prior center** for `o_x` in `fit_cis`.

**Implementation**: after `fit_ntc`, look up the cis gene's `o_y` from
`posterior_samples_ntc`, and either:

1. **Fix `o_x` to this value** (simplest — remove it as an inferred parameter in `fit_cis`)
2. **Use it as a tight informative prior center** (allows some flexibility around the estimate)

The Gamma(9,3) prior can remain unchanged in `fit_ntc` and `fit_trans`, where it behaves
as intended.

---

## Summary

| Context | Gamma(9,3) appropriate? | Reason |
|---------|------------------------|--------|
| `fit_ntc` | Yes | Containment prior across many genes; no latent x_true |
| `fit_trans` | Yes | x_true is fixed (not latent); same containment logic applies |
| `fit_cis` | **No** | x_true latent absorbs variance; single gene; prior pulls in wrong direction |
