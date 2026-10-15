# Pólya-Gamma CAVI for the negative binomial NTC model

`bayesDREAM/fitting/ntc_cavi.py` fits the negbinom NTC model without SVI. Use it through

```python
model.fit_ntc(sum_factor_col='sum_factor', method='cavi')
```

The fit draws posterior samples from the Gaussian approximation in (log mu, log a, log phi)
and stores them with the same keys and shapes as the SVI path (`log2_alpha_y`,
`alpha_y_mul`, `delta_y_add`, `mu_ntc`, `o_y`, `beta_o`). Feature reconstruction, cis-gene
extraction, `alpha_y_prefit_*`, saving and the later fits are therefore unchanged.
`lr`, `niters`, `minibatch_size` and `force_iaf` are ignored. Extra options go in
`cavi_options` (for example `tol`, `max_chunk_bytes`, `log_mu_prior_scale`).

## Model

For feature t and NTC cell i in technical group c(i), with c = 0 as the reference group:

    y_it ~ NB(mean = mu_t a_{c(i),t} s_i, total_count = phi_t)
    log mu_t ~ Normal(log of the normalised mean, 2^2)
    log2 a_ct ~ StudentT(3, 0, 20)          (as in _model_ntc)
    phi_t^(-1/2) ~ Exponential(beta_o),  beta_o ~ Gamma(9, 3)

## Algorithm

1. **CAVI.** With psi = log(mu a s) - log phi, the Pólya-Gamma identity makes
   theta = (log mu, log a_1, ..., log a_{C-1}) conditionally Gaussian. The Student-t prior
   is written as a Gamma scale mixture. q(theta), q(omega) and q(tau) all have closed-form
   updates, vectorised over features. log phi_t is updated by Newton steps on the expected
   log likelihood plus log prior; beta_o is updated to its conditional mode.
2. **Newton polish.** A few joint Newton steps on the exact log posterior in
   (theta, log phi) move the solution to the exact mode.
3. **Laplace covariance.** The covariance is the inverse negative Hessian at that mode.
   This step is needed: at convergence the PG bound is the Jaakkola-Jordan tangent bound,
   whose curvature exceeds the true curvature away from psi = 0, so the CAVI covariance
   alone is too narrow. Without the correction, 90 % intervals covered about 75 %.

## Memory and chunking

Features are independent given beta_o. They are fitted in chunks that keep the largest
working array ([cells, chunk, quadrature nodes] in float64) below `max_chunk_bytes`
(default 1 GB). beta_o is held fixed within a pass and updated between passes
(`beta_o_passes`, default 2); later passes are warm-started.

## Features without detectable overdispersion

For Poisson-like features, log phi runs to its upper bound (12), and the joint Hessian is
not positive definite there. For those features log phi is decoupled in the Laplace step
and given SD 1; the (log mu, log a) block is unaffected. `fit_ntc` reports how many such
features there are.

## Benchmark

`examples/diagnostics/benchmark_ntc_cavi.py` simulates 2000 NTC cells, 200 genes and
3 cell lines, with:
- mu between 0.1 and 100,
- phi between 1 and 50,
- log2 group effects ~ N(0, 0.5).

The table gives RMSE / coverage of 90 % intervals for one seed:

| method | log2 a | log mu | log phi | time |
|---|---|---|---|---|
| SVI `fit_ntc` (default; 100,000 steps) | 0.098 / 0.71 | 0.046 / 0.59 | 1.09 / 0.91 | 2910 s |
| PG-CAVI + Laplace | 0.096 / 0.87 | 0.045 / 0.87 | 0.42 / 0.93 | 12 s |

Over five seeds (same design, CAVI only), the mean coverage was 0.89 for log2 a, 0.89 for
log mu and 0.91 for log phi, with RMSE 0.105, 0.050 and 0.43.

## Real counts: split-half check

`examples/diagnostics/compare_ntc_svi_cavi_realdata.py` was run on the counts in
`toydata/gene_counts.csv` (92 genes), subsampled to 4000 cells.
- The barcode prefixes L1 and L2 are the technical groups, and sum factors are library
  size over its median.
- There is no matching metadata, so all cells, including perturbed ones, are fitted as if
  they were NTC. The check is therefore about computation and calibration, not biology.
- Each method fitted two random halves (stratified by group). If the reported SDs are
  right, |z| < 1.645 should hold for about 90 % of genes.

| | SVI (default, 100,000 steps) | PG-CAVI + Laplace |
|---|---|---|
| \|z\| < 1.645, log2 a (L2 vs L1) | 0.65 (sd(z) 1.80) | 0.74 (sd(z) 1.38) |
| \|z\| < 1.645, log mu | 0.87 | 0.96 |
| \|z\| < 1.645, log phi | 0.75 | 0.71 |
| mean abs. error, zero fraction per gene | 0.011 | 0.011 |
| time per half (2000 cells) | 2850 to 2880 s | 52 to 110 s |
| time on 4000 cells | 4934 s | 223 s |

The fits ran six at a time on four cores, so all times are inflated.

On the full 4000 cells the two methods agree closely:
- log mu: r = 0.99999, median difference 0.001;
- log phi: r = 0.9995, median difference 0.003;
- log2 a: r = 0.95, median difference 0.002 (the effects are small, which lowers r).

Both methods under-cover the group effect and log phi on these data. That points to
misspecification rather than to the inference. Perturbed cells are mixed in, and sum
factors from 92 genes are crude, so model-based SEs are too small for both methods.

## Replogle et al. 2022, K562 essential screen

Data: `K562_essential_raw_singlecell_01.h5ad` (figshare+ 10.25452/figshare.plus.20029387),
prepared with `examples/diagnostics/replogle_prepare.py ntc`. The subset is:
- 1752 non-targeting cells from gem groups 1 to 8 (the technical groups);
- 300 random genes with mean of at least 0.1 UMI;
- sum factors equal to UMI count over its median.

Split-half check as above (|z| < 1.645 should hold for about 90 %):

| | SVI (default, 100,000 steps) | PG-CAVI + Laplace |
|---|---|---|
| log2 a (2100 gene x group pairs) | 0.71 (sd(z) 1.57) | 0.88 (sd(z) 1.03) |
| log mu (300 genes) | 0.34 (sd(z) 3.8) | 0.89 (sd(z) 1.02) |
| log phi | 0.86 (sd(z) 0.97) | 0.98 (sd(z) 0.75) |
| mean abs. error, zero fraction per gene | 0.0039 | 0.0034 |
| time, one half | about 5500 s | about 80 s |
| time, all cells | 8759 s | 160 s |

The fits ran in parallel, so the times are inflated.

On all cells the two methods agree on the group effects (r = 0.987, median difference
0.017) and on log mu (r = 0.9997). They differ on log phi (r = 0.85, median difference 0.94).
On these genuine control cells CAVI intervals are close to nominal, while SVI intervals for
log mu and the group effects are clearly too narrow.

## Limitations

- negbinom only; the other distributions still use SVI.
- phi has a point estimate plus a Laplace interval, not a full variational factor.
- The prior on log mu differs from the SVI model: Normal(log normalised mean, 2^2) rather
  than the data-driven LogNormal.
