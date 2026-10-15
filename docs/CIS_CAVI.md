# Polya-Gamma CAVI for the negative binomial cis model (prototype)

`bayesDREAM/fitting/cis_cavi.py` fits the single-guide negbinom cis model without SVI.
It is standalone for now and is not called by `fit_cis`. It follows the NTC prototype
(`docs/NTC_CAVI.md`).

## Model

For cell i with guide g(i), technical group c(i), and target k of its guide:

    y_i ~ NB(mean = alpha_x[c(i)] x_i s_i, total_count = phi_x)    alpha_x, phi_x fixed (fit_ntc)
    ell_i = ln x_i ~ Normal(theta_g, v_g)
    theta_g ~ Normal(mu_k, sigma_k^2 / lambda_g),  lambda_g ~ Gamma(3/2, 3/2)
    mu_k ~ Normal(m0, s0^2)
    sigma_k ~ HalfCauchy(A), as sigma_k^2 | a_k ~ InvGamma(1/2, 1/a_k), a_k ~ InvGamma(1/2, 1/A^2)
    v_g ~ InvGamma(a_v, b_v),  b_v ~ Gamma(c_b, d_b)

theta_g / ln 2 is log2 x_eff_g and ell_i / ln 2 is log2 x_true_i of `_model_x`.
The Student-t(3) guide effects and the HalfCauchy on sigma are written as scale mixtures,
so every factor is conjugate.

Differences from `_model_x`:
- The per-guide cell-level spread has an inverse-gamma prior on the variance
  v_g = sigma_eff_g^2, with a shared Gamma hyperprior on its rate. `_model_x` puts a Gamma
  prior on the SD sigma_eff_g with Exponential hyperpriors on its shape and rate. The
  inverse-gamma is used because it is conjugate.
- Priors keep the data-driven defaults of `fit_cis` (mu prior from the mean and SD of the
  per-guide log2 means; HalfCauchy scale 2 in log2 units). The mu prior SD is floored at
  0.1 log2. The prior mean of v_g is set from the per-guide MADs, clamped to [0.05, 2] log2.
- `targets_per_guide` gives each target its own mu and sigma (as `independent_mu_sigma`).
- Single-guide mode only; high-MOI additive guide effects are not implemented.

## Algorithm

1. **PG-CAVI.** With psi_i = ell_i + ln(alpha s_i) - ln phi, the Polya-Gamma identity
   makes ell_i conditionally Gaussian. q(ell, theta, mu) is one structured Gaussian, not
   mean field. Its graph is a tree (cells -> guide -> target), so its means, marginal
   variances and the needed covariances come from eliminating ell_i and then theta_g in
   O(N + G). q(omega_i), q(lambda_g), q(sigma_k^2), q(a_k), q(v_g) and q(b_v) have closed-form
   updates.
2. **Newton + Laplace.** The hyperparameter factors are held at their expectations. Newton
   steps with a backtracking line search find the mode of the exact log posterior in
   (ell, theta, mu). The Newton step uses the same tree elimination with the exact
   curvature (y + phi) sigma(psi)(1 - sigma(psi)), so at the mode the elimination also gives
   the Laplace covariance. The hyperparameter factors are then refreshed from the Laplace
   moments, and the two steps alternate.
3. **Acceleration.** Both phases are fixed-point iterations that converge slowly when count
   noise per cell is large compared with v_g. Plain CAVI took 800 to 1400 sweeps and still
   had not reached the fixed point. Both phases are therefore accelerated with SQUAREM
   (Varadhan and Roland 2008). With SQUAREM, each phase needs 20 to 45 iterations.

Every hand-derived quantity is checked in `tests/test_cis_cavi.py`:
- the gradient and the likelihood curvature, against torch.autograd;
- the tree elimination, against a dense matrix inverse;
- the Newton step and the Laplace covariance, against the autograd Hessian.

### Are guide-level variances affected by the PG bound?

The cell-level variances are affected; the guide-level variances much less:
- **Cells.** The PG (Jaakkola-Jordan) curvature overstates the precision of each cell, so
  the CAVI per-cell SDs are about 20 % too small. Their 90 % coverage is about 0.81.
- **Guides.** Var(theta_g) is dominated by the spread between cells (v_g / n_g) and by the
  hierarchy, so the PG distortion shrinks the guide SDs by only about 7 %.
- **Mean field.** A mean-field q(ell) q(theta) would be much worse at the guide level. It
  drops the count noise from Var(theta_g); its guide SDs are about 0.27 of the Laplace SDs,
  and its 90 % coverage is about 0.3. The structured Gaussian block is therefore the main
  fix for theta_g, and the Laplace step is a smaller refinement.

## Benchmark

`examples/diagnostics/benchmark_cis_cavi.py` simulates one dataset:
- 3000 cells and 20 guides: 6 NTC guides at baseline 8 counts, and 14 targeting guides with
  log2 knockdown evenly spaced from -0.25 to -3;
- per-guide cell SD sigma_eff_g ~ U(0.2, 0.5) log2;
- 3 cell lines with alpha_x = (1, 1.5, 0.7), phi_x = 5, and log-normal size factors (SD 0.3).

Both methods are given the true alpha_x and phi_x. For SVI, `alpha_x_prefit` and
`posterior_samples_ntc['o_x']` of the cis modality are set directly.

The SVI intervals are the 5-95 % sample quantiles; the CAVI intervals are mean +/- 1.645 SD.
Torch ran single threaded. The machine was shared with other jobs, so the SVI times are
inflated and only their order of magnitude is meaningful.

Seed 0, with RMSE / 90 % coverage on the log2 scale:

| method | log2 x_eff_g (20 guides) | log2 x_i (3000 cells) | time |
|---|---|---|---|
| SVI `fit_cis` (default; 100,000 steps) | 0.079 / 0.60 | 0.356 / 0.94 | 898 s |
| SVI `fit_cis` (10,000 steps) | 0.124 / 0.25 | 0.368 / 0.90 | 93 s |
| PG-CAVI only (no Laplace) | 0.092 / 0.85 | 0.359 / 0.81 | |
| PG-CAVI + Laplace | 0.091 / 0.90 | 0.356 / 0.90 | 0.25 s |

Over 20 seeds (same design, CAVI only):

| | Laplace | CAVI only | mean field |
|---|---|---|---|
| log2 x_eff_g coverage | 0.88 (range 0.75 to 1.0) | 0.86 | 0.32 |
| log2 x_eff_g RMSE | 0.094 | 0.095 | |
| log2 x_i coverage | 0.90 | 0.81 | |
| log2 x_i RMSE | 0.352 | 0.354 | |

The mean runtime was 0.24 s, and every run converged.

With 20 guides per dataset, guide-level RMSE and coverage from one seed are noisy, so the
multi-seed numbers are the better guide. On seed 0, default SVI had a slightly lower guide
RMSE (0.079 vs 0.091), but its guide intervals were too narrow: they were less than half
as wide as the CAVI intervals and covered 60 %. Its per-cell intervals were slightly too
wide, covering 94 %. CAVI + Laplace was about 4000 times faster on this problem. SVI was
run on one seed only, because each default run takes about 15 minutes.

## Limitations

- Single-guide mode only. The high-MOI model (additive log2 effects with NTC centering)
  is not implemented.
- alpha_x and phi_x are fixed point values, as in `fit_cis`.
- The Laplace step conditions on the hyperparameter expectations (v_g, sigma, lambda_g), so
  their uncertainty does not reach the interval for theta_g. This is the likely cause of the
  slight under-coverage at the guide level (0.88).
- Not wired into `fit_cis`. The output gives means and SDs, not `posterior_samples_cis`
  draws. Draws could be generated from the Gaussian block if needed.
