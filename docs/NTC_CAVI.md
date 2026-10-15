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

## Limitations

- negbinom only; the other distributions still use SVI.
- phi has a point estimate plus a Laplace interval, not a full variational factor.
- The prior on log mu differs from the SVI model: Normal(log normalised mean, 2^2) rather
  than the data-driven LogNormal.
