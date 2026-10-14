# logistic_hill: a trans model reparameterised for SVI

`fit_trans(function_type='logistic_hill')` fits the same kind of dose-response curves as
`single_hill` and `additive_hill`. The parameterisation is chosen so that the posterior
is close to Gaussian in unconstrained space, which is the situation where stochastic
variational inference works well. It is implemented as a separate Pyro model,
`TransFitter._model_y_logistic`, and leaves the other function types unchanged.

## Model

Let x_i be the cis expression of cell i and

    u_i = (log2 x_i - u_center) / u_scale,

where `u_center` is the mean of log2 x over NTC cells and `u_scale` the SD of log2 x over
all cells. Thus u = 0 at the NTC level and the priors below are free of units. For
feature t the linear predictor is

    eta_t(u) = b_t + sum_j V_jt [ sigmoid(n_jt (u - k_jt)) - sigmoid(-n_jt k_jt) ]

and the likelihood is the same as for the other function types:

| distribution | link | mean passed to the likelihood |
|---|---|---|
| negbinom | log2 | 2^eta, times technical group effect and sum factor |
| binomial | logit | sigmoid(eta), technical effect added on the logit scale |
| normal, studentt | identity | eta, technical effect added |

Each logistic term is a Hill function of x. The Hill coefficient is `hill_n = n / (u_scale ln 2)`
and the half-maximal point is `log2_K = u_center + u_scale k`. Since the correction term vanishes
at u = 0, `b_t` is the value of the linear predictor at the NTC level.

Priors (s_t = 1 for negbinom/binomial, s_t = per-feature data SD for normal/studentt):

    b_t     ~ Normal(b0_t, b_scale_t)          b0 from NTC cells on the link scale
    tau     ~ HalfCauchy(tau0)                  global shrinkage, tau0 = 0.1
    lam_jt  ~ HalfCauchy(1)                     local shrinkage
    z_jt    ~ Normal(0, 1)
    V_jt    = s_t tau lam~_jt z_jt,   lam~ = c lam / sqrt(c^2 + tau^2 lam^2),  c = 3
    log n_jt ~ Normal(0.5, 0.75)                n > 0; the sign of the effect is in V
    k_1t    ~ Normal(0, 1.5),   k_jt = k_(j-1)t + softplus(r_jt),   r_jt ~ Normal(0, 1)

Dispersion: negbinom uses o_t ~ Exp(beta_o), beta_o ~ Gamma(9, 3), phi = 1/o^2, as in the
other models; normal/studentt use sigma_t ~ HalfNormal(2 s_t), and studentt nu_t ~ Gamma(10, 2).
Technical group effects are taken from `fit_technical` and held fixed.

## Differences from additive_hill, and why

1. **Log-x scale.** Hill functions in x are logistic functions in log x. Writing them this way
   reduces the correlation between K and n, and makes the midpoint prior meaningful relative to
   the observed range of x.
2. **Link scale.** The curve acts on log2 of the mean, not additively on counts. No positivity
   constraint is needed, and b and V decouple. The amplitudes are log2 fold changes. The
   components therefore combine multiplicatively, which is a modelling difference from
   `additive_hill`.
3. **Shrinkage instead of switches.** The relaxed-Bernoulli switches alpha and beta are replaced by
   a regularised horseshoe on a signed amplitude. This removes a bimodal latent variable and the
   temperature schedule.
4. **Ordered midpoints.** k_1 < k_2 < ... removes the label-switching symmetry between components.

## Variational family

`logistic_guide='block_mvn'` (default): the per-feature latent variables
(b, o_y or sigma_y/nu_y, z, lam, log_n, k_raw; d = 10 for J = 2 and negbinom) get a
full-covariance Gaussian per feature in unconstrained space. Features are independent of each
other, so the guide is block-diagonal and costs O(T d^2). The global scalars tau and beta_o get
independent Gaussians. The first half of the iterations fits a mean-field `AutoNormal`, and
the block guide starts from that solution. Started cold, the block guide reached clearly worse
optima in our tests.

`logistic_guide='normal'`: mean-field `AutoNormal` only.

Optimiser: `ClippedAdam`, default lr 1e-2, decaying to 10 % over `niters` (default 20,000).
Each phase stops when the mean loss over the last 1000 steps improves on the preceding
1000 steps by less than `logistic_tolerance` (relative, default 1e-4).

## Output

`posterior_samples_trans` holds `b` [S, T], `V`, `k`, `n`, `log2_K`, `hill_n` [S, J, T],
the raw sites (`z`, `lam`, `log_n`, `k_raw`, `tau`), the dispersion sites, and the scalars
`u_center`, `u_scale`, `n_components` and `link`. `save_trans_summary()` writes, with 95 %
intervals, `b`, `V_j`, `log2_K_j`, `hill_n_j`, and the range of the fitted curve over the
observed x range (`observed_log2fc` for negbinom, `observed_delta_p` for binomial,
`observed_delta` otherwise).

## Benchmark

`examples/diagnostics/benchmark_logistic_hill.py` simulates 3000 cells (5 NTC and 15 graded
knockdown guides) and 60 negative binomial trans genes: 20 null, 15 increasing, 15 decreasing
and 10 non-monotone (up then down). The true x is given to the model, so only the trans step is
tested. The table gives the median RMSE of the posterior-mean log2FC curve over the observed
x range, and the fraction of grid points where the truth lies in the 90 % pointwise band.
The logistic_hill runs used 2 CPU threads and the additive_hill runs 1, so the times are not directly comparable.

| model | steps | RMSE up / down / bump | 90 % coverage up / down / bump | time |
|---|---|---|---|---|
| additive_hill, default settings (EWMA rule stopped it) | 4,131 | 0.076 / 0.057 / 0.49 | 0.47 / 0.58 / 0.06 | 278 s |
| additive_hill | 20,000 | 0.025 / 0.021 / 0.42 | 0.83 / 0.84 / 0.30 | 883 s |
| logistic_hill, mean-field | 5,000 | 0.025 / 0.033 / 0.28 | 0.89 / 0.83 / 0.34 | 132 s |
| logistic_hill, block_mvn, fixed length | 5,000 | 0.025 / 0.021 / 0.27 | 0.91 / 0.89 / 0.48 | 130 s |
| logistic_hill, block_mvn, default settings | 8,000 | 0.030 / 0.026 / 0.26 | 1.00 / 0.94 / 0.62 | 208 s |

All models separate null from non-null genes perfectly in this simulation (AUC = 1). The
non-monotone genes remain the hardest case. Their bands are still too narrow, partly because
their rising flank lies where only the two or three strongest guides give data.

## Limitations

- Multinomial is not supported.
- The plotting functions for fitted trans curves (`plot_xy_data` with fitted curves,
  derivative-based classification in the summary) do not yet support `logistic_hill`.
- The simulation is a single data set with one seed. It shows that the reparameterisation behaves
  as intended; it is not a general performance estimate.
