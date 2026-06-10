"""
NTC fitting for bayesDREAM.

This module contains the NTC model and fitting logic (overdispersion estimation and batch effect correction).
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.distributions import constraints, transform_to
import pyro
import pyro.distributions as dist
from pyro.distributions.transforms import iterated, affine_autoregressive
import pyro.optim as optim
import pyro.infer as infer
import multiprocessing

# Optional dependency for memory detection
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False



class NTCFitter:
    """Handles NTC fitting: estimates gene-specific overdispersion and batch effects."""

    def __init__(self, model):
        """
        Initialize NTC fitter.

        Parameters
        ----------
        model : _BayesDREAMCore
            The parent model instance
        """
        self.model = model

    def _t(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self.model.device)

    def _to_cpu(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        return x

    def _estimate_predictive_memory_and_set_minibatch(
        self,
        N, T, C, nsamples, minibatch_size=None,
        distribution='negbinom',
        safety_factor=None  # Auto-determined: 0.7 for SLURM, 0.35 for shared nodes
    ):
        """
        Estimate memory requirements for Predictive and auto-set minibatch_size.

        Parameters
        ----------
        N : int
            Number of observations (cells)
        T : int
            Number of features
        C : int
            Number of technical groups
        nsamples : int
            Number of posterior samples requested
        minibatch_size : int or None
            If provided, use this value. If None, auto-compute.
        distribution : str
            Distribution type
        safety_factor : float or None
            Fraction of available RAM to use. If None (default), auto-determined:
            - 0.7 (70%) when SLURM allocation detected (dedicated resources)
            - 0.35 (35%) on shared nodes without SLURM (conservative)

        Returns
        -------
        minibatch_size : int or None
            Recommended minibatch size, or None if full batch fits
        use_parallel : bool
            Whether to use parallel=True in Predictive
        """
        # If user explicitly set minibatch_size, respect it
        if minibatch_size is not None:
            print(f"[MEMORY] Using user-specified minibatch_size={minibatch_size}")
            return minibatch_size, True

        # If psutil not available, use conservative defaults
        if not HAS_PSUTIL:
            print("[INFO] psutil not available - using conservative defaults (minibatch_size=50)")
            return 50, False

        # Get available CPU RAM
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)

        # Check for SLURM allocation (more accurate on HPC)
        slurm_mem_gb = None

        # Try SLURM_MEM_PER_NODE first (older SLURM versions)
        if 'SLURM_MEM_PER_NODE' in os.environ:
            try:
                slurm_mem_mb = int(os.environ['SLURM_MEM_PER_NODE'])
                slurm_mem_gb = slurm_mem_mb / 1024
            except (ValueError, KeyError):
                pass

        # Try SLURM_MEM_PER_CPU * SLURM_CPUS_ON_NODE (more common)
        if slurm_mem_gb is None and 'SLURM_MEM_PER_CPU' in os.environ:
            try:
                mem_per_cpu_mb = int(os.environ['SLURM_MEM_PER_CPU'])

                # Try different CPU count variables
                n_cpus = None
                if 'SLURM_CPUS_ON_NODE' in os.environ:
                    n_cpus = int(os.environ['SLURM_CPUS_ON_NODE'])
                elif 'SLURM_CPUS_PER_TASK' in os.environ:
                    n_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
                elif 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
                    n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])

                if n_cpus is not None:
                    slurm_mem_gb = (mem_per_cpu_mb * n_cpus) / 1024
            except (ValueError, KeyError):
                pass

        # Use SLURM allocation if available, otherwise use available memory
        # Auto-determine safety factor based on whether we have dedicated SLURM allocation
        if slurm_mem_gb is not None:
            usable_gb = slurm_mem_gb
            if safety_factor is None:
                safety_factor = 0.7  # 70% for dedicated SLURM allocation
        else:
            usable_gb = available_gb
            if safety_factor is None:
                safety_factor = 0.35  # 35% for shared nodes

        # Estimate memory per sample (in GB)
        params_per_sample_gb = (C * T + C * T + 3 * T) * 4 / (1024**3)
        input_data_gb = (N * T + N) * 4 / (1024**3)

        # CRITICAL: Multiple temporary [N, T] tensors exist simultaneously during forward pass:
        # 1. alpha_y_used [S, N, T] from gather
        # 2. mu_final [S, N, T] from broadcasting
        # 3. mean_param [S, N, T] from arithmetic
        # 4. observation sample [S, N, T] (filtered by keep_sites but still created)
        # Conservative estimate: 4× the single tensor size
        temp_per_sample_gb = 4 * (N * T) * 4 / (1024**3)
        memory_per_sample_gb = params_per_sample_gb + temp_per_sample_gb

        # Check if parallel execution fits
        n_workers = min(multiprocessing.cpu_count(), 32)
        mem_per_worker_gb = input_data_gb + params_per_sample_gb * 10
        total_parallel_gb = n_workers * mem_per_worker_gb
        use_parallel = total_parallel_gb <= usable_gb * safety_factor

        # Determine if minibatching is needed
        safe_memory_gb = usable_gb * safety_factor
        available_for_sampling_gb = safe_memory_gb - input_data_gb
        max_samples_at_once = int(available_for_sampling_gb / memory_per_sample_gb)

        if max_samples_at_once >= nsamples:
            return None, use_parallel
        else:
            # Need minibatching - disable parallel to avoid memory spike
            use_parallel = False
            recommended_batch = max(10, min(100, max_samples_at_once)) if max_samples_at_once >= 10 else max(1, max_samples_at_once)
            n_batches = int(np.ceil(nsamples / recommended_batch))
            print(f"[INFO] Minibatching: {recommended_batch} samples/batch × {n_batches} batches (parallel=False)")
            return recommended_batch, use_parallel

    def _model_ntc(
        self,
        N,
        T,
        C,
        groups_ntc_tensor,
        y_obs_ntc_tensor,
        sum_factor_ntc_tensor,
        beta_o_alpha_tensor,
        beta_o_beta_tensor,
        mu_x_mean_tensor,
        mu_x_sd_tensor,
        epsilon_tensor,
        distribution='negbinom',
        denominator_ntc_tensor=None,
        K=None,
        D=None,
        sigma_hat_tensor=None,
        skip_obs_sampling=False,
    ):
        """
        Technical model used for NTC-only prefit of cell-line effects.
        - negbinom: multiplicative effects on mu (alpha_full_mul), NB dispersion phi_y
        - normal/binomial/studentt: additive or logit-scale effects (alpha_full_add)
        - multinomial: Dirichlet probabilities per (feature, category); additive or logit-scale effects (alpha_full_add)
        """
    
        # ----------------------------
        # PLATES for groups and feats
        # ----------------------------
        c_plate  = pyro.plate("c_plate", C - 1, dim=-2)      # groups except baseline
        f_plate  = pyro.plate("feature_plate_technical", T, dim=-1)
    
        # ----------------------------
        # Cell-line effects (group × T [× K])
        # We sample a single latent matrix log2_alpha_y and
        # expose two parameterizations:
        #   multiplicative: alpha_full_mul  (baseline=1)
        #   additive/logit: alpha_full_add  (baseline=0)
        # ----------------------------
        if distribution == 'multinomial':
            assert K is not None, "multinomial requires K"

            # ---- zero-probability category mask per (T,K) from ALL NTC counts ----
            # (We need ALL NTC data to identify structurally absent categories)
            total_counts_per_feature = y_obs_ntc_tensor.sum(dim=0)  # [T, K]
            zero_cat_mask = (total_counts_per_feature == 0)         # [T, K] bool

            # ---- Per-group zero-count masking ----
            # Mask categories that are 0 in ANY technical group
            # This prevents fitting corrections for categories with no data in some groups
            any_group_zero_mask = torch.zeros_like(zero_cat_mask, dtype=torch.bool)
            for g in range(C):
                group_mask = (groups_ntc_tensor == g)
                if group_mask.sum() > 0:
                    group_counts = y_obs_ntc_tensor[group_mask, :, :].sum(dim=0)  # [T, K]
                    any_group_zero_mask = any_group_zero_mask | (group_counts == 0)

            # Combine: mask if zero across all data OR zero in any group
            zero_cat_mask = zero_cat_mask | any_group_zero_mask
            pyro.deterministic("zero_cat_mask", zero_cat_mask)

            # ---- Compute counts from reference group (group 0) for Dirichlet prior ----
            ref_mask = (groups_ntc_tensor == 0)
            if ref_mask.sum() > 0:
                # Use reference group only for more accurate baseline
                total_counts_ref = y_obs_ntc_tensor[ref_mask, :, :].sum(dim=0)  # [T, K]
            else:
                # Fallback to all data if no reference group
                total_counts_ref = total_counts_per_feature  # [T, K]
            # Count active categories after masking
            active_k = (~zero_cat_mask).sum(dim=-1)  # [T]
            
            if (active_k <= 1).any():
                bad = torch.nonzero(active_k <= 1, as_tuple=False).squeeze(-1).tolist()
                raise RuntimeError(
                    "Multinomial QC mismatch: feature(s) with ≤1 active category reached "
                    f"_model_ntc (indices in T_fit: {bad}). "
                    "These should have been filtered out in fit_technical()."
                )
        
            if C == 1:
                # No group effects to estimate — trivial correction
                alpha_full_mul = None
                alpha_full_add = torch.zeros(1, T, K, device=self.model.device)
            else:
                # ---- Cell-line logits α for NON-baseline groups (baseline=0 implicit) ----
                with pyro.plate("c_plate_multi", C - 1, dim=-3), \
                     pyro.plate("f_plate_multi", T, dim=-2), \
                     pyro.plate("k_plate", K, dim=-1):
                    alpha_logits_y = pyro.sample("alpha_logits_y", dist.StudentT(df=self._t(3), loc=self._t(0.0), scale=self._t(20.0)))

                # Force α to 0 where category is structurally absent
                alpha_logits_y = alpha_logits_y.masked_fill(zero_cat_mask.unsqueeze(0), 0.0)

                # Center over ACTIVE categories only.
                # Phantoms are already 0, so sum(alpha, dim=-1) = sum over active categories.
                # Dividing by n_active (not K) gives the correct softmax centering constraint
                # (sum of active alphas = 0).  Dividing by K would leave a residual constant
                # shift in the active alphas, breaking identifiability for features with
                # many phantom categories.
                n_active = (~zero_cat_mask).float().sum(dim=-1, keepdim=True).clamp_min(1.0)  # [T, 1]
                active_mean = alpha_logits_y.sum(dim=-1, keepdim=True) / n_active             # [C-1, T, 1]
                alpha_logits_y = alpha_logits_y - active_mean
                alpha_logits_y = alpha_logits_y.masked_fill(zero_cat_mask.unsqueeze(0), 0.0)
                alpha_logits_y = pyro.deterministic("alpha_logits_y_centered", alpha_logits_y)

                # Build full [C, T, K] (or [S, C, T, K]) with baseline=0
                if alpha_logits_y.dim() == 3:
                    alpha_full_add_logits = torch.cat(
                        [torch.zeros(1, T, K, device=self.model.device, dtype=alpha_logits_y.dtype),
                         alpha_logits_y.to(self.model.device)], dim=0
                    )
                elif alpha_logits_y.dim() == 4:
                    S = alpha_logits_y.size(0)
                    alpha_full_add_logits = torch.cat(
                        [torch.zeros(S, 1, T, K, device=self.model.device, dtype=alpha_logits_y.dtype),
                         alpha_logits_y.to(self.model.device)], dim=1
                    )
                else:
                    raise ValueError(f"Unexpected alpha_logits_y shape: {tuple(alpha_logits_y.shape)}")

                alpha_full_mul = None
                alpha_full_add = alpha_full_add_logits

            #print(f'[DEBUG], alpha_logits_y shape = {alpha_logits_y.shape}, expect [C-1={C-1}, T={T}, K={K}')
            #print(f'[DEBUG], alpha_full_add shape = {alpha_full_add.shape}')
        else:
            # ----------------------------
            # Shared cell-line effects for non-multinomial dists
            # ----------------------------
            if C == 1:
                # No group effects to estimate — trivial correction tensors
                alpha_full_mul = torch.ones(1, T, device=self.model.device)
                alpha_full_add = torch.zeros(1, T, device=self.model.device)
            else:
                if distribution in ("normal", "studentt"):
                    # For Normal we want additive shifts on the ORIGINAL scale,
                    # with a width that reflects the data:
                    #   alpha ~ StudentT(ν=3, loc=0, scale ~ robust_SD / sqrt(N))
                    #
                    # mu_x_sd_tensor is [T] and encodes robust SD (MAD-based) / sqrt(N).
                    alpha_scale = mu_x_sd_tensor.clamp_min(epsilon_tensor)

                    with f_plate:
                        with c_plate:  # [C-1, T]
                            # Name kept as log2_alpha_y for backwards compatibility;
                            # for Normal it is *literally* the additive shift in y-units.
                            log2_alpha_y = pyro.sample(
                                "log2_alpha_y",
                                dist.StudentT(
                                    df=self._t(3),
                                    loc=self._t(0.0),
                                    scale=alpha_scale,        # [T], broadcast to [C-1, T]
                                ),
                            )

                            # For Normal we don't *use* multiplicative effects at all:
                            # keep alpha_y_mul around for API compatibility, but fix it to 1.
                            alpha_y_mul = pyro.deterministic(
                                "alpha_y_mul",
                                torch.ones_like(log2_alpha_y)
                            )

                            # Additive shift on the mean
                            delta_y_add = pyro.deterministic(
                                "delta_y_add",
                                log2_alpha_y          # direct additive shift on mean
                            )

                else:
                    # Original behaviour for negbinom / binomial, etc.
                    with f_plate:
                        with c_plate:  # [C-1, T]
                            log2_alpha_y = pyro.sample(
                                "log2_alpha_y",
                                dist.StudentT(
                                    df=self._t(3),
                                    loc=self._t(0.0),
                                    scale=self._t(20.0),
                                ),
                            )
                            alpha_y_mul = pyro.deterministic(
                                "alpha_y_mul", self._t(2.0) ** log2_alpha_y
                            )
                            delta_y_add = pyro.deterministic(
                                "delta_y_add", log2_alpha_y
                            )

                if alpha_y_mul.ndim == 2:
                    alpha_full_mul = torch.cat(
                        [torch.ones(1, T, device=self.model.device), alpha_y_mul.to(self.model.device)],
                        dim=0
                    )
                    alpha_full_add = torch.cat(
                        [torch.zeros(1, T, device=self.model.device), delta_y_add.to(self.model.device)],
                        dim=0
                    )
                elif alpha_y_mul.ndim == 3:
                    S = alpha_y_mul.size(0)
                    alpha_full_mul = torch.cat(
                        [torch.ones(S, 1, T, device=self.model.device), alpha_y_mul.to(self.model.device)],
                        dim=1
                    )
                    alpha_full_add = torch.cat(
                        [torch.zeros(S, 1, T, device=self.model.device), delta_y_add.to(self.model.device)],
                        dim=1
                    )
                else:
                    raise ValueError(f"Unexpected alpha/log2 shapes: {alpha_y_mul.shape}, {delta_y_add.shape}")

        #print(f'[DEBUG], alpha_full_add shape = {alpha_full_add.shape}')
    
        # --------------------------------
        # Dispersion / variance priors
        # --------------------------------
        phi_y       = None  # NB overdispersion (as total_count)
        sigma_y     = None  # Normal per-feature std
        sigma_y_mv  = None  # MVN per-(feature,dim) std
    
        if distribution == 'negbinom':
            beta_o = pyro.sample("beta_o", dist.Gamma(beta_o_alpha_tensor, beta_o_beta_tensor))
            with f_plate:
                o_y = pyro.sample("o_y", dist.Exponential(beta_o))     # [T]
            phi_y = 1.0 / (o_y ** 2)                                   # [T]
    
        elif distribution in ('normal', 'studentt'):
            with f_plate:
                if sigma_hat_tensor is not None:
                    # Data-driven scale (per-feature)
                    sigma_y = pyro.sample(
                        "sigma_y",
                        dist.HalfNormal(2.0 * sigma_hat_tensor)
                    )
                else:
                    # Fallback broad prior
                    sigma_y = pyro.sample(
                        "sigma_y",
                        dist.HalfCauchy(self._t(10.0))
                    )
        
        # --------------------------------
        # Baseline means per distribution
        # --------------------------------
        # NOTE: we DO NOT pre-apply cell-line effects here; samplers will.
        mu_y = None
    
        if distribution in ('negbinom', 'normal', 'studentt'):
            # mu_ntc shape: [T]
            with f_plate:
                if distribution in ('normal', 'studentt'):
                    mu_ntc = pyro.sample("mu_ntc", dist.Normal(mu_x_mean_tensor, mu_x_sd_tensor))
                else:  # negbinom
                    # LogNormal prior — same mean and variance as the old Gamma(μ²/σ², μ/σ²),
                    # but parameterised natively in log-space so AutoNormal's unconstrained
                    # transformation aligns naturally with no clamping required.
                    # Matches the convention used by fit_trans for Vmax/K parameters.
                    # Floor on log_sigma prevents the prior from collapsing to a point mass
                    # when σ << μ (very stably expressed genes).
                    log_sigma = torch.sqrt(
                        torch.log1p((mu_x_sd_tensor / mu_x_mean_tensor) ** 2)
                    ).clamp(min=0.1)
                    log_mu = torch.log(mu_x_mean_tensor) - 0.5 * log_sigma ** 2
                    mu_ntc = pyro.sample(
                        "mu_ntc",
                        dist.LogNormal(log_mu, log_sigma)
                    )
            mu_y = mu_ntc  # [T]
    
        elif distribution == 'binomial':
            # Empirical-Bayes Beta with bounded concentration (stable when denom >> counts)
            # Use reference group (group 0) only for more accurate baseline
            ref_mask = (groups_ntc_tensor == 0)

            if ref_mask.sum() > 0:
                # Compute from reference group only
                y_sum_ref  = y_obs_ntc_tensor[ref_mask, :].sum(dim=0).float()    # [T]
                den_sum_ref = denominator_ntc_tensor[ref_mask, :].sum(dim=0).float()  # [T]
            else:
                # Fallback to all data if no reference group
                y_sum_ref  = y_obs_ntc_tensor.sum(dim=0).float()    # [T]
                den_sum_ref = denominator_ntc_tensor.sum(dim=0).float()  # [T]

            # Smooth p-hat to keep it off 0/1 even if den_sum==0
            p_hat = (y_sum_ref + 0.5) / (den_sum_ref + 1.0)         # [T] in (0,1)
            p_hat = torch.clamp(p_hat, 1e-6, 1 - 1e-6)

            # Cap effective sample size: informative but not razor-sharp
            # tune these if needed (e.g., 20..100)
            kappa = torch.clamp(den_sum_ref, min=20.0, max=200.0)

            # Tiny floor to avoid exactly 0 concentration parameters
            a = p_hat * kappa + 1e-3
            b = (1.0 - p_hat) * kappa + 1e-3

            with f_plate:
                mu_ntc = pyro.sample("mu_ntc", dist.Beta(a, b))  # [T]
            mu_y = mu_ntc
    
        elif distribution == 'multinomial':
            # ---- Baseline category probabilities per feature: DO NOT wrap in a plate ----
            # Use reference group counts for more accurate baseline (concentration is [T, K])
            concentration = total_counts_ref + 1.0  # [T, K], strictly > 0
            # Zero concentration for phantom categories (masked in zero_cat_mask) so the
            # Dirichlet prior doesn't leak mass to padding positions and bias real categories.
            concentration = torch.where(
                zero_cat_mask,
                torch.full_like(concentration, 1e-6),
                concentration,
            )
            with f_plate:
                probs0 = pyro.sample("probs_baseline_raw", dist.Dirichlet(concentration))  # [T, K]

            # Hard-zero masked categories and renormalize across active ones
            probs_masked = probs0 * (~zero_cat_mask).to(probs0.dtype)                 # [T, K]
            row_sums = probs_masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            probs_baseline = probs_masked / row_sums                                   # [T, K]
            probs_baseline = pyro.deterministic("probs_baseline", probs_baseline)

            mu_y = probs_baseline  # [T, K]

    
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
        # --------------------------------
        # Call distribution-specific sampler
        # --------------------------------
        # --------------------------------
        # Call distribution-specific sampler
        # --------------------------------
        # CRITICAL: Skip observation sampling during Predictive to avoid NaN/Inf issues
        # During Predictive, posterior samples of alpha_y can be extreme due to heavy-tailed priors
        # We only need parameter posteriors (alpha, mu, phi), not observations
        if not skip_obs_sampling:
            from .distributions import get_observation_sampler
            observation_sampler = get_observation_sampler(distribution, 'trans')

        if not skip_obs_sampling and distribution == 'negbinom':
            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,           # [N, T]
                mu_y=mu_y,                               # [T]
                phi_y_used=phi_y.unsqueeze(-2),          # [1, T]
                alpha_y_full=alpha_full_mul,             # [C, T]
                groups_tensor=groups_ntc_tensor,
                sum_factor_tensor=sum_factor_ntc_tensor,
                N=N, T=T, C=C
            )

        elif not skip_obs_sampling and distribution == 'normal':
            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,           # [N, T]
                mu_y=mu_y,                               # [T]
                sigma_y=sigma_y,                         # [T]
                alpha_y_full=alpha_full_add,             # [C, T]
                groups_tensor=groups_ntc_tensor,
                N=N, T=T, C=C
            )

        elif not skip_obs_sampling and distribution == 'studentt':
            # Degrees of freedom (nu) - two options:
            # Option 1: Fixed value (simpler, faster)
            nu_y = self._t(3.0)
            # Option 2: Sample per-feature (more flexible, slower)
            # with f_plate:
            #     nu_y = pyro.sample("nu_y", dist.Gamma(self._t(10.0), self._t(2.0)))  # mean~5, ensures df>2

            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,           # [N, T]
                mu_y=mu_y,                               # [T]
                sigma_y=sigma_y,                         # [T]
                nu_y=nu_y,                               # scalar or [T]
                alpha_y_full=alpha_full_add,             # [C, T]
                groups_tensor=groups_ntc_tensor,
                N=N, T=T, C=C
            )

        elif not skip_obs_sampling and distribution == 'binomial':
            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,           # [N, T]
                denominator_tensor=denominator_ntc_tensor,
                mu_y=mu_y,                               # [T] in [0,1]
                alpha_y_full=alpha_full_add,             # [C, T] (add on logit)
                groups_tensor=groups_ntc_tensor,
                N=N, T=T, C=C
            )

        elif not skip_obs_sampling and distribution == 'multinomial':
            # mu_y is [T, K] baseline; expand to [N, T, K]
            #mu_y_multi = mu_y.unsqueeze(0).expand(N, T, K)
            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,
                #mu_y=mu_y_multi,
                mu_y=mu_y,
                alpha_y_full=alpha_full_add,     # <— ADD THIS
                groups_tensor=groups_ntc_tensor, # <— ADD THIS
                N=N, T=T, K=K, C=C
            )

        elif not skip_obs_sampling:
            # Only raise error if we're actually trying to sample observations
            raise ValueError(f"Unknown distribution: {distribution}")
    
    def set_technical_groups(self, covariates: list[str]):
        # Allow empty covariates list to create a single technical group
        if not covariates:
            print("[INFO] No covariates specified - creating single technical group (C=1)")
            self.model.meta["technical_group_code"] = 0
            return

        missing_cols = [c for c in covariates if c not in self.model.meta.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in meta: {missing_cols}")

        self.model.meta["technical_group_code"] = self.model.meta.groupby(covariates).ngroup()
        print(f"[INFO] Set technical_group_code with {self.model.meta['technical_group_code'].nunique()} groups based on {covariates}")
    
        # ---- SAFEGUARD: ensure every technical group is represented among NTCs ----
        # Skip this check if require_ntc=False (e.g., when NTC cells are not needed)
        require_ntc = getattr(self.model, 'require_ntc', True)
        if require_ntc and "target" in self.model.meta.columns:
            all_groups  = set(self.model.meta["technical_group_code"].unique().tolist())
            ntc_groups  = set(self.model.meta.loc[self.model.meta["target"] == "ntc", "technical_group_code"].unique().tolist())
            missing_ntc = sorted(all_groups - ntc_groups)
            if missing_ntc:
                n_drop = int(self.model.meta["technical_group_code"].isin(missing_ntc).sum())
                warnings.warn(
                    "[WARNING] Dropping cells from technical groups with no NTC representation. "
                    f"Groups: {missing_ntc} | Cells dropped: {n_drop}",
                    UserWarning
                )
                self.model.meta = self.model.meta.loc[~self.model.meta["technical_group_code"].isin(missing_ntc)].copy()
                # Optional: re-announce counts after drop
                print(f"[INFO] After dropping non-NTC groups, {self.model.meta['technical_group_code'].nunique()} technical group(s) remain.")


    ########################################################
    # Step 1: Optional Prefit for alpha_y (NTC only)
    ########################################################
    def fit_ntc(
        self,
        sum_factor_col: str = 'sum_factor',
        lr: float = 1e-3,
        niters: int = None,
        nsamples: int = 1_000,
        alpha_ewma: float = 0.05,
        tolerance: float = 1e-4,   # recommended to keep based on cell2location
        beta_o_beta: float = 3,    # recommended to keep based on cell2location
        beta_o_alpha: float = 9,   # recommended to keep based on cell2location
        epsilon: float = 1e-6,
        minibatch_size: int = None,
        distribution: str = None,
        denominator: np.ndarray = None,
        modality_name: str = None,
        use_all_cells: bool = False,
        force_iaf: bool = False,
        **kwargs
    ):
        """
        Prefit cell-line technical effects for a given modality.
        Stores both multiplicative ('alpha_y_mult') and additive/logit ('alpha_y_add') effects.

        Parameters
        ----------
        use_all_cells : bool, default False
            If False (default): Fit using NTC cells only (standard approach).
            If True: Fit using all cells in the dataset.

            **When to use use_all_cells=True:**
            - High MOI experiments where technical effects are batch/lane specific
            - Technical variation is independent of perturbation effects
            - Saves compute: fit_technical only needs to run once per dataset (not per cis gene)

            **When NOT to use use_all_cells=True (use default NTC-only):**
            - Technical groups correlate with cis gene expression
              Example: cell lines that differ in perturbation type
            - Low MOI experiments with clear NTC vs perturbed distinction
            - When technical correction should be based solely on unperturbed cells

        Warnings
        --------
        Using use_all_cells=True when technical effects correlate with cis expression
        may lead to over-correction and spurious trans effects.
        """

        # ---------------------------
        # Resolve modality/distribution
        # ---------------------------
        if modality_name is None:
            modality_name = self.model.primary_modality
        modality = self.model.get_modality(modality_name)

        if distribution is None:
            distribution = modality.distribution

        # ---------------------------
        # Set conditional default for niters
        # ---------------------------
        niters_was_default = (niters is None)
        if niters is None:
            # Default: 50,000 unless multivariate (multinomial), then 100,000
            if distribution in ('multinomial'):
                niters = 100_000
                print(f"[INFO] Using default niters=100,000 for multivariate distribution '{distribution}'")
            else:
                niters = 50_000
                print(f"[INFO] Using default niters=50,000 for distribution '{distribution}'")

        if denominator is None and modality.denominator is not None:
            denominator = modality.denominator

        # CRITICAL: Convert denominator numpy.matrix to numpy.ndarray if needed
        if denominator is not None and isinstance(denominator, np.matrix):
            print(f"[INFO] Converting deprecated numpy.matrix denominator to numpy.ndarray")
            denominator = np.asarray(denominator)

        # ========================================================================
        # CRITICAL VALIDATION: Cis extraction requires negative binomial distribution
        # ========================================================================
        if (
            modality_name == self.model.primary_modality
            and hasattr(self.model, 'counts')
            and self.model.cis_gene is not None
            and 'cis' in self.model.modalities
        ):
            if distribution != 'negbinom':
                raise ValueError(
                    f"Primary modality has distribution '{distribution}', but cis effect extraction "
                    f"requires 'negbinom' distribution. "
                    f"The cis gene/feature must represent count data following a negative binomial distribution. "
                    f"Cannot perform cis modeling with '{distribution}' distribution."
                )

        # Always use the modality's filtered counts directly (densify if sparse)
        counts_to_fit = modality.counts
        if hasattr(counts_to_fit, 'toarray'):
            counts_to_fit = counts_to_fit.toarray()
        if modality.cell_names is not None:
            modality_cells = modality.cell_names
        else:
            modality_cells = self.model.meta['cell'].values[:counts_to_fit.shape[modality.cells_axis]]

        # Track number of trans features; we may append the cis gene row below
        n_trans_features = modality.dims['n_features']
        cis_idx_in_counts = None  # set if cis gene is appended as last row

        # For primary modality with a cis gene: append the cis gene row so we can
        # estimate alpha_x_prefit from the technical fit (NTC cells only).
        from scipy import sparse
        if (modality_name == self.model.primary_modality
                and self.model.cis_gene is not None
                and 'cis' in self.model.modalities):
            cis_mod = self.model.get_modality('cis')
            # cis modality has cells_axis=1, so counts is [1, N]
            cis_row = cis_mod.counts if cis_mod.cells_axis == 1 else cis_mod.counts.T

            if sparse.issparse(counts_to_fit):
                cis_row_sp = (cis_row.tocsr() if sparse.issparse(cis_row)
                              else sparse.csr_matrix(np.asarray(cis_row).reshape(1, -1)))
                counts_to_fit = sparse.vstack([counts_to_fit, cis_row_sp], format='csr')
            else:
                cis_row_dense = np.asarray(cis_row).reshape(1, -1)
                counts_to_fit = np.vstack([np.asarray(counts_to_fit), cis_row_dense])

            cis_idx_in_counts = n_trans_features  # cis gene is the last row
            print(f"[INFO] Appended cis gene '{self.model.cis_gene}' for alpha_x estimation")

        # CRITICAL: Convert numpy.matrix to numpy.ndarray if needed
        # numpy.matrix is deprecated and has problematic indexing behavior with boolean arrays
        if isinstance(counts_to_fit, np.matrix):
            print(f"[INFO] Converting deprecated numpy.matrix to numpy.ndarray")
            counts_to_fit = np.asarray(counts_to_fit)
        elif not isinstance(counts_to_fit, (np.ndarray, type(None))) and hasattr(counts_to_fit, '__array__'):
            # Handle any other array-like objects that aren't standard numpy arrays
            counts_to_fit = np.asarray(counts_to_fit)

        # CRITICAL: Convert COO sparse matrices to CSR for efficient row indexing
        # COO matrices don't support indexing operations needed throughout this function
        # CSR is optimal for row-based operations (subsetting, slicing, means)
        if sparse.issparse(counts_to_fit):
            if sparse.isspmatrix_coo(counts_to_fit):
                print(f"[INFO] Converting COO sparse matrix to CSR for efficient row indexing")
                counts_to_fit = counts_to_fit.tocsr()

        print(f"[INFO] Fitting technical model for modality '{modality_name}' (distribution: {distribution})")
    
        # ---------------------------
        # Validate requirements
        # ---------------------------
        from .distributions import requires_sum_factor, requires_denominator, is_3d_distribution
    
        if requires_sum_factor(distribution) and sum_factor_col is None:
            raise ValueError(f"Distribution '{distribution}' requires sum_factor_col parameter")

        if requires_sum_factor(distribution) and modality.sum_factors is None:
            raise ValueError(
                f"Modality '{modality_name}' has distribution '{distribution}' but no sum_factors "
                f"DataFrame has been set. Assign a cell-indexed DataFrame to "
                f"modality.sum_factors before calling fit_technical()."
            )

        if requires_denominator(distribution) and denominator is None:
            raise ValueError(f"Distribution '{distribution}' requires denominator parameter")
    
        if "technical_group_code" not in self.model.meta.columns:
            warnings.warn(
                "technical_group_code not set — assuming single group (no technical covariate correction). "
                "Call set_technical_groups(['cell_line', ...]) before fit_ntc() if you have multiple batches or cell lines.",
                UserWarning,
            )
            self.model.meta["technical_group_code"] = 0
    
        print("Running prefit_cellline...")
    
        # ---------------------------
        # Subset to NTC cells (or use all cells if requested)
        # ---------------------------
        modality_cell_set = set(modality_cells)
        meta_subset = self.model.meta[self.model.meta['cell'].isin(modality_cell_set)].copy()

        if use_all_cells:
            # Use all cells in the dataset
            warnings.warn(
                "[WARNING] use_all_cells=True: Fitting technical effects on ALL cells. "
                "Only use this mode if technical effects are independent of perturbation effects. "
                "If technical groups correlate with cis gene expression, "
                "use the default NTC-only mode to avoid over-correction.",
                UserWarning
            )
            meta_ntc = meta_subset.copy()  # Use all cells
            ntc_cell_list = meta_ntc["cell"].tolist()
            ntc_indices = list(range(len(modality_cells)))  # All cell indices

            # Subset counts -> ALL cells (no subsetting needed)
            counts_ntc_array = counts_to_fit

            print(f"[INFO] Modality '{modality_name}': {len(modality_cells)} total cells, using ALL cells for technical fit")
        else:
            # Standard NTC-only mode
            meta_ntc = meta_subset[meta_subset["target"] == "ntc"].copy()

            ntc_cell_list = meta_ntc["cell"].tolist()
            ntc_indices = [i for i, c in enumerate(modality_cells) if c in ntc_cell_list]

            # Subset counts -> NTC
            if counts_to_fit.ndim == 2:
                counts_ntc_array = counts_to_fit[:, ntc_indices] if modality.cells_axis == 1 else counts_to_fit[ntc_indices, :]
            elif counts_to_fit.ndim == 3:
                counts_ntc_array = counts_to_fit[:, ntc_indices, :]  # cells are axis 1 for 3D
            else:
                raise ValueError(f"Unexpected number of dimensions: {counts_to_fit.ndim}")

            print(f"[INFO] Modality '{modality_name}': {len(modality_cells)} total cells, {len(ntc_indices)} NTC cells")
    
        # ---------------------------
        # Quality filters per feature
        # ---------------------------
        if counts_ntc_array.ndim == 2:
            feature_sums_ntc = counts_ntc_array.sum(axis=1 if modality.cells_axis == 1 else 0)
            # CRITICAL: Sparse matrices return numpy.matrix of shape (n, 1) from sum()
            # This must be flattened to 1D to avoid broadcasting issues with masks
            from scipy import sparse
            if sparse.issparse(counts_ntc_array):
                feature_sums_ntc = np.asarray(feature_sums_ntc).flatten()
        elif counts_ntc_array.ndim == 3:
            feature_sums_ntc = counts_ntc_array.sum(axis=(1, 2))
        else:
            raise ValueError(f"Unexpected number of dimensions: {counts_ntc_array.ndim}")

        zero_count_mask = feature_sums_ntc == 0
        zero_std_mask = np.zeros(len(feature_sums_ntc), dtype=bool)

        # --- Per-group zero-count check for negbinom ---
        # Two cases need different treatment:
        #  - Reference group (group 0) all-zero: baseline expression can't be estimated;
        #    exclude the feature entirely (all groups get alpha_y = 1.0).
        #  - Non-reference group all-zero: fit the feature on the other groups normally;
        #    after fitting, patch alpha_y = 1.0 only for the zero-count group(s).
        any_group_zero_mask = np.zeros(len(feature_sums_ntc), dtype=bool)
        _negbinom_nonref_patch = {}   # {group_code (int): bool array [T_all]} — applied post-fit
        if distribution == 'negbinom' and 'technical_group_code' in meta_ntc.columns:
            groups_ntc_codes = meta_ntc['technical_group_code'].values
            unique_groups_nb = np.unique(groups_ntc_codes)
            if len(unique_groups_nb) > 1 and counts_ntc_array.ndim == 2:
                for g in unique_groups_nb:
                    g_mask = (groups_ntc_codes == g)
                    if g_mask.sum() == 0:
                        continue
                    if modality.cells_axis == 1:
                        group_sums = counts_ntc_array[:, g_mask].sum(axis=1)
                    else:
                        group_sums = counts_ntc_array[g_mask, :].sum(axis=0)
                    if sparse.issparse(group_sums):
                        group_sums = np.asarray(group_sums).flatten()
                    feat_zero = (group_sums == 0)
                    if g == 0:
                        # Reference group zero → baseline corrupted → exclude entirely
                        n_new = int((feat_zero & ~zero_count_mask).sum())
                        if n_new > 0:
                            print(f"[INFO] {modality_name}: {n_new} feature(s) excluded — "
                                  f"all-zero NTC counts in reference group (baseline can't be estimated).")
                        any_group_zero_mask |= feat_zero
                    else:
                        # Non-reference group zero → fit on other groups, patch this group after
                        # Only track features that aren't already excluded for other reasons
                        patchable = feat_zero & ~zero_count_mask
                        n_patch = int(patchable.sum())
                        if n_patch > 0:
                            print(f"[INFO] {modality_name}: {n_patch} feature(s) have all-zero NTC counts "
                                  f"in non-reference group {g}; will fit on other groups and "
                                  f"set alpha_y=1.0 for group {g} after fitting.")
                            _negbinom_nonref_patch[int(g)] = patchable

        if distribution == 'multinomial':
            for f_idx in range(counts_ntc_array.shape[0]):
                feature_counts = counts_ntc_array[f_idx, :, :]  # (cells, K)
                totals = feature_counts.sum(axis=1, keepdims=True)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratios = np.where(totals > 0, feature_counts / totals, 0)
                ratio_stds = ratios.std(axis=0)
                if np.all(ratio_stds == 0):
                    zero_std_mask[f_idx] = True
    
        elif distribution == 'binomial':
            if denominator is None:
                raise ValueError("Binomial distribution requires denominator for variance check")
            denom_ntc = denominator[:, ntc_indices] if denominator.ndim == 2 and modality.cells_axis == 1 else \
                        denominator[ntc_indices, :] if denominator.ndim == 2 else \
                        denominator[:, ntc_indices, :]

            # Get technical group assignments for NTC cells
            groups_ntc_codes = meta_ntc['technical_group_code'].values

            for f_idx in range(counts_ntc_array.shape[0]):
                if modality.cells_axis == 0:
                    numer = counts_ntc_array[:, f_idx]
                    denom = denom_ntc[:, f_idx]
                else:
                    numer = counts_ntc_array[f_idx, :]
                    denom = denom_ntc[f_idx, :]

                # Global check: all denominators zero
                valid = denom > 0
                if valid.sum() == 0:
                    zero_std_mask[f_idx] = True
                    continue

                # Global check: zero std across all valid cells
                ratios = numer[valid] / denom[valid]
                if ratios.std() == 0:
                    zero_std_mask[f_idx] = True
                    continue

                # Per-group boundary checks
                # Exclude if ANY group has all cells at a boundary (PSI=0, PSI=1, or denom=0)
                exclude_feature = False
                for g in np.unique(groups_ntc_codes):
                    group_mask = (groups_ntc_codes == g)
                    group_numer = numer[group_mask]
                    group_denom = denom[group_mask]

                    # Check 1: All denominators zero in this group
                    if (group_denom == 0).all():
                        exclude_feature = True
                        break

                    # Check 2: Among cells with valid denominators, check for boundary PSI
                    group_valid = group_denom > 0
                    if group_valid.sum() > 0:
                        group_numer_valid = group_numer[group_valid]
                        group_denom_valid = group_denom[group_valid]

                        # All PSI=0 in this group (all numerators zero)
                        if (group_numer_valid == 0).all():
                            exclude_feature = True
                            break

                        # All PSI=1 in this group (all numerator==denominator)
                        if (group_numer_valid == group_denom_valid).all():
                            exclude_feature = True
                            break

                if exclude_feature:
                    zero_std_mask[f_idx] = True
        
        elif distribution in ("normal", "studentt"):
            # ---------------------------------------------
            # NORMAL: exclude features that
            #  - have zero variance globally (ignoring NaNs), OR
            #  - are all-NaN in *any* technical group
            # ---------------------------------------------
            if counts_ntc_array.ndim != 2:
                raise ValueError(
                    f"Unexpected dims for distribution '{distribution}': {counts_ntc_array.ndim}"
                )

            y_np = counts_ntc_array.astype(float)

            # Figure out which axis is features vs cells
            if modality.cells_axis == 1:
                # counts_ntc_array: [features, cells]
                F = y_np.shape[0]
                def get_feat_vals(f_idx):
                    return y_np[f_idx, :]   # [cells]
            else:
                # counts_ntc_array: [cells, features]
                F = y_np.shape[1]
                def get_feat_vals(f_idx):
                    return y_np[:, f_idx]   # [cells]

            groups_ntc_codes = meta_ntc['technical_group_code'].values
            unique_groups = np.unique(groups_ntc_codes)

            zero_std_mask = np.zeros(F, dtype=bool)
            all_nan_any_group_mask = np.zeros(F, dtype=bool)

            for f_idx in range(F):
                feat_vals = get_feat_vals(f_idx)          # [cells]
                finite_global = np.isfinite(feat_vals)

                # If *no* finite values at all across NTC cells:
                #   feature is hopeless -> exclude
                if not finite_global.any():
                    zero_std_mask[f_idx] = True
                    all_nan_any_group_mask[f_idx] = True
                    continue

                # Global std ignoring NaNs
                if np.nanstd(feat_vals[finite_global]) == 0:
                    zero_std_mask[f_idx] = True

                # Per-group all-NaN check
                for g in unique_groups:
                    g_mask = (groups_ntc_codes == g)
                    if not g_mask.any():
                        continue  # shouldn't happen, but safe

                    group_vals = feat_vals[g_mask]
                    group_finite = np.isfinite(group_vals)

                    # If *within this group* there are no finite values,
                    # mark feature for exclusion.
                    if not group_finite.any():
                        all_nan_any_group_mask[f_idx] = True
                        break  # no need to check other groups

            # A feature is excluded if it has zero variance OR is all-NaN in any group
            zero_std_mask = zero_std_mask | all_nan_any_group_mask
    
        else:
            if counts_ntc_array.ndim == 2:
                # Handle sparse matrices which don't have .std() method
                from scipy import sparse
                if sparse.issparse(counts_ntc_array):
                    # Sparse matrix: convert to dense for std calculation per feature
                    axis = 1 if modality.cells_axis == 1 else 0
                    feature_std_ntc = np.array([np.std(counts_ntc_array[i, :].toarray() if modality.cells_axis == 1 else counts_ntc_array[:, i].toarray())
                                                for i in range(counts_ntc_array.shape[0 if modality.cells_axis == 1 else 1])])
                    zero_std_mask = feature_std_ntc == 0
                else:
                    # Dense array: use standard method
                    feature_std_ntc = counts_ntc_array.std(axis=1 if modality.cells_axis == 1 else 0)
                    zero_std_mask = feature_std_ntc == 0
            else:
                raise ValueError(f"Unexpected dims for distribution '{distribution}': {counts_ntc_array.ndim}")
    
        only_one_category_mask = np.zeros(len(feature_sums_ntc), dtype=bool)
        
        if distribution == 'multinomial':
            # per-feature categories must be present in *all* technical groups
            groups_ntc_codes = meta_ntc['technical_group_code'].values
            unique_groups = np.unique(groups_ntc_codes)

            F, _, K = counts_ntc_array.shape  # [features, cells, categories]

            for f_idx in range(F):
                feature_counts = counts_ntc_array[f_idx, :, :]  # (cells, K)

                shared_present = None  # will become boolean [K]
                for g in unique_groups:
                    g_mask = (groups_ntc_codes == g)
                    if not np.any(g_mask):
                        continue  # should not happen, but just in case

                    counts_g = feature_counts[g_mask, :].sum(axis=0)  # [K]
                    present_g = counts_g > 0                          # category present in this group?

                    if shared_present is None:
                        shared_present = present_g
                    else:
                        shared_present &= present_g  # intersection across groups

                if shared_present is None:
                    n_shared = 0
                else:
                    n_shared = int(shared_present.sum())

                # If 0 or 1 categories survive the "present in all groups" criterion,
                # the feature cannot support a multinomial cell-line effect.
                if n_shared <= 1:
                    only_one_category_mask[f_idx] = True

    
        needs_filtering_mask = zero_std_mask | only_one_category_mask
        needs_exclusion_mask = zero_count_mask | needs_filtering_mask | any_group_zero_mask

        # Count features that are excluded ONLY for each reason (no overlap)
        num_zero_count_only = (zero_count_mask & ~zero_std_mask & ~only_one_category_mask & ~any_group_zero_mask).sum()
        num_zero_std_only = (zero_std_mask & ~only_one_category_mask & ~any_group_zero_mask).sum()
        num_single_category = only_one_category_mask.sum()
        num_group_zero = (any_group_zero_mask & ~zero_count_mask).sum()  # ref-group-zero exclusions only
        num_excluded = needs_exclusion_mask.sum()

        if num_excluded > 0:
            warnings.warn(
                f"[WARNING] {num_excluded} feature(s) excluded from fitting: "
                f"{num_zero_count_only} zero-count-only, {num_zero_std_only} zero-std, "
                f"{num_single_category} ≤1 category present in all technical groups, "
                f"{num_group_zero} zero-in-one-group (negbinom only). "
                "Alpha set to baseline for excluded features.",
                UserWarning,
            )
    
        # ---------------------------
        # Subset to features that can be fit
        # ---------------------------
        fit_mask = ~needs_exclusion_mask
        if fit_mask.sum() == 0:
            raise ValueError("No features left to fit after filtering!")

        # CRITICAL: Convert boolean mask to integer indices for sparse matrix compatibility
        # Scipy sparse matrices don't support boolean indexing like dense arrays
        from scipy import sparse
        fit_indices = np.where(fit_mask)[0]

        if counts_ntc_array.ndim == 2:
            if sparse.issparse(counts_ntc_array):
                # Sparse matrix: use integer indices
                counts_ntc_for_fit = counts_ntc_array[fit_indices, :] if modality.cells_axis == 1 else counts_ntc_array[:, fit_indices]
            else:
                # Dense array: boolean indexing works fine
                counts_ntc_for_fit = counts_ntc_array[fit_mask, :] if modality.cells_axis == 1 else counts_ntc_array[:, fit_mask]
        else:
            # 3D arrays (multinomial) - use boolean indexing
            counts_ntc_for_fit = counts_ntc_array[fit_mask, :, :]

        denominator_ntc_for_fit = None
        if denominator is not None:
            if denominator.ndim == 2:
                denom_ntc = denominator[:, ntc_indices] if modality.cells_axis == 1 else denominator[ntc_indices, :]
                if sparse.issparse(denom_ntc):
                    # Sparse matrix: use integer indices
                    denominator_ntc_for_fit = denom_ntc[fit_indices, :] if modality.cells_axis == 1 else denom_ntc[:, fit_indices]
                else:
                    # Dense array: boolean indexing works fine
                    denominator_ntc_for_fit = denom_ntc[fit_mask, :] if modality.cells_axis == 1 else denom_ntc[:, fit_mask]
            elif denominator.ndim == 3:
                denom_ntc = denominator[:, ntc_indices, :]
                denominator_ntc_for_fit = denom_ntc[fit_mask, :, :]
    
        # ---------------------------
        # Build tensors for model
        # ---------------------------
        N = meta_ntc.shape[0]
    
        if counts_ntc_for_fit.ndim == 2:
            if modality.cells_axis == 1:
                y_obs_ntc = counts_ntc_for_fit.T   # [T, N] -> [N, T]
                T_fit = counts_ntc_for_fit.shape[0]
            else:
                y_obs_ntc = counts_ntc_for_fit     # [N, T]
                T_fit = counts_ntc_for_fit.shape[1]
            # Convert sparse to dense for PyTorch tensor creation
            if sparse.issparse(y_obs_ntc):
                y_obs_ntc = y_obs_ntc.toarray()
        else:
            y_obs_ntc = counts_ntc_for_fit.transpose(1, 0, 2)  # [T, N, K/D] -> [N, T, K/D]
            T_fit = counts_ntc_for_fit.shape[0]
    
        C = meta_ntc['technical_group_code'].nunique()
        groups_ntc = meta_ntc['technical_group_code'].values
    
        # Detect K/D
        K = None
        D = None
        if is_3d_distribution(distribution):
            if distribution == 'multinomial':
                K = y_obs_ntc.shape[2]
    
        # ---------------------------
        # Build data for priors per distribution
        # ---------------------------
        if y_obs_ntc.ndim == 3:
            if distribution == 'multinomial':
                y_obs_ntc_for_priors = y_obs_ntc.sum(axis=2)   # [N, T]
            else:
                y_obs_ntc_for_priors = y_obs_ntc.sum(axis=2)
        else:
            y_obs_ntc_for_priors = y_obs_ntc


        # Size-factor use only for NB; otherwise ignore
        if distribution == 'negbinom' and sum_factor_col is not None:
            ntc_sf = modality.sum_factors.loc[meta_ntc['cell'].values, sum_factor_col].values
            y_obs_ntc_factored = y_obs_ntc_for_priors / ntc_sf.reshape(-1, 1)
        else:
            y_obs_ntc_factored = y_obs_ntc_for_priors

        # Helper function to extract rows by mask from sparse or dense matrix
        # Maintains sparsity until the final computation
        from scipy import sparse
        def _extract_rows_sparse_safe(matrix, mask):
            """Extract rows using mask, handling sparse matrices efficiently."""
            if sparse.issparse(matrix):
                # Convert boolean mask to integer indices for sparse compatibility
                indices = np.where(mask)[0]
                # COO matrices don't support indexing - convert to CSR first
                if sparse.isspmatrix_coo(matrix):
                    matrix = matrix.tocsr()
                # Extract subset - stays sparse
                subset = matrix[indices, :]
                # Convert to dense only for the subset
                return np.asarray(subset.toarray() if hasattr(subset, 'toarray') else subset)
            else:
                # Dense array: use boolean indexing directly
                return matrix[mask, :]

        if distribution in ('normal', 'studentt'):
            baseline_mask = (groups_ntc == 0)
            y_baseline = _extract_rows_sparse_safe(y_obs_ntc_factored, baseline_mask)
            sigma_hat = np.nanstd(y_baseline, axis=0) + epsilon
            sigma_hat = np.where(np.isfinite(sigma_hat), sigma_hat, 1.0)
            sigma_hat_tensor = torch.tensor(sigma_hat, dtype=torch.float32, device=self.model.device)
        else:
            sigma_hat_tensor = None

        # ---- Helper: compute robust SD using MAD with SD fallback ----
        def _compute_robust_sd(y_data, epsilon, use_nan=False):
            """
            Compute robust SD using MAD (median absolute deviation) with SD fallback.

            MAD is more robust to outliers and heavy tails common in count data.
            Falls back to SD when MAD is 0 (e.g., >50% of values equal the median).

            Parameters
            ----------
            y_data : np.ndarray
                Data array of shape [N, T]
            epsilon : float
                Small constant to avoid division by zero
            use_nan : bool
                If True, use nanmedian/nanstd for NaN-aware computation

            Returns
            -------
            robust_sd : np.ndarray
                Robust SD estimate of shape [T]
            """
            if use_nan:
                median_vals = np.nanmedian(y_data, axis=0)
                mad = np.nanmedian(np.abs(y_data - median_vals), axis=0)
                cell_sd = np.nanstd(y_data, axis=0)
            else:
                median_vals = np.median(y_data, axis=0)
                mad = np.median(np.abs(y_data - median_vals), axis=0)
                cell_sd = np.std(y_data, axis=0)

            # Convert MAD to SD (scaling factor for normal distribution)
            robust_sd = 1.4826 * mad

            # Fallback to regular SD when MAD is 0 (e.g., >50% zeros)
            robust_sd = np.where(robust_sd > epsilon, robust_sd, cell_sd)

            return robust_sd

        # ---- mu_x priors per distribution ----
        if distribution == 'negbinom':
            baseline_mask = (groups_ntc == 0)
            y_baseline = _extract_rows_sparse_safe(y_obs_ntc_factored, baseline_mask)
            mu_x_mean = np.mean(y_baseline, axis=0)  # [T_fit]

            # Compute robust SD using MAD with SD fallback
            robust_sd = _compute_robust_sd(y_obs_ntc_factored, epsilon, use_nan=False)

            # Convert to SE of mean, with floor at 10% of mean
            N_cells = y_obs_ntc_factored.shape[0]
            mu_x_sd = robust_sd / np.sqrt(N_cells) + epsilon
            mu_x_sd = np.maximum(mu_x_sd, (mu_x_mean + epsilon) * 0.1)  # floor at 10% of mean

            mu_x_mean = mu_x_mean + epsilon  # strictly positive for Gamma

        elif distribution in ('normal', 'studentt'):
            baseline_mask = (groups_ntc == 0)
            y_baseline = _extract_rows_sparse_safe(y_obs_ntc_factored, baseline_mask)

            # NaN-aware baseline mean
            mu_x_mean = np.nanmean(y_baseline, axis=0)  # [T_fit]

            # Compute robust SD using MAD with SD fallback (NaN-aware)
            robust_sd = _compute_robust_sd(y_obs_ntc_factored, epsilon, use_nan=True)

            # Convert to SE of mean, with floor at 10% of |mean| (or 0.1 if mean is ~0)
            N_cells = y_obs_ntc_factored.shape[0]
            mu_x_sd = robust_sd / np.sqrt(N_cells) + epsilon
            mean_floor = np.maximum(np.abs(mu_x_mean) * 0.1, 0.1)
            mu_x_sd = np.maximum(mu_x_sd, mean_floor)

            # Safety: replace any remaining non-finite with defaults
            mu_x_mean = np.where(np.isfinite(mu_x_mean), mu_x_mean, 0.0)
            mu_x_sd   = np.where(np.isfinite(mu_x_sd),   mu_x_sd,   1.0)
    
        else:
            # binomial & multinomial: handled with Beta/Dirichlet in the model
            mu_x_mean = np.zeros((T_fit,), dtype=float)
            mu_x_sd   = np.ones((T_fit,), dtype=float)
    
        # CRITICAL MEMORY OPTIMIZATION: Use torch.from_numpy() to avoid copies
        # torch.tensor() creates a NEW copy, torch.from_numpy() shares memory with numpy array
        # Then explicitly delete numpy arrays and move tensors to device
        import gc

        # Tensors (scalars and small arrays - copy is fine)
        beta_o_beta_tensor  = torch.tensor(beta_o_beta,  dtype=torch.float32, device=self.model.device)
        beta_o_alpha_tensor = torch.tensor(beta_o_alpha, dtype=torch.float32, device=self.model.device)
        epsilon_tensor      = torch.tensor(epsilon, dtype=torch.float32, device=self.model.device)

        # Large arrays: use from_numpy() to share memory, then move to device
        mu_x_mean_tensor = torch.from_numpy(mu_x_mean.astype(np.float32)).to(self.model.device)  # [T_fit]
        mu_x_sd_tensor   = torch.from_numpy(mu_x_sd.astype(np.float32)).to(self.model.device)    # [T_fit]
        del mu_x_mean, mu_x_sd  # Free numpy arrays

        if sum_factor_col is not None and distribution == 'negbinom':
            sum_factor_np = modality.sum_factors.loc[
                meta_ntc['cell'].values, sum_factor_col
            ].values.astype(np.float32)
            sum_factor_ntc_tensor = torch.from_numpy(sum_factor_np).to(self.model.device)
            del sum_factor_np
        else:
            sum_factor_ntc_tensor = torch.ones(N, dtype=torch.float32, device=self.model.device)

        groups_ntc_tensor = torch.from_numpy(groups_ntc.astype(np.int64)).to(self.model.device)
        del groups_ntc

        # CRITICAL: y_obs_ntc is the LARGEST array (2.5 GB)
        # Ensure it's contiguous numpy array (required for from_numpy)
        if not y_obs_ntc.flags['C_CONTIGUOUS']:
            y_obs_ntc = np.ascontiguousarray(y_obs_ntc)
        y_obs_ntc_tensor = torch.from_numpy(y_obs_ntc.astype(np.float32)).to(self.model.device)
        del y_obs_ntc  # Free the 2.5 GB numpy array immediately
        gc.collect()  # Force garbage collection

        # TODO: PERFORMANCE OPTIMIZATION
        # Pre-compute sums that are currently recomputed in every _model_ntc call:
        # - For multinomial: total_counts_per_feature, per_group_counts, ref_counts
        # - For binomial: ref_y_sum, ref_denom_sum
        # These should be computed once here and passed as parameters to _model_ntc
    
        denominator_ntc_tensor = None
        if denominator_ntc_for_fit is not None:
            if modality.cells_axis == 1:
                denom_for_tensor = denominator_ntc_for_fit.T   # [T_fit, N] -> [N, T_fit]
            else:
                denom_for_tensor = denominator_ntc_for_fit     # [N, T_fit]

            # CRITICAL: Use from_numpy() to avoid copy (denominator can be 2.5 GB!)
            # Handle sparse matrices first
            from scipy import sparse
            if sparse.issparse(denom_for_tensor):
                denom_for_tensor = denom_for_tensor.toarray()
            if not denom_for_tensor.flags['C_CONTIGUOUS']:
                denom_for_tensor = np.ascontiguousarray(denom_for_tensor)
            denominator_ntc_tensor = torch.from_numpy(denom_for_tensor.astype(np.float32)).to(self.model.device)
            del denom_for_tensor  # Free numpy array immediately
            gc.collect()
    
        # ---------------------------
        # Multinomial precomputations (done once, before guide + SVI)
        # ---------------------------
        if distribution == 'multinomial':
            # Replicate the zero-category mask from _model_ntc so we can use it here.
            _total_all = y_obs_ntc_tensor.sum(dim=0)                          # [T, K]
            _zmask = (_total_all == 0)
            for _g in range(C):
                _gm = (groups_ntc_tensor == _g)
                if _gm.sum() > 0:
                    _zmask = _zmask | (y_obs_ntc_tensor[_gm].sum(dim=0) == 0)

            # Reference group empirical proportions → log-space init for softmax guide
            _ref_m = (groups_ntc_tensor == 0)
            _ref_counts = (y_obs_ntc_tensor[_ref_m].sum(dim=0).float() + 1.0
                           if _ref_m.sum() > 0 else y_obs_ntc_tensor.sum(dim=0).float() + 1.0)
            _ref_counts = _ref_counts.masked_fill(_zmask, 0.0)
            _ref_props = _ref_counts / _ref_counts.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            _log_p_init = torch.log(_ref_props.clamp(min=1e-8))              # [T, K]

            # Store for the guide closure
            self._multinomial_zero_cat_mask = _zmask                         # [T, K]
            self._multinomial_log_p_init    = _log_p_init                    # [T, K]

        # ---------------------------
        # Guide + init functions
        # ---------------------------
        import pyro.poutine as poutine
        from pyro.infer.autoguide import AutoNormal, AutoNormalMessenger, AutoIAFNormal
        from pyro.infer.autoguide.initialization import init_to_median
        
        def _safe_simplex_from_counts(total_counts_TK: torch.Tensor, interior_floor=1e-3) -> torch.Tensor:
            """
            total_counts_TK: [T, K] float
            Returns p: [T, K] on the simplex, strictly interior (all entries in (eps, 1-eps)).
            """
            p = total_counts_TK + 1.0
            p = p / p.sum(dim=-1, keepdim=True)
            K = p.shape[-1]
            eps = interior_floor / K
            p = (1.0 - K * eps) * p + eps
            p = p / p.sum(dim=-1, keepdim=True)
            return p
        
        def init_loc_fn(site):
            name = site["name"]

            # Derive local shapes from the actual tensors used in the model
            if y_obs_ntc_tensor.dim() == 3:
                T_local = int(y_obs_ntc_tensor.shape[1])
                K_local = int(y_obs_ntc_tensor.shape[2])
            else:
                T_local = int(y_obs_ntc_tensor.shape[1])
                K_local = None

            # -------------------------
            # Multinomial-only inits
            # -------------------------
            if distribution == 'multinomial':
                # probs_baseline_raw is guided by the softmax pyro.param directly —
                # init_loc_fn is never called for it. Only alpha_logits_y reaches here.
                if name == "alpha_logits_y":
                    # Initialize with logit-scale differences between groups vs reference
                    shape = site["fn"].batch_shape  # [C-1, T_fit, K]
                    if len(shape) != 3:
                        # Fallback to zeros if unexpected shape
                        return torch.zeros(shape, dtype=torch.float32, device=self.model.device)

                    Cminus1, T_fit_local, K_local = shape

                    # Compute empirical proportions per group
                    group_codes_np = groups_ntc_tensor.cpu().numpy()
                    y_obs_np = y_obs_ntc_tensor.cpu().numpy()  # [N, T, K]

                    # Reference group (group 0) proportions
                    ref_mask = (group_codes_np == 0)
                    if ref_mask.sum() > 0:
                        ref_counts = y_obs_np[ref_mask, :, :].sum(axis=0) + 0.5  # [T, K] with pseudocount
                        ref_props = ref_counts / ref_counts.sum(axis=1, keepdims=True)
                    else:
                        # Fallback uniform
                        ref_props = np.ones((T_fit_local, K_local)) / K_local

                    # Initialize alpha for each non-reference group
                    alpha_init = np.zeros((Cminus1, T_fit_local, K_local), dtype=np.float32)

                    non_ref_groups = sorted(set(group_codes_np) - {0})
                    for idx, g in enumerate(non_ref_groups[:Cminus1]):  # Ensure we don't exceed C-1
                        group_mask = (group_codes_np == g)
                        if group_mask.sum() > 0:
                            group_counts = y_obs_np[group_mask, :, :].sum(axis=0) + 0.5  # [T, K]
                            group_props = group_counts / group_counts.sum(axis=1, keepdims=True)

                            # Logit-scale difference: log(P_group / P_ref)
                            epsilon = 1e-6
                            ref_clipped = np.clip(ref_props, epsilon, 1 - epsilon)
                            group_clipped = np.clip(group_props, epsilon, 1 - epsilon)
                            alpha_init[idx, :, :] = np.log(group_clipped / ref_clipped)

                    return torch.tensor(alpha_init, dtype=torch.float32, device=self.model.device)
        
            # -------------------------
            # Other distributions — with binomial and negbinom inits
            # -------------------------
            if name == "log2_alpha_y":
                if distribution == 'negbinom':
                    group_codes = meta_ntc["technical_group_code"].values
                    group_labels = np.array(sorted(set(group_codes) - {0}))
                    init_values = []
                    # Use helper to safely extract baseline group (handles sparse matrices)
                    baseline_data = _extract_rows_sparse_safe(y_obs_ntc_factored, group_codes == 0)
                    baseline_mean = np.mean(baseline_data, axis=0) + 1e-8
                    for g in group_labels:
                        # Use helper for each group
                        group_data = _extract_rows_sparse_safe(y_obs_ntc_factored, group_codes == g)
                        group_mean = np.mean(group_data, axis=0) + 1e-8
                        init_values.append(np.log2(group_mean / baseline_mean))
                    init_arr = np.stack(init_values) if len(init_values) else np.zeros((0, T_local), dtype=np.float32)
                    return torch.tensor(init_arr, dtype=torch.float32, device=self.model.device)

                elif distribution == 'binomial':
                    # Initialize with logit-scale differences between groups (on logit scale)
                    group_codes_np = groups_ntc_tensor.cpu().numpy()
                    y_obs_np = y_obs_ntc_tensor.cpu().numpy()  # [N, T]
                    denom_np = denominator_ntc_tensor.cpu().numpy()  # [N, T]

                    # Compute PSI per group
                    epsilon = 1e-6

                    # Reference group (group 0) PSI
                    ref_mask = (group_codes_np == 0)
                    if ref_mask.sum() > 0:
                        ref_numer = y_obs_np[ref_mask, :].sum(axis=0) + 0.5  # [T] with pseudocount
                        ref_denom = denom_np[ref_mask, :].sum(axis=0) + 1.0   # [T] with pseudocount
                        ref_psi = np.clip(ref_numer / ref_denom, epsilon, 1 - epsilon)
                    else:
                        # Fallback uniform
                        ref_psi = np.ones(T_local) * 0.5

                    # Compute logit of reference PSI
                    ref_logit = np.log(ref_psi / (1 - ref_psi))

                    # Initialize alpha for each non-reference group
                    init_values = []
                    non_ref_groups = sorted(set(group_codes_np) - {0})

                    for g in non_ref_groups[:C-1]:  # Ensure we don't exceed C-1
                        group_mask = (group_codes_np == g)
                        if group_mask.sum() > 0:
                            group_numer = y_obs_np[group_mask, :].sum(axis=0) + 0.5  # [T]
                            group_denom = denom_np[group_mask, :].sum(axis=0) + 1.0   # [T]
                            group_psi = np.clip(group_numer / group_denom, epsilon, 1 - epsilon)

                            # Logit-scale difference: logit(PSI_group) - logit(PSI_ref)
                            group_logit = np.log(group_psi / (1 - group_psi))
                            init_values.append(group_logit - ref_logit)
                        else:
                            init_values.append(np.zeros(T_local, dtype=np.float32))

                    init_arr = np.stack(init_values) if len(init_values) else np.zeros((0, T_local), dtype=np.float32)
                    return torch.tensor(init_arr, dtype=torch.float32, device=self.model.device)

                else:
                    return torch.zeros((C - 1, T_local), dtype=torch.float32, device=self.model.device)
        
            # Default for everything else
            return init_to_median(site)
        
        pyro.clear_param_store()

        # ---------------------------
        # Guide choice with memory check
        # ---------------------------
        if distribution == 'multinomial':
            # Softmax guide for probs_baseline_raw: avoids the stick-breaking gradient
            # compression that caused AutoDelta to overshoot near the simplex boundary.
            # We parameterise log-proportions in ℝ^K and apply softmax — gradients are
            # uniform everywhere, no vanishing near p→1.
            _zmask_guide  = self._multinomial_zero_cat_mask   # [T, K]  captured in closure
            _log_p_init_g = self._multinomial_log_p_init      # [T, K]
            _T_guide      = T_fit
            _plate_name   = "feature_plate_technical"

            # AutoNormal for alpha_logits_y (all sites except probs_baseline_raw)
            _guide_alpha = AutoNormal(
                poutine.block(self._model_ntc, hide=["probs_baseline_raw"]),
                init_loc_fn=init_loc_fn,
            )

            def guide_cellline(*args, **kwargs):
                # ---- probs_baseline_raw: softmax-parameterised point estimate ----
                log_p = pyro.param("probs_baseline_logits", _log_p_init_g)  # [T, K]
                log_p_masked = log_p.masked_fill(_zmask_guide, -1e9)
                probs = torch.softmax(log_p_masked, dim=-1)                  # [T, K]
                with pyro.plate(_plate_name, _T_guide, dim=-1):
                    pyro.sample("probs_baseline_raw", dist.Delta(probs, event_dim=1))
                # ---- everything else: AutoNormal ----
                _guide_alpha(*args, **kwargs)

        elif distribution in ['binomial', 'normal', 'studentt']:
            # Calmer guide for these distributions
            guide_cellline = AutoNormalMessenger(self._model_ntc, init_loc_fn=init_loc_fn)

        else:
            # For negbinom and other distributions: check if IAF is feasible
            # Estimate number of latent variables
            if distribution == 'negbinom':
                # log2_alpha_y: (C-1) × T_fit
                # o_y: T_fit
                # mu_ntc: T_fit
                n_latent = (C - 1 + 2) * T_fit
            else:
                # Conservative estimate for unknown distributions
                n_latent = C * T_fit

            # IAF memory estimate (rough approximation):
            # - Transformation matrices: n_latent × n_latent × 4 bytes (float32)
            # - Hidden states: ~2× the matrix size
            # - Safety margin: 1.5×
            iaf_memory_gb = (n_latent ** 2 * 4 * 3 * 1.5) / 1e9

            # Check available VRAM
            use_iaf = False
            if self.model.device.type == 'cuda':
                try:
                    # torch is already imported at module level
                    total_memory_gb = torch.cuda.get_device_properties(self.model.device).total_memory / 1e9
                    # Reserve 10 GB for data, gradients, and other operations
                    available_for_guide_gb = total_memory_gb - 10.0

                    if force_iaf:
                        use_iaf = True
                        print(f"[INFO] force_iaf=True: using AutoIAFNormal (estimated {iaf_memory_gb:.1f} GB, "
                              f"{available_for_guide_gb:.1f} GB available — OOM risk if estimate is accurate)")
                    elif iaf_memory_gb < available_for_guide_gb:
                        use_iaf = True
                        print(f"[INFO] Using AutoIAFNormal guide (estimated {iaf_memory_gb:.1f} GB < {available_for_guide_gb:.1f} GB available)")
                    else:
                        print(f"[WARNING] AutoIAFNormal would require ~{iaf_memory_gb:.1f} GB VRAM (>{available_for_guide_gb:.1f} GB available)")
                        print(f"[WARNING] Falling back to AutoNormal (mean-field approximation) for {n_latent} latent variables")
                        # Increase niters for AutoNormal if using default (AutoNormal needs more iterations)
                        if niters_was_default and niters < 100_000:
                            old_niters = niters
                            niters = 100_000
                            print(f"[INFO] Increasing niters from {old_niters:,} to {niters:,} for AutoNormal convergence")
                except Exception as e:
                    print(f"[WARNING] Could not check VRAM ({e}), using AutoNormal for safety")
                    # Increase niters for AutoNormal if using default
                    if niters_was_default and niters < 100_000:
                        old_niters = niters
                        niters = 100_000
                        print(f"[INFO] Increasing niters from {old_niters:,} to {niters:,} for AutoNormal convergence")
            else:
                # CPU: always use AutoNormal for large models
                if n_latent > 5000:
                    print(f"[INFO] Using AutoNormal for CPU fitting with {n_latent} latent variables")
                    # Increase niters for AutoNormal if using default
                    if niters_was_default and niters < 100_000:
                        old_niters = niters
                        niters = 100_000
                        print(f"[INFO] Increasing niters from {old_niters:,} to {niters:,} for AutoNormal convergence")
                else:
                    use_iaf = True

            if use_iaf:
                guide_cellline = AutoIAFNormal(self._model_ntc, init_loc_fn=init_loc_fn)
                # Initialize the guide by calling it once with the model
                with torch.no_grad():
                    guide_cellline(
                        N, T_fit, C,
                        groups_ntc_tensor,
                        y_obs_ntc_tensor,
                        sum_factor_ntc_tensor,
                        beta_o_alpha_tensor,
                        beta_o_beta_tensor,
                        mu_x_mean_tensor,
                        mu_x_sd_tensor,
                        epsilon_tensor,
                        distribution,
                        denominator_ntc_tensor,
                        K, D,
                        sigma_hat_tensor,
                    )
                guide_cellline.to(self.model.device)
            else:
                # Fallback to memory-efficient AutoNormal
                guide_cellline = AutoNormalMessenger(self._model_ntc, init_loc_fn=init_loc_fn)

        # Choose ELBO based on guide type
        if isinstance(guide_cellline, (infer.autoguide.AutoNormal, infer.autoguide.AutoNormalMessenger)):
            # lower-variance estimator for mean-field Normal
            elbo = pyro.infer.TraceMeanField_ELBO(num_particles=1)
        else:
            elbo = pyro.infer.Trace_ELBO(num_particles=1)   # you can bump num_particles later if needed

        # Use OneCycleLR scheduler for stable convergence
        from torch.optim.lr_scheduler import OneCycleLR
        base_lr = 1e-3 if lr is None else lr

        # OneCycleLR requires pct_start * total_steps > 1 to avoid ZeroDivisionError.
        # With pct_start=0.1, minimum total_steps is 11 (0.1*11-1=0.1>0).
        _pct_start = 0.1
        _min_total_steps = int(1.0 / _pct_start) + 1  # = 11
        _scheduler_total_steps = max(niters, _min_total_steps)

        optimizer = pyro.optim.PyroLRScheduler(
            scheduler_constructor=OneCycleLR,
            optim_args={
                # underlying torch optimizer
                "optimizer": torch.optim.Adam,
                "optim_args": {
                    "lr": base_lr,      # initial lr (OneCycle will move it)
                    "betas": (0.9, 0.999),
                },
                # OneCycleLR hyperparameters
                "max_lr":          base_lr * 10,  # peak at 10x base_lr
                "total_steps":     _scheduler_total_steps,
                "pct_start":       _pct_start,    # 10% warmup
                "div_factor":      25.0,          # initial_lr = max_lr/25
                "final_div_factor": 1e4,          # final_lr = max_lr/1e4
            },
            clip_args={"clip_norm": 10.0},  # gradient clipping
        )

        svi = pyro.infer.SVI(self._model_ntc, guide_cellline, optimizer, loss=elbo)
        guide_cellline.to(self.model.device)
    
        # ---------------------------
        # Optimize
        # ---------------------------
        losses = []
        smoothed_loss = None
        for step in range(niters):
            loss = svi.step(
                N, T_fit, C,
                groups_ntc_tensor,
                y_obs_ntc_tensor,
                sum_factor_ntc_tensor,
                beta_o_alpha_tensor,
                beta_o_beta_tensor,
                mu_x_mean_tensor,
                mu_x_sd_tensor,
                epsilon_tensor,
                distribution,
                denominator_ntc_tensor,
                K, D,
                sigma_hat_tensor,   # ← ADD HERE
            )
            losses.append(loss)
            if step % 1000 == 0:
                print(f"Step {step} : loss = {loss:.5e}, device: {mu_x_mean_tensor.device}")
            # Detect NaN/Inf loss early and give an actionable message
            import math
            if not math.isfinite(loss):
                raise RuntimeError(
                    f"fit_technical: loss became {loss} at step {step}. "
                    "This usually indicates numerical instability. Try: "
                    "(1) lower learning rate (e.g. lr=1e-4), "
                    "(2) check for extreme values in sum_factor, "
                    "(3) reduce OneCycleLR peak (pass lr=3e-4 to use a smaller cycle)."
                )
            if smoothed_loss is None:
                smoothed_loss = loss
            else:
                if abs(alpha_ewma * (loss - smoothed_loss)) < tolerance:
                    print(f"Converged at step {step}! Loss = {loss:.5e}")
                    break
                smoothed_loss = alpha_ewma * loss + (1 - alpha_ewma) * smoothed_loss
    
        # ---------------------------
        # Predictive (optionally on CPU)
        # ---------------------------
        _original_device = self.model.device  # Save exact device (e.g. cuda:6) before any switch
        run_on_cpu = self.model.device.type != "cpu"
        if run_on_cpu:
            guide_cellline.to("cpu")
            self.model.device = torch.device("cpu")
    
            model_inputs = {
                "N": N, "T": T_fit, "C": C,
                "groups_ntc_tensor": self._to_cpu(groups_ntc_tensor),
                "y_obs_ntc_tensor": self._to_cpu(y_obs_ntc_tensor),
                "sum_factor_ntc_tensor": self._to_cpu(sum_factor_ntc_tensor),
                "beta_o_alpha_tensor": self._to_cpu(beta_o_alpha_tensor),
                "beta_o_beta_tensor": self._to_cpu(beta_o_beta_tensor),
                "mu_x_mean_tensor": self._to_cpu(mu_x_mean_tensor),
                "mu_x_sd_tensor": self._to_cpu(mu_x_sd_tensor),
                "epsilon_tensor": self._to_cpu(epsilon_tensor),
                "distribution": distribution,
                "denominator_ntc_tensor": self._to_cpu(denominator_ntc_tensor),
                "K": K, "D": D,
                "sigma_hat_tensor": self._to_cpu(sigma_hat_tensor),
            }
        else:
            model_inputs = {
                "N": N, "T": T_fit, "C": C,
                "groups_ntc_tensor": groups_ntc_tensor,
                "y_obs_ntc_tensor": y_obs_ntc_tensor,
                "sum_factor_ntc_tensor": sum_factor_ntc_tensor,
                "beta_o_alpha_tensor": beta_o_alpha_tensor,
                "beta_o_beta_tensor": beta_o_beta_tensor,
                "mu_x_mean_tensor": mu_x_mean_tensor,
                "mu_x_sd_tensor": mu_x_sd_tensor,
                "epsilon_tensor": epsilon_tensor,
                "distribution": distribution,
                "denominator_ntc_tensor": denominator_ntc_tensor,
                "K": K, "D": D,
                "sigma_hat_tensor": sigma_hat_tensor,
            }
    
        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        import gc; gc.collect()

        # ========================================
        # AUTO-DETECT MEMORY AND SET MINIBATCH
        # ========================================
        minibatch_size, use_parallel = self._estimate_predictive_memory_and_set_minibatch(
            N=N, T=T_fit, C=C, nsamples=nsamples,
            minibatch_size=minibatch_size,
            distribution=distribution
        )

        # CRITICAL MEMORY OPTIMIZATION: Exclude observation predictions from posterior
        # These are enormous ([S, N, T]) and redundant since we already have the data
        # Default filter excludes: y_obs_ntc, y_obs (for any distribution)
        def default_keep_sites(name, site):
            # Exclude all observation-level predictions (huge memory waste)
            if name in ('y_obs_ntc', 'y_obs', 'obs'):
                return False
            # Keep all parameter posteriors (small: alpha, mu, phi, sigma, etc.)
            return True

        keep_sites = kwargs.get("keep_sites", default_keep_sites)

        # CRITICAL: Skip observation sampling during Predictive
        # During Predictive, posterior samples of alpha_y can be extreme due to heavy-tailed priors
        # This causes NaN/Inf in likelihood evaluation
        # We only need parameter posteriors (alpha, mu, phi), not observations
        model_inputs_with_skip = model_inputs.copy()
        model_inputs_with_skip['skip_obs_sampling'] = True

        if minibatch_size is not None:
            predictive_technical = pyro.infer.Predictive(
                self._model_ntc, guide=guide_cellline, num_samples=minibatch_size,
                parallel=use_parallel
            )

            # Run first batch to get shapes, then preallocate full tensors
            posterior_samples = {}
            batch_idx = 0
            with torch.no_grad():
                for i in range(0, nsamples, minibatch_size):
                    current_batch_size = min(minibatch_size, nsamples - i)

                    # Generate batch
                    if current_batch_size < minibatch_size:
                        # Last batch might be smaller
                        predictive_batch = pyro.infer.Predictive(
                            self._model_ntc, guide=guide_cellline, num_samples=current_batch_size,
                            parallel=use_parallel
                        )
                        samples = predictive_batch(**model_inputs_with_skip)
                    else:
                        samples = predictive_technical(**model_inputs_with_skip)

                    # First batch: preallocate full tensors
                    if batch_idx == 0:
                        for k, v in samples.items():
                            if keep_sites(k, {"value": v}):
                                v_cpu = self._to_cpu(v)
                                # Preallocate: [nsamples, ...] with same dtype/shape as v
                                full_shape = (nsamples,) + v_cpu.shape[1:]
                                posterior_samples[k] = torch.empty(full_shape, dtype=v_cpu.dtype)
                                # Fill first batch
                                posterior_samples[k][:current_batch_size] = v_cpu
                    else:
                        # Subsequent batches: fill in-place
                        start_idx = i
                        end_idx = i + current_batch_size
                        for k, v in samples.items():
                            if k in posterior_samples:
                                posterior_samples[k][start_idx:end_idx] = self._to_cpu(v)

                    batch_idx += 1

                    # Clean up after each batch
                    del samples
                    if self.model.device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
        else:
            predictive_technical = pyro.infer.Predictive(
                self._model_ntc, guide=guide_cellline, num_samples=nsamples,
                parallel=use_parallel
            )
            with torch.no_grad():
                posterior_samples = predictive_technical(**model_inputs_with_skip)
                if self.model.device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
    
        if run_on_cpu:
            self.model.device = _original_device  # Restore exact original device (not generic "cuda")
    
        for k, v in posterior_samples.items():
            posterior_samples[k] = v.cpu()
    
        # ----------------------------------------
        # Reconstruct full α (all original T)
        # multiplicative baseline=1; additive baseline=0
        # ----------------------------------------
        # Back-compat: some guides name the mult version "alpha_y_mul"
        if "alpha_y" not in posterior_samples and "alpha_y_mul" in posterior_samples:
            posterior_samples["alpha_y"] = posterior_samples["alpha_y_mul"]

        # C == 1: no group effects were sampled — synthesise trivial posterior entries so
        # the reconstruction code below sees consistent shapes.
        if C == 1:
            _ref_tensor = posterior_samples.get("o_y", posterior_samples.get("mu_ntc"))
            _S = _ref_tensor.shape[0] if _ref_tensor is not None else 1
            if distribution != "multinomial":
                posterior_samples.setdefault("log2_alpha_y", torch.zeros(_S, 0, T_fit))
                posterior_samples.setdefault("alpha_y_mul",  torch.ones( _S, 0, T_fit))
                posterior_samples.setdefault("delta_y_add",  torch.zeros(_S, 0, T_fit))
                posterior_samples.setdefault("alpha_y",      torch.ones( _S, 0, T_fit))
            else:
                _K_local = K if K is not None else 1
                posterior_samples.setdefault("alpha_logits_y",          torch.zeros(_S, 0, T_fit, _K_local))
                posterior_samples.setdefault("alpha_logits_y_centered", torch.zeros(_S, 0, T_fit, _K_local))
                posterior_samples.setdefault("alpha_y",                 torch.zeros(_S, 0, T_fit, _K_local))

        def _reconstruct_full_2d(alpha_fit, baseline_value, fit_mask_bool):
            fit_idx = np.where(fit_mask_bool)[0]
            if alpha_fit.dim() == 3:           # [S, C-1, T_fit]
                S, Cminus1, _ = alpha_fit.shape
                full = torch.full((S, Cminus1 + 1, len(fit_mask_bool)),
                                  baseline_value, dtype=alpha_fit.dtype, device=alpha_fit.device)
                full[:, 1:, fit_idx] = alpha_fit
            elif alpha_fit.dim() == 2:         # [C-1, T_fit]
                Cminus1, _ = alpha_fit.shape
                full = torch.full((Cminus1 + 1, len(fit_mask_bool)),
                                  baseline_value, dtype=alpha_fit.dtype, device=alpha_fit.device)
                full[1:, fit_idx] = alpha_fit
            else:
                raise ValueError(f"Unexpected alpha shape: {alpha_fit.shape}")
            return full

        def _reconstruct_full_3d(alpha_fit, baseline_value, fit_mask_bool):
            """
            Normalize alpha_fit to [S, C-1, T_fit, K] (S may be 1),
            then expand to full [S, C, T_all, K] inserting the baseline row
            and placing T_fit back into original T_all via fit_mask_bool.
            """
            # last two dims must be (T_fit, K)
            Tfit = alpha_fit.size(-2)
            K_   = alpha_fit.size(-1)
            if Tfit != T_fit or K_ != K:
                raise ValueError(f"alpha_fit trailing dims mismatch: got (T={Tfit},K={K_}) expected (T={T_fit},K={K})")
        
            # ----- Step 1: ensure a leading sample dim S -----
            t = alpha_fit
            if t.dim() == 3:
                # [C-1, T_fit, K] -> [1, C-1, T_fit, K]
                t = t.unsqueeze(0)
            elif t.dim() == 4:
                # [S, C-1, T_fit, K] -> OK
                pass
            elif t.dim() >= 5:
                # e.g. [S, 1, 1, T_fit, K] or even [S, a, b, T_fit, K]
                # Collapse all dims between S and the trailing (T_fit,K) into a single (C-1) axis.
                S = t.size(0)
                # product over dims 1..(dim-3)
                pre_shape = t.shape[1:-2]
                Cminus1 = 1
                for d in pre_shape:
                    Cminus1 *= int(d)
                t = t.reshape(S, Cminus1, Tfit, K_)
            else:
                raise ValueError(f"Unexpected 3D alpha shape: {tuple(alpha_fit.shape)}")
        
            # t is now [S, C-1, T_fit, K]
            S, Cminus1, Tfit, K_ = t.shape
        
            # ----- Step 2: build full [S, C, T_all, K] and scatter T_fit back -----
            fit_idx = torch.as_tensor(np.where(fit_mask_bool)[0], device=t.device)
            full = torch.full(
                (S, Cminus1 + 1, len(fit_mask_bool), K_),
                baseline_value,
                dtype=t.dtype,
                device=t.device,
            )
            full[:, 1:, fit_idx, :] = t  # insert non-baseline rows into fit positions
            return full

    
        if distribution == "multinomial":
            # Use the logits that actually entered the likelihood
            alpha_logits_fit = (
                posterior_samples.get("alpha_logits_y_centered",
                                      posterior_samples["alpha_logits_y"])
            )  # [S?, C-1, T_fit, K]
            alpha_add_full = _reconstruct_full_3d(
                alpha_logits_fit, baseline_value=0.0, fit_mask_bool=fit_mask
            )
        
            # Store consistently
            posterior_samples["alpha_y_add"]  = alpha_add_full          # [S?, C, T_all, K]
            posterior_samples["alpha_y_mult"] = None
            posterior_samples["alpha_y"]      = alpha_add_full
        else:
            # existing path for 2D α
            alpha_y_mult_fit = posterior_samples["alpha_y"]
            alpha_y_mult_full = _reconstruct_full_2d(alpha_y_mult_fit, baseline_value=1.0, fit_mask_bool=fit_mask)
            log2_alpha_fit = posterior_samples["log2_alpha_y"]
            alpha_y_add_full = _reconstruct_full_2d(log2_alpha_fit, baseline_value=0.0, fit_mask_bool=fit_mask)

            # Patch non-reference zero-count (group, feature) pairs to baseline (negbinom only).
            # These features were included in fitting but the zero-count group's alpha_y estimate
            # is unreliable (prior-dominated). Overwrite with 1.0 (no technical correction).
            # alpha_y_mult_full shape: [C, T_all]; group index matches technical_group_code.
            if _negbinom_nonref_patch:
                n_pairs = sum(int(m.sum()) for m in _negbinom_nonref_patch.values())
                print(f"[INFO] Patching alpha_y=1.0 (mult) / 0.0 (add) for {n_pairs} "
                      f"(group, feature) pair(s) with all-zero NTC counts in a non-reference group.")
                for g, feat_mask_patch in _negbinom_nonref_patch.items():
                    # Use torch LongTensor for feature indices — numpy arrays
                    # cause PyTorch to mis-route indices to the wrong dimension.
                    # Pyro Predictive always produces a leading S dimension, so
                    # _reconstruct_full_2d always returns [S, C, T_all] (3D).
                    # The 2D branch [C, T_all] is kept for robustness only.
                    # `row` is the non-reference group code (int ≥ 1, < C).
                    # `feat_idx_t` indexes T_all, matching the last dimension.
                    row = int(g)
                    feat_idx_t = torch.as_tensor(
                        np.where(feat_mask_patch)[0], dtype=torch.long
                    )
                    if alpha_y_mult_full.dim() == 2:    # [C, T_all]  — fallback
                        alpha_y_mult_full[row, feat_idx_t] = 1.0
                        alpha_y_add_full[row, feat_idx_t] = 0.0
                    else:                               # [S, C, T_all] — normal
                        alpha_y_mult_full[:, row, feat_idx_t] = 1.0
                        alpha_y_add_full[:, row, feat_idx_t] = 0.0

            # Choose correct type based on distribution
            # negbinom: multiplicative, others: additive
            if distribution == 'negbinom':
                posterior_samples["alpha_y"] = alpha_y_mult_full
            else:  # binomial, normal, studentt use additive
                posterior_samples["alpha_y"] = alpha_y_add_full

            posterior_samples["alpha_y_mult"]  = alpha_y_mult_full
            posterior_samples["alpha_y_add"]   = alpha_y_add_full

        # ----------------------------------------
        # Generic reconstruction of feature-wise posteriors: T_fit -> T_all
        # ----------------------------------------

        T_all = len(fit_mask)                  # original feature count
        T_fit_detected = int(fit_mask.sum())   # number actually fit
        # build once, on CPU (safe because all posterior tensors were .cpu()'d)
        fit_idx_cpu = torch.as_tensor(np.where(fit_mask)[0], device=torch.device("cpu"))
        
        def _scatter_T(full_template, t, axis=-1):
            index = [slice(None)] * full_template.dim()
            index[axis] = fit_idx_cpu.to(full_template.device)  # <-- ensure same device
            full_template[tuple(index)] = t
            return full_template
        
        def _reconstruct_feature_axis(x, *, fill_value_nan=True):
            """
            Detect the feature axis (either -1 or -2 when a trailing K/D exists),
            create a full tensor with T_all on that axis, fill with NaN or
            zeros (for 'no-effect' defaults), and scatter the T_fit slice back.
            Returns reconstructed tensor or the original if no T_fit axis detected.
            """
            if not isinstance(x, torch.Tensor):
                return x

            # figure out where T_fit lives
            axis = None
            if x.dim() >= 1 and x.shape[-1] == T_fit_detected:
                axis = -1
            elif x.dim() >= 2 and x.shape[-2] == T_fit_detected:
                axis = -2

            if axis is None:
                return x  # not a feature-wise tensor, skip

            # shape with T_all inserted
            full_shape = list(x.shape)
            full_shape[axis] = T_all

            if fill_value_nan:
                fill_val = torch.tensor(float('nan'), dtype=x.dtype, device=x.device)
            else:
                fill_val = torch.tensor(0.0, dtype=x.dtype, device=x.device)

            full = torch.full(full_shape, fill_val, dtype=x.dtype, device=x.device)
            return _scatter_T(full, x, axis=axis)

        # Decide which keys are "effects" (fill with 0.0) vs "baselines/scales" (fill with NaN)
        # You can extend/tweak these sets as your model grows.
        effect_like_keys = {
            # common across dists
            "log2_alpha_y", "delta_y_add", "alpha_y_mul",  # raw effect params
            # multinomial logits
            "alpha_logits_y", "alpha_logits_y_centered",
        }

        baseline_like_keys = {
            # NB / Normal / Binomial baselines & variances
            "mu_ntc", "o_y", "sigma_y",
            # MVN
            "mu_ntc_mv", "sigma_y_mv",
            # Multinomial baseline probs
            "probs_baseline",
        }

        # α’s already reconstructed above; skip final forms to avoid double work
        skip_keys = {
            "alpha_y", "alpha_y_mult", "alpha_y_add"
        }

        # Reconstruct any posterior that carries T_fit
        for k, v in list(posterior_samples.items()):
            if k in skip_keys:
                continue
            if not isinstance(v, torch.Tensor):
                continue

            # Heuristic: only attempt if a T_fit-looking axis exists
            has_Tfit_axis = (
                (v.dim() >= 1 and v.shape[-1] == T_fit_detected) or
                (v.dim() >= 2 and v.shape[-2] == T_fit_detected)
            )
            if not has_Tfit_axis:
                continue

            # Choose fill: 0.0 for effect-like, NaN for baseline-like (default)
            fill_nan = k not in effect_like_keys
            try:
                posterior_samples[k] = _reconstruct_feature_axis(v, fill_value_nan=fill_nan)
            except Exception as e:
                print(f"[WARN] Skipped reconstruction for '{k}' due to: {e}")
        
        # ----------------------------------------
        # Feature metadata flags
        # ----------------------------------------
        # Masks cover rows 0..n_trans_features-1 (trans genes) and optionally
        # row n_trans_features (cis gene, if appended).  Store only the trans
        # portion in the modality's feature_meta.
        if len(modality.feature_meta) == n_trans_features:
            modality.feature_meta["ntc_zero_count"]               = zero_count_mask[:n_trans_features].tolist()
            modality.feature_meta["ntc_zero_std"]                 = zero_std_mask[:n_trans_features].tolist()
            modality.feature_meta["ntc_single_category"]          = only_one_category_mask[:n_trans_features].tolist()
            modality.feature_meta["ntc_excluded_from_fit"]        = needs_exclusion_mask[:n_trans_features].tolist()
            modality.feature_meta["technical_correction_applied"] = (~needs_exclusion_mask[:n_trans_features]).tolist()

    
        # ----------------------------------------
        # Persist results at modality level
        # ----------------------------------------
        if cis_idx_in_counts is not None:
            # Cis gene was appended as the last row.  Extract its alpha for alpha_x_prefit,
            # then trim it from all primary-modality posteriors so they align with the modality.

            if distribution == 'multinomial':
                raise ValueError(
                    "Primary modality uses a multinomial distribution, "
                    "but cis effect extraction is undefined for multinomial. "
                    "Ensure the primary modality is not multinomial when cis is present."
                )

            full_alpha_y_mult = posterior_samples["alpha_y_mult"]
            full_alpha_y_add  = posterior_samples["alpha_y_add"]

            if full_alpha_y_mult is None:
                raise RuntimeError(
                    "alpha_y_mult is None while attempting cis extraction. "
                    "This can happen if the distribution is multinomial; "
                    "check distribution handling earlier."
                )

            cis_idx_orig = cis_idx_in_counts          # always == n_trans_features
            trans_idx    = list(range(cis_idx_orig))  # 0 .. n_trans_features-1
            T_all_with_cis = n_trans_features + 1

            # alpha_x_prefit: mean over posterior samples → [C]
            self.model.alpha_x_prefit = full_alpha_y_mult[..., cis_idx_orig].mean(dim=0)

            # Modality alpha_y: trans genes only → mean → [C, T]
            if modality.distribution == 'negbinom':
                modality.alpha_y_prefit_mult = full_alpha_y_mult[..., trans_idx].mean(dim=0)
            else:
                modality.alpha_y_prefit_add = full_alpha_y_add[..., trans_idx].mean(dim=0)

            # ---- Store cis gene posteriors in cis modality ----
            cis_modality = self.model.get_modality('cis')
            cis_posterior = {}
            for src_key, dst_key in [
                ('log2_alpha_y', 'log2_alpha_x'), ('alpha_y_mul', 'alpha_x_mul'),
                ('delta_y_add', 'delta_x_add'),   ('o_y',         'o_x'),
                ('mu_ntc',      'mu_ntc'),
            ]:
                if src_key in posterior_samples:
                    cis_posterior[dst_key] = posterior_samples[src_key][..., cis_idx_orig:cis_idx_orig+1]

            if 'alpha_x_mul' in cis_posterior:
                raw = cis_posterior['alpha_x_mul']  # [S, C-1, 1]
                if raw.dim() == 3:
                    S, Cminus1, _ = raw.shape
                    cis_posterior['alpha_x_mult'] = torch.cat(
                        [torch.ones(S, 1, 1, device=raw.device), raw], dim=1)
                    cis_posterior['alpha_x'] = cis_posterior['alpha_x_mult']

            if 'delta_x_add' in cis_posterior:
                raw = cis_posterior['delta_x_add']  # [S, C-1, 1]
                if raw.dim() == 3:
                    S, Cminus1, _ = raw.shape
                    cis_posterior['alpha_x_add'] = torch.cat(
                        [torch.zeros(S, 1, 1, device=raw.device), raw], dim=1)

            cis_modality.posterior_samples_ntc = cis_posterior
            print(f"[INFO] Stored cis gene posteriors in 'cis' modality: {list(cis_posterior.keys())}")

            # ---- Trim cis gene from ALL primary-modality posteriors ----
            if modality.distribution == 'negbinom':
                posterior_samples["alpha_y_mult"] = full_alpha_y_mult[..., trans_idx]
                posterior_samples["alpha_y"]      = posterior_samples["alpha_y_mult"]
            else:
                posterior_samples["alpha_y_add"] = full_alpha_y_add[..., trans_idx]
                posterior_samples["alpha_y"]     = posterior_samples["alpha_y_add"]

            for key in ['log2_alpha_y', 'alpha_y_mul', 'delta_y_add', 'o_y', 'mu_ntc']:
                if key in posterior_samples and isinstance(posterior_samples[key], torch.Tensor):
                    v = posterior_samples[key]
                    if v.shape[-1] == T_all_with_cis:
                        posterior_samples[key] = v[..., trans_idx]

            print(f"[INFO] Extracted alpha_x for '{self.model.cis_gene}' (index {cis_idx_orig}) "
                  f"and trimmed it from primary modality posteriors")

        else:
            # No cis gene appended (non-primary modality, or no cis_gene set)
            if modality.distribution == 'negbinom':
                modality.alpha_y_prefit_mult = posterior_samples["alpha_y_mult"].mean(dim=0)
            else:
                modality.alpha_y_prefit_add = posterior_samples["alpha_y_add"].mean(dim=0)

        modality.posterior_samples_ntc = posterior_samples

        if modality.is_exon_skipping():
            modality.mark_technical_fit_complete()

        # Store losses on the modality (always)
        modality.loss_ntc = losses

        print(f"[INFO] Stored technical fit results in modality '{modality_name}'")
    
        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        pyro.clear_param_store()
        import gc; gc.collect()
    
        print("Finished fit_ntc.")

