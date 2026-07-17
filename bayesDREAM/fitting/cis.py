"""
Cis gene expression fitting for bayesDREAM.

This module contains the cis model and fitting logic.
"""

import gc
import os
import warnings
import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.distributions.transforms import iterated, affine_autoregressive
import pyro.optim as optim
import pyro.infer as infer



class CisFitter:
    """Handles cis gene expression fitting."""

    def __init__(self, model):
        """
        Initialize cis fitter.

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

    ########################################################
    # Step 2: Fit cis effects (model_x)
    ########################################################
    def _model_x(
        self,
        N,
        G,
        guides_tensor,
        x_obs_tensor,
        sum_factor_tensor,
        mu_x_mean_tensor,
        mu_x_sd_tensor,
        sigma_eff_mean_tensor,
        sigma_eff_sd_tensor,
        C=None,
        groups_tensor=None,
        alpha_x_sample=None,
        target_per_guide_tensor=None,
        independent_mu_sigma=False,
        phi_x_precomputed=None,
        rate_alpha_precomputed=None,
        rate_beta_precomputed=None,
        scale_2_tensor=None,
    ):

        ###################################
        ## Technical covariate modelling ##
        ###################################
        if alpha_x_sample is not None:
            alpha_x = alpha_x_sample
        else:
            alpha_x = None
        
        ####################
        ## Overdispersion ##
        ####################
        if phi_x_precomputed is None:
            raise ValueError(
                "phi_x_precomputed is required. "
                "Run fit_ntc() before fit_cis() to estimate cis gene overdispersion from NTC cells."
            )
        phi_x_used = phi_x_precomputed

        ###############################
        ## Mixture model for x_eff_g ##
        ###############################
        if independent_mu_sigma:
            unique_targets = torch.unique(target_per_guide_tensor)
            mu_targets = {}
            sigma_targets = {}
            _hc_scale = scale_2_tensor if scale_2_tensor is not None else self._t(2.0)
            for t in unique_targets:
                mu_targets[int(t.item())] = pyro.sample(f"mu_target_{int(t.item())}", dist.Normal(mu_x_mean_tensor, mu_x_sd_tensor))
                sigma_targets[int(t.item())] = pyro.sample(f"sigma_target_{int(t.item())}", dist.HalfCauchy(scale=_hc_scale))
            # gather mu and sigma for each guide
            mu_target_tensor = torch.stack([mu_targets[int(t.item())] for t in unique_targets], dim=0)
            sigma_target_tensor = torch.stack([sigma_targets[int(t.item())] for t in unique_targets], dim=0)
            mu = mu_target_tensor[target_per_guide_tensor]
            sigma = sigma_target_tensor[target_per_guide_tensor]
        else:
            mu = pyro.sample("mu", dist.Normal(mu_x_mean_tensor, mu_x_sd_tensor))
            _hc_scale = scale_2_tensor if scale_2_tensor is not None else self._t(2.0)
            sigma = pyro.sample("sigma", dist.HalfCauchy(scale=_hc_scale))
            mu = mu.expand(G)
            sigma = sigma.expand(G)

        # Non-centered parameterization
        if rate_alpha_precomputed is not None:
            sigma_eff_alpha = pyro.sample("sigma_eff_alpha", dist.Exponential(rate_alpha_precomputed))
            sigma_eff_beta = pyro.sample("sigma_eff_beta", dist.Exponential(rate_beta_precomputed))
        elif (sigma_eff_mean_tensor >= 0.01) and (sigma_eff_sd_tensor >= 0.01):
            rate_alpha = (sigma_eff_sd_tensor ** 2) / (sigma_eff_mean_tensor ** 2)
            rate_beta = (sigma_eff_sd_tensor ** 2) / sigma_eff_mean_tensor
            sigma_eff_alpha = pyro.sample("sigma_eff_alpha", dist.Exponential(rate_alpha))
            sigma_eff_beta = pyro.sample("sigma_eff_beta", dist.Exponential(rate_beta))
        else:
            sigma_eff_alpha = pyro.sample("sigma_eff_alpha", dist.Gamma(self._t(1.0), self._t(0.01)))
            sigma_eff_beta = pyro.sample("sigma_eff_beta", dist.Gamma(self._t(1.0), self._t(0.01)))
        with pyro.plate("guides_plate", G):
            eps_x_eff_g = pyro.sample("eps_x_eff_g", dist.StudentT(df=self._t(3.0), loc=self._t(0.0), scale=self._t(1.0)))
            log2_x_eff_g = mu + sigma * eps_x_eff_g
            x_eff_g = pyro.deterministic("x_eff_g", 2.0 ** log2_x_eff_g)
        
            sigma_eff = pyro.sample("sigma_eff", dist.Gamma(sigma_eff_alpha, sigma_eff_beta))
                
        ##########################
        ## Cell-level variables ##
        ##########################
        if alpha_x is not None:
            # alpha_x shape [C]: already includes reference group (index 0 = 1.0)
            # or shape [C-1]: needs reference group prepended
            if alpha_x.shape[-1] == C:
                alpha_x_full = alpha_x
            else:
                ones_ = torch.ones(1, device=self.model.device)
                alpha_x_full = torch.cat([ones_, alpha_x], dim=-1)  # shape = [C]
            alpha_x_used = alpha_x_full[groups_tensor]  # shape = [N]
        else:
            alpha_x_used = torch.ones_like(sum_factor_tensor)  # shape = [N] or [S,N]

        # Aggregate guide effects to cells
        if self.model.is_high_moi:
            # High MOI: effects are additive in log2FC space.
            # log2FC for guide g = log2(x_eff_g[g]) - log2(NTC_mean)
            # For a cell with guides A, B: log2FC_cell = log2FC_A + log2FC_B
            # => log2(x_mean) = log2(NTC) + log2FC_cell
            #                 = sum(log2(x_eff_g)) - (n_guides - 1) * log2(NTC)
            # This is always finite since x_eff_g > 0 and NTC_mean > 0.

            # NTC reference: weighted mean of NTC guide effects
            # Use pre-computed mask (set in fit_cis before SVI loop to avoid pandas access here)
            ntc_mask = self.model._ntc_guide_mask
            cells_per_guide = self.model.guide_assignment_tensor.sum(dim=0)  # [G]
            weights = cells_per_guide / sigma_eff.clamp(min=1e-6)            # [G]
            ntc_weights = weights[ntc_mask]
            ntc_effects  = x_eff_g[ntc_mask]
            weighted_mean_NTC = (ntc_weights * ntc_effects).sum(dim=-1) / ntc_weights.sum()
            weighted_mean_NTC = pyro.deterministic("weighted_mean_NTC", weighted_mean_NTC)

            log2_x_eff_g  = torch.log2(x_eff_g.clamp(min=1e-12))                               # [G]
            log2_NTC      = torch.log2(weighted_mean_NTC.clamp(min=1e-12))                       # scalar
            guides_per_cell = self.model.guide_assignment_tensor.sum(dim=1).clamp(min=1)         # [N]
            sum_log2_effects = torch.matmul(self.model.guide_assignment_tensor, log2_x_eff_g)    # [N]
            log2_x_mean   = sum_log2_effects - (guides_per_cell - 1) * log2_NTC                 # [N]
            x_mean        = 2.0 ** log2_x_mean                                                   # [N]
            sigma_mean    = torch.matmul(self.model.guide_assignment_tensor, sigma_eff) / guides_per_cell  # [N]
        else:
            # Single guide per cell: use indexing (unchanged)
            x_mean = x_eff_g[..., guides_tensor]  # [N]
            sigma_mean = sigma_eff[..., guides_tensor]  # [N]

        ######################
        ## Cell-level plate ##
        ######################
        with pyro.plate("data_plate", N):
            log_x_true = pyro.sample( # use log2 of xtrue to allow small values of xtrue
                "log_x_true",
                dist.Normal(torch.log2(x_mean), sigma_mean)
            )
            x_true = pyro.deterministic("x_true", 2.0 ** log_x_true)
            mu_obs = alpha_x_used * x_true * sum_factor_tensor
            pyro.sample(
                "x_obs",
                dist.NegativeBinomial(total_count=phi_x_used,
                                      logits=torch.log(mu_obs) - torch.log(phi_x_used),
                                      validate_args=False
                                     ), # important to use logits to allow small x_true
                obs=x_obs_tensor
            )

    def fit_cis(
        self,
        technical_covariates: list[str] = None,
        sum_factor_col: str = 'sum_factor',
        cis_feature: str = None,
        manual_guide_effects: pd.DataFrame = None,
        prior_strength: float = 1.0,
        lr: float = 1e-3,
        niters: int = 100_000,
        nsamples: int = 1000,
        alpha_ewma: float = 0.05,
        epsilon: float = 1e-6,
        minibatch_size: int = None,
        predictive_on_cpu: bool = True,
        independent_mu_sigma: bool = False,
        **kwargs
    ):
        """
        Fits the cis effects (model_x) for your gene_of_interest.
        This step can be repeated multiple times with different priors
        or hyperparameters.

        Parameters
        ----------
        technical_covariates : list, optional
            Technical covariates for correction
        sum_factor_col : str
            Column name for size factors
        cis_feature : str, optional
            Feature ID to use as cis proxy from the primary modality.
            If None, uses self.model.cis_gene (must exist in primary modality).
            For ATAC: region ID (e.g., 'chr9:132283881-132284881')
            For genes: gene name
        manual_guide_effects : pd.DataFrame, optional
            Manual guide effect estimates as priors. DataFrame with columns:
            - guide: guide identifier (matches meta['guide'])
            - log2FC: expected log2 fold-change vs NTC
        prior_strength : float
            Weight for manual guide effects (default: 1.0)
            0 = ignore manual effects, higher = trust more
        lr : float
            Learning rate for Adam
        niters : int
            Number of SVI iterations
        nsamples : int
            Number of posterior samples
        predictive_on_cpu : bool
            If True (default), run posterior Predictive sampling on CPU after SVI
            to reduce GPU memory pressure. Set False to keep Predictive on GPU.
        alpha_ewma : float
            Exponential weight for smoothing the ELBO
        independent_mu_sigma : bool
            Whether to use independent mu/sigma per target type
        kwargs :
            Additional arguments controlling priors, etc.
        """
        print("Running fit_cis...")

        if self.model.cis_gene is None:
            raise ValueError("self.model.cis_gene must be set.")

        # fit_cis ALWAYS uses the 'cis' modality
        # This modality is created automatically when bayesDREAM is initialized

        # Get cis modality
        if 'cis' not in self.model.modalities:
            raise ValueError(
                "No 'cis' modality found. The 'cis' modality should be created automatically "
                "when bayesDREAM is initialized with a cis_gene. "
                "Make sure cis_gene parameter is set during initialization."
            )

        cis_modality = self.model.get_modality('cis')

        # Determine which feature to use as cis proxy
        if cis_feature is None:
            # Default: use the first (and typically only) feature in cis modality
            cis_feature_idx = cis_modality.feature_meta.index[0]
            # Get actual feature name from metadata (not just numeric index)
            if 'gene_name' in cis_modality.feature_meta.columns:
                cis_feature_name = cis_modality.feature_meta.loc[cis_feature_idx, 'gene_name']
            elif 'gene' in cis_modality.feature_meta.columns:
                cis_feature_name = cis_modality.feature_meta.loc[cis_feature_idx, 'gene']
            elif 'feature' in cis_modality.feature_meta.columns:
                cis_feature_name = cis_modality.feature_meta.loc[cis_feature_idx, 'feature']
            else:
                # Fallback to index if no name column found
                cis_feature_name = cis_feature_idx
            cis_feature = cis_feature_idx  # Use numeric index for actual data retrieval
            print(f"[INFO] Using cis feature '{cis_feature_name}' from 'cis' modality")
        else:
            # User specified explicit cis_feature
            if cis_feature not in cis_modality.feature_meta.index:
                raise ValueError(
                    f"cis_feature '{cis_feature}' not found in 'cis' modality.\n"
                    f"Available features: {cis_modality.feature_meta.index.tolist()}"
                )
            print(f"[INFO] Using cis_feature '{cis_feature}' from 'cis' modality")

        # Get counts for cis feature from cis modality
        if isinstance(cis_modality.counts, pd.DataFrame):
            cis_counts = cis_modality.counts.loc[cis_feature].values
        else:
            # numpy/sparse array - need to find index
            feature_idx = cis_modality.feature_meta.index.get_loc(cis_feature)
            if cis_modality.cells_axis == 1:
                cis_counts = cis_modality.counts[feature_idx, :]
            else:
                cis_counts = cis_modality.counts[:, feature_idx]
            # Densify if sparse (slicing sparse returns sparse row/col matrix)
            if hasattr(cis_counts, 'toarray'):
                cis_counts = cis_counts.toarray().ravel()

        # convert to gpu for fitting
        if self.model.alpha_x_prefit is not None and self.model.alpha_x_prefit.device != self.model.device:
            self.model.alpha_x_prefit = self.model.alpha_x_prefit.to(self.model.device)

        if technical_covariates:
            new_technical_group_code = self.model.meta.groupby(technical_covariates).ngroup()
            if "technical_group_code" in self.model.meta.columns:
                old_technical_group_code = self.model.meta["technical_group_code"].values
                groups_changed = not np.array_equal(old_technical_group_code, new_technical_group_code.values)
                if groups_changed:
                    warnings.warn("technical_group already set. Overwriting with new covariate grouping.")
                    if self.model.alpha_x_prefit is not None:
                        warnings.warn("Technical groups changed; discarding alpha_x_prefit and refitting.")
                        self.model.alpha_x_prefit = None
                else:
                    # Keep loaded alpha_x_prefit when grouping is unchanged (common in stepwise workflows)
                    pass

            self.model.meta["technical_group_code"] = new_technical_group_code
            C = self.model.meta['technical_group_code'].nunique()
            groups_tensor = torch.tensor(self.model.meta['technical_group_code'].values, dtype=torch.long, device=self.model.device)

            # alpha_x_prefit is required when technical_covariates are specified
            if self.model.alpha_x_prefit is None:
                raise ValueError(
                    f"Technical covariates provided but alpha_x_prefit not set. "
                    f"Run fit_ntc() on the primary modality ('{self.model.primary_modality}') before fit_cis() "
                    f"to estimate technical effects for the cis gene."
                )

        elif self.model.alpha_x_prefit is None:
            C = None
            groups_tensor = None
            warnings.warn("no alpha_x_prefit and no technical_covariates provided, assuming no confounding effect.")
        else:
            # alpha_x_prefit exists but no new technical_covariates specified
            # Use existing technical groups
            C = self.model.meta['technical_group_code'].nunique()
            groups_tensor = torch.tensor(self.model.meta['technical_group_code'].values, dtype=torch.long, device=self.model.device)
        
        N = self.model.meta.shape[0]

        # Handle G (number of guides) differently for high MOI vs single-guide mode
        if self.model.is_high_moi:
            G = self.model.guide_assignment.shape[1]  # Number of guides
            guides_tensor = None  # Not used in high MOI mode
        else:
            G = self.model.meta['guide_code'].nunique()
            guides_tensor = torch.tensor(self.model.meta['guide_code'].values, dtype=torch.long, device=self.model.device)

        # Validate minimum data requirements for stable initialization
        min_guides = 3
        min_cells_per_guide = 5
        if self.model.is_high_moi:
            cells_per_guide = self.model.guide_assignment.sum(axis=0)  # [G]
            n_adequate = int((cells_per_guide >= min_cells_per_guide).sum())
        else:
            guide_counts = self.model.meta['guide_code'].value_counts()
            n_adequate = int((guide_counts >= min_cells_per_guide).sum())
        if n_adequate < min_guides:
            raise ValueError(
                f"fit_cis requires at least {min_guides} guides with >= {min_cells_per_guide} cells each, "
                f"but only {n_adequate} guides meet this criterion. "
                f"Subsetting to too few cells/guides will produce unreliable initializations."
            )

        # Use cis_counts from modality-specific lookup (or traditional self.model.counts)
        x_obs_tensor = torch.tensor(cis_counts, dtype=torch.float32, device=self.model.device)

        # ========================================================================
        # MANUAL GUIDE EFFECTS INFRASTRUCTURE
        # ========================================================================
        # If user provides manual guide effects (log2FC estimates), prepare them
        # as tensors that can be used as priors in the Pyro model.
        manual_guide_log2fc_tensor = None
        manual_guide_mask_tensor = None

        if manual_guide_effects is not None:
            raise NotImplementedError(
                "manual_guide_effects is not yet implemented. "
                "The infrastructure for guide-level priors is planned but not yet integrated into _model_x."
            )
        # ========================================================================
        if independent_mu_sigma:
            if ('target' not in self.model.meta.columns):
                raise ValueError("independent_mu_sigma is True, self.model.meta['target'] column not found.")
            elif self.model.meta['target'].nunique() < 2:
                raise ValueError("independent_mu_sigma is True, but only 1 target type found in self.model.meta['target'] column.")

            ### BUILD target_per_guide_tensor [G] based on guide → target
            if self.model.is_high_moi:
                # High MOI: derive one target label per guide for grouping mu/sigma.
                # Two sources: guide_meta['target'] (simple) or guide_targets_dict (many-to-many).
                _ntc_variants = {'ntc', 'NTC', 'non-targeting', 'non-targeting-control', 'Non-Targeting'}

                if 'target' in self.model.guide_meta.columns:
                    guide_target_labels = self.model.guide_meta['target'].tolist()
                elif hasattr(self.model, 'guide_targets_dict') and self.model.guide_targets_dict:
                    # Derive a single representative target per guide.
                    # Priority: cis_gene > any NTC variant > first target.
                    def _primary_target(targets):
                        if self.model.cis_gene and self.model.cis_gene in targets:
                            return self.model.cis_gene
                        for t in targets:
                            if t in _ntc_variants:
                                return 'ntc'
                        return targets[0] if targets else 'ntc'

                    guide_target_labels = [
                        _primary_target(self.model.guide_targets_dict.get(row['guide'], ['ntc']))
                        for _, row in self.model.guide_meta.iterrows()
                    ]
                else:
                    raise ValueError(
                        "independent_mu_sigma=True in high MOI mode requires either "
                        "guide_meta['target'] or guide_targets_dict."
                    )

                target_factorized, target_unique = pd.factorize(guide_target_labels)
                target_per_guide_tensor = torch.tensor(target_factorized, dtype=torch.long, device=self.model.device)
                print(f"[INFO] independent_mu_sigma (high MOI): {len(target_unique)} unique targets")
            else:
                # Single-guide mode: one target code per guide (raises if a guide has multiple targets)
                self.model.meta['target_code'] = pd.factorize(self.model.meta['target'])[0]
                target_codes_tensor = torch.tensor(self.model.meta['target_code'].values, dtype=torch.long, device=self.model.device)

                target_per_guide_tensor = torch.empty(G, dtype=torch.long, device=self.model.device)
                for g in range(G):
                    idx = (guides_tensor == g)
                    guide_targets = torch.unique(target_codes_tensor[idx])
                    if guide_targets.shape[0] != 1:
                        raise ValueError(f"Guide {g} maps to multiple targets: {guide_targets}. independent_mu_sigma=True requires unambiguous target assignment per guide.")
                    target_per_guide_tensor[g] = guide_targets[0]
        else:
            target_per_guide_tensor = None
        primary_mod = self.model.get_modality(self.model.primary_modality)
        sum_factor_tensor = torch.tensor(
            primary_mod.sum_factors.loc[self.model.meta['cell'].values, sum_factor_col].values,
            dtype=torch.float32, device=self.model.device
        )

        # Compute guide means (adjusting for alpha_x if provided)
        x_obs_factored = x_obs_tensor / sum_factor_tensor
        
        if self.model.alpha_x_prefit is not None:
            # alpha_x_prefit is always a [C] point estimate (mean already taken at fit_ntc time)
            alpha_x_full = self.model.alpha_x_prefit.flatten()
            # Select the correct alpha_x for each observation (expand for broadcasting)
            alpha_x_used = alpha_x_full[groups_tensor]  # groups_tensor indexes into (C,)

            # Adjust x_obs_factored by alpha_x_used
            x_obs_factored /= alpha_x_used  # Ensure correct shape for division
        
        # --- Extract o_x from the technical fit (cis modality posterior) ---
        cis_modality = self.model.get_modality('cis')
        if (cis_modality.posterior_samples_ntc is None
                or 'o_x' not in cis_modality.posterior_samples_ntc):
            raise ValueError(
                "Cis gene overdispersion not found. Run fit_ntc() before fit_cis().\n"
                "fit_ntc() estimates o_x for the cis gene from NTC cells, which is required "
                "to set a data-driven overdispersion prior instead of the generic Gamma(9,3)."
            )
        o_x_ntc = float(cis_modality.posterior_samples_ntc['o_x'].mean().item())
        print(f"[INFO] fit_cis: using technical o_x = {o_x_ntc:.4f} (phi = {1/o_x_ntc**2:.2f})")

        # Compute guide-level means and MADs.
        # Uses numpy on CPU for per-guide median computation (faster than G GPU boolean masks).
        x_np = x_obs_factored.detach().cpu().numpy().astype(np.float32)

        if self.model.is_high_moi:
            # High MOI: vectorized means via matrix-vector multiply, MADs via sorted numpy slices
            assign_np = self.model.guide_assignment  # [N, G] numpy array
            cell_counts_np = assign_np.sum(axis=0).clip(min=1)  # [G]
            guide_sums_np = x_np @ assign_np  # [G]
            guide_means_np = np.log2(np.clip(guide_sums_np / cell_counts_np, epsilon, None))

            log2_x_np = np.log2(np.clip(x_np, epsilon, None))
            guide_mads_np = np.zeros(G, dtype=np.float32)
            for g in range(G):
                cell_mask = assign_np[:, g].astype(bool)
                if cell_mask.sum() > 0:
                    vals = log2_x_np[cell_mask]
                    med = np.median(vals)
                    guide_mads_np[g] = float(np.median(np.abs(vals - med)))

            guide_means = torch.from_numpy(guide_means_np).to(self.model.device)
            guide_mads_tensor = torch.from_numpy(guide_mads_np).to(self.model.device)
        else:
            # Single-guide mode: vectorized means via bincount, MADs via sorted numpy slices
            guides_np = guides_tensor.cpu().numpy()
            n_per_guide_np = np.bincount(guides_np, minlength=G).clip(min=1).astype(np.float32)
            sum_per_guide_np = np.bincount(guides_np, weights=x_np, minlength=G).astype(np.float32)
            guide_means_np = np.log2(np.clip(sum_per_guide_np / n_per_guide_np, epsilon, None))

            log2_x_np = np.log2(np.clip(x_np, epsilon, None))
            sort_idx = np.argsort(guides_np, kind='stable')
            guides_sorted = guides_np[sort_idx]
            log2_x_sorted = log2_x_np[sort_idx]
            boundaries = np.searchsorted(guides_sorted, np.arange(G + 1))

            guide_mads_np = np.zeros(G, dtype=np.float32)
            for g in range(G):
                s, e = boundaries[g], boundaries[g + 1]
                if e > s:
                    vals = log2_x_sorted[s:e]
                    med = np.median(vals)
                    guide_mads_np[g] = float(np.median(np.abs(vals - med)))

            guide_means = torch.from_numpy(guide_means_np).to(self.model.device)
            guide_mads_tensor = torch.from_numpy(guide_mads_np).to(self.model.device)

        mu_x_mean_tensor = torch.mean(guide_means)
        mu_x_sd_tensor = torch.std(guide_means)

        guide_mads_tensor = guide_mads_tensor * 1.4826  # Gaussian-equivalent spread
        sigma_eff_mean_tensor = torch.mean(guide_mads_tensor)#.to(self.model.device)
        sigma_eff_sd_tensor = torch.std(guide_mads_tensor)#.to(self.model.device)

        def init_loc_fn(site):
            name = site["name"]
        
            if name == "mu":
                return mu_x_mean_tensor
            elif name == "sigma":
                return mu_x_sd_tensor.clamp(min=1e-3)
        
            elif name == "sigma_eff_alpha":
                # Corresponds to (mean^2 / var)
                sigma_eff_mean_tensor_tmp = sigma_eff_mean_tensor.clamp(min=1e-2)
                sigma_eff_sd_tensor_tmp = sigma_eff_sd_tensor.clamp(min=1e-2)
                return ((sigma_eff_mean_tensor_tmp ** 2) / (sigma_eff_sd_tensor_tmp ** 2))
            elif name == "sigma_eff_beta":
                # Corresponds to (mean / var)
                sigma_eff_mean_tensor_tmp = sigma_eff_mean_tensor.clamp(min=1e-2)
                sigma_eff_sd_tensor_tmp = sigma_eff_sd_tensor.clamp(min=1e-2)
                return (sigma_eff_mean_tensor_tmp / (sigma_eff_sd_tensor_tmp ** 2))
        
            elif name == "sigma_eff":
                # Assume you’re sampling per-guide (e.g. G = 37)
                return sigma_eff_mean_tensor.clamp(min=1e-2).expand(G)
        
            return pyro.infer.autoguide.initialization.init_to_median(site)

        # Pre-compute NTC guide mask for high MOI mode (avoids pandas access inside Pyro model)
        if self.model.is_high_moi:
            _ntc_variants = {'ntc', 'NTC', 'non-targeting', 'non-targeting-control', 'Non-Targeting'}
            if 'target' in self.model.guide_meta.columns:
                ntc_flags = [self.model.guide_meta.iloc[g]['target'] in _ntc_variants
                             for g in range(G)]
            elif hasattr(self.model, 'guide_targets_dict') and self.model.guide_targets_dict:
                ntc_flags = [
                    any(t in _ntc_variants for t in self.model.guide_targets_dict.get(row['guide'], []))
                    for _, row in self.model.guide_meta.iterrows()
                ]
            else:
                raise ValueError(
                    "High MOI mode requires either guide_meta['target'] or guide_targets_dict "
                    "to identify NTC guides."
                )
            self.model._ntc_guide_mask = torch.tensor(ntc_flags, dtype=torch.bool,
                                                       device=self.model.device)

        # Precompute model invariants once to avoid repeated computation in every SVI step
        phi_x_precomputed = torch.tensor(1.0 / (o_x_ntc ** 2), dtype=torch.float32, device=self.model.device)
        scale_2_tensor = self._t(2.0)
        _seff_mean = sigma_eff_mean_tensor.clamp(min=1e-2)
        _seff_sd = sigma_eff_sd_tensor.clamp(min=1e-2)
        if (_seff_mean >= 0.01) and (_seff_sd >= 0.01):
            rate_alpha_precomputed = (_seff_sd ** 2) / (_seff_mean ** 2)
            rate_beta_precomputed = (_seff_sd ** 2) / _seff_mean
        else:
            rate_alpha_precomputed = None
            rate_beta_precomputed = None

        guide_x = pyro.infer.autoguide.AutoNormalMessenger(self._model_x, init_loc_fn=init_loc_fn)
        guide_x.to(self.model.device)
        optimizer = pyro.optim.Adam({"lr": lr})
        svi = pyro.infer.SVI(self._model_x, guide_x, optimizer,
                             loss=pyro.infer.Trace_ELBO())

        original_device = self.model.device
        losses = []
        smoothed_loss = None
        alpha_x_sample_loop = self.model.alpha_x_prefit

        for step in range(niters):
            loss = svi.step(
                N,
                G,
                guides_tensor,
                x_obs_tensor,
                sum_factor_tensor,
                mu_x_mean_tensor,
                mu_x_sd_tensor,
                _seff_mean,
                _seff_sd,
                C=C,
                groups_tensor=groups_tensor,
                alpha_x_sample=alpha_x_sample_loop,
                target_per_guide_tensor=target_per_guide_tensor,
                independent_mu_sigma=independent_mu_sigma,
                phi_x_precomputed=phi_x_precomputed,
                rate_alpha_precomputed=rate_alpha_precomputed,
                rate_beta_precomputed=rate_beta_precomputed,
                scale_2_tensor=scale_2_tensor,
            )
            losses.append(loss)
            if step % 1000 == 0:
                print(f"Step {step} : loss = {loss:.5e}, device: {mu_x_mean_tensor.device}")
            if smoothed_loss is None:
                smoothed_loss = loss
            else:
                smoothed_loss = alpha_ewma * loss + (1 - alpha_ewma) * smoothed_loss

        # Move to CPU if using too much GPU memory for Predictive
        run_on_cpu = predictive_on_cpu and self.model.device.type != "cpu"

        if run_on_cpu:
            print(f"[INFO] SVI completed on {original_device}. Running Predictive on CPU to reduce GPU memory pressure...")
            guide_x.to("cpu")
            self.model.device = torch.device("cpu")
            # Move high-MOI tensors that are accessed directly in _model_x
            if self.model.is_high_moi:
                self.model.guide_assignment_tensor = self.model.guide_assignment_tensor.cpu()
                self.model._ntc_guide_mask = self.model._ntc_guide_mask.cpu()

            model_inputs = {
                "N": N,
                "G": G,
                "guides_tensor": self._to_cpu(guides_tensor),
                "x_obs_tensor": self._to_cpu(x_obs_tensor),
                "sum_factor_tensor": self._to_cpu(sum_factor_tensor),
                "mu_x_mean_tensor": self._to_cpu(mu_x_mean_tensor),
                "mu_x_sd_tensor": self._to_cpu(mu_x_sd_tensor),
                "sigma_eff_mean_tensor": self._to_cpu(_seff_mean),
                "sigma_eff_sd_tensor": self._to_cpu(_seff_sd),
                "C": C,
                "groups_tensor": self._to_cpu(groups_tensor),
                "alpha_x_sample": self._to_cpu(self.model.alpha_x_prefit),
                "target_per_guide_tensor": self._to_cpu(target_per_guide_tensor),
                "independent_mu_sigma": independent_mu_sigma,
                "phi_x_precomputed": self._to_cpu(phi_x_precomputed),
                "rate_alpha_precomputed": self._to_cpu(rate_alpha_precomputed),
                "rate_beta_precomputed": self._to_cpu(rate_beta_precomputed),
                "scale_2_tensor": self._to_cpu(scale_2_tensor),
            }
        else:
            model_inputs = {
                "N": N,
                "G": G,
                "guides_tensor": guides_tensor,
                "x_obs_tensor": x_obs_tensor,
                "sum_factor_tensor": sum_factor_tensor,
                "mu_x_mean_tensor": mu_x_mean_tensor,
                "mu_x_sd_tensor": mu_x_sd_tensor,
                "sigma_eff_mean_tensor": _seff_mean,
                "sigma_eff_sd_tensor": _seff_sd,
                "C": C,
                "groups_tensor": groups_tensor,
                "alpha_x_sample": self.model.alpha_x_prefit,
                "target_per_guide_tensor": target_per_guide_tensor,
                "independent_mu_sigma": independent_mu_sigma,
                "phi_x_precomputed": phi_x_precomputed,
                "rate_alpha_precomputed": rate_alpha_precomputed,
                "rate_beta_precomputed": rate_beta_precomputed,
                "scale_2_tensor": scale_2_tensor,
            }

        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        max_samples = nsamples
        keep_sites = kwargs.get("keep_sites", lambda name, site: site["value"].ndim <= 2 or name != "x_obs")

        if minibatch_size is not None:
            from collections import defaultdict
            print(f"[INFO] Running Predictive in minibatches of {minibatch_size}...")
            predictive_x = pyro.infer.Predictive(
                self._model_x,
                guide=guide_x,
                num_samples=minibatch_size,
                parallel=True
            )
            all_samples = defaultdict(list)
            with torch.no_grad():
                for i in range(0, max_samples, minibatch_size):
                    samples = predictive_x(**model_inputs)
                    for k, v in samples.items():
                        if keep_sites(k, {"value": v}):
                            all_samples[k].append(self._to_cpu(v))
                    if self.model.device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
            posterior_samples_x = {k: torch.cat(v, dim=0) for k, v in all_samples.items()}
        else:
            predictive_x = pyro.infer.Predictive(
                self._model_x,
                guide=guide_x,
                num_samples=nsamples
            )
            with torch.no_grad():
                posterior_samples_x = predictive_x(**model_inputs)
                if self.model.device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        if run_on_cpu:
            self.model.device = original_device
            print("[INFO] Reset self.model.device to:", self.model.device)
            if self.model.is_high_moi:
                self.model.guide_assignment_tensor = self.model.guide_assignment_tensor.to(original_device)
                self.model._ntc_guide_mask = self.model._ntc_guide_mask.to(original_device)

        for k, v in posterior_samples_x.items():
            posterior_samples_x[k] = self._to_cpu(v)

        self.model.loss_x = losses
        # Store full posterior on model (not just on CisFitter)
        self.model.posterior_samples_cis = posterior_samples_x
        self.model.x_true = posterior_samples_x['x_true'].median(dim=0).values
        self.model.log2_x_true = posterior_samples_x['log_x_true'].median(dim=0).values

        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        pyro.clear_param_store()

        print("Finished fit_cis.")
