"""
Save methods for bayesDREAM fitted parameters.
"""

import os
import torch

from ..utils import is_lean_posterior

class ModelSaver:
    """Handles saving fitted parameters."""

    def __init__(self, model):
        """
        Initialize model saver.

        Parameters
        ----------
        model : bayesDREAM
            The parent model instance
        """
        self.model = model

    ########################################################
    # Save/Load fitted parameters
    ########################################################

    def save_ntc_fit(self, output_dir: str = None, modalities: list = None, verbose: bool = False,
                          save_model_level: bool = None):
        """
        Save fitted NTC parameters from fit_ntc().

        Parameters
        ----------
        output_dir : str, optional
            Directory to save to. If None, uses self.model.output_dir.
        modalities : list of str, optional
            List of modality names to save. If None, saves all modalities.
            Example: ['gene', 'atac']
        verbose : bool
            If True, print detailed save information. Default False (summary only).

        Returns
        -------
        dict
            Paths to saved files

        Notes
        -----
        Saves per-modality:
        - alpha_y_prefit_{modality}.pt: Appropriate alpha_y for distribution (add or mult)
        - posterior_samples_ntc_{modality}.pt: Full posterior samples

        Saves model-level (automatically when primary modality is included):
        - alpha_x_prefit.pt: Cis gene overdispersion (if set)
        - alpha_y_prefit.pt: Trans gene overdispersion (from primary modality)
        """
        if output_dir is None:
            output_dir = os.path.join(self.model.output_dir, self.model.label)

        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}
        saved_summary = []

        # Determine which modalities to save
        if modalities is None:
            modalities_to_save = list(self.model.modalities.keys())
        else:
            # Validate requested modalities
            invalid = set(modalities) - set(self.model.modalities.keys())
            if invalid:
                raise ValueError(f"Unknown modalities: {invalid}. Available: {list(self.model.modalities.keys())}")
            modalities_to_save = modalities

        # Refuse to re-save lean-loaded posteriors as if they were a full fit:
        # load_ntc_fit(lean=True) irreversibly collapses the per-draw samples
        # to point estimates (see io.load._reduce_posterior_samples), so the
        # resulting file would silently claim to be a full posterior archive
        # while actually containing single-sample tensors.
        for mod_name in modalities_to_save:
            ps = getattr(self.model.modalities[mod_name], 'posterior_samples_ntc', None)
            if is_lean_posterior(ps):
                raise ValueError(
                    f"Cannot save_ntc_fit(): modality '{mod_name}' was loaded with "
                    f"load_ntc_fit(lean=True), so its posterior_samples_ntc only "
                    f"contains collapsed point estimates, not the full posterior. "
                    f"Reload with lean=False if you need to re-save a full fit."
                )

        # Determine whether to save model-level parameters
        if save_model_level is None:
            should_save_model_level = self.model.primary_modality in modalities_to_save
        else:
            should_save_model_level = save_model_level

        # Save model-level parameters (when primary modality is being saved)
        if should_save_model_level:
            if hasattr(self.model, 'alpha_x_prefit') and self.model.alpha_x_prefit is not None:
                path = os.path.join(output_dir, 'alpha_x_prefit.pt')
                torch.save(self.model.alpha_x_prefit, path)
                saved_files['alpha_x_prefit'] = path
                saved_summary.append('alpha_x')
                if verbose:
                    print(f"[SAVE] alpha_x_prefit → {path}")

        # Save per-modality alpha_y_prefit and posterior_samples_ntc
        for mod_name in modalities_to_save:
            mod = self.model.modalities[mod_name]
            mod_saved = []

            # Save distribution-appropriate alpha_y
            if hasattr(mod, 'alpha_y_prefit') and mod.alpha_y_prefit is not None:
                # Determine which alpha_y to save based on distribution
                if mod.distribution == 'negbinom':
                    # Save multiplicative for negbinom
                    if hasattr(mod, 'alpha_y_prefit_mult') and mod.alpha_y_prefit_mult is not None:
                        alpha_to_save = mod.alpha_y_prefit_mult
                        alpha_type = 'mult'
                    else:
                        alpha_to_save = mod.alpha_y_prefit
                        alpha_type = 'mult'
                else:
                    # Save additive for normal, binomial, multinomial
                    if hasattr(mod, 'alpha_y_prefit_add') and mod.alpha_y_prefit_add is not None:
                        alpha_to_save = mod.alpha_y_prefit_add
                        alpha_type = 'add'
                    else:
                        alpha_to_save = mod.alpha_y_prefit
                        alpha_type = 'add'

                path = os.path.join(output_dir, f'alpha_y_prefit_{mod_name}.pt')
                torch.save(alpha_to_save, path)
                saved_files[f'alpha_y_prefit_{mod_name}'] = path
                mod_saved.append('alpha_y')
                if verbose:
                    print(f"[SAVE] {mod_name}.alpha_y_prefit ({alpha_type}) → {path}")

            # Save modality-specific posterior_samples_ntc
            if hasattr(mod, 'posterior_samples_ntc') and mod.posterior_samples_ntc is not None:
                # Remove large observation arrays before saving
                # Ensure alpha_y_add and alpha_y_mult are present for downstream loading
                posterior_clean = {k: v for k, v in mod.posterior_samples_ntc.items()
                                 if k not in ['y_obs_ntc', 'y_obs']}

                # Verify critical keys are present for downstream loading
                if mod.distribution != 'negbinom' and 'alpha_y_add' not in posterior_clean:
                    if hasattr(mod, 'alpha_y_prefit_add') and mod.alpha_y_prefit_add is not None:
                        posterior_clean['alpha_y_add'] = mod.alpha_y_prefit_add

                if mod.distribution == 'negbinom' and 'alpha_y_mult' not in posterior_clean and 'alpha_y' not in posterior_clean:
                    if hasattr(mod, 'alpha_y_prefit_mult') and mod.alpha_y_prefit_mult is not None:
                        posterior_clean['alpha_y_mult'] = mod.alpha_y_prefit_mult

                # Add feature metadata (including full DataFrame for excluded features tracking)
                # Use the modality's authoritative feature count (same reason as trans save)
                n_features = mod.dims.get('n_features', None)
                if n_features is None:
                    _sample_tensor = next((v for v in posterior_clean.values()
                                           if isinstance(v, torch.Tensor) and v.ndim >= 2), None)
                    n_features = _sample_tensor.shape[-1] if _sample_tensor is not None else None
                posterior_with_meta = {
                    'posterior_samples': posterior_clean,
                    'modality_name': mod_name,
                    'distribution': mod.distribution,
                    'feature_names': mod.feature_names if hasattr(mod, 'feature_names') else None,
                    'n_features': n_features,
                    'feature_meta': mod.feature_meta.to_dict('records') if hasattr(mod, 'feature_meta') and mod.feature_meta is not None else None,
                    'loss_ntc': mod.loss_ntc if hasattr(mod, 'loss_ntc') else None
                }

                path = os.path.join(output_dir, f'posterior_samples_ntc_{mod_name}.pt')
                torch.save(posterior_with_meta, path)
                saved_files[f'posterior_samples_ntc_{mod_name}'] = path
                mod_saved.append(f'posterior({n_features} features)')
                if verbose:
                    print(f"[SAVE] {mod_name}.posterior_samples_ntc ({n_features} features) → {path}")

            if mod_saved:
                saved_summary.append(f"{mod_name}: {', '.join(mod_saved)}")

        # Print summary
        print(f"[SAVE] NTC fit to {output_dir}")
        if saved_summary:
            print(f"[SAVE] Saved: {'; '.join(saved_summary)}")

        return saved_files


    def save_cis_fit(self, output_dir: str = None, verbose: bool = False):
        """
        Save fitted cis parameters from fit_cis().

        Saves:
        - x_true: True cis gene expression (posterior samples or point estimate)
        - log2_x_true: Log2-transformed x_true (posterior samples or point estimate)
        - posterior_samples_cis: Full posterior samples

        Parameters
        ----------
        output_dir : str, optional
            Directory to save to. If None, uses self.model.output_dir.
        verbose : bool
            If True, print detailed save information. Default False (summary only).

        Returns
        -------
        dict
            Paths to saved files
        """
        if output_dir is None:
            output_dir = os.path.join(self.model.output_dir, self.model.label)

        os.makedirs(output_dir, exist_ok=True)

        # Refuse to re-save lean-loaded posteriors as if they were a full fit
        # (see the matching guard in save_ntc_fit for why).
        ps_cis = getattr(self.model, 'posterior_samples_cis', None)
        if is_lean_posterior(ps_cis):
            raise ValueError(
                "Cannot save_cis_fit(): posterior_samples_cis was loaded with "
                "load_cis_fit(lean=True), so it only contains collapsed point "
                "estimates, not the full posterior. Reload with lean=False if "
                "you need to re-save a full fit."
            )

        saved_files = {}
        saved_summary = []

        # Save x_true
        if hasattr(self.model, 'x_true') and self.model.x_true is not None:
            path = os.path.join(output_dir, 'x_true.pt')
            torch.save(self.model.x_true, path)
            saved_files['x_true'] = path
            saved_summary.append('x_true')
            if verbose:
                print(f"[SAVE] x_true → {path}")

        # Save log2_x_true
        if hasattr(self.model, 'log2_x_true') and self.model.log2_x_true is not None:
            path = os.path.join(output_dir, 'log2_x_true.pt')
            torch.save(self.model.log2_x_true, path)
            saved_files['log2_x_true'] = path
            saved_summary.append('log2_x_true')
            if verbose:
                print(f"[SAVE] log2_x_true → {path}")

        # Save posterior samples
        if hasattr(self.model, 'posterior_samples_cis') and self.model.posterior_samples_cis is not None:
            # Remove large observation arrays
            posterior_clean = {k: v for k, v in self.model.posterior_samples_cis.items()
                             if k not in ['x_obs', 'y_obs']}

            # Add cis gene metadata (including feature_meta from cis modality)
            cis_mod = self.model.get_modality('cis')
            posterior_with_meta = {
                'posterior_samples': posterior_clean,
                'cis_gene': self.model.cis_gene,
                'modality_name': 'cis',  # Cis always uses 'cis' modality
                'feature_meta': cis_mod.feature_meta.to_dict('records') if hasattr(cis_mod, 'feature_meta') and cis_mod.feature_meta is not None else None,
                'cell_names': self.model.meta['cell'].tolist() if 'cell' in self.model.meta.columns else None,
                'loss_x': self.model.loss_x if hasattr(self.model, 'loss_x') else None
            }

            path = os.path.join(output_dir, 'posterior_samples_cis.pt')
            torch.save(posterior_with_meta, path)
            saved_files['posterior_samples_cis'] = path
            saved_summary.append(f'posterior_cis ({self.model.cis_gene})')
            if verbose:
                print(f"[SAVE] posterior_samples_cis (cis_gene: {self.model.cis_gene}) → {path}")

        # Print summary
        print(f"[SAVE] Cis fit to {output_dir}")
        if saved_summary:
            print(f"[SAVE] Saved: {', '.join(saved_summary)}")

        return saved_files


    def save_trans_fit(self, output_dir: str = None, modalities: list = None, verbose: bool = False):
        """
        Save fitted trans parameters from fit_trans().

        Parameters
        ----------
        output_dir : str, optional
            Directory to save to. If None, uses self.model.output_dir.
        modalities : list of str, optional
            List of modality names to save. If None, saves all modalities.
            Example: ['gene', 'atac']
        verbose : bool
            If True, print detailed save information. Default False (summary only).

        Returns
        -------
        dict
            Paths to saved files

        Notes
        -----
        Saves per-modality:
        - posterior_samples_trans_{modality}.pt: Full posterior samples for each modality

        Saves model-level (automatically when primary modality is included):
        - posterior_samples_trans.pt: Model-level posterior samples
        """
        if output_dir is None:
            output_dir = os.path.join(self.model.output_dir, self.model.label)

        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}
        saved_summary = []

        # Determine which modalities to save
        if modalities is None:
            modalities_to_save = list(self.model.modalities.keys())
        else:
            # Validate requested modalities
            invalid = set(modalities) - set(self.model.modalities.keys())
            if invalid:
                raise ValueError(f"Unknown modalities: {invalid}. Available: {list(self.model.modalities.keys())}")
            modalities_to_save = modalities

        # Automatically save model-level parameters if primary modality is included
        should_save_model_level = self.model.primary_modality in modalities_to_save

        # Save model-level posterior samples (when primary modality is being saved)
        if should_save_model_level:
            if hasattr(self.model, 'posterior_samples_trans') and self.model.posterior_samples_trans is not None:
                # Remove large observation arrays and add metadata
                posterior_clean = {k: v for k, v in self.model.posterior_samples_trans.items()
                                 if k not in ['y_obs', 'x_obs']}

                # Get modality for feature info
                primary_mod = self.model.get_modality(self.model.primary_modality)

                # Add modality and feature metadata (including full feature_meta DataFrame)
                # Use the modality's authoritative feature count (not sample tensor shape[-1],
                # which can pick a non-feature tensor such as alpha_y [S, C-1, T] whose
                # last dim could coincidentally be 1 when C-1=1, saving wrong n_features).
                n_features_primary = primary_mod.dims.get('n_features', None)
                if n_features_primary is None:
                    _sample_tensor = next((v for v in posterior_clean.values()
                                           if isinstance(v, torch.Tensor) and v.ndim >= 2), None)
                    n_features_primary = _sample_tensor.shape[-1] if _sample_tensor is not None else None
                posterior_with_meta = {
                    'posterior_samples': posterior_clean,
                    'modality_name': self.model.primary_modality,
                    'distribution': primary_mod.distribution,
                    'feature_names': primary_mod.feature_names if hasattr(primary_mod, 'feature_names') else None,
                    'n_features': n_features_primary,
                    'feature_meta': primary_mod.feature_meta.to_dict('records') if hasattr(primary_mod, 'feature_meta') and primary_mod.feature_meta is not None else None,
                    'cis_gene': self.model.cis_gene,
                    'losses_trans': self.model.losses_trans if hasattr(self.model, 'losses_trans') else None,
                    'trans_prior_params': self.model.trans_prior_params if hasattr(self.model, 'trans_prior_params') else None,
                }

                # Include modality name in filename to prevent overwrites
                path = os.path.join(output_dir, f'posterior_samples_trans_{self.model.primary_modality}.pt')
                torch.save(posterior_with_meta, path)
                saved_files['posterior_samples_trans'] = path
                if verbose:
                    print(f"[SAVE] posterior_samples_trans (modality: {self.model.primary_modality}, {n_features_primary} features) → {path}")

        # Save per-modality posterior samples
        for mod_name in modalities_to_save:
            mod = self.model.modalities[mod_name]
            if hasattr(mod, 'posterior_samples_trans') and mod.posterior_samples_trans is not None:
                posterior_clean = {k: v for k, v in mod.posterior_samples_trans.items()
                                 if k not in ['y_obs', 'x_obs']}

                # Use the modality's authoritative feature count (same reason as primary modality)
                n_features = mod.dims.get('n_features', None)
                if n_features is None:
                    _sample_tensor = next((v for v in posterior_clean.values()
                                           if isinstance(v, torch.Tensor) and v.ndim >= 2), None)
                    n_features = _sample_tensor.shape[-1] if _sample_tensor is not None else None

                # Add modality and feature metadata (including full feature_meta DataFrame)
                posterior_with_meta = {
                    'posterior_samples': posterior_clean,
                    'modality_name': mod_name,
                    'distribution': mod.distribution,
                    'feature_names': mod.feature_names if hasattr(mod, 'feature_names') else None,
                    'n_features': n_features,
                    'feature_meta': mod.feature_meta.to_dict('records') if hasattr(mod, 'feature_meta') and mod.feature_meta is not None else None,
                    'cis_gene': self.model.cis_gene,
                    'losses_trans': mod.losses_trans if hasattr(mod, 'losses_trans') else None,
                    'trans_prior_params': mod.trans_prior_params if hasattr(mod, 'trans_prior_params') else None,
                }

                path = os.path.join(output_dir, f'posterior_samples_trans_{mod_name}.pt')
                torch.save(posterior_with_meta, path)
                saved_files[f'posterior_samples_trans_{mod_name}'] = path
                saved_summary.append(f"{mod_name}: posterior({n_features} features)")
                if verbose:
                    print(f"[SAVE] {mod_name}.posterior_samples_trans (distribution: {mod.distribution}, {n_features} features) → {path}")

        # Print summary
        print(f"[SAVE] Trans fit to {output_dir}")
        if saved_summary:
            print(f"[SAVE] Saved: {'; '.join(saved_summary)}")

        return saved_files
