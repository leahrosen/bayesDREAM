"""
Load methods for bayesDREAM fitted parameters.
"""

import os
import pickle
import torch
import pandas as pd


def _torch_load(path, map_location=None):
    """
    Load a torch checkpoint, falling back to weights_only=False for files
    that contain numpy arrays or other non-tensor globals.

    PyTorch 2.6 changed the default of weights_only from False to True.
    Old checkpoints saved with numpy arrays (e.g. via torch.save on a dict
    containing numpy arrays) will fail with weights_only=True.  We retry
    with weights_only=False when that happens.
    """
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError as e:
        if "Weights only load failed" in str(e):
            return torch.load(path, map_location=map_location, weights_only=False)
        raise

class ModelLoader:
    """Handles loading fitted parameters."""

    def __init__(self, model):
        """
        Initialize model loader.

        Parameters
        ----------
        model : bayesDREAM
            The parent model instance
        """
        self.model = model

    def load_technical_fit(self, input_dir: str = None,
                          modalities: list = None, verbose: bool = False,
                          load_model_level: bool = None):
        """
        Load fitted technical parameters.

        Parameters
        ----------
        input_dir : str, optional
            Directory to load from. If None, uses self.model.output_dir.
        modalities : list of str, optional
            List of modality names to load. If None, attempts to load all existing modalities.
            Example: ['gene', 'atac']
        verbose : bool
            If True, print detailed loading information. Default False (summary only).

        Returns
        -------
        dict
            Loaded parameters (keys only, not full tensors)
        """
        if input_dir is None:
            input_dir = os.path.join(self.model.output_dir, self.model.label)

        loaded = {}
        loaded_summary = []  # Track what was loaded for summary

        # Determine which modalities to load
        if modalities is None:
            modalities_to_load = list(self.model.modalities.keys())
        else:
            # Validate requested modalities
            invalid = set(modalities) - set(self.model.modalities.keys())
            if invalid:
                raise ValueError(f"Unknown modalities: {invalid}. Available: {list(self.model.modalities.keys())}")
            modalities_to_load = modalities

        # Determine whether to load model-level parameters
        if load_model_level is None:
            should_load_model_level = self.model.primary_modality in modalities_to_load
        else:
            should_load_model_level = load_model_level

        # Load model-level parameters (when primary modality is being loaded)
        if should_load_model_level:
            # Load alpha_x_prefit
            alpha_x_path = os.path.join(input_dir, 'alpha_x_prefit.pt')
            if os.path.exists(alpha_x_path):
                alpha_x = _torch_load(alpha_x_path)
                # Backward compat: if saved as 3D posterior, collapse to mean
                if isinstance(alpha_x, torch.Tensor) and alpha_x.ndim >= 2 and alpha_x.shape[0] > 1:
                    alpha_x = alpha_x.mean(dim=0)
                self.model.alpha_x_prefit = alpha_x.flatten()
                loaded['alpha_x_prefit'] = True
                loaded_summary.append('alpha_x')
                if verbose:
                    print(f"[LOAD] alpha_x_prefit ← {alpha_x_path}")

            # Load alpha_y_prefit (legacy model-level file → primary modality)
            alpha_y_path = os.path.join(input_dir, 'alpha_y_prefit.pt')
            if os.path.exists(alpha_y_path):
                alpha_y = _torch_load(alpha_y_path)
                # Backward compat: if saved as 3D posterior, collapse to mean
                if isinstance(alpha_y, torch.Tensor) and alpha_y.ndim >= 3:
                    alpha_y = alpha_y.mean(dim=0)
                primary_mod = self.model.get_modality(self.model.primary_modality)
                primary_mod.alpha_y_prefit = alpha_y  # Uses property to set distribution-specific attr
                loaded['alpha_y_prefit'] = True
                if verbose:
                    print(f"[LOAD] alpha_y_prefit → {self.model.primary_modality} modality ← {alpha_y_path}")

        # Load per-modality alpha_y_prefit and posterior_samples_technical
        for mod_name in modalities_to_load:
            mod = self.model.modalities[mod_name]
            mod_loaded = []

            mod_path = os.path.join(input_dir, f'alpha_y_prefit_{mod_name}.pt')
            if os.path.exists(mod_path):
                alpha_y_to_set = _torch_load(mod_path)
                # Backward compat: collapse old multi-sample posteriors to point estimate.
                # Valid point-estimate shapes: [C, T, K] (3D) for multinomial, [C, T] (2D) for others.
                # Old saved format: [S, C, T, K] (4D) for multinomial, [S, C, T] (3D) for others.
                if isinstance(alpha_y_to_set, torch.Tensor):
                    collapse_threshold = 4 if mod.distribution == 'multinomial' else 3
                    if alpha_y_to_set.ndim >= collapse_threshold:
                        alpha_y_to_set = alpha_y_to_set.mean(dim=0)

                mod.alpha_y_prefit = alpha_y_to_set

                if mod.distribution == 'negbinom':
                    mod.alpha_y_prefit_mult = alpha_y_to_set
                else:
                    mod.alpha_y_prefit_add = alpha_y_to_set

                loaded[f'alpha_y_prefit_{mod_name}'] = True
                mod_loaded.append('alpha_y')
                if verbose:
                    print(f"[LOAD] {mod_name}.alpha_y_prefit ← {mod_path}")

            # Load modality-specific posterior_samples_technical
            posterior_path = os.path.join(input_dir, f'posterior_samples_technical_{mod_name}.pt')
            if os.path.exists(posterior_path):
                loaded_data = _torch_load(posterior_path)
                n_features = None

                # Check if new format (with metadata) or old format (just dict)
                if isinstance(loaded_data, dict) and 'posterior_samples' in loaded_data:
                    # New format with metadata
                    mod.posterior_samples_technical = loaded_data['posterior_samples']
                    n_features = loaded_data.get('n_features')

                    # Reconstruct feature_meta DataFrame if present
                    feature_meta_df = None
                    if loaded_data.get('feature_meta') is not None:
                        feature_meta_df = pd.DataFrame(loaded_data['feature_meta'])

                    loaded[f'posterior_samples_technical_{mod_name}_metadata'] = {
                        'modality_name': loaded_data.get('modality_name'),
                        'distribution': loaded_data.get('distribution'),
                        'n_features': n_features
                    }

                    # Load loss_technical if present
                    if loaded_data.get('loss_technical') is not None:
                        mod.loss_technical = loaded_data['loss_technical']
                        mod_loaded.append(f'loss({len(mod.loss_technical)})')

                    if verbose:
                        print(f"[LOAD] {mod_name}.posterior_samples_technical ({n_features} features) ← {posterior_path}")
                else:
                    # Old format (backward compatibility)
                    mod.posterior_samples_technical = loaded_data
                    if verbose:
                        print(f"[LOAD] {mod_name}.posterior_samples_technical (legacy format) ← {posterior_path}")

                loaded[f'posterior_samples_technical_{mod_name}'] = True
                mod_loaded.append(f'posterior({n_features or "?"} features)')

                # Also extract and set specific alpha attributes from posterior_samples
                # This ensures backward compatibility even if files were saved without the specific attributes
                if 'alpha_y_add' in mod.posterior_samples_technical:
                    if not hasattr(mod, 'alpha_y_prefit_add') or mod.alpha_y_prefit_add is None:
                        alpha_y_add = mod.posterior_samples_technical['alpha_y_add']
                        # Backward compat: collapse old multi-sample posteriors to point estimate.
                        # Valid shapes: [C, T, K] (3D) for multinomial, [C, T] (2D) for others.
                        if isinstance(alpha_y_add, torch.Tensor):
                            collapse_threshold = 4 if mod.distribution == 'multinomial' else 3
                            if alpha_y_add.ndim >= collapse_threshold:
                                alpha_y_add = alpha_y_add.mean(dim=0)
                        mod.alpha_y_prefit_add = alpha_y_add
                        if not hasattr(mod, 'alpha_y_prefit') or mod.alpha_y_prefit is None:
                            if mod.distribution != 'negbinom':
                                mod.alpha_y_prefit = alpha_y_add
                        if verbose:
                            print(f"[LOAD] {mod_name}.alpha_y_prefit_add ← extracted from posterior_samples_technical")

                if 'alpha_y_mult' in mod.posterior_samples_technical or 'alpha_y' in mod.posterior_samples_technical:
                    alpha_y_mult_key = 'alpha_y_mult' if 'alpha_y_mult' in mod.posterior_samples_technical else 'alpha_y'
                    if not hasattr(mod, 'alpha_y_prefit_mult') or mod.alpha_y_prefit_mult is None:
                        alpha_y_mult = mod.posterior_samples_technical[alpha_y_mult_key]
                        # Backward compat: collapse 3D posteriors to mean
                        if isinstance(alpha_y_mult, torch.Tensor) and alpha_y_mult.ndim >= 3:
                            alpha_y_mult = alpha_y_mult.mean(dim=0)
                        mod.alpha_y_prefit_mult = alpha_y_mult
                        if not hasattr(mod, 'alpha_y_prefit') or mod.alpha_y_prefit is None:
                            if mod.distribution == 'negbinom':
                                mod.alpha_y_prefit = alpha_y_mult
                        if verbose:
                            print(f"[LOAD] {mod_name}.alpha_y_prefit_mult ← extracted from posterior_samples_technical")

            if mod_loaded:
                loaded_summary.append(f"{mod_name}: {', '.join(mod_loaded)}")

        # Print summary
        print(f"[LOAD] Technical fit from {input_dir}")
        if loaded_summary:
            print(f"[LOAD] Loaded: {'; '.join(loaded_summary)}")

        # Warn if alpha_y_prefit was not loaded for modalities that need it
        # Note: 'cis' modality doesn't need alpha_y_prefit (it uses alpha_x instead)
        modalities_missing_alpha_y = []
        for mod_name in modalities_to_load:
            # Skip 'cis' modality - it's for cis gene fitting, not trans fitting
            if mod_name == 'cis':
                continue
            mod = self.model.modalities[mod_name]
            if mod.alpha_y_prefit is None:
                modalities_missing_alpha_y.append(mod_name)

        if modalities_missing_alpha_y:
            import warnings
            warnings.warn(
                f"[WARNING] alpha_y_prefit was NOT loaded for modalities: {modalities_missing_alpha_y}. "
                f"This will cause fit_trans() to fail for these modalities. "
                f"Check that the following files exist in {input_dir}:\n"
                f"  - alpha_y_prefit.pt (legacy format for primary modality)\n"
                f"  - alpha_y_prefit_<modality>.pt (per-modality format)\n"
                f"  - posterior_samples_technical_<modality>.pt (contains alpha_y in posterior samples)\n"
                f"If files are in a different directory, use load_technical_fit(input_dir='path/to/saved/fit')",
                UserWarning
            )

        return loaded

    def load_cis_fit(self, input_dir: str = None, verbose: bool = False):
        """
        Load fitted cis parameters.

        Parameters
        ----------
        input_dir : str, optional
            Directory to load from. If None, uses self.model.output_dir.
        verbose : bool
            If True, print detailed loading information. Default False (summary only).

        Returns
        -------
        dict
            Loaded parameters (keys only, not full tensors)
        """
        if input_dir is None:
            input_dir = os.path.join(self.model.output_dir, self.model.label)

        loaded = {}
        loaded_summary = []

        # Load x_true
        x_true_path = os.path.join(input_dir, 'x_true.pt')
        if os.path.exists(x_true_path):
            x_true = _torch_load(x_true_path)
            # Backward compat: if saved as 2D/3D posterior, collapse to mean
            if isinstance(x_true, torch.Tensor) and x_true.ndim >= 2:
                x_true = x_true.mean(dim=0)
            self.model.x_true = x_true
            loaded['x_true'] = True
            loaded_summary.append('x_true')
            if verbose:
                print(f"[LOAD] x_true ← {x_true_path}")

        # Load log2_x_true if saved separately
        log2_x_true_path = os.path.join(input_dir, 'log2_x_true.pt')
        if os.path.exists(log2_x_true_path):
            log2_x_true = _torch_load(log2_x_true_path)
            # Backward compat: if saved as 2D/3D posterior, collapse to mean
            if isinstance(log2_x_true, torch.Tensor) and log2_x_true.ndim >= 2:
                log2_x_true = log2_x_true.mean(dim=0)
            self.model.log2_x_true = log2_x_true
            loaded['log2_x_true'] = True
            loaded_summary.append('log2_x_true')
            if verbose:
                print(f"[LOAD] log2_x_true ← {log2_x_true_path}")

        # Load posterior samples
        posterior_path = os.path.join(input_dir, 'posterior_samples_cis.pt')
        if os.path.exists(posterior_path):
            loaded_data = _torch_load(posterior_path)
            cis_gene = None

            # Check if new format (with metadata) or old format (just dict)
            if isinstance(loaded_data, dict) and 'posterior_samples' in loaded_data:
                # New format with metadata
                self.model.posterior_samples_cis = loaded_data['posterior_samples']
                cis_gene = loaded_data.get('cis_gene')

                # Reconstruct feature_meta DataFrame if present
                feature_meta_df = None
                if loaded_data.get('feature_meta') is not None:
                    feature_meta_df = pd.DataFrame(loaded_data['feature_meta'])

                loaded['posterior_samples_cis_metadata'] = {
                    'cis_gene': cis_gene,
                    'modality_name': loaded_data.get('modality_name'),
                }

                # Load loss_x if present
                if loaded_data.get('loss_x') is not None:
                    self.model.loss_x = loaded_data['loss_x']
                    loaded_summary.append(f"loss_x({len(self.model.loss_x)})")

                if verbose:
                    print(f"[LOAD] posterior_samples_cis (cis_gene: {cis_gene}) ← {posterior_path}")
            else:
                # Old format (backward compatibility)
                self.model.posterior_samples_cis = loaded_data
                if verbose:
                    print(f"[LOAD] posterior_samples_cis (legacy format) ← {posterior_path}")

            loaded['posterior_samples_cis'] = True
            loaded_summary.append(f"posterior_cis" + (f" ({cis_gene})" if cis_gene else ""))

            # Extract log2_x_true from posterior_samples_cis if not already loaded
            if not hasattr(self.model, 'log2_x_true') or self.model.log2_x_true is None:
                if 'log_x_true' in self.model.posterior_samples_cis:
                    log_x_true = self.model.posterior_samples_cis['log_x_true']
                    # Backward compat: collapse 2D/3D posteriors to mean
                    if isinstance(log_x_true, torch.Tensor) and log_x_true.ndim >= 2:
                        log_x_true = log_x_true.mean(dim=0)
                    self.model.log2_x_true = log_x_true
                    loaded['log2_x_true'] = True
                    if verbose:
                        print(f"[LOAD] log2_x_true ← extracted from posterior_samples_cis")
                elif hasattr(self.model, 'x_true') and self.model.x_true is not None:
                    self.model.log2_x_true = torch.log2(self.model.x_true)
                    loaded['log2_x_true'] = True
                    if verbose:
                        print(f"[LOAD] log2_x_true ← computed from x_true")

        # Print summary
        print(f"[LOAD] Cis fit from {input_dir}")
        if loaded_summary:
            print(f"[LOAD] Loaded: {', '.join(loaded_summary)}")

        return loaded


    def load_trans_fit(self, input_dir: str = None, modalities: list = None, verbose: bool = False):
        """
        Load fitted trans parameters.

        Parameters
        ----------
        input_dir : str, optional
            Directory to load from. If None, uses self.model.output_dir.
        modalities : list of str, optional
            List of modality names to load. If None, attempts to load all existing modalities.
            Example: ['gene', 'atac']
        verbose : bool
            If True, print detailed loading information. Default False (summary only).

        Returns
        -------
        dict
            Loaded parameters (keys only, not full tensors)
        """
        if input_dir is None:
            input_dir = os.path.join(self.model.output_dir, self.model.label)

        loaded = {}
        loaded_summary = []

        # Determine which modalities to load
        if modalities is None:
            modalities_to_load = list(self.model.modalities.keys())
        else:
            # Validate requested modalities
            invalid = set(modalities) - set(self.model.modalities.keys())
            if invalid:
                raise ValueError(f"Unknown modalities: {invalid}. Available: {list(self.model.modalities.keys())}")
            modalities_to_load = modalities

        # Automatically load model-level parameters if primary modality is included
        should_load_model_level = self.model.primary_modality in modalities_to_load

        # Load model-level posterior samples (when primary modality is being loaded)
        if should_load_model_level:
            # Try new filename pattern first (includes modality name)
            posterior_path = os.path.join(input_dir, f'posterior_samples_trans_{self.model.primary_modality}.pt')
            if not os.path.exists(posterior_path):
                # Fall back to old filename pattern (backward compatibility)
                posterior_path = os.path.join(input_dir, 'posterior_samples_trans.pt')

            if os.path.exists(posterior_path):
                loaded_data = _torch_load(posterior_path)
                # Check if new format (with metadata) or old format (just dict)
                if isinstance(loaded_data, dict) and 'posterior_samples' in loaded_data:
                    # New format with metadata
                    self.model.posterior_samples_trans = loaded_data['posterior_samples']
                    loaded['posterior_samples_trans'] = True

                    loaded['posterior_samples_trans_metadata'] = {
                        'modality_name': loaded_data.get('modality_name'),
                        'distribution': loaded_data.get('distribution'),
                        'n_features': loaded_data.get('n_features'),
                        'cis_gene': loaded_data.get('cis_gene'),
                    }

                    # Load losses_trans if present
                    if loaded_data.get('losses_trans') is not None:
                        self.model.losses_trans = loaded_data['losses_trans']

                    # Load trans_prior_params if present
                    if loaded_data.get('trans_prior_params') is not None:
                        self.model.trans_prior_params = loaded_data['trans_prior_params']

                    if verbose:
                        print(f"[LOAD] posterior_samples_trans (modality: {loaded_data.get('modality_name')}, {loaded_data.get('n_features')} features) ← {posterior_path}")
                else:
                    # Old format (backward compatibility)
                    self.model.posterior_samples_trans = loaded_data
                    loaded['posterior_samples_trans'] = True
                    if verbose:
                        print(f"[LOAD] posterior_samples_trans (legacy format) ← {posterior_path}")

        # Load per-modality posterior samples
        for mod_name in modalities_to_load:
            mod_path = os.path.join(input_dir, f'posterior_samples_trans_{mod_name}.pt')
            if os.path.exists(mod_path):
                loaded_data = _torch_load(mod_path)
                mod_loaded = []
                n_features = None

                # Check if new format (with metadata) or old format (just dict)
                if isinstance(loaded_data, dict) and 'posterior_samples' in loaded_data:
                    # New format with metadata
                    self.model.modalities[mod_name].posterior_samples_trans = loaded_data['posterior_samples']
                    n_features = loaded_data.get('n_features')

                    loaded[f'posterior_samples_trans_{mod_name}_metadata'] = {
                        'modality_name': loaded_data.get('modality_name'),
                        'distribution': loaded_data.get('distribution'),
                        'n_features': n_features,
                        'cis_gene': loaded_data.get('cis_gene'),
                    }

                    # Load losses_trans if present
                    if loaded_data.get('losses_trans') is not None:
                        self.model.modalities[mod_name].losses_trans = loaded_data['losses_trans']
                        mod_loaded.append(f"loss({len(self.model.modalities[mod_name].losses_trans)})")

                    # Load trans_prior_params if present
                    if loaded_data.get('trans_prior_params') is not None:
                        self.model.modalities[mod_name].trans_prior_params = loaded_data['trans_prior_params']

                    if verbose:
                        print(f"[LOAD] {mod_name}.posterior_samples_trans (distribution: {loaded_data.get('distribution')}, {n_features} features) ← {mod_path}")
                else:
                    # Old format (backward compatibility)
                    self.model.modalities[mod_name].posterior_samples_trans = loaded_data
                    if verbose:
                        print(f"[LOAD] {mod_name}.posterior_samples_trans (legacy format) ← {mod_path}")

                loaded[f'posterior_samples_trans_{mod_name}'] = True
                mod_loaded.append(f"posterior({n_features or '?'} features)")
                loaded_summary.append(f"{mod_name}: {', '.join(mod_loaded)}")

        # Print summary
        print(f"[LOAD] Trans fit from {input_dir}")
        if loaded_summary:
            print(f"[LOAD] Loaded: {'; '.join(loaded_summary)}")

        return loaded
