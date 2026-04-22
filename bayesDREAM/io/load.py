"""
Load methods for bayesDREAM fitted parameters.
"""

import os
import pickle
import torch
import pandas as pd


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

def _align_tensor(tensor, saved_names, target_names, dim):
    """
    Align one dimension of `tensor` from `saved_names` ordering to `target_names`.

    Features present in target_names but absent from saved_names are filled with
    NaN (float tensors) or 0 (integer tensors).  Features present in saved_names
    but absent from target_names are dropped.

    Parameters
    ----------
    tensor : torch.Tensor
    saved_names : list of str
    target_names : list of str
    dim : int  (may be negative)

    Returns
    -------
    aligned : torch.Tensor  shape has target_names length along `dim`
    mask    : torch.BoolTensor  shape (len(target_names),)
              True for positions that were present in the saved fit
    """
    if dim < 0:
        dim = tensor.ndim + dim

    saved_idx = {n: i for i, n in enumerate(saved_names)}
    n_target = len(target_names)
    mapping = [saved_idx.get(n, -1) for n in target_names]
    mask = torch.tensor([i >= 0 for i in mapping], dtype=torch.bool)

    # Move feature axis to last, do the reindex, move back
    t = tensor.transpose(dim, tensor.ndim - 1)        # [..., T_saved]
    out_shape = t.shape[:-1] + (n_target,)
    if t.is_floating_point():
        out = torch.full(out_shape, float('nan'), dtype=t.dtype)
    else:
        out = torch.zeros(out_shape, dtype=t.dtype)
    for ci, si in enumerate(mapping):
        if si >= 0:
            out[..., ci] = t[..., si]
    aligned = out.transpose(dim, tensor.ndim - 1)     # restore original axis order

    return aligned, mask


def _detect_feature_dim(tensor, n_features_saved):
    """
    Return the dimension index that corresponds to features, or None.

    Checks last dim first, then second-to-last (for multinomial [C, T, K] tensors).
    Returns None when no dimension matches n_features_saved (scalars, per-group
    tensors, etc.).
    """
    if tensor.ndim >= 1 and tensor.shape[-1] == n_features_saved:
        return tensor.ndim - 1
    if tensor.ndim >= 2 and tensor.shape[-2] == n_features_saved:
        return tensor.ndim - 2
    return None


def _align_posterior_features(posterior_dict, saved_names, target_names, n_features_saved):
    """
    Align every tensor in `posterior_dict` whose feature dimension equals
    `n_features_saved`, reindexing from `saved_names` to `target_names`.

    Returns
    -------
    aligned_dict : dict
    mask         : torch.BoolTensor | None   (None when no tensor needed aligning)
    """
    if saved_names is None or target_names is None:
        return posterior_dict, None
    if list(saved_names) == list(target_names):
        return posterior_dict, torch.ones(len(target_names), dtype=torch.bool)

    aligned = {}
    mask = None
    for k, v in posterior_dict.items():
        if not isinstance(v, torch.Tensor):
            aligned[k] = v
            continue
        dim = _detect_feature_dim(v, n_features_saved)
        if dim is None:
            aligned[k] = v
        else:
            av, m = _align_tensor(v, saved_names, target_names, dim)
            aligned[k] = av
            if mask is None:
                mask = m
    return aligned, mask


def _align_cell_tensor(tensor, saved_cell_names, target_cell_names):
    """
    Align a 1-D per-cell tensor from `saved_cell_names` to `target_cell_names`.

    Returns
    -------
    aligned : torch.Tensor  shape (len(target_cell_names),)
    mask    : torch.BoolTensor  shape (len(target_cell_names),)
    """
    if saved_cell_names is None or target_cell_names is None:
        return tensor, None
    if list(saved_cell_names) == list(target_cell_names):
        return tensor, torch.ones(len(target_cell_names), dtype=torch.bool)
    return _align_tensor(tensor, saved_cell_names, target_cell_names, dim=0)


def _report_alignment(label, saved_names, target_names, mask):
    """Print a one-line alignment summary."""
    if mask is None:
        return
    n_matched = int(mask.sum())
    n_target = len(target_names)
    n_saved = len(saved_names)
    n_dropped = n_saved - n_matched       # in saved but not in target → dropped
    n_missing = n_target - n_matched      # in target but not in saved  → NaN
    if n_dropped == 0 and n_missing == 0:
        return  # perfect match, nothing to report
    parts = []
    if n_dropped:
        parts.append(f"{n_dropped} dropped (not in current modality)")
    if n_missing:
        parts.append(f"{n_missing} missing → NaN")
    print(f"[LOAD] {label}: {n_matched}/{n_target} matched; {', '.join(parts)}")


def _check_nan_features(feat_mask, mod_name, mask_features):
    """
    Raise or warn when alignment would introduce NaN feature values.

    Parameters
    ----------
    feat_mask : torch.BoolTensor | None
    mod_name  : str
    mask_features : bool
        If False, raise an error.  If True, proceed silently (caller will fill NaNs).
    """
    if feat_mask is None or feat_mask.all():
        return
    n_missing = int((~feat_mask).sum())
    if not mask_features:
        raise ValueError(
            f"[LOAD] {mod_name}: {n_missing} feature(s) in the current modality were not present "
            f"in the saved technical fit and would be filled with NaN.\n"
            f"Options:\n"
            f"  • load_technical_fit(..., mask_features=True)  — fill missing features with "
            f"the per-group median alpha_y and mark them in modality.fitted_feature_mask; "
            f"then use fit_trans(..., subset_features=True) to exclude them from fitting.\n"
            f"  • Rerun fit_technical() on the current feature set to produce a matching fit."
        )


def _fill_nan_with_median(tensor, feat_mask):
    """
    Replace NaN positions (where feat_mask is False) along the last matching dimension
    with the nanmedian of the non-NaN positions (per leading dimension slice).
    Works for 2D [C, T] and 3D [C, T, K] tensors.
    """
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return tensor
    if feat_mask.all():
        return tensor
    t = tensor.clone()
    # Identify the feature axis (last dim or second-to-last)
    feat_dim = _detect_feature_dim(t, len(feat_mask))
    if feat_dim is None:
        return t
    # Move feature axis to last for uniform treatment
    t = t.transpose(feat_dim, t.ndim - 1)  # [..., T]
    # Flatten leading dims, fill row-wise
    leading = t.shape[:-1]
    t_flat = t.reshape(-1, t.shape[-1])         # [L, T]
    for row in t_flat:
        finite = row[feat_mask]
        if finite.numel() > 0:
            med = finite.nanmedian().item() if hasattr(finite, 'nanmedian') else float(torch.nanmedian(finite))
            row[~feat_mask] = med
    t = t_flat.reshape(*leading, -1)
    return t.transpose(feat_dim, tensor.ndim - 1)


def _check_nan_cells(cell_mask, subset_cells):
    """
    Raise when alignment would introduce NaN cell values and subset_cells is False.
    """
    if cell_mask is None or cell_mask.all():
        return
    n_missing = int((~cell_mask).sum())
    if not subset_cells:
        raise ValueError(
            f"[LOAD] {n_missing} cell(s) in the current model were not present in the saved "
            f"cis fit and would receive NaN x_true values.\n"
            f"Options:\n"
            f"  • load_cis_fit(..., subset_cells=True)  — drop those cells from the model so "
            f"every cell has a fitted x_true.\n"
            f"  • Rerun fit_cis() on the current cell set to produce a matching fit."
        )


def _subset_model_cells_inplace(model, keep_cells):
    """
    Drop all cells not in `keep_cells` from model.meta and every modality in-place.

    This is used by load_cis_fit(subset_cells=True) to shrink the model so that
    every remaining cell has a fitted x_true value.
    """
    import numpy as np
    keep_set = set(keep_cells)

    # Subset model.meta
    model.meta = model.meta[model.meta['cell'].isin(keep_set)].reset_index(drop=True)

    # Subset each modality
    for mod in model.modalities.values():
        if mod.counts is None:
            continue
        # Build boolean index aligned to current cell_names or positional order
        if mod.cell_names is not None:
            keep_idx = [i for i, c in enumerate(mod.cell_names) if c in keep_set]
            mod.cell_names = [mod.cell_names[i] for i in keep_idx]
        else:
            # No explicit cell names — assume same positional order as model.meta
            n_cells = (mod.counts.shape[1] if mod.cells_axis == 1
                       else mod.counts.shape[0])
            keep_idx = list(range(min(n_cells, len(keep_cells))))

        # Subset counts array
        if hasattr(mod.counts, 'toarray'):
            arr = mod.counts.toarray()
        else:
            arr = mod.counts
        if mod.cells_axis == 1:
            mod.counts = arr[:, keep_idx]
        else:
            mod.counts = arr[keep_idx, :]

        # Subset denominator (binomial)
        if mod.denominator is not None:
            if mod.cells_axis == 1:
                mod.denominator = mod.denominator[:, keep_idx]
            else:
                mod.denominator = mod.denominator[keep_idx, :]

        # Subset sum_factors DataFrame
        if hasattr(mod, 'sum_factors') and mod.sum_factors is not None:
            if hasattr(mod.sum_factors, 'loc'):
                mod.sum_factors = mod.sum_factors.loc[
                    mod.sum_factors.index.isin(keep_set)
                ].copy()

    # Also subset guide-level info if it lives on meta
    if hasattr(model, 'guide_meta') and 'cell' in model.guide_meta.columns:
        model.guide_meta = model.guide_meta[
            model.guide_meta['cell'].isin(keep_set)
        ].reset_index(drop=True)


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
                          load_model_level: bool = None,
                          mask_features: bool = False):
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

            # Resolve current feature names for alignment
            current_feature_names = (mod.feature_names
                                     if mod.feature_names is not None
                                     else (list(mod.feature_meta.index)
                                           if mod.feature_meta is not None else None))

            # ── posterior_samples_technical (load first to get saved feature names) ──
            saved_feature_names = None
            n_features_saved = None

            posterior_path = os.path.join(input_dir, f'posterior_samples_technical_{mod_name}.pt')
            if os.path.exists(posterior_path):
                loaded_data = _torch_load(posterior_path)
                n_features = None

                if isinstance(loaded_data, dict) and 'posterior_samples' in loaded_data:
                    posterior_raw = loaded_data['posterior_samples']
                    n_features = loaded_data.get('n_features')
                    saved_feature_names = loaded_data.get('feature_names')
                    n_features_saved = n_features

                    # Align posteriors to current modality's feature set
                    if (current_feature_names is not None and saved_feature_names is not None
                            and current_feature_names != saved_feature_names):
                        posterior_raw, feat_mask = _align_posterior_features(
                            posterior_raw, saved_feature_names, current_feature_names, n_features_saved)
                        _report_alignment(f"{mod_name} technical posterior",
                                          saved_feature_names, current_feature_names, feat_mask)
                        _check_nan_features(feat_mask, mod_name, mask_features)
                        if feat_mask is not None:
                            mod.fitted_feature_mask = feat_mask
                            if mask_features and not feat_mask.all():
                                # Fill NaN positions with per-group median — same effective
                                # treatment as zero-NTC-count genes in fit_technical
                                for k in list(posterior_raw.keys()):
                                    if isinstance(posterior_raw[k], torch.Tensor):
                                        posterior_raw[k] = _fill_nan_with_median(
                                            posterior_raw[k], feat_mask)
                                n_missing = int((~feat_mask).sum())
                                print(f"[LOAD] {mod_name}: {n_missing} missing feature(s) filled "
                                      f"with per-group median alpha_y (mask_features=True). "
                                      f"Use fit_trans(..., subset_features=True) to exclude them.")
                    mod.posterior_samples_technical = posterior_raw

                    # Reconstruct feature_meta DataFrame if present
                    if loaded_data.get('feature_meta') is not None:
                        _ = pd.DataFrame(loaded_data['feature_meta'])  # available if needed

                    loaded[f'posterior_samples_technical_{mod_name}_metadata'] = {
                        'modality_name': loaded_data.get('modality_name'),
                        'distribution': loaded_data.get('distribution'),
                        'n_features': n_features
                    }

                    if loaded_data.get('loss_technical') is not None:
                        mod.loss_technical = loaded_data['loss_technical']
                        mod_loaded.append(f'loss({len(mod.loss_technical)})')

                    if verbose:
                        print(f"[LOAD] {mod_name}.posterior_samples_technical ({n_features} features) ← {posterior_path}")
                else:
                    # Old format (backward compatibility) — no alignment possible
                    mod.posterior_samples_technical = loaded_data
                    if verbose:
                        print(f"[LOAD] {mod_name}.posterior_samples_technical (legacy format) ← {posterior_path}")

                loaded[f'posterior_samples_technical_{mod_name}'] = True
                mod_loaded.append(f'posterior({n_features or "?"} features)')

                # Extract and set specific alpha attributes from posterior_samples
                if 'alpha_y_add' in mod.posterior_samples_technical:
                    if not hasattr(mod, 'alpha_y_prefit_add') or mod.alpha_y_prefit_add is None:
                        alpha_y_add = mod.posterior_samples_technical['alpha_y_add']
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
                        if isinstance(alpha_y_mult, torch.Tensor) and alpha_y_mult.ndim >= 3:
                            alpha_y_mult = alpha_y_mult.mean(dim=0)
                        mod.alpha_y_prefit_mult = alpha_y_mult
                        if not hasattr(mod, 'alpha_y_prefit') or mod.alpha_y_prefit is None:
                            if mod.distribution == 'negbinom':
                                mod.alpha_y_prefit = alpha_y_mult
                        if verbose:
                            print(f"[LOAD] {mod_name}.alpha_y_prefit_mult ← extracted from posterior_samples_technical")

            # ── alpha_y_prefit standalone file (align using names from posterior file) ──
            mod_path = os.path.join(input_dir, f'alpha_y_prefit_{mod_name}.pt')
            if os.path.exists(mod_path):
                alpha_y_to_set = _torch_load(mod_path)
                if isinstance(alpha_y_to_set, torch.Tensor):
                    collapse_threshold = 4 if mod.distribution == 'multinomial' else 3
                    if alpha_y_to_set.ndim >= collapse_threshold:
                        alpha_y_to_set = alpha_y_to_set.mean(dim=0)

                # Align if we have both saved and current feature names
                if (current_feature_names is not None and saved_feature_names is not None
                        and current_feature_names != saved_feature_names
                        and n_features_saved is not None):
                    dim = _detect_feature_dim(alpha_y_to_set, n_features_saved)
                    if dim is not None:
                        alpha_y_to_set, feat_mask = _align_tensor(
                            alpha_y_to_set, saved_feature_names, current_feature_names, dim)
                        _report_alignment(f"{mod_name} alpha_y_prefit",
                                          saved_feature_names, current_feature_names, feat_mask)
                        _check_nan_features(feat_mask, mod_name, mask_features)
                        if feat_mask is not None:
                            if not hasattr(mod, 'fitted_feature_mask'):
                                mod.fitted_feature_mask = feat_mask
                            if mask_features and not feat_mask.all():
                                alpha_y_to_set = _fill_nan_with_median(alpha_y_to_set, feat_mask)

                mod.alpha_y_prefit = alpha_y_to_set
                if mod.distribution == 'negbinom':
                    mod.alpha_y_prefit_mult = alpha_y_to_set
                else:
                    mod.alpha_y_prefit_add = alpha_y_to_set

                loaded[f'alpha_y_prefit_{mod_name}'] = True
                mod_loaded.append('alpha_y')
                if verbose:
                    print(f"[LOAD] {mod_name}.alpha_y_prefit ← {mod_path}")

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

    def load_cis_fit(self, input_dir: str = None, verbose: bool = False,
                     subset_cells: bool = False):
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

        # Current cell names for alignment
        current_cell_names = (self.model.meta['cell'].tolist()
                              if 'cell' in self.model.meta.columns else None)

        # Load posterior samples first so we have saved_cell_names for aligning x_true files
        saved_cell_names = None
        cis_gene = None

        posterior_path = os.path.join(input_dir, 'posterior_samples_cis.pt')
        if os.path.exists(posterior_path):
            loaded_data = _torch_load(posterior_path)

            if isinstance(loaded_data, dict) and 'posterior_samples' in loaded_data:
                posterior_raw = loaded_data['posterior_samples']
                cis_gene = loaded_data.get('cis_gene')
                saved_cell_names = loaded_data.get('cell_names')

                # Align per-cell tensors in the posterior (x_true, log_x_true)
                if (current_cell_names is not None and saved_cell_names is not None
                        and current_cell_names != saved_cell_names):
                    # Pre-compute cell mask to check for missing cells
                    _saved_set = set(saved_cell_names)
                    _cell_mask_preview = torch.tensor(
                        [c in _saved_set for c in current_cell_names], dtype=torch.bool)

                    # Error or subset before doing any NaN alignment
                    _check_nan_cells(_cell_mask_preview, subset_cells)

                    if subset_cells and not _cell_mask_preview.all():
                        # Reduce model to only cells present in the saved fit
                        keep_cells = [c for c in current_cell_names if c in _saved_set]
                        _subset_model_cells_inplace(self.model, keep_cells)
                        current_cell_names = keep_cells
                        print(f"[LOAD] subset_cells=True: model reduced to "
                              f"{len(keep_cells)} cells present in saved cis fit.")

                    # Now align (if sets still differ, e.g. saved has cells not in current)
                    if current_cell_names != saved_cell_names:
                        n_cells_saved = len(saved_cell_names)
                        per_cell_keys = [k for k, v in posterior_raw.items()
                                         if isinstance(v, torch.Tensor)
                                         and v.shape[-1] == n_cells_saved]
                        cell_mask = None
                        for k in per_cell_keys:
                            posterior_raw[k], cell_mask = _align_tensor(
                                posterior_raw[k], saved_cell_names, current_cell_names,
                                dim=posterior_raw[k].ndim - 1)
                        if cell_mask is not None:
                            _report_alignment("cis posterior cells",
                                              saved_cell_names, current_cell_names, cell_mask)
                            self.model.fitted_cell_mask = cell_mask

                self.model.posterior_samples_cis = posterior_raw

                if loaded_data.get('feature_meta') is not None:
                    _ = pd.DataFrame(loaded_data['feature_meta'])  # available if needed

                loaded['posterior_samples_cis_metadata'] = {
                    'cis_gene': cis_gene,
                    'modality_name': loaded_data.get('modality_name'),
                }

                if loaded_data.get('loss_x') is not None:
                    self.model.loss_x = loaded_data['loss_x']
                    loaded_summary.append(f"loss_x({len(self.model.loss_x)})")

                if verbose:
                    print(f"[LOAD] posterior_samples_cis (cis_gene: {cis_gene}) ← {posterior_path}")
            else:
                # Old format (backward compatibility) — no alignment possible
                self.model.posterior_samples_cis = loaded_data
                if verbose:
                    print(f"[LOAD] posterior_samples_cis (legacy format) ← {posterior_path}")

            loaded['posterior_samples_cis'] = True
            loaded_summary.append("posterior_cis" + (f" ({cis_gene})" if cis_gene else ""))

        # Load x_true (standalone file — align by cell if names available)
        # Note: if subset_cells=True was applied above, current_cell_names is already reduced
        x_true_path = os.path.join(input_dir, 'x_true.pt')
        if os.path.exists(x_true_path):
            x_true = _torch_load(x_true_path)
            if isinstance(x_true, torch.Tensor) and x_true.ndim >= 2:
                x_true = x_true.mean(dim=0)
            if (current_cell_names is not None and saved_cell_names is not None
                    and current_cell_names != saved_cell_names
                    and x_true.ndim == 1 and x_true.shape[0] == len(saved_cell_names)):
                # Check only — subset was already handled above (or will raise)
                _saved_set = set(saved_cell_names)
                _mask = torch.tensor([c in _saved_set for c in current_cell_names], dtype=torch.bool)
                _check_nan_cells(_mask, subset_cells)
                x_true, cell_mask = _align_cell_tensor(x_true, saved_cell_names, current_cell_names)
                _report_alignment("x_true cells", saved_cell_names, current_cell_names, cell_mask)
                if cell_mask is not None and not hasattr(self.model, 'fitted_cell_mask'):
                    self.model.fitted_cell_mask = cell_mask
            self.model.x_true = x_true
            loaded['x_true'] = True
            loaded_summary.append('x_true')
            if verbose:
                print(f"[LOAD] x_true ← {x_true_path}")

        # Load log2_x_true (standalone file — align by cell if names available)
        log2_x_true_path = os.path.join(input_dir, 'log2_x_true.pt')
        if os.path.exists(log2_x_true_path):
            log2_x_true = _torch_load(log2_x_true_path)
            if isinstance(log2_x_true, torch.Tensor) and log2_x_true.ndim >= 2:
                log2_x_true = log2_x_true.mean(dim=0)
            if (current_cell_names is not None and saved_cell_names is not None
                    and current_cell_names != saved_cell_names
                    and log2_x_true.ndim == 1 and log2_x_true.shape[0] == len(saved_cell_names)):
                log2_x_true, _ = _align_cell_tensor(log2_x_true, saved_cell_names, current_cell_names)
            self.model.log2_x_true = log2_x_true
            loaded['log2_x_true'] = True
            loaded_summary.append('log2_x_true')
            if verbose:
                print(f"[LOAD] log2_x_true ← {log2_x_true_path}")

        # Extract log2_x_true from posterior_samples_cis if not already loaded
        if not hasattr(self.model, 'log2_x_true') or self.model.log2_x_true is None:
            if (hasattr(self.model, 'posterior_samples_cis')
                    and self.model.posterior_samples_cis is not None
                    and 'log_x_true' in self.model.posterior_samples_cis):
                log_x_true = self.model.posterior_samples_cis['log_x_true']
                if isinstance(log_x_true, torch.Tensor) and log_x_true.ndim >= 2:
                    log_x_true = log_x_true.mean(dim=0)
                self.model.log2_x_true = log_x_true
                loaded['log2_x_true'] = True
                if verbose:
                    print(f"[LOAD] log2_x_true ← extracted from posterior_samples_cis")
            elif hasattr(self.model, 'x_true') and self.model.x_true is not None:
                self.model.log2_x_true = torch.log2(self.model.x_true.clamp(min=1e-12))
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

        def _load_trans_posterior(loaded_data, label, n_features_hint=None):
            """
            Parse a trans posterior file and align features to current modality.
            Returns (posterior_dict, n_features, feat_mask, extra_meta).
            """
            if not (isinstance(loaded_data, dict) and 'posterior_samples' in loaded_data):
                # Old format — no alignment possible
                return loaded_data, None, None, {}

            posterior_raw = loaded_data['posterior_samples']
            n_features = loaded_data.get('n_features') or n_features_hint
            saved_names = loaded_data.get('feature_names')
            mod_name_saved = loaded_data.get('modality_name')
            feat_mask = None

            # Align to current modality features if possible
            if mod_name_saved and mod_name_saved in self.model.modalities:
                mod = self.model.modalities[mod_name_saved]
                cur_names = (mod.feature_names
                             if mod.feature_names is not None
                             else (list(mod.feature_meta.index)
                                   if mod.feature_meta is not None else None))
                if (cur_names is not None and saved_names is not None
                        and cur_names != saved_names and n_features is not None):
                    posterior_raw, feat_mask = _align_posterior_features(
                        posterior_raw, saved_names, cur_names, n_features)
                    _report_alignment(f"{label} features", saved_names, cur_names, feat_mask)
                    if feat_mask is not None:
                        mod.fitted_feature_mask = feat_mask

            extra = {
                'modality_name': mod_name_saved,
                'distribution': loaded_data.get('distribution'),
                'n_features': n_features,
                'cis_gene': loaded_data.get('cis_gene'),
            }
            return posterior_raw, n_features, feat_mask, extra

        # Load model-level posterior samples (when primary modality is being loaded)
        if should_load_model_level:
            posterior_path = os.path.join(input_dir, f'posterior_samples_trans_{self.model.primary_modality}.pt')
            if not os.path.exists(posterior_path):
                posterior_path = os.path.join(input_dir, 'posterior_samples_trans.pt')

            if os.path.exists(posterior_path):
                loaded_data = _torch_load(posterior_path)
                if isinstance(loaded_data, dict) and 'posterior_samples' in loaded_data:
                    posterior_dict, n_features, _, extra = _load_trans_posterior(
                        loaded_data, f"trans {self.model.primary_modality}")
                    self.model.posterior_samples_trans = posterior_dict
                    loaded['posterior_samples_trans'] = True
                    loaded['posterior_samples_trans_metadata'] = extra
                    if loaded_data.get('losses_trans') is not None:
                        self.model.losses_trans = loaded_data['losses_trans']
                    if loaded_data.get('trans_prior_params') is not None:
                        self.model.trans_prior_params = loaded_data['trans_prior_params']
                    if verbose:
                        print(f"[LOAD] posterior_samples_trans (modality: {extra.get('modality_name')}, "
                              f"{extra.get('n_features')} features) ← {posterior_path}")
                else:
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

                if isinstance(loaded_data, dict) and 'posterior_samples' in loaded_data:
                    posterior_dict, n_features, _, extra = _load_trans_posterior(
                        loaded_data, f"trans {mod_name}")
                    self.model.modalities[mod_name].posterior_samples_trans = posterior_dict
                    loaded[f'posterior_samples_trans_{mod_name}_metadata'] = extra

                    if loaded_data.get('losses_trans') is not None:
                        self.model.modalities[mod_name].losses_trans = loaded_data['losses_trans']
                        mod_loaded.append(f"loss({len(self.model.modalities[mod_name].losses_trans)})")
                    if loaded_data.get('trans_prior_params') is not None:
                        self.model.modalities[mod_name].trans_prior_params = loaded_data['trans_prior_params']

                    if verbose:
                        print(f"[LOAD] {mod_name}.posterior_samples_trans "
                              f"(distribution: {extra.get('distribution')}, {n_features} features) ← {mod_path}")
                else:
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
