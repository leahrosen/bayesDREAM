"""
Run this in your current session (where model_a and model_b are fitted)
to save everything that compare_models.py will need.

Files written per model to <output_dir>/<label>/:
  meta_plot.csv           - full cell metadata (includes sum_factor_new,
                            technical_group_code, guide_code, etc.)
  counts_plot.npz         - compressed count matrix + gene/cell name arrays
  gene_meta_plot.csv      - gene-level feature annotations
  alpha_y_prefit_*.pt     - NTC technical fit parameters
  posterior_samples_ntc_*.pt
  x_true.pt / log2_x_true.pt
  posterior_samples_cis.pt
  posterior_samples_trans_*.pt
"""

import os
import numpy as np
from scipy import sparse


def save_model_for_plotting(model, verbose=True):
    """
    Save everything compare_models.py needs to re-load this model.

    Writes files to model.output_dir/model.label/ alongside the fitted
    parameter files that save_ntc_fit/save_cis_fit/save_trans_fit produce.

    Parameters
    ----------
    model : bayesDREAM
        A fitted model (fit_technical, fit_cis, and fit_trans all called).
    verbose : bool
        Print a summary of saved files.
    """
    save_dir = os.path.join(model.output_dir, model.label)
    os.makedirs(save_dir, exist_ok=True)

    # ── 1. Cell metadata ────────────────────────────────────────────────────
    # model.meta already has all computed columns:
    #   technical_group_code, guide_code, guide_used,
    #   sum_factor, sum_factor_adj, sum_factor_new, etc.
    meta_path = os.path.join(save_dir, 'meta_plot.csv')
    model.meta.to_csv(meta_path, index=False)
    if verbose:
        print(f"[SAVE] meta ({len(model.meta)} cells, {len(model.meta.columns)} cols) → {meta_path}")

    # ── 2. Count matrix — primary modality + cis gene ───────────────────────
    # self.counts is None after init; data lives in modality objects.
    # The cis gene is split out into its own 'cis' modality at init time, so it
    # is NOT in the primary modality counts.  Re-init with cis_gene= requires
    # the cis gene to be present in the counts, so we prepend it here.
    import pandas as pd
    mod = model.get_modality(model.primary_modality)
    arr_primary = mod.counts.toarray() if sparse.issparse(mod.counts) else np.asarray(mod.counts)
    names_primary = list(mod.feature_names)
    meta_primary = mod.feature_meta.copy()

    if 'cis' in model.modalities:
        cis_mod = model.get_modality('cis')
        arr_cis = cis_mod.counts.toarray() if sparse.issparse(cis_mod.counts) else np.asarray(cis_mod.counts)
        if arr_cis.ndim == 1:
            arr_cis = arr_cis[np.newaxis, :]
        names_cis = list(cis_mod.feature_names) if cis_mod.feature_names else [model.cis_gene]
        meta_cis = cis_mod.feature_meta.copy()

        arr = np.vstack([arr_cis, arr_primary])
        feature_names = names_cis + names_primary
        # Combined feature_meta: cis row(s) first so iloc positions match counts rows
        combined_meta = pd.concat([meta_cis, meta_primary], ignore_index=True)
    else:
        arr = arr_primary
        feature_names = names_primary
        combined_meta = meta_primary

    cell_names = model.meta['cell'].tolist()
    counts_path = os.path.join(save_dir, 'counts_plot.npz')
    np.savez_compressed(
        counts_path,
        counts=arr,
        feature_names=np.array(feature_names),
        cell_names=np.array(cell_names),
    )
    if verbose:
        print(f"[SAVE] counts {arr.shape} (includes cis gene row) → {counts_path}")

    # ── 3. Gene-level feature metadata ──────────────────────────────────────
    gene_meta_path = os.path.join(save_dir, 'gene_meta_plot.csv')
    combined_meta.to_csv(gene_meta_path, index=False)
    if verbose:
        print(f"[SAVE] gene_meta ({len(combined_meta)} genes) → {gene_meta_path}")

    # ── 3b. Modality sum_factors (sum_factor, sum_factor_adj, sum_factor_new, …) ──
    # These are stored on the modality, not model.meta, and are not saved by
    # save_ntc_fit/save_cis_fit/save_trans_fit.
    sf = mod.sum_factors
    if sf is not None and not sf.empty:
        sf_path = os.path.join(save_dir, 'sum_factors_plot.csv')
        sf.to_csv(sf_path)   # index = cell barcodes
        if verbose:
            print(f"[SAVE] sum_factors {sf.shape} cols={list(sf.columns)} → {sf_path}")

    # ── 4. High-MOI guide assignment (if applicable) ─────────────────────────
    if getattr(model, 'is_high_moi', False):
        ga_path = os.path.join(save_dir, 'guide_assignment_plot.npz')
        np.savez_compressed(ga_path, guide_assignment=np.asarray(model.guide_assignment))
        model.guide_meta.to_csv(os.path.join(save_dir, 'guide_meta_plot.csv'), index=False)

        # Reconstruct guide_target DataFrame from guide_targets_dict so the
        # init can re-establish guide→target relationships without needing
        # guide_meta['target'] (which may not be present).
        if hasattr(model, 'guide_targets_dict') and model.guide_targets_dict:
            import pandas as pd
            rows = [{'guide': g, 'target': t}
                    for g, targets in model.guide_targets_dict.items()
                    for t in targets]
            pd.DataFrame(rows).to_csv(
                os.path.join(save_dir, 'guide_target_plot.csv'), index=False)

        if verbose:
            print(f"[SAVE] guide_assignment {model.guide_assignment.shape} → {ga_path}")

    # ── 5. Fitted parameters ─────────────────────────────────────────────────
    # save_ntc_fit / save_cis_fit / save_trans_fit write to save_dir by default.
    model.save_ntc_fit()
    model.save_cis_fit()
    model.save_trans_fit()

    if verbose:
        print(f"[SAVE] fitted params → {save_dir}/")
        print(f"[DONE] {model.label} saved to {save_dir}/\n")


# ── Call for each model ──────────────────────────────────────────────────────
# Replace model_a / model_b with your actual variable names.

save_model_for_plotting(model_a)
save_model_for_plotting(model_b)
