"""
Per-gene dose-response curve panels, comparing two OR MORE fitted bayesDREAM
models -- a generalized, dataset-agnostic version of the original
GEX_comp_Doming_Morris.ipynb / compare_models.py (Domingo vs Morris, GFI1B
only, hardcoded paths).

Panel shape auto-adapts to however many datasets have a completed fit for a
given cis gene: 2 datasets -> 2x2 panel, 3 datasets -> 2x3 panel, etc. Row 0
is each dataset standalone (its own data + its own curve); row 1 is each
dataset's own data + own curve again, with every *other* available dataset's
curve overlaid on top (so for 3 datasets, row 1 shows up to 2 extra curves
per subplot). See `make_panel`.

Prerequisites
-------------
For each (dataset, cis_gene) pair you want a panel for, `save_model_for_plotting()`
(see save_for_plotting.py at the repo root) must have been run once, in the
original fitting session, and its output directory registered as that
dataset's `save_for_plotting_dir_fn` in comparative/datasets.py. This is a
full-model reload, so it only scales to a bounded number of genes (Domingo's
~91 shared trans genes is fine; doing this transcriptome-wide for Morris/
Replogle is not the intended use -- see trans_param_compare.py for that).

Quick start
-----------
    from comparative.datasets import DOMINGO, MORRIS
    from comparative.dose_response_panels import compare_datasets

    compare_datasets([DOMINGO, MORRIS], cis_gene='GFI1B', out_dir='./dose_response_plots')

Or all three datasets in one panel per gene, for a cis gene all three have exported::

    from comparative.datasets import DOMINGO, MORRIS, REPLOGLE
    compare_datasets([DOMINGO, MORRIS, REPLOGLE], cis_gene='GFI1B', out_dir='./dose_response_plots')

To automate this across every one of Domingo's cis genes -- using whichever
subset of {Domingo, Morris, Replogle} actually has a completed fit for each
one (see `DatasetSpec.cis_genes`) -- see `compare_all_domingo_cis_genes()`.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from bayesDREAM import bayesDREAM
from bayesDREAM.plotting.xy_plots import predict_hill_from_summary_row
from .datasets import DatasetSpec, morris_symbol_to_id, morris_id_to_symbol, DOMINGO


# ── Loading ────────────────────────────────────────────────────────────────

def load_model_for_plotting(spec: DatasetSpec, cis_gene: str, device: Optional[str] = None,
                              lean: bool = True, load_trans: bool = True) -> "bayesDREAM":
    """Re-initialise a bayesDREAM model from files written by
    save_model_for_plotting() (save_for_plotting.py) and load the fitted
    NTC/cis/trans parameters.

    lean : bool
        Passed through to load_ntc_fit()/load_cis_fit() -- collapses each
        posterior to point estimates (median + `<key>_lower`/`<key>_upper`)
        instead of keeping the full multi-sample tensors. Does NOT reach
        load_trans_fit(): lean=True there is not implemented (bayesDREAM/
        io/load.py raises NotImplementedError unconditionally -- nearly
        every save_trans_summary() column depends on the FULL joint
        per-draw posterior of alpha/beta/Vmax_a/Vmax_b/K_a/K_b/n_a/n_b/A,
        not just their marginals, so collapsing them first would silently
        produce wrong CIs/FDR). The trans-fit posterior is also the
        dominant share of peak memory for Morris/Replogle's ~8-11k-feature
        panels ([1000 samples, n_features] per raw parameter, ~10
        parameters for additive_hill) -- lean alone does NOT fix that. This
        function still loads the FULL trans-fit posterior for every trans
        gene regardless of `lean`; for interactive "plot any gene on demand"
        use at Morris/Replogle scale, use compare_datasets_lightweight()/
        make_panel_lightweight() instead, which never loads a model or raw
        counts at all (see their docstrings). This function stays the
        heavier "give me a live model" path -- e.g. for the notebook's
        "inspect a single panel inline" cell, or any use that genuinely
        needs the live posterior. save_trans_summary()'s own extract_param()
        (used for the NTC/cis point estimates `lean` DOES affect) degrades
        gracefully on a lean array -- verified directly against
        predict_trans_function()/_extract_param_mean()/_compute_hill_markers()
        in xy_plots.py too (same `.mean(dim=0)`-then-squeeze idiom
        throughout, and none of them draw a shaded band around the Hill
        curve in this code path, so there's no visual degradation from THIS
        part at all). Default True: real, repeated kernel-killing OOMs on
        any Morris/Replogle comparison without it -- see
        reconstruct_export_replogle.py's reconstruct_model() comment for an
        identical prior incident on Replogle's shared NTC fit specifically.
    load_trans : bool
        If False, skips model.load_trans_fit() entirely -- this is what
        makes ensure_smoothed_curve()'s on-demand single-feature smoothing
        cheap: compute_smoothed_curves() never touches the trans-fit
        posterior (only x_true, alpha_y_prefit, raw counts, sum_factors),
        all of which come from load_ntc_fit()/load_cis_fit()/sum_factors_plot.csv
        alone. Skipping the trans-fit load avoids paying for Morris/
        Replogle's full ~10-20k-feature joint posterior just to smooth one
        gene's curve. The returned model's `posterior_samples_trans` is
        left unset/empty in this case -- do not use it for anything that
        needs the trans-fit posterior (e.g. plot_xy_data's Hill curve,
        save_trans_summary()).
    """
    save_dir = spec.plotting_save_dir(cis_gene)

    meta = pd.read_csv(os.path.join(save_dir, 'meta_plot.csv'))

    data = np.load(os.path.join(save_dir, 'counts_plot.npz'))
    counts = pd.DataFrame(
        data['counts'],
        index=data['feature_names'].tolist(),
        columns=data['cell_names'].tolist(),
    )

    gene_meta_path = os.path.join(save_dir, 'gene_meta_plot.csv')
    gene_meta = pd.read_csv(gene_meta_path) if os.path.exists(gene_meta_path) else None

    ga_path = os.path.join(save_dir, 'guide_assignment_plot.npz')
    gm_path = os.path.join(save_dir, 'guide_meta_plot.csv')
    gt_path = os.path.join(save_dir, 'guide_target_plot.csv')
    if os.path.exists(ga_path) and os.path.exists(gm_path):
        guide_assignment = np.load(ga_path)['guide_assignment']
        guide_meta = pd.read_csv(gm_path)
        guide_target = pd.read_csv(gt_path) if os.path.exists(gt_path) else None
    else:
        guide_assignment = None
        guide_meta = None
        guide_target = None

    # Translate symbol -> feature-index identifier where they differ (Replogle:
    # Ensembl gene ID -- counts.index above comes straight from counts_plot.npz's
    # feature_names, which for Replogle IS the Ensembl ID, not the symbol; see
    # DatasetSpec.cis_gene_id_fn's docstring). None for Domingo/Morris, where
    # the feature index already is the symbol.
    feature_cis_gene = cis_gene
    if spec.cis_gene_id_fn is not None:
        feature_cis_gene = spec.cis_gene_id_fn(cis_gene)
        if feature_cis_gene is None:
            raise KeyError(
                f"[{spec.name}] cis_gene_id_fn has no mapping for symbol {cis_gene!r} -- "
                f"add one (e.g. to REPLOGLE_GENE_TO_ID in comparative/datasets.py)."
            )

    model_kwargs = dict(
        meta=meta,
        counts=counts,
        feature_meta=gene_meta,
        cis_gene=feature_cis_gene,
        output_dir=spec.name,
        label=cis_gene,
        sum_factor_col=spec.init_sum_factor_col,
        guide_assignment=guide_assignment,
        guide_meta=guide_meta,
        guide_target=guide_target,
        require_ntc=False,
    )
    if device is not None:
        model_kwargs['device'] = device

    model = bayesDREAM(**model_kwargs)

    model.load_ntc_fit(input_dir=save_dir, lean=lean)
    model.load_cis_fit(input_dir=save_dir, lean=lean)
    # load_trans_fit(lean=True) is NOT implemented (bayesDREAM/io/load.py
    # raises NotImplementedError unconditionally -- collapsing the joint
    # per-draw posterior would silently break the Hill-curve/FDR/derivative
    # evaluations save_trans_summary() computes downstream). `lean` here
    # only ever collapses the NTC/cis posteriors -- see this function's own
    # docstring, corrected 2026-09 after this line originally (wrongly)
    # forwarded `lean` here too.
    if load_trans:
        model.load_trans_fit(input_dir=save_dir, lean=False)

    sf_path = os.path.join(save_dir, 'sum_factors_plot.csv')
    if os.path.exists(sf_path):
        sf = pd.read_csv(sf_path, index_col=0)
        primary_mod = model.get_modality(model.primary_modality)
        primary_mod.sum_factors = sf
        if 'cis' in model.modalities:
            model.modalities['cis'].sum_factors = sf

    if spec.force_single_cell_line:
        model.meta['cell_line'] = spec.force_single_cell_line

    return model


def pick_sum_factor_col(model: "bayesDREAM", preference: Sequence[str] = (
    'sum_factor_new', 'sum_factor_refit', 'sum_factor_adj', 'sum_factor',
)) -> str:
    """Pick the most-adjusted sum_factor column actually available on this
    model's primary modality, trying `preference` in order. Which column a
    given export has depends on which of compute_scran/adjust_ntc_sum_factor/
    refit_sumfactor were run for that dataset -- rather than hardcode a
    per-dataset assumption that can go stale, just use whatever's furthest
    along the chain.
    """
    mod = model.get_modality(model.primary_modality)
    cols = set(mod.sum_factors.columns) if mod.sum_factors is not None else set()
    for c in preference:
        if c in cols:
            return c
    if 'sum_factor' in model.meta.columns:
        return 'sum_factor'
    raise ValueError(f"No usable sum_factor column found among {preference} or 'sum_factor' in meta.")


def resolve_sum_factor_col(spec: DatasetSpec, model: "bayesDREAM") -> str:
    """DatasetSpec.plot_sum_factor_col if set (known-correct per dataset --
    Domingo's own refit_sumfactor() writes 'sum_factor_refit'; Morris/Replogle
    stop at adjust_ntc_sum_factor() and use 'sum_factor_adj'), else fall back
    to probing the reloaded model via pick_sum_factor_col().
    """
    if spec.plot_sum_factor_col:
        mod = model.get_modality(model.primary_modality)
        cols = set(mod.sum_factors.columns) if mod.sum_factors is not None else set()
        if spec.plot_sum_factor_col not in cols and spec.plot_sum_factor_col not in model.meta.columns:
            raise ValueError(
                f"[{spec.name}] configured plot_sum_factor_col={spec.plot_sum_factor_col!r} "
                f"not found on the reloaded model (available: {sorted(cols)}). Update "
                "this DatasetSpec's plot_sum_factor_col in comparative/datasets.py."
            )
        return spec.plot_sum_factor_col
    return pick_sum_factor_col(model)


def allsig_copy(summary: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Copy of a trans summary with FDR columns zeroed (so the Hill shape
    always draws, not just FDR-significant features), gene_name forced to
    equal feature (feature_meta merges can otherwise populate gene_name with
    a non-matching identifier, e.g. an Ensembl ID for Replogle, breaking the
    reference_df lookup used by plot_xy_data's overlay), and a 'gene_id'
    (Ensembl) column attached for cross-dataset matching.

    'gene_id' is what compare_datasets()/make_panel() actually intersect and
    look genes up by -- Domingo/Morris's own 'feature' column already IS the
    gene symbol (spec.symbol_col == 'feature'), so 'gene_id' is translated
    via comparative.datasets.morris_symbol_to_id() (the master symbol->
    Ensembl-ID map, built from Morris's transcriptome-wide gene_meta.csv --
    see its docstring for why that source, not Domingo's or Replogle's own
    feature_meta). Replogle's 'feature' already IS the Ensembl ID, so
    'gene_id' is just a copy of it. A Domingo/Morris gene whose symbol isn't
    in the map (ambiguous, or absent from Morris's panel) gets NaN --
    excluded from any cross-dataset intersection, same as a real mismatch.
    """
    out = summary.copy()
    for col in [c for c in out.columns if c.startswith('fdr_')]:
        out[col] = 0.0
    out['gene_name'] = out['feature']
    if spec.symbol_col == 'feature':
        out['gene_id'] = out['feature'].map(morris_symbol_to_id())
    else:
        out['gene_id'] = out['feature']
    return out


# ── Pre-computed smoothed curves (model-free plotting) ───────────────────────
#
# compare_datasets()/make_panel() need a full model (raw counts + trans-fit
# posterior) per dataset -- too much memory to hold for Morris/Replogle's
# transcriptome-wide panels if the notebook should stay usable to plot ANY
# trans gene on demand, not just a pre-chosen subset. The functions below
# instead precompute, ONCE per (dataset, cis_gene) at backfill time (when the
# full model is already resident anyway -- see reconstruct_export.py/
# reconstruct_export_replogle.py), a small per-gene "smoothed trend" array
# (compute_smoothed_curves()) that gets saved alongside the existing
# trans_feature_summary_{modality}.csv. At plot time, load_smoothed_curves()
# + that CSV are the ONLY two things read -- no model, no raw counts, no
# trans-fit posterior -- so every trans gene in the FULL panel can be plotted
# interactively (make_panel_lightweight()/compare_datasets_lightweight()) at
# a memory cost of a few tens of MB even for Morris/Replogle.

def compute_smoothed_curves(
    model, modality_name: str = 'gene', color_by: str = 'cell_line',
    sum_factor_col: str = 'sum_factor', window: int = 100, n_points: int = 150,
    verbose: bool = True, features: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """For every feature in `model`'s `modality_name` modality (or just
    `features`, if given -- see below), compute the SAME alpha_y-corrected,
    k-NN-smoothed (x_true, y_expr) trend that
    model.plot_xy_data(show_correction='corrected') draws -- reusing that
    exact machinery (bayesDREAM.plotting.xy_plots' private
    _align_cells_to_modality/_knn_k/_smooth_knn, imported rather than
    reimplemented, so this can never silently drift from what the live plot
    would show), grouped by `color_by` (matching dose_response_panels.py's
    own color_by='cell_line' convention).

    Deliberately does NOT bake in the log2FC(x)/log2FC(y) NTC offset that
    plot_xy_data(log2fc=True) applies -- returns absolute log2(x)/log2(y)
    curves instead. plot_gene_lightweight() applies the offset later, at
    plot time, from that gene's own trans_feature_summary row (x_ntc/y_ntc)
    -- the SAME offset source _overlay_extra_curve() already uses for a
    cross-dataset Hill-curve overlay, so a lightweight panel's smoothed
    trend and its Hill curve are guaranteed to line up using one consistent
    convention, even though this differs slightly from plot_xy_data's own
    live mu_ntc-based offset (already true today: _overlay_extra_curve's
    curves use row-based offsets while plot_xy_data's OWN curve+data use its
    live offset -- an established, working precedent this reuses, not a new
    inconsistency).

    Meant to be called ONCE per (dataset, cis_gene), on the fully-loaded
    model, right where reconstruct_export.py/reconstruct_export_replogle.py
    already call save_model_for_plotting() -- see save_smoothed_curves().

    Cost: dominated by _smooth_knn's per-feature Python loop over ~n_cells
    windows -- for Morris/Replogle's full ~10-20k-feature panel this is
    HOURS (confirmed 2026-09-07: ~2.1s/feature on Morris, ~6.5h projected for
    all 11045). reconstruct_export.py/reconstruct_export_replogle.py's own
    backfill therefore does NOT default to the full panel for Morris/Replogle
    -- see their `features=domingo_union_features(...)` call, which bounds
    this to the union of Domingo's own (much smaller) trans-gene panels
    across all its cis genes. Anything outside that gets computed on demand
    instead, at PLOT time, by ensure_smoothed_curve() (a handful of
    features, not the whole panel -- see its docstring).

    features : sequence of str, optional
        Restrict the loop to just these feature names (a KeyError-free
        subset -- anything in `features` not actually present in this
        modality is silently skipped, not an error, since that's the normal
        case for a Domingo-derived gene list against Morris/Replogle's own,
        differently-sized panel). None (default) computes every feature in
        the modality, same as before this parameter existed.

    Returns {'feature_names': [F] (== `features`, filtered to what's actually
    present, when given), 'group_labels': [G], 'x_log2': float32
    [F, G, n_points], 'y_log2': float32 [F, G, n_points]}, NaN-padded past
    each (feature, group)'s actual smoothed-curve length (or entirely, for a
    group with no valid cells for that feature -- e.g. an all-zero-count gene
    in one cell_line).
    """
    from bayesDREAM.plotting.xy_plots import _align_cells_to_modality, _knn_k, _smooth_knn

    modality = model.get_modality(modality_name)
    if modality.alpha_y_prefit is None:
        raise ValueError(
            f"modality {modality_name!r} has no alpha_y_prefit -- fit_ntc()/load_ntc_fit() "
            "must be loaded before precomputing smoothed curves."
        )
    if modality.sum_factors is None or sum_factor_col not in modality.sum_factors.columns:
        raise ValueError(
            f"sum_factor_col={sum_factor_col!r} not found in modality.sum_factors "
            f"(available: {list(modality.sum_factors.columns) if modality.sum_factors is not None else '(none)'})."
        )

    x_true = model.x_true
    if hasattr(x_true, 'cpu'):
        x_true = x_true.cpu().numpy()
    x_true = np.asarray(x_true)

    alpha_y_full = modality.alpha_y_prefit
    if hasattr(alpha_y_full, 'cpu'):
        alpha_y_full = alpha_y_full.cpu().numpy()
    alpha_y_full = np.asarray(alpha_y_full)  # [C, T] or [S, C, T]

    all_feature_names = list(modality.feature_names)
    if features is None:
        selected = list(enumerate(all_feature_names))  # [(fi, name), ...], full panel
    else:
        name_to_idx = {n: i for i, n in enumerate(all_feature_names)}
        selected = [(name_to_idx[f], f) for f in features if f in name_to_idx]
        n_missing = len(features) - len(selected)
        if n_missing and verbose:
            print(f"[{modality_name}] {n_missing}/{len(features)} requested feature(s) not present "
                  f"in this modality's panel -- skipped (not an error).")
    feature_names = [f for _, f in selected]
    n_features = len(feature_names)

    # Group labels are fixed across every feature (same cell population,
    # modulo per-feature NaN/zero filtering below) -- computed once here
    # rather than per feature, so every feature shares the same [F, G, ...]
    # slot layout (a feature missing a group just gets an all-NaN slot).
    if color_by in model.meta.columns:
        group_labels = sorted(model.meta[color_by].dropna().astype(str).unique())
    else:
        group_labels = ['All']
    n_groups = len(group_labels)

    x_arr = np.full((n_features, n_groups, n_points), np.nan, dtype=np.float32)
    y_arr = np.full((n_features, n_groups, n_points), np.nan, dtype=np.float32)

    iterator = range(n_features)
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f'[{modality_name}] precomputing smoothed curves')
        except ImportError:
            print(f"[{modality_name}] precomputing smoothed curves for {n_features} features...")

    for out_i in iterator:
        fi = selected[out_i][0]
        y_obs = modality.counts[fi, :] if modality.cells_axis == 1 else modality.counts[:, fi]
        x_true_aligned, y_obs_aligned, meta_aligned = _align_cells_to_modality(model, modality, x_true, y_obs)
        sum_factor = modality.sum_factors.loc[meta_aligned['cell'].values, sum_factor_col].values

        y_expr = np.empty(len(meta_aligned))
        if 'technical_group_code' in meta_aligned.columns:
            tgc = meta_aligned['technical_group_code'].values
            for gc in np.unique(tgc):
                gc = int(gc)
                a = (float(alpha_y_full[:, gc, fi].mean()) if alpha_y_full.ndim == 3
                     else float(alpha_y_full[gc, fi]))
                m = tgc == gc
                y_expr[m] = y_obs_aligned[m] / (sum_factor[m] * a)
        else:
            y_expr = y_obs_aligned / sum_factor

        cb_vals = (meta_aligned[color_by].astype(str).values if color_by in meta_aligned.columns
                   else np.full(len(meta_aligned), 'All'))

        for gi, glabel in enumerate(group_labels):
            gmask = cb_vals == glabel
            xg, yg = x_true_aligned[gmask], y_expr[gmask]
            valid = (xg > 0) & np.isfinite(yg)
            xg, yg = xg[valid], yg[valid]
            if len(xg) == 0:
                continue
            k = _knn_k(len(xg), window)
            x_smooth, y_smooth = _smooth_knn(xg, yg, k)
            vs = y_smooth > 0
            x_smooth, y_smooth = x_smooth[vs], y_smooth[vs]
            if len(x_smooth) == 0:
                continue
            xl, yl = np.log2(x_smooth), np.log2(y_smooth)
            if len(xl) > n_points:
                # x_smooth is sorted (guaranteed by _smooth_knn) -- take
                # evenly-spaced INDICES rather than interpolating, so the
                # stored curve is a subset of real smoothed values, not a
                # re-derived approximation of them.
                idx = np.unique(np.linspace(0, len(xl) - 1, n_points).round().astype(int))
                xl, yl = xl[idx], yl[idx]
            x_arr[out_i, gi, :len(xl)] = xl
            y_arr[out_i, gi, :len(yl)] = yl

    return {
        'feature_names': feature_names,
        'group_labels': group_labels,
        'x_log2': x_arr,
        'y_log2': y_arr,
    }


def save_smoothed_curves(save_dir: str, smoothed: Dict[str, object], modality_name: str = 'gene') -> str:
    """Save compute_smoothed_curves()'s output to
    `save_dir`/smoothed_xy_{modality_name}.npz -- alongside
    save_model_for_plotting()'s other exports, so load_smoothed_curves()
    finds it via the same DatasetSpec.plotting_save_dir(cis_gene).
    """
    path = os.path.join(save_dir, f'smoothed_xy_{modality_name}.npz')
    np.savez_compressed(
        path,
        feature_names=np.array(smoothed['feature_names']),
        group_labels=np.array(smoothed['group_labels']),
        x_log2=smoothed['x_log2'],
        y_log2=smoothed['y_log2'],
    )
    return path


def load_smoothed_curves(spec: DatasetSpec, cis_gene: str) -> Dict[str, object]:
    """Load ONLY the precomputed smoothed-curve artifact (compute_smoothed_
    curves()/save_smoothed_curves()) for (spec, cis_gene) -- no model, no
    raw counts. Returns {'feature_names', 'feature_index' (name -> position,
    for O(1) lookup), 'group_labels', 'x_log2', 'y_log2'}.
    """
    save_dir = spec.plotting_save_dir(cis_gene)
    path = os.path.join(save_dir, f'smoothed_xy_{spec.modality_name}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[{spec.name}] no precomputed smoothed-curve artifact at {path!r} for cis gene "
            f"{cis_gene!r}. This is written by reconstruct_export(_replogle).py's "
            f"reconstruct_and_export() alongside save_model_for_plotting() -- if this export "
            f"predates that addition, re-run reconstruct_and_export(..., force=True) for this "
            f"(dataset, cis_gene) to backfill it."
        )
    data = np.load(path)
    feature_names = data['feature_names'].tolist()
    return {
        'feature_names': feature_names,
        'feature_index': {f: i for i, f in enumerate(feature_names)},
        'group_labels': data['group_labels'].tolist(),
        'x_log2': data['x_log2'],
        'y_log2': data['y_log2'],
    }


def load_gene_summary(spec: DatasetSpec, cis_gene: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load ONLY `spec`'s trans_feature_summary_{modality}.csv for `cis_gene`
    -- the CSV every completed fit_trans run already writes, regardless of
    whether save_model_for_plotting()/compute_smoothed_curves() have been
    run. Returns (summary, summary_allsig) -- see allsig_copy().
    """
    summary = pd.read_csv(spec.trans_summary_path(cis_gene), low_memory=False)
    return summary, allsig_copy(summary, spec)


_DOMINGO_UNION_CACHE: Dict[str, List[str]] = {}


def domingo_union_features(spec: DatasetSpec) -> List[str]:
    """Union of every trans gene appearing in ANY of Domingo's own
    trans_feature_summary panels -- one per DOMINGO.cis_genes (GFI1B, NFE2,
    MYB, TET2) -- translated into `spec`'s native feature identifier (its
    own 'feature' column convention -- gene symbol for Domingo/Morris,
    Ensembl ID for Replogle, via the same morris_symbol_to_id()/
    morris_id_to_symbol() maps allsig_copy() uses for cross-dataset
    matching).

    A single, cis-gene-INDEPENDENT bound (unlike an earlier per-cis-gene
    version of this function): reconstruct_export.py/
    reconstruct_export_replogle.py use the SAME set here for every one of
    `spec`'s cis genes, bounding compute_smoothed_curves()'s precompute for
    Morris/Replogle down from their full ~10-20k-feature transcriptome-wide
    panel (hours) to this much smaller union (seconds) -- anything outside
    it is computed on demand instead, at plot time, by
    ensure_smoothed_curve(). Domingo's own per-cis-gene panels overlap
    heavily in practice (largely the same readout genes tested against
    every cis gene), so the union stays small, not anywhere near the full
    transcriptome.

    Being cis-gene-independent is deliberate: a per-cis-gene bound (only the
    Domingo panel for THIS cis gene) has nothing to bound against for a cis
    gene Domingo never fit at all -- HHEX/IKZF1/RUNX1, Morris/Replogle-only
    cis genes -- leaving their precompute empty and every gene on the
    on-demand path. Using the union of ALL of Domingo's panels instead still
    gives a meaningful, small bound even there.

    For Domingo itself, returns the union of its own 4 panels directly (no
    translation needed) -- included for completeness; reconstruct_export.py
    doesn't actually restrict Domingo's own (already small) per-cis-gene
    precompute with it.

    Cached per `spec.name` (module-level `_DOMINGO_UNION_CACHE`) since it's
    identical across every cis gene of that dataset -- reconstruct_and_export()
    calls this once per cis gene, and re-reading Domingo's 4 summary CSVs
    each time would be wasteful.
    """
    if spec.name in _DOMINGO_UNION_CACHE:
        return _DOMINGO_UNION_CACHE[spec.name]

    gene_ids: set = set()
    for cis_gene in DOMINGO.cis_genes:
        domingo_summary, _ = load_gene_summary(DOMINGO, cis_gene)
        if spec.name == DOMINGO.name:
            gene_ids.update(domingo_summary['feature'].dropna())
        else:
            domingo_allsig = allsig_copy(domingo_summary, DOMINGO)
            gene_ids.update(domingo_allsig['gene_id'].dropna())

    if spec.name == DOMINGO.name:
        result = sorted(gene_ids)
    elif spec.symbol_col == 'feature':
        id_to_symbol = morris_id_to_symbol()
        result = sorted({id_to_symbol[g] for g in gene_ids if g in id_to_symbol})
    else:
        result = sorted(gene_ids)

    _DOMINGO_UNION_CACHE[spec.name] = result
    return result


# ── On-demand smoothing (fallback for a gene missing from the precompute) ────

_ON_DEMAND_MODEL_CACHE: Dict[Tuple[str, str], "bayesDREAM"] = {}


def _get_on_demand_model(spec: DatasetSpec, cis_gene: str, device: Optional[str] = None) -> "bayesDREAM":
    """Cached (per dataset, cis_gene) model reload backing on-the-fly
    smoothing -- see ensure_smoothed_curve(). Loaded WITHOUT the trans-fit
    posterior (load_model_for_plotting(..., load_trans=False)):
    compute_smoothed_curves() never reads it (only x_true, alpha_y_prefit,
    raw counts, sum_factors), and skipping it is what keeps a single
    on-demand gene cheap even for Morris/Replogle -- no per-call reload of
    their full ~10-20k-feature joint posterior (the actual cost that made
    the FULL-panel precompute take hours, confirmed 2026-09-07). Cached
    across calls in this process so plotting several missing genes in a row
    for the same (dataset, cis_gene) only pays the reload once.
    """
    key = (spec.name, cis_gene)
    if key not in _ON_DEMAND_MODEL_CACHE:
        _ON_DEMAND_MODEL_CACHE[key] = load_model_for_plotting(spec, cis_gene, device=device, load_trans=False)
    return _ON_DEMAND_MODEL_CACHE[key]


def clear_on_demand_model_cache() -> None:
    """Free every model held by _get_on_demand_model()'s cache -- these hold
    raw counts (non-trivial for Morris/Replogle's full panel) for as long as
    the process lives otherwise.
    """
    _ON_DEMAND_MODEL_CACHE.clear()


def compute_smoothed_curve_on_demand(
    spec: DatasetSpec, cis_gene: str, features: Sequence[str], *, device: Optional[str] = None,
    sum_factor_col: Optional[str] = None,
) -> Dict[str, object]:
    """compute_smoothed_curves(), for just `features`, against a cached
    trans-posterior-free model for (spec, cis_gene) reloaded on demand (see
    _get_on_demand_model()) rather than requiring a live model be handed in.
    Used by ensure_smoothed_curve() to backfill a gene missing from a
    precomputed smoothed_xy_{modality}.npz.
    """
    model = _get_on_demand_model(spec, cis_gene, device=device)
    sf_col = sum_factor_col or resolve_sum_factor_col(spec, model)
    return compute_smoothed_curves(model, modality_name=spec.modality_name, sum_factor_col=sf_col,
                                    features=list(features), verbose=False)


def _merge_smoothed_curves_inplace(base: Dict[str, object], addition: Dict[str, object]) -> None:
    """Append `addition`'s (compute_smoothed_curves() output) features into
    `base` (a load_smoothed_curves()-shaped dict -- i.e. has 'feature_index')
    IN PLACE. Silently skips any feature already present in `base`. Both must
    share the same group_labels (guaranteed by ensure_smoothed_curve(): both
    come from the same underlying model's meta/color_by).
    """
    if base['group_labels'] != addition['group_labels']:
        raise ValueError(
            f"group_labels mismatch merging smoothed curves: {base['group_labels']!r} vs "
            f"{addition['group_labels']!r} -- on-demand compute must use the same color_by "
            f"grouping as the original precompute."
        )
    new = [(i, f) for i, f in enumerate(addition['feature_names']) if f not in base['feature_index']]
    if not new:
        return
    idxs = [i for i, _ in new]
    names = [f for _, f in new]
    base['x_log2'] = np.concatenate([base['x_log2'], addition['x_log2'][idxs]], axis=0)
    base['y_log2'] = np.concatenate([base['y_log2'], addition['y_log2'][idxs]], axis=0)
    start = len(base['feature_names'])
    base['feature_names'] = base['feature_names'] + names
    base['feature_index'].update({f: start + j for j, f in enumerate(names)})


def ensure_smoothed_curve(
    spec: DatasetSpec, cis_gene: str, smoothed: Dict[str, object], feature: str, *,
    persist: bool = True, device: Optional[str] = None,
) -> bool:
    """If `feature` is already in `smoothed` (a load_smoothed_curves()/
    compute_smoothed_curves() dict), no-op. Otherwise computes it on the fly
    (compute_smoothed_curve_on_demand()) and merges it into `smoothed` IN
    PLACE, so subsequent lookups against the same dict (e.g. row-0/row-1 of
    the same panel) succeed without recomputing.

    persist : bool
        If True (default), also appends the newly-computed curve to the
        on-disk smoothed_xy_{modality}.npz for (spec, cis_gene) -- so a
        later notebook session (a fresh load_smoothed_curves() call) gets it
        for free too, not just this one. Re-reads the file fresh from disk
        before appending (not just `smoothed`, which may already hold other
        in-memory-only curves from earlier persist=False calls) so this
        never clobbers anything else already written there.

    Returns True if `feature` is now available in `smoothed` (whether it
    already was, or was just computed) -- False only if the on-demand
    compute itself found nothing (e.g. `feature` isn't a real feature in
    this dataset's own panel at all).
    """
    if feature in smoothed['feature_index']:
        return True

    print(f"[{spec.name}] {feature!r} not in the precomputed smoothed-curve artifact for "
          f"cis gene {cis_gene!r} -- computing on demand...")
    addition = compute_smoothed_curve_on_demand(spec, cis_gene, [feature], device=device)
    if feature not in addition['feature_names']:
        print(f"[{spec.name}] {feature!r} isn't a feature in this dataset's own panel -- skipping.")
        return False

    _merge_smoothed_curves_inplace(smoothed, addition)

    if persist:
        save_dir = spec.plotting_save_dir(cis_gene)
        path = os.path.join(save_dir, f'smoothed_xy_{spec.modality_name}.npz')
        on_disk = load_smoothed_curves(spec, cis_gene) if os.path.exists(path) else {
            'feature_names': [], 'feature_index': {}, 'group_labels': addition['group_labels'],
            'x_log2': np.zeros((0, len(addition['group_labels']), addition['x_log2'].shape[2]), dtype=np.float32),
            'y_log2': np.zeros((0, len(addition['group_labels']), addition['y_log2'].shape[2]), dtype=np.float32),
        }
        if feature not in on_disk['feature_index']:
            _merge_smoothed_curves_inplace(on_disk, addition)
            save_smoothed_curves(save_dir, on_disk, modality_name=spec.modality_name)
            print(f"[{spec.name}] appended {feature!r} to {path}")

    return True


def plot_gene_lightweight(
    ax: plt.Axes, feature: str, row: Optional[pd.Series], smoothed: Dict[str, object], spec: DatasetSpec,
    *, cis_gene: Optional[str] = None, fdr_threshold: float = 0.05, show_hill_function: bool = True,
    hill_color: Optional[str] = None, hill_label: Optional[str] = None,
    allow_on_demand: bool = True, device: Optional[str] = None,
) -> bool:
    """Model-free equivalent of _plot_into()/model.plot_xy_data() for one
    (dataset, feature): draws one smoothed trend line per color_by group
    from `smoothed` (compute_smoothed_curves()/load_smoothed_curves()) plus
    the fitted Hill curve from `row` (a trans_feature_summary row -- pass
    the REAL row for genuine FDR gating, or an allsig_copy() row to
    force-render regardless of significance, exactly as make_panel() does
    for its row-0/row-1 respectively), via predict_hill_from_summary_row()
    (the same function _overlay_extra_curve() already uses for cross-dataset
    overlays).

    Both the smoothed trend and the Hill curve are offset into log2FC(x)/
    log2FC(y) space using `row`'s own x_ntc/y_ntc -- see
    compute_smoothed_curves()'s docstring for why that's the right offset
    source here.

    If `feature` isn't in `smoothed` (e.g. it fell outside the Domingo-
    bounded precompute -- see domingo_union_features()), and `cis_gene` is
    given and `allow_on_demand` is True (both default-on for
    make_panel_lightweight()'s calls), computes it on the fly via
    ensure_smoothed_curve() and mutates `smoothed` in place so later calls
    against the same dict find it already there. Pass allow_on_demand=False
    (or leave `cis_gene` unset) to only ever draw what's already precomputed.

    Returns False if nothing was drawn (row missing/lacks x_ntc,y_ntc, or no
    smoothed data for this feature).
    """
    if row is None:
        return False
    y_ntc = float(row.get('y_ntc', np.nan))
    x_ntc = float(row.get('x_ntc', np.nan))
    if not (np.isfinite(y_ntc) and y_ntc > 0 and np.isfinite(x_ntc) and x_ntc > 0):
        return False
    x_off, y_off = np.log2(x_ntc), np.log2(y_ntc)

    drew_any = False
    if feature not in smoothed['feature_index'] and allow_on_demand and cis_gene is not None:
        ensure_smoothed_curve(spec, cis_gene, smoothed, feature, device=device)
    fi = smoothed['feature_index'].get(feature)
    if fi is not None:
        for gi, glabel in enumerate(smoothed['group_labels']):
            xl, yl = smoothed['x_log2'][fi, gi], smoothed['y_log2'][fi, gi]
            valid = np.isfinite(xl) & np.isfinite(yl)
            if not valid.any():
                continue
            color = spec.cell_line_palette.get(glabel)
            ax.plot(xl[valid] - x_off, yl[valid] - y_off, color=color, linewidth=2, label=glabel)
            drew_any = True

    if show_hill_function:
        xlim = ax.get_xlim() if drew_any else (-6.0, 6.0)
        x_abs = 2 ** (np.linspace(xlim[0], xlim[1], 2000) + x_off)
        y_pred = predict_hill_from_summary_row(row, x_abs, fdr_threshold=fdr_threshold)
        if y_pred is not None:
            valid = y_pred > 0
            if valid.any():
                ax.plot(np.log2(x_abs[valid]) - x_off, np.log2(y_pred[valid]) - y_off,
                        color=hill_color or spec.color, linewidth=2, label=hill_label or spec.name)
                drew_any = True

    ax.axhline(0, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
    ax.set_xlabel('log2FC(x_true)')
    ax.set_ylabel('log2FC(y)')
    return drew_any


# ── Panel plotting ────────────────────────────────────────────────────────

def _plot_into(model, goi, ax, spec: DatasetSpec, sum_factor_col: str,
                mark_params, fdr_df: Optional[pd.DataFrame] = None,
                show_hill_function: bool = True):
    kw = dict(
        show_hill_function=show_hill_function,
        sum_factor_col=sum_factor_col,
        log2fc=True,
        show_correction='corrected',
        legend_outside=False,
        color_by='cell_line',
        color_palette=spec.cell_line_palette,
        hill_color=spec.color,
        hill_label=spec.name,
    )
    if fdr_df is not None:
        # fdr_df (as opposed to reference_df) only overrides which FDR values
        # gate THIS dataset's own curve -- it does NOT trigger plot_xy_data's
        # built-in single reference-curve overlay (that's handled separately,
        # for any number of extra datasets, by _overlay_extra_curve below).
        kw['fdr_df'] = fdr_df
    model.plot_xy_data(goi, mark_params=mark_params, ax=ax, **kw)


def _overlay_extra_curve(ax: plt.Axes, ref_row: pd.Series, spec: DatasetSpec,
                          fdr_threshold: float = 0.05) -> bool:
    """Draw one additional dataset's fitted Hill curve on `ax`, in log2FC
    space, replicating plot_negbinom_xy's own 'reference curve overlay' math
    (bayesDREAM/plotting/xy_plots.py) -- but done externally so more than one
    extra curve can be layered onto the same ax (plot_xy_data's own
    `reference_df` argument only accepts a single overlay per call, which
    isn't enough once >2 datasets are being compared at once).

    `ref_row` should come from an "allsig" copy of that dataset's trans
    summary (see allsig_copy()) so the curve renders regardless of that
    dataset's own significance -- matching the original 2-way panels' intent
    ("show the shape even where a formal FDR call is not (yet) significant").

    Returns True if a curve was actually drawn (False if the row lacked
    x_ntc/y_ntc or all predicted y were non-positive).
    """
    y_ntc = float(ref_row.get('y_ntc', np.nan)) if hasattr(ref_row, 'get') else float('nan')
    x_ntc = float(ref_row.get('x_ntc', np.nan)) if hasattr(ref_row, 'get') else float('nan')
    if not (np.isfinite(y_ntc) and y_ntc > 0 and np.isfinite(x_ntc) and x_ntc > 0):
        return False

    x_off, y_off = np.log2(x_ntc), np.log2(y_ntc)
    xlim = ax.get_xlim()  # current visible log2FC(x) range for this panel
    x_abs = 2 ** (np.linspace(xlim[0], xlim[1], 2000) + x_off)
    y_pred = predict_hill_from_summary_row(ref_row, x_abs, fdr_threshold=fdr_threshold)
    if y_pred is None:
        return False
    valid = y_pred > 0
    if not valid.any():
        return False

    ax.plot(np.log2(x_abs[valid]) - x_off, np.log2(y_pred[valid]) - y_off,
            color=spec.color, linestyle='--', linewidth=2, alpha=0.8, label=spec.name)
    return True


def _lookup_row(summary_allsig: pd.DataFrame, gene_id: str) -> Optional[pd.Series]:
    """`gene_id` is the canonical Ensembl ID (see allsig_copy()) -- matching
    on this instead of 'gene_name'/'feature' is what makes this work across
    Domingo/Morris (symbol-indexed) and Replogle (Ensembl-ID-indexed) at
    once.
    """
    match = summary_allsig.loc[summary_allsig['gene_id'] == gene_id]
    return match.iloc[0] if not match.empty else None


def _native_feature(summary_allsig: pd.DataFrame, gene_id: str) -> Optional[str]:
    """This dataset's OWN feature identifier for `gene_id` (Ensembl) -- the
    symbol for Domingo/Morris, the Ensembl ID itself for Replogle.
    plot_xy_data()/_get_feature_index() (bayesDREAM/plotting/xy_plots.py)
    resolve features by a model's own native identifier and explicitly do
    NOT check 'gene_name'/'gene_id' columns, so the canonical `gene_id` used
    for cross-dataset matching must be translated back to this before being
    handed to model.plot_xy_data().
    """
    row = _lookup_row(summary_allsig, gene_id)
    return None if row is None else row['feature']


def make_panel(
    goi: str,
    specs: List[DatasetSpec], models: list, summaries_allsig: List[pd.DataFrame], sfcols: List[str],
    *, cis_gene: str, show_param_markers: bool = True, fdr_threshold: float = 0.05,
    figsize_per: Tuple[float, float] = (3.6, 3.0), display_name: Optional[str] = None,
) -> Tuple[plt.Figure, Tuple[float, float]]:
    """2xN panel (N = len(specs)): row 0 is each dataset standalone (own data
    + own curve, with fitted-parameter markers if show_param_markers); row 1
    is each dataset's own data + own curve again, with every *other*
    dataset's curve overlaid on top (so for N=3, up to 2 extra curves per
    row-1 subplot). For N=2 this reduces to the original 2x2 design.

    `goi` is the canonical gene_id (Ensembl -- see allsig_copy()), not a raw
    'feature'/symbol: a single shared identifier can't be handed directly to
    every dataset's model (Domingo/Morris resolve features by symbol,
    Replogle by Ensembl ID), so it's translated to each dataset's own native
    identifier via _native_feature() before being passed to plot_xy_data().
    Raises KeyError if `goi` is missing from any of `specs` -- shouldn't
    happen when called from compare_datasets() (which only ever passes a
    gene_id already confirmed present in every participating dataset's
    summary), but is a real bug (not silently-droppable) if it does.

    `display_name` (defaults to `goi` itself) is used for the figure title
    only -- pass the human-readable symbol so a panel doesn't show a bare
    Ensembl ID.

    show_param_markers controls whether row-0 standalone panels draw the
    fitted parameter markers (EC50/inflection/etc as small annotated lines)
    -- set False if they read as confusing next to the dataset-colour curves.
    """
    n = len(specs)
    assert n >= 2, "make_panel needs at least 2 datasets to compare"

    native = [_native_feature(summaries_allsig[j], goi) for j in range(n)]
    missing_j = [j for j, f in enumerate(native) if f is None]
    if missing_j:
        raise KeyError(
            f"gene_id {goi!r} not present in {[specs[j].name for j in missing_j]}'s trans "
            f"summary. make_panel() expects `goi` to already be a gene_id every dataset in "
            f"`specs` shares -- compare_datasets() only calls this after intersecting on gene_id."
        )

    fig, axes = plt.subplots(2, n, figsize=(figsize_per[0] * n, figsize_per[1] * 2),
                              constrained_layout=True, squeeze=False)

    mark = 'fit' if show_param_markers else False

    # Row 0: standalone (real per-dataset FDR gating, as usual).
    for j in range(n):
        _plot_into(models[j], native[j], axes[0][j], specs[j], sfcols[j], mark_params=mark)
        axes[0][j].set_title(specs[j].name)

    # Row 1: dataset j's own data + own curve (force-rendered regardless of
    # dataset j's own significance, via fdr_df=its own allsig summary -- same
    # "always show the shape" intent as the original 2-way overlay panels),
    # plus every other dataset's curve manually overlaid on top.
    for j in range(n):
        ax = axes[1][j]
        _plot_into(models[j], native[j], ax, specs[j], sfcols[j], mark_params=False,
                   fdr_df=summaries_allsig[j])
        overlaid_names = []
        for k in range(n):
            if k == j:
                continue
            row = _lookup_row(summaries_allsig[k], goi)
            if row is not None and _overlay_extra_curve(ax, row, specs[k], fdr_threshold=fdr_threshold):
                overlaid_names.append(specs[k].name)
        title = specs[j].name if not overlaid_names else f"{specs[j].name} + {' + '.join(overlaid_names)}"
        ax.set_title(title)
        # Rebuild this ax's own legend from scratch: plot_xy_data already drew
        # one internally, but the manually-added extra curves above were
        # plotted afterwards and aren't in it yet. get_legend_handles_labels()
        # picks up every labeled artist on the ax regardless, old and new.
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        handles, labels = ax.get_legend_handles_labels()
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                h2.append(h)
                l2.append(l)
        if h2:
            ax.legend(h2, l2, fontsize=7, frameon=False)

    # Unified axis limits, derived from the row-1 (fully-overlaid) panels so
    # row-0's parameter markers get clipped rather than expanding the view.
    xlims = [axes[1][j].get_xlim() for j in range(n)]
    ylims = [axes[1][j].get_ylim() for j in range(n)]
    unified_x = (min(x[0] for x in xlims), max(x[1] for x in xlims))
    unified_y = (min(y[0] for y in ylims), max(y[1] for y in ylims))
    for ax in axes.ravel():
        ax.set_xlim(unified_x)
        ax.set_ylim(unified_y)

    fig.suptitle(f'{cis_gene} → {display_name or goi}', fontsize=11, fontweight='bold')

    # Single shared legend (dedup across all 2N subplots' own legends), then
    # drop the per-subplot ones so it isn't shown twice.
    seen, handles, labels = set(), [], []
    for ax in axes.ravel():
        leg = ax.get_legend()
        if leg is None:
            continue
        for h, t in zip(leg.legend_handles, [t.get_text() for t in leg.get_texts()]):
            if t not in seen:
                seen.add(t)
                handles.append(h)
                labels.append(t)
        leg.remove()
    if handles:
        fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.5), loc='center left',
                   frameon=False, fontsize=8)

    return fig, unified_x


def make_panel_lightweight(
    goi: str,
    specs: List[DatasetSpec], summaries: List[pd.DataFrame], summaries_allsig: List[pd.DataFrame],
    smoothed_list: List[Dict[str, object]],
    *, cis_gene: str, fdr_threshold: float = 0.05,
    figsize_per: Tuple[float, float] = (3.6, 3.0), display_name: Optional[str] = None,
) -> Tuple[plt.Figure, Tuple[float, float]]:
    """Model-free equivalent of make_panel(): same 2xN row-0 (standalone,
    real per-dataset FDR gating)/row-1 (own data force-rendered + every
    other dataset's curve overlaid) layout and intent, but every trend/curve
    is drawn from a precomputed smoothed array (compute_smoothed_curves()/
    load_smoothed_curves()) + a trans_feature_summary row
    (plot_gene_lightweight()) instead of a live model + plot_xy_data(). No
    raw counts or trans-fit posterior are ever touched, so unlike
    compare_datasets()/make_panel(), the caller can iterate this over a
    dataset's ENTIRE trans panel -- see compare_datasets_lightweight().

    `summaries` (real FDR values) drives row-0's standalone significance
    gating -- these are the same fdr_alpha/fdr_beta columns
    save_trans_summary() itself wrote to trans_feature_summary_{modality}.csv,
    so this gates identically to make_panel()'s row-0 (which re-derives the
    same values live from the model), just read from disk instead. Does NOT
    support show_param_markers (row-0 EC50/inflection annotations) -- those
    need the live posterior's per-sample derivative/root-finding, not
    available from a summary row alone.

    `goi`/`display_name` and the missing-gene KeyError behavior match
    make_panel() exactly (see its docstring) -- `summaries_allsig` is what's
    actually intersected/looked-up by gene_id.
    """
    n = len(specs)
    assert n >= 2, "make_panel_lightweight needs at least 2 datasets to compare"

    native = [_native_feature(summaries_allsig[j], goi) for j in range(n)]
    missing_j = [j for j, f in enumerate(native) if f is None]
    if missing_j:
        raise KeyError(
            f"gene_id {goi!r} not present in {[specs[j].name for j in missing_j]}'s trans "
            f"summary. make_panel_lightweight() expects `goi` to already be a gene_id every "
            f"dataset in `specs` shares -- compare_datasets_lightweight() only calls this "
            f"after intersecting on gene_id."
        )

    def _real_row(j: int) -> Optional[pd.Series]:
        match = summaries[j].loc[summaries[j]['feature'] == native[j]]
        return match.iloc[0] if not match.empty else None

    real_rows = [_real_row(j) for j in range(n)]
    allsig_rows = [_lookup_row(summaries_allsig[j], goi) for j in range(n)]

    fig, axes = plt.subplots(2, n, figsize=(figsize_per[0] * n, figsize_per[1] * 2),
                              constrained_layout=True, squeeze=False)

    # Row 0: standalone, real per-dataset FDR gating (from the row's own
    # on-disk fdr_alpha/fdr_beta -- see docstring above).
    for j in range(n):
        plot_gene_lightweight(axes[0][j], native[j], real_rows[j], smoothed_list[j], specs[j],
                               cis_gene=cis_gene, fdr_threshold=fdr_threshold, hill_color=specs[j].color,
                               hill_label=specs[j].name)
        axes[0][j].set_title(specs[j].name)

    # Row 1: dataset j's own data + own curve (force-rendered via the allsig
    # row, same "always show the shape" intent as make_panel()), plus every
    # other dataset's curve overlaid on top via _overlay_extra_curve (already
    # model-free -- it only ever reads a summary row).
    for j in range(n):
        ax = axes[1][j]
        plot_gene_lightweight(ax, native[j], allsig_rows[j], smoothed_list[j], specs[j],
                               cis_gene=cis_gene, fdr_threshold=fdr_threshold, hill_color=specs[j].color,
                               hill_label=specs[j].name)
        overlaid_names = []
        for k in range(n):
            if k == j:
                continue
            if allsig_rows[k] is not None and _overlay_extra_curve(ax, allsig_rows[k], specs[k],
                                                                     fdr_threshold=fdr_threshold):
                overlaid_names.append(specs[k].name)
        title = specs[j].name if not overlaid_names else f"{specs[j].name} + {' + '.join(overlaid_names)}"
        ax.set_title(title)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        handles, labels = ax.get_legend_handles_labels()
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                h2.append(h)
                l2.append(l)
        if h2:
            ax.legend(h2, l2, fontsize=7, frameon=False)

    xlims = [axes[1][j].get_xlim() for j in range(n)]
    ylims = [axes[1][j].get_ylim() for j in range(n)]
    unified_x = (min(x[0] for x in xlims), max(x[1] for x in xlims))
    unified_y = (min(y[0] for y in ylims), max(y[1] for y in ylims))
    for ax in axes.ravel():
        ax.set_xlim(unified_x)
        ax.set_ylim(unified_y)

    fig.suptitle(f'{cis_gene} → {display_name or goi}', fontsize=11, fontweight='bold')

    seen, handles, labels = set(), [], []
    for ax in axes.ravel():
        leg = ax.get_legend()
        if leg is None:
            continue
        for h, t in zip(leg.legend_handles, [t.get_text() for t in leg.get_texts()]):
            if t not in seen:
                seen.add(t)
                handles.append(h)
                labels.append(t)
        leg.remove()
    if handles:
        fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.5), loc='center left',
                   frameon=False, fontsize=8)

    return fig, unified_x


# ── Cis-side guide-density panel ─────────────────────────────────────────────

def _get_x_ntc_log2(model) -> float:
    try:
        cis_mod = model.get_modality('cis')
        psn = getattr(cis_mod, 'posterior_samples_ntc', None)
        if psn is not None and 'mu_ntc' in psn:
            mu = np.asarray(psn['mu_ntc']).mean()
            if np.isfinite(mu) and mu > 0:
                return float(np.log2(mu))
    except Exception:
        pass
    ntc = model.meta.loc[model.meta['target'] == 'ntc', 'x_true']
    ntc = ntc[ntc > 0]
    return float(np.log2(ntc.mean())) if len(ntc) > 0 else 0.0


def _expand_cell_guide_data(model, meta_filtered: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, targeting guide) for per-guide KDE plotting.
    High-MOI: reads model.guide_assignment directly. Low-MOI: parses the
    'guide' string column.
    """
    if getattr(model, 'is_high_moi', False) and model.guide_assignment is not None:
        guide_names = model.guide_meta['guide'].tolist()
        gtd = getattr(model, 'guide_targets_dict', {})
        ntc_guides = {g for g, targets in gtd.items() if all(str(t).lower() == 'ntc' for t in targets)}

        ga = model.guide_assignment
        cell_to_idx = {cell: i for i, cell in enumerate(model.meta['cell'])}

        expanded = []
        for _, row in meta_filtered.iterrows():
            ci = cell_to_idx.get(row['cell'])
            if ci is None:
                continue
            active = [guide_names[j] for j in np.where(ga[ci] > 0)[0]]
            targeting = [g for g in active if g not in ntc_guides]
            if not targeting:
                r = row.to_dict()
                r['eff_guide'] = 'ntc'
                expanded.append(r)
            else:
                for g in targeting:
                    r = row.to_dict()
                    r['eff_guide'] = g
                    expanded.append(r)

        if not expanded:
            return meta_filtered.assign(eff_guide='unknown')
        return pd.DataFrame(expanded)

    gtd = getattr(model, 'guide_targets_dict', None)
    ntc_names = set()
    if gtd:
        ntc_names = {g for g, targets in gtd.items() if all(str(t).lower() == 'ntc' for t in targets)}

    def _parse(g):
        if pd.isna(g):
            return 'unknown'
        g = str(g)
        for sep in ('+', ',', '|', ';'):
            if sep in g:
                parts = [p.strip() for p in g.split(sep)]
                tgt = [p for p in parts if p not in ntc_names and 'ntc' not in p.lower()]
                if len(tgt) == 1:
                    return tgt[0]
                if len(tgt) > 1:
                    return '+'.join(sorted(tgt))
                return parts[0]
        return g

    out = meta_filtered.copy()
    out['eff_guide'] = out.get('guide', pd.Series('unknown', index=out.index)).map(_parse)
    return out


def make_density_panel(cis_gene: str, specs_and_models: List[Tuple[DatasetSpec, "bayesDREAM"]],
                        unified_x: Tuple[float, float], figsize_per=(4.5, 3)) -> plt.Figure:
    """1xN panel of log2FC(x_true) KDE density curves, one column per model
    (x_true is the cis gene's own expression -- constant across trans genes,
    so this panel is produced once per (cis_gene, dataset set), not per gene).
    """
    n = len(specs_and_models)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per[0] * n, figsize_per[1]), constrained_layout=True)
    axes = np.atleast_1d(axes)
    x_grid = np.linspace(unified_x[0], unified_x[1], 500)

    def _kde_plot(ax, vals, color, alpha_fill=0.15, lw=1.2):
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            return
        y = gaussian_kde(vals, bw_method='scott')(x_grid)
        ax.fill_between(x_grid, 0, y, alpha=alpha_fill, color=color)
        ax.plot(x_grid, y, color=color, linewidth=lw)

    for ax, (spec, model) in zip(axes, specs_and_models):
        x_ntc_log2 = _get_x_ntc_log2(model)
        meta = model.meta.copy()
        if 'x_true' not in meta.columns:
            xt = getattr(model, 'x_true', None)
            if xt is None:
                raise ValueError(f"x_true not found for {spec.name}")
            meta['x_true'] = np.asarray(xt.cpu() if hasattr(xt, 'cpu') else xt)
        meta = meta[meta['x_true'] > 0].copy()
        meta['log2fc_x'] = np.log2(meta['x_true']) - x_ntc_log2

        expanded = _expand_cell_guide_data(model, meta)
        legend_patches = []

        ntc_vals = expanded.loc[expanded['target'] == 'ntc', 'log2fc_x'].values
        if len(ntc_vals[np.isfinite(ntc_vals)]) >= 2:
            _kde_plot(ax, ntc_vals, color='gray')
            legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.5, label='NTC'))

        tgt = expanded[expanded['target'] != 'ntc']
        crispr_i = sorted(tgt.loc[tgt['cell_line'] == 'CRISPRi', 'eff_guide'].unique())
        crispr_a = sorted(tgt.loc[tgt['cell_line'] == 'CRISPRa', 'eff_guide'].unique()) if 'CRISPRa' in spec.cell_line_palette else []

        if crispr_i:
            blues = plt.cm.Blues(np.linspace(0.4, 0.85, max(len(crispr_i), 1)))
            for j, guide in enumerate(crispr_i):
                vals = tgt.loc[(tgt['eff_guide'] == guide) & (tgt['cell_line'] == 'CRISPRi'), 'log2fc_x'].values
                _kde_plot(ax, vals, color=blues[j])
            legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc=plt.cm.Blues(0.65), alpha=0.6, label='CRISPRi'))

        if crispr_a:
            reds = plt.cm.Reds(np.linspace(0.4, 0.85, max(len(crispr_a), 1)))
            for j, guide in enumerate(crispr_a):
                vals = tgt.loc[(tgt['eff_guide'] == guide) & (tgt['cell_line'] == 'CRISPRa'), 'log2fc_x'].values
                _kde_plot(ax, vals, color=reds[j])
            legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc=plt.cm.Reds(0.65), alpha=0.6, label='CRISPRa'))

        ax.set_xlim(unified_x)
        ax.set_xlabel('log2FC(x_true)')
        ax.set_ylabel('Density')
        ax.set_title(spec.name)
        ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.legend(handles=legend_patches, frameon=False, fontsize=8)

    fig.suptitle(f'{cis_gene}  |  guide distributions', fontsize=10)
    return fig


# ── Top-level orchestrator ───────────────────────────────────────────────────

def _tag(specs: List[DatasetSpec]) -> str:
    return '_vs_'.join(s.name for s in specs)


def _missing_exports(specs: List[DatasetSpec], cis_gene: str) -> List[Tuple[DatasetSpec, str]]:
    """Cheap upfront check (no model loading) of which `specs` are missing
    their save_model_for_plotting() export for `cis_gene`. Returns
    [(spec, error_message), ...] for each one that's missing (empty if all
    present). Used to fail fast / fall back gracefully instead of fully
    reloading N-1 (potentially large) models before discovering the Nth is
    missing.
    """
    missing = []
    for spec in specs:
        try:
            spec.plotting_save_dir(cis_gene)
        except (FileNotFoundError, ValueError) as e:
            missing.append((spec, str(e)))
    return missing


def compare_datasets(
    specs: List[DatasetSpec], cis_gene: str,
    *, out_dir: str = './dose_response_plots', genes: Optional[List[str]] = None,
    show_param_markers: bool = True, device: Optional[str] = None,
    panel_figsize_per: Tuple[float, float] = (3.6, 3.0), lean: bool = True,
) -> List[str]:
    """Full pipeline for N (>=2) datasets: load all N *live models* for
    `cis_gene`, summarise each, find trans genes present in *every*
    dataset's summary (or use `genes` if given), and write one 2xN panel PNG
    per gene plus one guide-density panel to `out_dir`. Panel width
    auto-scales with N.

    This is the heavy, full-model path -- every trans gene's full posterior
    for every dataset in `specs` is loaded, which for Morris/Replogle's
    transcriptome-wide panels is large even with `lean=True` (see
    load_model_for_plotting()'s docstring: lean never touches the trans-fit
    posterior). For interactively plotting genes across a dataset's FULL
    panel on demand, use compare_datasets_lightweight() instead, which loads
    only each dataset's trans_feature_summary CSV + a small precomputed
    smoothed-curve artifact -- no model, no raw counts, no trans-fit
    posterior at all. This function stays useful for genuinely needing the
    live posterior (e.g. show_param_markers below, which needs per-sample
    root-finding not available from a summary row alone), or for a short
    hand-picked `genes` list where the model-reload cost is bounded anyway.

    Genes are matched across datasets by Ensembl gene_id, not raw 'feature'
    -- Domingo/Morris's own 'feature' column already is the gene symbol,
    Replogle's is the Ensembl ID itself; see allsig_copy()'s docstring for
    how the two get reconciled via comparative.datasets.morris_symbol_to_id().
    `genes`, if given, is also treated as a symbol/gene_id list and
    translated the same way (an already-Ensembl-ID entry passes through
    unchanged) -- pass real gene_ids directly if you already have them.

    Checks that every spec's save_model_for_plotting() export actually
    exists BEFORE loading any model -- so a missing dataset fails immediately
    rather than after wastefully reloading the others (which, for a
    transcriptome-wide dataset like Morris, can take a while). If you'd
    rather silently drop whatever's missing and continue with the rest, use
    compare_all_domingo_cis_genes() (or pre-filter `specs` yourself via
    _missing_exports()).

    lean : bool
        Passed through to load_model_for_plotting() for every dataset --
        see its docstring for what this does and doesn't cover (NTC/cis
        only, not trans).

    Returns the list of trans genes actually plotted (display symbols, not
    gene_ids).
    """
    assert len(specs) >= 2, "compare_datasets needs at least 2 datasets"
    names = [s.name for s in specs]
    tag = _tag(specs)

    missing = _missing_exports(specs, cis_gene)
    if missing:
        lines = "\n".join(f"  - {msg}" for _, msg in missing)
        raise FileNotFoundError(
            f"Cannot compare {names} for {cis_gene!r}: {len(missing)}/{len(specs)} dataset(s) "
            f"missing their save_model_for_plotting() export (checked before loading any "
            f"models, to avoid wasting time loading the others):\n{lines}"
        )

    models = []
    for spec in specs:
        print(f"[{spec.name}] loading model for {cis_gene}...")
        models.append(load_model_for_plotting(spec, cis_gene, device=device, lean=lean))

    sfcols = [resolve_sum_factor_col(s, m) for s, m in zip(specs, models)]
    for s, sf in zip(specs, sfcols):
        print(f"[{s.name}] plotting sum_factor_col={sf!r}")

    # compute_derivative_roots isn't needed for dose-response curve plotting
    # (only for finding local optima elsewhere) -- skip it everywhere for speed.
    summaries = []
    for spec, model in zip(specs, models):
        print(f"[{spec.name}] summarising trans fit...")
        summaries.append(model.save_trans_summary(compute_lfc_ci=False, compute_derivative_roots=False))
    summaries_allsig = [allsig_copy(s, spec) for s, spec in zip(summaries, specs)]

    if genes is None:
        gene_sets = [set(s['gene_id'].dropna()) for s in summaries_allsig]
        gene_ids = sorted(set.intersection(*gene_sets))
    else:
        sym_to_id = morris_symbol_to_id()
        gene_ids = sorted({sym_to_id.get(g, g) for g in genes})
    id_to_symbol = morris_id_to_symbol()
    print(f"\n{' vs '.join(names)} ({cis_gene}): {len(gene_ids)} trans genes to plot")

    os.makedirs(out_dir, exist_ok=True)

    unified_x_all = None
    plotted = []
    for i, gid in enumerate(gene_ids, 1):
        display = id_to_symbol.get(gid, gid)
        print(f"  [{i}/{len(gene_ids)}] {display}", end='', flush=True)
        fig, unified_x = make_panel(
            gid, specs, models, summaries_allsig, sfcols,
            cis_gene=cis_gene, show_param_markers=show_param_markers, figsize_per=panel_figsize_per,
            display_name=display,
        )
        fig.savefig(os.path.join(out_dir, f'{cis_gene}_{tag}_{display}_panel.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        plotted.append(display)
        unified_x_all = unified_x if unified_x_all is None else (
            min(unified_x_all[0], unified_x[0]), max(unified_x_all[1], unified_x[1]))
        print("  done")

    if unified_x_all is not None:
        print("Plotting guide density panel...", end='', flush=True)
        fig_density = make_density_panel(cis_gene, list(zip(specs, models)), unified_x_all)
        fig_density.savefig(os.path.join(out_dir, f'{cis_gene}_{tag}_guide_density.png'),
                             dpi=150, bbox_inches='tight')
        plt.close(fig_density)
        print("  done")

    print(f"\nDone. {len(plotted)} panels + density plot written to {out_dir}/")
    return plotted


def compare_datasets_lightweight(
    specs: List[DatasetSpec], cis_gene: str,
    *, out_dir: str = './dose_response_plots', genes: Optional[List[str]] = None,
    fdr_threshold: float = 0.05, panel_figsize_per: Tuple[float, float] = (3.6, 3.0),
) -> List[str]:
    """Model-free equivalent of compare_datasets(): for each dataset in
    `specs`, loads ONLY its trans_feature_summary CSV (load_gene_summary(),
    already required for every completed fit_trans run) and its precomputed
    smoothed-curve artifact (load_smoothed_curves(), written by
    reconstruct_export.py/reconstruct_export_replogle.py's backfill) --
    never a full model, raw counts, or trans-fit posterior. Both are cheap
    (tens of MB) even for Morris/Replogle's full ~10-20k-gene panel, so
    unlike compare_datasets() this scales to plotting the ENTIRE shared
    trans-gene set (or any `genes` subset) on demand -- the intended default
    for interactive use in the notebook.

    Same 2xN panel semantics as compare_datasets()/make_panel() (see
    make_panel_lightweight()'s docstring for the one real difference: row-0
    gating reads on-disk FDR values instead of live per-sample posteriors,
    and show_param_markers isn't available). Unlike compare_datasets(), does
    NOT write a guide-density panel -- that needs per-cell x_true/guide data,
    which this path never loads.

    Raises FileNotFoundError up front (via load_smoothed_curves()) if any
    dataset in `specs` hasn't had its smoothed-curve artifact computed yet.
    """
    assert len(specs) >= 2, "compare_datasets_lightweight needs at least 2 datasets"
    names = [s.name for s in specs]
    tag = _tag(specs)

    print(f"[{' vs '.join(names)}] {cis_gene}: loading trans summaries + smoothed curves...")
    summary_pairs = [load_gene_summary(spec, cis_gene) for spec in specs]
    summaries = [p[0] for p in summary_pairs]
    summaries_allsig = [p[1] for p in summary_pairs]
    smoothed_list = [load_smoothed_curves(spec, cis_gene) for spec in specs]

    if genes is None:
        gene_sets = [set(s['gene_id'].dropna()) for s in summaries_allsig]
        gene_ids = sorted(set.intersection(*gene_sets))
    else:
        sym_to_id = morris_symbol_to_id()
        gene_ids = sorted({sym_to_id.get(g, g) for g in genes})
    id_to_symbol = morris_id_to_symbol()
    print(f"{' vs '.join(names)} ({cis_gene}): {len(gene_ids)} trans genes to plot")

    os.makedirs(out_dir, exist_ok=True)

    plotted = []
    for i, gid in enumerate(gene_ids, 1):
        display = id_to_symbol.get(gid, gid)
        print(f"  [{i}/{len(gene_ids)}] {display}", end='', flush=True)
        fig, _ = make_panel_lightweight(
            gid, specs, summaries, summaries_allsig, smoothed_list,
            cis_gene=cis_gene, fdr_threshold=fdr_threshold, figsize_per=panel_figsize_per,
            display_name=display,
        )
        fig.savefig(os.path.join(out_dir, f'{cis_gene}_{tag}_{display}_panel.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        plotted.append(display)
        print("  done")

    print(f"\nDone. {len(plotted)} panels written to {out_dir}/ (no guide-density panel -- "
          f"lightweight path never loads per-cell data).")
    return plotted


def compare_pair(
    spec_a: DatasetSpec, spec_b: DatasetSpec, cis_gene: str, **kwargs,
) -> List[str]:
    """Thin 2-dataset convenience wrapper around compare_datasets()."""
    return compare_datasets([spec_a, spec_b], cis_gene, **kwargs)


def compare_all_domingo_cis_genes(
    *, out_dir: str = './dose_response_plots', datasets: Optional[List[DatasetSpec]] = None,
    bounding_dataset: Optional[DatasetSpec] = None, require_all_exports: bool = True,
    lightweight: bool = False, **kwargs,
) -> Dict[str, List[str]]:
    """Automate compare_datasets() (or compare_datasets_lightweight(), see
    `lightweight`) across every cis gene, using whichever subset of
    `datasets` (default: [DOMINGO, MORRIS, REPLOGLE]) structurally has a
    completed fit_trans run for that gene (per each DatasetSpec.cis_genes)
    -- so GFI1B/NFE2 (all 3 fit) get a 2x3 panel, while TET2/MYB (Morris
    never fit these; see publication_runs/morris/config.yaml's primary_genes)
    are Domingo-vs-Replogle from the start. That drop is expected/structural,
    not an error.

    lightweight : bool
        If True, calls compare_datasets_lightweight() instead of
        compare_datasets() for each gene -- no model/raw counts/trans-fit
        posterior loaded, so this scales to plotting every trans gene in
        every participating dataset's full panel, not just a hand-picked
        subset (see that function's docstring). require_all_exports still
        gates on save_model_for_plotting()'s export existing (that's where
        the precomputed smoothed-curve artifact + trans_feature_summary CSV
        live) -- it does not separately check for the smoothed-curve file;
        compare_datasets_lightweight() raises its own clear error if that's
        missing for a dataset whose model export otherwise exists.

    require_all_exports : bool
        If True (default), raise immediately if a dataset that DOES list a
        given cis gene in its own cis_genes is still missing its
        save_model_for_plotting() export -- run
        comparative.reconstruct_export(_replogle).reconstruct_and_export_all()
        first. Pass False to instead silently drop that dataset from the
        gene's panel (the old, pre-automation behavior, for partial/manual
        adoption before every gene has been exported).

    The cis gene list iterated is `bounding_dataset.cis_genes` (default: the
    first dataset in `datasets`, i.e. Domingo).

    Writes into `out_dir/<cis_gene>/`. Returns {cis_gene: [genes plotted]}.
    """
    from .datasets import DOMINGO, MORRIS, REPLOGLE
    datasets = datasets or [DOMINGO, MORRIS, REPLOGLE]
    bounding_dataset = bounding_dataset or datasets[0]
    compare_fn = compare_datasets_lightweight if lightweight else compare_datasets

    results = {}
    for cis_gene in bounding_dataset.cis_genes:
        participating = [s for s in datasets if cis_gene in s.cis_genes]
        if len(participating) < 2:
            print(f"=== {cis_gene}: only {[s.name for s in participating]} has a completed fit -- skipping ===")
            continue

        missing = _missing_exports(participating, cis_gene)
        if missing:
            lines = "\n".join(f"  - {msg}" for _, msg in missing)
            if require_all_exports:
                raise FileNotFoundError(
                    f"{cis_gene}: {len(missing)} dataset(s) have a completed fit_trans but no "
                    f"save_model_for_plotting() export yet -- run "
                    f"comparative.reconstruct_export(_replogle).reconstruct_and_export_all() "
                    f"first, or pass require_all_exports=False to drop them instead:\n{lines}"
                )
            missing_names = {spec.name for spec, _ in missing}
            participating = [s for s in participating if s.name not in missing_names]
            print(f"    ({cis_gene}: {sorted(missing_names)} dropped -- no export yet)")
            if len(participating) < 2:
                print(f"=== {cis_gene}: only {[s.name for s in participating]} usable -- skipping ===")
                continue

        print(f"=== {cis_gene}: {[s.name for s in participating]} ===")
        results[cis_gene] = compare_fn(
            participating, cis_gene, out_dir=os.path.join(out_dir, cis_gene), **kwargs,
        )
    return results
