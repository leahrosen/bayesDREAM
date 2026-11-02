"""
Shared helpers for the fix_xtrue test notebooks (x_true uncertainty in fit_trans).

Each notebook rebuilds one production (publication_runs) trans model -- same
data subset, NTC/cis fits and sum-factor adjustments as the production
trans stage (mirrors publication_runs/common/run_trans.py) -- then refits ONLY
fit_trans under several x-input conditions (fit_trans(x_level=..., x_uncertainty=...))
into a separate test output directory, and plots the fits side by side.
Production outputs are only read, never written.
"""
import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "publication_runs" / "common"))
sys.path.insert(0, str(REPO))

from config_utils import (  # noqa: E402
    apply_device_override,
    apply_sum_factor_adjustments,
    build_model_from_config,
    is_enabled,
    load_bayesdream_yaml,
    normalize_stage_args,
    read_table,
)
from bayesDREAM.utils import make_names_unique  # noqa: E402

# condition name -> fit_trans x-input kwargs
CONDITIONS = {
    "cell_median": dict(x_level="cell", x_uncertainty="median"),   # old behaviour; not recommended
    "cell_sample": dict(x_level="cell", x_uncertainty="sample"),   # fit_trans default; recommended
    "guide_sample": dict(x_level="guide", x_uncertainty="sample"), # not recommended
}
COND_COLORS = {"cell_median": "#7f7f7f", "cell_sample": "#1f77b4", "guide_sample": "#d62728",
               "guide_median": "#ff7f0e"}


def production_config(dataset, label, stage="trans"):
    """Path of a rendered production stage config (publication_runs/<dataset>/slurm/configs)."""
    path = REPO / "publication_runs" / dataset / "slurm" / "configs" / f"{label}_{stage}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Production config not found: {path}")
    return path


def build_trans_model(cfg_path, device, model_overrides=None, legacy_feature_names_col=None):
    """
    Rebuild the production trans-stage model up to (not including)
    exclude_trans_genes/fit_trans: build -> load_ntc_fit -> load_cis_fit ->
    technical groups -> sum-factor adjustments. Returns (model, cfg).

    model_overrides: keys replaced in the config's `model:` block before
    building (e.g. guide_covariates=[] so the guide panel matches a cis fit
    made when high-MOI guide_covariates were still ignored).

    legacy_feature_names_col: name features as pre-2026-09-09 bayesDREAM did --
    this feature_meta column with repeats disambiguated scanpy-style
    (X, X-1, ...) -- so they match fits saved by that code. Current code
    rejects duplicates in feature_name_col, so the list is passed explicitly
    as feature_names.
    """
    cfg = load_bayesdream_yaml(Path(cfg_path))
    apply_device_override(cfg, str(device))
    if model_overrides:
        for k, v in model_overrides.items():
            print(f"[override] model.{k}: {cfg['model'].get(k)!r} -> {v!r}")
        cfg["model"].update(model_overrides)
    if legacy_feature_names_col:
        data_cfg = cfg["data"]
        fm = read_table(data_cfg["feature_meta"],
                        data_cfg.get("feature_meta_read_csv_kwargs", {"index_col": 0}))
        names = make_names_unique(fm[legacy_feature_names_col].tolist())
        n_dup = sum(a != b for a, b in zip(names, fm[legacy_feature_names_col].astype(str)))
        print(f"[override] model.feature_names: {legacy_feature_names_col!r} with "
              f"{n_dup} duplicate(s) suffixed -1, -2, ... (pre-2026-09-09 naming)")
        cfg["model"].pop("feature_name_col", None)
        cfg["model"]["feature_names"] = names
    model = build_model_from_config(cfg)

    trans_cfg = cfg.get("trans") or {}
    ntc_cfg = cfg.get("ntc") or cfg.get("technical") or {}
    if is_enabled(trans_cfg.get("load_ntc"), default=True):
        model.load_ntc_fit(**normalize_stage_args(trans_cfg.get("load_ntc")))
        _check_ntc_alignment(model)
    # full (non-lean) cis posterior: x_uncertainty='sample' needs the raw draws
    model.load_cis_fit(**normalize_stage_args(trans_cfg.get("load_cis")))
    if "technical_group_code" not in model.meta.columns:
        covariates = ntc_cfg.get("set_technical_groups")
        if covariates:
            model.set_technical_groups(covariates)
    apply_sum_factor_adjustments(model, cfg.get("sum_factor") or {})
    return model, cfg


def _check_ntc_alignment(model, min_matched=0.9):
    """
    load_ntc_fit aligns the saved NTC posterior to the model's features by name
    and, with mask_features=True, silently fills unmatched features with the
    baseline (1.0 -- for every key, mu_ntc included). If the model's feature_ids
    use a different identifier than the fit (e.g. Ensembl vs gene symbol), almost
    nothing matches; stop instead of fitting with alpha_y = mu_ntc = 1.
    """
    mod = model.get_modality(model.primary_modality)
    mask = getattr(mod, "fitted_feature_mask", None)
    if mask is None:
        return
    mask = np.asarray(mask, dtype=bool)
    frac = mask.mean()
    print(f"NTC fit matched {mask.sum()}/{mask.size} features by name ({frac:.1%})")
    if frac < min_matched:
        raise ValueError(
            f"Only {frac:.1%} of the model's features match the saved NTC fit by name -- "
            f"feature_ids look like {list(mod.feature_ids[:3])}. The model is probably using a "
            "different identifier than the fit (set model_overrides={'feature_name_col': ...}).")


def build_modality_model(cfg_path, device):
    """
    Rebuild a Domingo production modality-stage model (mirrors
    publication_runs/domingo/load_modalities.py main(), before the modality is
    attached): build -> load_cis_fit -> technical groups. Returns (model, cfg).
    """
    cfg = load_bayesdream_yaml(Path(cfg_path))
    apply_device_override(cfg, str(device))
    model = build_model_from_config(cfg)
    cis_cfg = cfg.get("cis") or {}
    ntc_cfg = cfg.get("ntc") or cfg.get("technical") or {}
    model.load_cis_fit(**normalize_stage_args(cis_cfg.get("load_cis")))
    if "technical_group_code" not in model.meta.columns:
        covariates = ntc_cfg.get("set_technical_groups")
        if covariates:
            model.set_technical_groups(covariates)
    return model, cfg


def production_dir(model_or_cfg):
    """<output_dir>/<label> of a production config (where its fits live)."""
    model_cfg = model_or_cfg.get("model") or {}
    return os.path.join(model_cfg.get("output_dir", "output"), model_cfg["label"])


def x_uncertainty_report(model):
    """
    How much of the x-axis ordering is cis-posterior noise?

    Per cell: posterior SD of log2 x_true. Compared with the spread of the
    per-cell medians within each guide (the ordering fit_trans used to treat as
    exact) and across guides.
    """
    ps = model.posterior_samples_cis
    log2_draws = ps["log_x_true"].detach().float().cpu()
    log2_draws = log2_draws.reshape(log2_draws.shape[0], -1)               # [S, N]
    med = log2_draws.median(dim=0).values.numpy()   # torch (lower) median, as x_true uses
    log2_draws = log2_draws.numpy()
    if not np.allclose(2.0 ** med, model.x_true.detach().cpu().numpy(), rtol=1e-3):
        print("[WARN] model.x_true is not the median of posterior_samples_cis -- "
              "x_uncertainty='sample' will refuse to run (stale/pre-median cis fit).")
    post_sd = log2_draws.std(axis=0)
    df = pd.DataFrame({"guide": _cell_guide_labels(model), "median": med, "post_sd": post_sd})
    within_sd = df.groupby("guide")["median"].std().median()
    between_sd = df.groupby("guide")["median"].median().std()
    print(f"cells: {len(df)}, guides: {df['guide'].nunique()}")
    print(f"median per-cell posterior SD of log2 x_true : {np.median(post_sd):.3f}")
    print(f"median within-guide SD of per-cell medians  : {within_sd:.3f}")
    print(f"SD of guide-level medians (between guides)  : {between_sd:.3f}")
    return df


def _cell_guide_labels(model):
    if getattr(model, "is_high_moi", False):
        ga = np.asarray(model.guide_assignment)
        names = model.guide_meta["guide"].astype(str).values
        return ["+".join(sorted(names[np.flatnonzero(row)])) for row in ga]
    return model.meta["guide_code"].values


def select_morris_genes(model, summary_path, domingo_genes, n_total=200, modality_name="gene"):
    """
    Pick the Morris trans-gene subset: every Domingo trans gene present in the
    modality, then the steepest dependent Hill components from the production
    full-transcriptome summary, alternating most-positive / most-negative n,
    until n_total genes.

    Steepness uses only active components (which_active a/b/both); per gene the
    signed n of its steepest active component.
    """
    mod = model.get_modality(modality_name)
    available = list(mod.feature_ids)
    avail_set = set(available)
    to_id = _feature_id_lookup(mod)

    domingo_ids = {}
    for g in dict.fromkeys(domingo_genes):
        fid = to_id.get(str(g))
        if fid is not None and fid not in domingo_ids.values():
            domingo_ids[g] = fid
    domingo_keep = list(domingo_ids.values())

    df = pd.read_csv(summary_path)
    df["feature"] = df["feature"].astype(str).map(lambda f: to_id.get(f, f))
    n_summary_matched = df["feature"].isin(avail_set).sum()
    print(f"modality feature_ids look like: {available[:3]}")
    print(f"Domingo genes matched: {len(domingo_keep)}/{len(set(domingo_genes))}; "
          f"summary features matched: {n_summary_matched}/{len(df)}")
    if n_summary_matched == 0:
        raise ValueError(
            f"No feature in {summary_path} maps to this modality's feature_ids "
            f"(summary examples: {df['feature'].head(3).tolist()}). Check the summary is "
            "from this cis gene's production trans run.")
    rows = []
    for comp, active in (("a", ("a", "both")), ("b", ("b", "both"))):
        sub = df[df["which_active"].isin(active)]
        rows.append(pd.DataFrame({"feature": sub["feature"], "component": comp,
                                  "n": sub[f"n_{comp}_median"]}))
    steep = pd.concat(rows, ignore_index=True).dropna(subset=["n"])
    steep = (steep.assign(abs_n=steep["n"].abs())
                  .sort_values("abs_n", ascending=False)
                  .drop_duplicates("feature"))
    steep = steep[steep["feature"].isin(avail_set) & ~steep["feature"].isin(domingo_keep)]
    pos = steep[steep["n"] > 0].sort_values("n", ascending=False).to_dict("records")
    neg = steep[steep["n"] < 0].sort_values("n", ascending=True).to_dict("records")

    n_needed = max(n_total - len(domingo_keep), 0)
    picked = []
    while len(picked) < n_needed and (pos or neg):
        for pool in (pos, neg):
            if pool and len(picked) < n_needed:
                picked.append(pool.pop(0))

    selection = pd.concat([
        pd.DataFrame({"feature": domingo_keep, "source": "domingo"}),
        pd.DataFrame(picked).assign(source=lambda d: np.where(d["n"] > 0, "steep_pos", "steep_neg"))
        if picked else pd.DataFrame(columns=["feature", "source"]),
    ], ignore_index=True)
    print(f"{len(domingo_keep)} Domingo genes + {len(picked)} steep genes "
          f"({sum(p['n'] > 0 for p in picked)} pos, {sum(p['n'] < 0 for p in picked)} neg) "
          f"= {len(selection)} / {n_total} requested")
    missing = [g for g in dict.fromkeys(domingo_genes) if g not in domingo_ids]
    if missing:
        print(f"Domingo genes not in this modality (cis gene / filtered / absent): {missing}")
    if "gene_name" in mod.feature_meta.columns:   # readable names alongside the ids
        id_to_name = dict(zip(available, mod.feature_meta["gene_name"].astype(str)))
        selection.insert(1, "gene_name", selection["feature"].map(id_to_name))
    return selection


def _feature_id_lookup(mod):
    """
    Map any identifier in the modality's feature_meta (gene_name / gene /
    gene_id, plus the feature_ids themselves) to its feature_id, so gene
    symbols and Ensembl IDs both resolve whichever one the model uses.
    """
    ids = [str(f) for f in mod.feature_ids]
    lookup = {}
    fm = mod.feature_meta.reset_index(drop=True)
    for col in ("gene_id", "gene", "gene_name"):   # later columns don't overwrite earlier ones
        if col in fm.columns:
            for val, fid in zip(fm[col].astype(str), ids):
                lookup.setdefault(val, fid)
    lookup.update({fid: fid for fid in ids})
    return lookup


def fit_conditions(base_model, fit_kwargs, test_output_dir, base_label, conditions=None,
                   modality_name=None, refit=False):
    """
    For each condition: deepcopy the prepared model, point it at
    <test_output_dir>/<base_label>_<condition>, fit_trans with that condition's
    x-input kwargs (or load it if already saved there), save fit + summary.

    Returns (models, summaries) dicts keyed by condition name.
    """
    conditions = conditions or CONDITIONS
    mod = modality_name or base_model.primary_modality
    models, summaries = {}, {}
    for name, xargs in conditions.items():
        m = copy.deepcopy(base_model)
        m.output_dir = test_output_dir
        m.label = f"{base_label}_{name}"
        out = os.path.join(test_output_dir, m.label)
        os.makedirs(out, exist_ok=True)
        post_path = os.path.join(out, f"posterior_samples_trans_{mod}.pt")
        summ_path = os.path.join(out, f"trans_feature_summary_{mod}.csv")

        if os.path.exists(post_path) and os.path.exists(summ_path) and not refit:
            print(f"[{name}] loading existing fit from {out}")
            m.load_trans_fit(input_dir=out, modalities=[mod])
            summaries[name] = pd.read_csv(summ_path)
        else:
            print(f"[{name}] fit_trans({xargs}) -> {out}")
            m.fit_trans(modality_name=mod, checkpoint_dir=out, **fit_kwargs, **xargs)
            m.save_trans_fit(modalities=[mod])
            summaries[name] = m.save_trans_summary(output_dir=out, modality_name=mod,
                                                     compute_derivative_roots=False)
        models[name] = m
    return models, summaries


def dependence_table(summaries):
    """Counts of fit_type / which_active per condition."""
    out = {name: df["which_active"].fillna("none").value_counts() for name, df in summaries.items()}
    tab = pd.DataFrame(out).fillna(0).astype(int)
    tab.loc["dependent (any)"] = tab.drop(index="none", errors="ignore").sum()
    return tab


def plot_n_shift(summaries, reference="cell_median"):
    """Hill coefficients per gene: each condition vs the reference condition."""
    others = [c for c in summaries if c != reference]
    fig, axes = plt.subplots(len(others), 2, figsize=(8, 3.6 * len(others)), squeeze=False)
    ref = summaries[reference].set_index("feature")
    for i, name in enumerate(others):
        df = summaries[name].set_index("feature").reindex(ref.index)
        for j, comp in enumerate(("a", "b")):
            ax = axes[i, j]
            x, y = ref[f"n_{comp}_median"], df[f"n_{comp}_median"]
            ax.scatter(x, y, s=6, alpha=0.6, c=COND_COLORS.get(name, "k"))
            lim = np.nanmax(np.abs(np.r_[x, y])) * 1.05
            ax.plot([-lim, lim], [-lim, lim], ls="--", c="grey", lw=0.8)
            ax.set_xlabel(f"n_{comp} ({reference})")
            ax.set_ylabel(f"n_{comp} ({name})")
    fig.tight_layout()
    return fig


def _dep_marks(summaries, feature):
    marks = []
    for name, df in summaries.items():
        row = df.loc[df["feature"] == feature, "which_active"]
        wa = row.iloc[0] if len(row) else np.nan
        marks.append(f"{name.split('_')[0][0]}{name.split('_')[1][0]}:{wa if isinstance(wa, str) else '-'}")
    return " ".join(marks)


def plot_condition_overlays(models, summaries, features, modality_name=None, pdf_path=None,
                            ncols=4, per_page=16, show=True, **trans_fn_kwargs):
    """
    One panel per feature: fitted trans function (+95% CI) of every condition
    overlaid. Title lists which_active per condition (cm = cell_median, ...).
    All pages are written to pdf_path (if given) and optionally shown inline.
    """
    trans_fn_kwargs.setdefault("show_ci", True)
    features = list(features)
    first = next(iter(models.values()))
    mod_name = modality_name or first.primary_modality
    # Binomial: plot Δp on a log2FC x-axis, then label y as absolute p (p_ntc + Δp),
    # so the dashed line plot_trans_functions draws at Δp = 0 sits at y_ntc.
    absolute_p = (first.get_modality(mod_name).distribution == "binomial"
                  and not trans_fn_kwargs.get("use_log2fc", False))
    if absolute_p:
        trans_fn_kwargs = {**trans_fn_kwargs, "use_log2fc": False, "use_delta_p": True}
        p_ntc = _ntc_reference(first, mod_name)
    handles = [Line2D([], [], color=COND_COLORS.get(n, "k"), label=n) for n in models]
    pdf = PdfPages(pdf_path) if pdf_path else None
    try:
        for start in range(0, len(features), per_page):
            page = features[start:start + per_page]
            nrows = int(np.ceil(len(page) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
            for ax, feat in zip(axes.flat, page):
                for name, m in models.items():
                    try:
                        m.plot_trans_functions(feat, modality_name=modality_name, ax=ax, legend=False,
                                               colors=COND_COLORS.get(name, "k"), **trans_fn_kwargs)
                    except Exception as e:  # keep the page going; note the failure on the panel
                        ax.text(0.02, 0.02, f"{name}: {type(e).__name__}", transform=ax.transAxes,
                                fontsize=6, color="red")
                ax.set_title(f"{feat}\n{_dep_marks(summaries, feat)}", fontsize=8)
                if absolute_p and feat in p_ntc:
                    ref = p_ntc[feat]
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos, r=ref: f"{v + r:.2f}"))
                    ax.set_ylabel(f"p   (dashed: NTC = {ref:.2f})")
            for ax in list(axes.flat)[len(page):]:
                ax.axis("off")
            fig.suptitle("")  # plot_trans_functions sets its own figure title
            fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            if pdf:
                pdf.savefig(fig)
            if show:
                plt.show()
            else:
                plt.close(fig)
    finally:
        if pdf:
            pdf.close()
            print(f"Saved {pdf_path}")


def _ntc_reference(model, modality_name):
    """
    Per-feature NTC level of a trans modality, {feature_id: value}: the
    posterior median of mu_ntc, as plot_trans_functions' delta-p/log2FC
    modes use for their reference.
    """
    mod = model.get_modality(modality_name)
    mu_ntc = (getattr(mod, "posterior_samples_ntc", None) or {}).get("mu_ntc")
    if mu_ntc is None:
        return {}
    if hasattr(mu_ntc, "float"):   # torch tensor [S, ...]
        vals = mu_ntc.float().quantile(0.5, dim=0).squeeze().cpu().numpy()
    else:
        vals = np.median(np.asarray(mu_ntc), axis=0).squeeze()
    vals = np.atleast_1d(vals)
    return {f: float(vals[i]) for i, f in enumerate(mod.feature_ids) if i < len(vals)}


def save_xy_plots(models, features, plot_dir, modality_name=None, **xy_kwargs):
    """
    plot_xy_data (data + fitted curve) for every feature x condition, saved as
    <plot_dir>/<condition>/<feature>.png (not shown inline). x positions are
    the per-cell posterior-median x_true in every condition.
    """
    for name, m in models.items():
        d = os.path.join(plot_dir, name)
        os.makedirs(d, exist_ok=True)
        for feat in features:
            safe = str(feat).replace("/", "_").replace(":", "_")
            try:
                fig = m.plot_xy_data(feat, modality_name=modality_name,
                                     filename=os.path.join(d, f"{safe}.png"), **xy_kwargs)
            except Exception as e:
                print(f"[{name}] {feat}: {type(e).__name__}: {e}")
                continue
            plt.close("all")
    print(f"Saved xy plots under {plot_dir}")
