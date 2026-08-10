"""
Domingo-specific hook: attach the extra (non-primary) splicing/velocity
modalities listed in config_modalities.yaml to an already-cis-fit model,
then fit that modality's OWN fit_ntc() (a SEPARATE technical fit per
modality -- not reused from the primary 'gene' modality) followed by
fit_trans(). Based on real, working loader code for six modality types
(sj/exon_skip/intron_retention/gene_velocity/mxe [binomial],
donor_choice [multinomial]) -- see config_modalities.yaml's comments for
what's confirmed vs. inferred-by-symmetry (acceptor_choice).

Directory layout: `data_dir` (config.yaml's `modalities.data_dir`) is ONE
shared directory used for EVERY cis gene -- NOT a per-gene subdirectory.
Each modality lives at `<data_dir>/<stype>/{cell_meta.tsv.gz,
feature_meta.tsv.gz, counts.npz[, denominator.npz]}`, covering the FULL
cell population; per-gene cell subsetting happens inside the loader by
aligning against that gene's own `model.meta['L_cell_barcode']` (missing
cells zero-filled, matching `add_custom_modality()`'s own internal
intersection-based re-alignment against `model.meta['cell']` -- both work
together correctly because Domingo's preprocessing sets
`meta['cell'] = meta['L_cell_barcode']`, see domingo/README.md).

Contract: `attach_modality(model, spec, data_dir)` attaches the modality via
`model.add_custom_modality(...)` and returns the resulting modality name
(`f'{name_prefix}_{stype}'`), ready for `fit_ntc(modality_name=...)` /
`fit_trans(modality_name=...)`.

`attach_modality_precomputed(model, spec, precomputed_dir)`: same contract,
but reads a directory ALREADY subsetted to this model's own cells (written
by subset_modality_per_gene.py, which calls attach_modality() itself and
saves the result) instead of the raw shared `data_dir` -- no cell alignment
or (for SJ) gene-expression-denominator computation needed here, both
already done when the precomputed file was written. This is what
07_modality_<gene>_<mod>.sh actually uses now; attach_modality() itself is
only still called by subset_modality_per_gene.py and (via the
attach_modality: config block) run_permutation_null.py/
run_recapitulation_sim.py -- those also switched to
attach_modality_precomputed, since permutation/recapitulation reruns don't
need to re-read the raw splicing directory either.
"""

import os
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse


# ---------------------------------------------------------------------- #
# Low-level loaders (shared across every modality type)                    #
# ---------------------------------------------------------------------- #

def _load_npz_dense(path: str) -> np.ndarray:
    """Load a .npz file as a dense 2-D numpy array."""
    try:
        return scipy.sparse.load_npz(path).toarray()
    except Exception:
        pass
    data = np.load(path, allow_pickle=False)
    keys = list(data.keys())
    for key in ("data", "arr_0", "matrix", "counts"):
        if key in keys:
            return data[key]
    return data[keys[0]]


def _align_cells(counts: np.ndarray, file_cells: list, model_cells: list) -> np.ndarray:
    """Reorder/subset the cell axis (axis 1) of counts to match model_cells.
    Cells in model_cells but absent from file_cells are filled with zeros."""
    file_idx = {c: i for i, c in enumerate(file_cells)}
    n_missing = sum(1 for c in model_cells if c not in file_idx)
    if n_missing:
        warnings.warn(
            f"{n_missing}/{len(model_cells)} model cells not found in file; "
            "filling missing cells with zeros."
        )
    out_shape = list(counts.shape)
    out_shape[1] = len(model_cells)
    out = np.zeros(out_shape, dtype=counts.dtype)
    for j, cell in enumerate(model_cells):
        if cell in file_idx:
            if counts.ndim == 2:
                out[:, j] = counts[:, file_idx[cell]]
            elif counts.ndim == 3:
                out[:, j, :] = counts[:, file_idx[cell], :]
    return out


def _reconstruct_multinomial_3d(counts_2d: np.ndarray, feature_meta: pd.DataFrame) -> np.ndarray:
    """Reshape a flat 2-D counts matrix (total_rows x n_cells) into a 3-D
    multinomial array (n_features x n_cells x max_categories) using
    row_start/row_end columns in feature_meta -- rows row_start:row_end
    belong to feature i."""
    n_features = len(feature_meta)
    n_cells = counts_2d.shape[1]
    max_cats = int(feature_meta["n_categories"].max())
    counts_3d = np.zeros((n_features, n_cells, max_cats), dtype=counts_2d.dtype)
    for i, (_, row) in enumerate(feature_meta.iterrows()):
        s, e = int(row["row_start"]), int(row["row_end"])
        n_cat = e - s
        counts_3d[i, :, :n_cat] = counts_2d[s:e, :].T
    return counts_3d


def _read_type_meta(type_dir: str) -> Tuple[list, pd.DataFrame]:
    cell_meta = pd.read_csv(os.path.join(type_dir, "cell_meta.tsv.gz"), sep="\t")
    feature_meta = pd.read_csv(os.path.join(type_dir, "feature_meta.tsv.gz"), sep="\t")
    return cell_meta["cell_barcode"].tolist(), feature_meta


def _to_dense(counts) -> np.ndarray:
    return counts.toarray() if scipy.sparse.issparse(counts) else np.asarray(counts)


def _plot_counts_vs_denominator(counts, denominator, feature_meta=None, gene_col="gene",
                                 n_worst_genes=20, max_points=200_000, seed=0):
    """Diagnostic scatter + per-gene violation breakdown for counts >
    denominator entries (SJ's gene-expression-derived denominator only).
    Returns (fig, axes); caller is responsible for saving/closing."""
    import matplotlib.pyplot as plt

    c = _to_dense(counts)
    d = _to_dense(denominator)

    viol = c > d
    n_viol = int(viol.sum())
    n_total = c.size
    print(f"Violations: {n_viol} / {n_total} entries ({100 * n_viol / max(n_total, 1):.4f}%)")

    keep = (c > 0) | (d > 0)
    cf, df, vf = c[keep], d[keep], viol[keep]

    rng = np.random.default_rng(seed)
    ok_idx = np.flatnonzero(~vf)
    if len(ok_idx) > max_points:
        ok_idx = rng.choice(ok_idx, size=max_points, replace=False)
    viol_idx = np.flatnonzero(vf)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(df[ok_idx] + 1, cf[ok_idx] + 1, s=4, alpha=0.15, color="#4C72B0", linewidths=0, label="valid")
    ax.scatter(df[viol_idx] + 1, cf[viol_idx] + 1, s=8, alpha=0.6, color="#C44E52", linewidths=0, label="counts > denom")
    lims = [1, max(cf.max(), df.max()) + 1]
    ax.plot(lims, lims, color="gray", lw=1, ls="--", label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("denominator (+1)")
    ax.set_ylabel("SJ counts (+1)")
    ax.set_title(f"Counts vs denominator ({n_viol} violations)")
    ax.legend(frameon=False, markerscale=3)

    ax2 = axes[1]
    if feature_meta is not None and gene_col in feature_meta.columns:
        genes = feature_meta[gene_col].to_numpy()
        viol_counts_per_feature = viol.sum(axis=1)
        df_g = pd.DataFrame({"gene": genes, "n_viol_entries": viol_counts_per_feature})
        top = (df_g.groupby("gene")["n_viol_entries"].sum().sort_values(ascending=False).head(n_worst_genes))
        ax2.barh(top.index[::-1], top.values[::-1], color="#C44E52")
        ax2.set_xlabel("# violating entries")
        ax2.set_xscale("log")
        ax2.set_title(f"Top {n_worst_genes} genes by violation count")
    else:
        viol_counts_per_feature = viol.sum(axis=1)
        top_idx = np.argsort(viol_counts_per_feature)[::-1][:n_worst_genes]
        ax2.barh([str(i) for i in top_idx[::-1]], viol_counts_per_feature[top_idx][::-1], color="#C44E52")
        ax2.set_xlabel("# violating entries")
        ax2.set_xscale("log")
        ax2.set_title(f"Top {n_worst_genes} features (rows) by violation count")

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------- #
# Per-distribution modality loaders                                        #
# ---------------------------------------------------------------------- #

def load_binomial_modality(
    model, base_dir: str, stype: str,
    name_prefix: str = "splicing",
    gene_alias_col: str = "gene_name",
    denominator_mode: str = "file",
    clip_violations: bool = False,
    diagnostic_plot_path: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """denominator_mode: 'file' (separate denominator.npz) or
    'gene_expression' (SJ-style: denominator from the primary+cis modality's
    gene counts, features filtered to genes present in the model,
    optionally clipping counts>denominator violations)."""
    type_dir = os.path.join(base_dir, stype)
    model_cells = model.meta["L_cell_barcode"].tolist()
    file_cells, feature_meta = _read_type_meta(type_dir)

    if gene_alias_col and "gene" not in feature_meta.columns and gene_alias_col in feature_meta.columns:
        feature_meta = feature_meta.assign(gene=feature_meta[gene_alias_col])

    counts = _load_npz_dense(os.path.join(type_dir, "counts.npz"))

    if denominator_mode == "file":
        denominator = _load_npz_dense(os.path.join(type_dir, "denominator.npz"))
        counts = _align_cells(counts, file_cells, model_cells)
        denominator = _align_cells(denominator, file_cells, model_cells)

    elif denominator_mode == "gene_expression":
        gene_mod = model.get_modality(model.primary_modality)
        gene_counts = _to_dense(gene_mod.counts)
        gene_names = list(gene_mod.feature_names)
        cell_index = {c: i for i, c in enumerate(gene_mod.cell_names)}

        # add_cis_gene() extracts the cis gene out of the primary modality into
        # its own 'cis' modality, so it's no longer in gene_mod above -- add it
        # back so features belonging to the cis gene aren't dropped below.
        if "cis" in model.modalities:
            cis_mod = model.get_modality("cis")
            cis_counts = _to_dense(cis_mod.counts)
            gene_counts = np.vstack([gene_counts, cis_counts])
            gene_names = gene_names + [cis_mod.feature_meta["gene_name"].iloc[0]]

        gene_index = {g: i for i, g in enumerate(gene_names)}
        feature_keep_idx = feature_meta["gene"].isin(gene_index)
        feature_meta = feature_meta[feature_keep_idx]

        counts = _align_cells(counts, file_cells, model_cells)
        counts = counts[feature_keep_idx.values, :]

        denom_gene_col = "gene_for_denominator" if "gene_for_denominator" in feature_meta.columns else "gene"
        genes = feature_meta[denom_gene_col].tolist()
        gene_rows = [gene_index[g] for g in genes]
        cell_cols = [cell_index[c] for c in model_cells]
        denominator = gene_counts[np.ix_(gene_rows, cell_cols)]

        if diagnostic_plot_path:
            import matplotlib.pyplot as plt
            fig, _ = _plot_counts_vs_denominator(counts, denominator, feature_meta, gene_col="gene")
            os.makedirs(os.path.dirname(diagnostic_plot_path), exist_ok=True)
            fig.savefig(diagnostic_plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[diagnostic] wrote {diagnostic_plot_path}")

        violations = counts > denominator
        if clip_violations and violations.sum() > 0:
            print(f"Clipping {int(violations.sum())} entries where {stype} counts > gene denominator")
            counts = np.minimum(counts, denominator)
    else:
        raise ValueError(f"load_binomial_modality: unknown denominator_mode {denominator_mode!r}")

    modality_name = f"{name_prefix}_{stype}"
    model.add_custom_modality(
        name=modality_name, counts=counts, feature_meta=feature_meta.reset_index(drop=True),
        distribution="binomial", denominator=denominator, cell_names=model_cells, overwrite=overwrite,
    )
    print(f"Added '{modality_name}': {counts.shape[0]} features")
    return modality_name


def load_multinomial_modality(
    model, base_dir: str, stype: str, name_prefix: str = "splicing", overwrite: bool = False,
) -> str:
    type_dir = os.path.join(base_dir, stype)
    model_cells = model.meta["L_cell_barcode"].tolist()
    file_cells, feature_meta = _read_type_meta(type_dir)

    counts_2d = _load_npz_dense(os.path.join(type_dir, "counts.npz"))
    counts_2d = _align_cells(counts_2d, file_cells, model_cells)
    counts_3d = _reconstruct_multinomial_3d(counts_2d, feature_meta)
    print(f"  shape: {counts_3d.shape}  (features x cells x max_categories={counts_3d.shape[2]})")

    modality_name = f"{name_prefix}_{stype}"
    model.add_custom_modality(
        name=modality_name, counts=counts_3d, feature_meta=feature_meta.reset_index(drop=True),
        distribution="multinomial", cell_names=model_cells, overwrite=overwrite,
    )
    # bayesDREAM/utils.py's resolve_feature_names should already pick
    # 'feature_id' automatically when present; set it explicitly too, matching
    # the reference notebook code (defensive, not redundant if that ever
    # changes upstream).
    if "feature_id" in feature_meta.columns:
        model.get_modality(modality_name).feature_names = list(feature_meta["feature_id"].values)
    print(f"Added '{modality_name}': {counts_3d.shape[0]} features")
    return modality_name


def attach_modality(model, spec: Dict, base_dir: str, diagnostic_plot_path: Optional[str] = None) -> str:
    """Load and attach one modality (per config_modalities.yaml `spec`) to
    `model`. Returns the resulting bayesDREAM modality name."""
    stype = spec["stype"]
    name_prefix = spec.get("name_prefix", "splicing")
    distribution = spec["distribution"]

    if distribution == "multinomial":
        return load_multinomial_modality(model, base_dir, stype, name_prefix=name_prefix)
    elif distribution == "binomial":
        return load_binomial_modality(
            model, base_dir, stype, name_prefix=name_prefix,
            gene_alias_col=spec.get("gene_alias_col", "gene_name"),
            denominator_mode=spec.get("denominator_mode", "file"),
            clip_violations=spec.get("clip_violations", False),
            diagnostic_plot_path=diagnostic_plot_path if spec.get("save_diagnostic_plot") else None,
        )
    else:
        raise ValueError(f"attach_modality: unsupported distribution {distribution!r} for stype={stype!r}")


def attach_modality_precomputed(model, spec: Dict, precomputed_dir: str) -> str:
    """Like attach_modality(), but reads precomputed_dir (written by
    subset_modality_per_gene.py -- already cell-aligned to THIS model and
    min_count-filtered) instead of the raw shared splicing directory. See
    this module's docstring for why this is what the real pipeline uses
    now."""
    stype = spec["stype"]
    name_prefix = spec.get("name_prefix", "splicing")
    distribution = spec["distribution"]
    modality_name = f"{name_prefix}_{stype}"

    feature_meta = pd.read_csv(os.path.join(precomputed_dir, "feature_meta.csv"))
    with open(os.path.join(precomputed_dir, "cell_names.txt")) as f:
        cell_names = [line.rstrip("\n") for line in f if line.strip()]

    if distribution == "multinomial":
        counts = np.load(os.path.join(precomputed_dir, "counts.npy"))
        model.add_custom_modality(
            name=modality_name, counts=counts, feature_meta=feature_meta,
            distribution="multinomial", cell_names=cell_names,
        )
        if "feature_id" in feature_meta.columns:
            model.get_modality(modality_name).feature_names = list(feature_meta["feature_id"].values)
    else:
        counts = _to_dense(scipy.sparse.load_npz(os.path.join(precomputed_dir, "counts.npz")))
        denom_path = os.path.join(precomputed_dir, "denominator.npz")
        denominator = _to_dense(scipy.sparse.load_npz(denom_path)) if os.path.exists(denom_path) else None
        model.add_custom_modality(
            name=modality_name, counts=counts, feature_meta=feature_meta,
            distribution=distribution, denominator=denominator, cell_names=cell_names,
        )

    print(f"Added '{modality_name}' (precomputed from {precomputed_dir}): {counts.shape[0]} features")
    return modality_name


def main() -> None:
    """CLI entry point for one (gene, modality) attach + fit_ntc + fit_trans task.

    Fits a SEPARATE fit_ntc() for this modality (not reused from the primary
    'gene' modality's ntc fit -- matches the reference notebook pattern of
    one fit_ntc(modality_name=...) call per custom modality) before
    fit_trans(). Both use the same load-if-exists-else-fit-and-save pattern
    as every other stage script in this pipeline.

    Usage:
        # normal (precomputed) path -- see subset_modality_per_gene.py:
        python load_modalities.py --config <gene_full_subset_config.yaml> \\
            --modality-name sj --modality-spec config_modalities.yaml \\
            --precomputed-dir <that gene+modality's precomputed subset dir>

        # raw path -- only subset_modality_per_gene.py itself should need this:
        python load_modalities.py --config <gene_cis_stage_config.yaml> \\
            --modality-name sj --modality-spec config_modalities.yaml \\
            --data-dir <modalities.data_dir>   # shared across all genes, NOT per-gene

    Exactly one of --data-dir/--precomputed-dir must be given.
    """
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from config_utils import build_model_from_config, load_bayesdream_yaml, normalize_stage_args  # noqa: E402
    from git_provenance import save_provenance_json  # noqa: E402
    from resource_stats import (  # noqa: E402
        is_cuda, new_stats_dict, load_prior_stats, step_completed,
        carry_forward_step, timed_step, timed_attempt, record_trans_step,
    )
    import yaml  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True, help="Gene's cis-stage bayesdream config (needs cis already fit and saved).")
    parser.add_argument("--modality-name", required=True, help="stype key in config_modalities.yaml, e.g. 'sj'.")
    parser.add_argument("--modality-spec", required=True, help="Path to config_modalities.yaml")
    parser.add_argument("--data-dir", default=None, help="Shared loader_inputs/ dir (same for every gene). Mutually exclusive with --precomputed-dir.")
    parser.add_argument("--precomputed-dir", default=None, help="Per-(gene,modality) precomputed subset dir (see subset_modality_per_gene.py). Mutually exclusive with --data-dir.")
    args = parser.parse_args()
    if bool(args.data_dir) == bool(args.precomputed_dir):
        parser.error("exactly one of --data-dir/--precomputed-dir is required")

    cfg = load_bayesdream_yaml(Path(args.config))
    with open(args.modality_spec) as f:
        specs = {m["stype"]: m for m in yaml.safe_load(f)["modalities"]}
    spec = specs[args.modality_name]

    model = build_model_from_config(cfg)
    device_is_cuda = is_cuda(model)
    cis_cfg = cfg.get("cis") or {}
    ntc_cfg = cfg.get("ntc") or cfg.get("technical") or {}
    # NOTE: deliberately does NOT call model.load_ntc_fit() here for the
    # primary 'gene' modality (earlier versions of this script did, cargo-
    # culted from bayesDREAM.cli._run_fit_cis's load-then-fit_cis pattern).
    # It has no `input_dir` in this script's config (only `load_cis` is
    # rendered -- see domingo/generate_slurm.py's mod_cfg), so it would
    # default to this gene's OWN output_dir/label, where no ntc fit is ever
    # saved (only ntc_shared_dir has one, and the cis stage here only saves
    # via save_cis_fit(), never save_ntc_fit()) -- a silent no-op. Nothing
    # below consumes the primary modality's alpha_x_prefit/alpha_y_prefit/
    # posterior_samples_ntc: fit_ntc()/fit_trans() below are both scoped to
    # THIS job's own new custom modality via modality_name=, and
    # fitting/trans.py never reads alpha_x_prefit at all.
    model.load_cis_fit(**normalize_stage_args(cis_cfg.get("load_cis")))
    if "technical_group_code" not in model.meta.columns:
        covariates = ntc_cfg.get("set_technical_groups")
        if covariates:
            model.set_technical_groups(covariates)

    model_cfg = cfg.get("model") or {}
    output_dir = os.path.join(model_cfg.get("output_dir", "output"), model_cfg.get("label"))

    if args.precomputed_dir:
        modality_name = attach_modality_precomputed(model, spec, args.precomputed_dir)
    else:
        diagnostic_plot_path = os.path.join(output_dir, f"diagnostic_{args.modality_name}_counts_vs_denominator.png")
        modality_name = attach_modality(model, spec, args.data_dir, diagnostic_plot_path=diagnostic_plot_path)

    stats_path = os.path.join(output_dir, f"stats_{modality_name}.json")
    prior_stats = load_prior_stats(stats_path)
    stats = new_stats_dict(model, extra={"stage": "modality_fit", "label": model_cfg.get("label"),
                                          "modality": modality_name})

    # --- modality-specific NTC fit: load if exists, otherwise fit and save ---
    # fit_ntc() has no internal checkpoint (see resource_stats.py's module
    # docstring) -- gated on stats_path (not the .pt file's mere existence,
    # since save_ntc_fit() uses plain torch.save(), not an atomic write, so a
    # file that exists isn't proof it's complete/uncorrupted if a prior
    # attempt died mid-write).
    if step_completed(prior_stats, "fit_ntc"):
        carry_forward_step(stats, stats_path, "fit_ntc", prior_stats)
        model.load_ntc_fit(output_dir, modalities=[modality_name])
    else:
        print("[INFO] Running ntc fit for modality (this may take a while)...")
        with timed_step("fit_ntc", stats, device_is_cuda, stats_path):
            model.fit_ntc(modality_name=modality_name, tolerance=0)
        model.save_ntc_fit()

    # --- modality-specific trans fit ---
    # Unlike fit_ntc above, no existence/skip check needed here: fit_trans()
    # has its OWN internal checkpoint/resume (restart_from_checkpoint=True by
    # default) -- if a prior attempt already completed, it detects that from
    # its own checkpoint almost instantly rather than re-fitting, so always
    # calling it (same as run_trans.py) is both simpler and more robust than
    # trusting a plain torch.save()'d output file's mere existence.
    # checkpoint_dir left at fit_trans()'s own default (output_dir, computed
    # above) -- this is the sole writer for this (gene, modality) pair, no
    # collision risk (unlike permutation/recapitulation reps, which must
    # override it -- see resource_stats.py's module docstring).
    print("[INFO] Running trans fit for modality (loads/resumes from its own checkpoint if already complete)...")
    with timed_attempt(device_is_cuda) as this_attempt:
        model.fit_trans(
            modality_name=modality_name, tolerance=0, function_type=spec["function_type"],
            min_denominator=spec.get("min_denominator", 0), checkpoint_dir=output_dir,
        )
    record_trans_step(stats, stats_path, "fit_trans", modality_name, output_dir, this_attempt, prior_stats)
    model.save_trans_fit(modalities=[modality_name])

    model.save_trans_summary(output_dir=output_dir, modality_name=modality_name)

    save_provenance_json(
        os.path.join(output_dir, f"provenance_trans_{modality_name}.json"),
        extra={"stage": "trans_modality", "modality": modality_name, "label": model_cfg.get("label")},
    )


if __name__ == "__main__":
    main()
