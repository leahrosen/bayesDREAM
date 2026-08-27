"""
Automates save_model_for_plotting() for Domingo and Morris -- plus
backfilling hill_eval's y_at_x_log2fc{...} columns into each gene's real
trans_feature_summary_{modality}.csv -- by replaying that gene's REAL
rendered <label>_trans.yaml config (publication_runs/<dataset>/slurm/configs/),
the exact same recipe publication_runs/common/backfill_trans_summary.py uses
to regenerate a summary from an already-completed fit. Nothing is re-fit;
this only reloads what fit_ntc/fit_cis/fit_trans already saved to disk.

Not reimplementing bayesDREAM's own data-loading/sum-factor logic by hand
here is deliberate: build_model_from_config()/apply_sum_factor_adjustments()
already encode exactly which meta/counts/feature_meta paths and which
compute_scran/adjust_ntc_sum_factor/refit_sumfactor args each dataset's real
fit actually used -- replaying the config is the only way to guarantee this
matches, rather than an independently-guessed (and possibly subtly wrong)
reconstruction.

Does NOT cover Replogle: that pipeline has no rendered YAML config (a
hand-written papermill notebook instead, see 10_bayesDREAM_fit_trans_MYB.ipynb).
Reconstructing it correctly means reusing THAT notebook's own already-tested
load_gene_model_inputs()/build_trans_model() functions in their own session
-- see the "Replogle" section of comparative/README (or ask for the
ready-to-paste snippet) rather than an independent reimplementation here.

Usage
-----
    from comparative.reconstruct_export import reconstruct_and_export, reconstruct_and_export_all

    reconstruct_and_export('Domingo', 'GFI1B')          # one (dataset, gene)
    reconstruct_and_export_all()                        # every Domingo + Morris cis gene
"""

import gc
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import bayesDREAM as _bayesdream_pkg

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(_bayesdream_pkg.__file__)))
_COMMON_DIR = os.path.join(REPO_ROOT, 'publication_runs', 'common')
for _p in (REPO_ROOT, _COMMON_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config_utils import (  # noqa: E402
    build_model_from_config,
    load_bayesdream_yaml,
    normalize_stage_args,
    is_enabled,
    apply_sum_factor_adjustments,
    apply_device_override,
)
from save_for_plotting import save_model_for_plotting  # noqa: E402

from .datasets import DatasetSpec, DOMINGO, MORRIS  # noqa: E402
from .hill_eval import add_log2fc_at_columns, get_x_ntc, HILL_LOG2FC_TARGETS, is_already_backfilled  # noqa: E402

_SPEC_BY_NAME = {'Domingo': DOMINGO, 'Morris': MORRIS}
_DATASET_DIRNAME = {'Domingo': 'domingo', 'Morris': 'morris'}
_LABEL_PREFIX = {'Domingo': 'domingo_20260806', 'Morris': 'morris_20260806'}


def _trans_config_path(dataset_name: str, cis_gene: str, label_prefix: str) -> str:
    dirname = _DATASET_DIRNAME[dataset_name]
    return os.path.join(REPO_ROOT, 'publication_runs', dirname, 'slurm', 'configs',
                         f'{label_prefix}_{cis_gene}_trans.yaml')


def _load_trans_config(dataset_name: str, cis_gene: str, label_prefix: Optional[str] = None) -> dict:
    label_prefix = label_prefix or _LABEL_PREFIX[dataset_name]
    cfg_path = _trans_config_path(dataset_name, cis_gene, label_prefix)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"[{dataset_name}/{cis_gene}] no rendered trans config at {cfg_path!r} -- "
            f"has generate_slurm.py been run for this dataset/gene?"
        )
    return load_bayesdream_yaml(Path(cfg_path))


def _output_dir_and_modality(cfg: dict) -> tuple:
    """Cheap peek at a loaded trans config for the (output_dir, modality_name)
    reconstruct_model() will end up using -- without building the model --
    so reconstruct_and_export() can check is_already_backfilled() before
    paying for a full reconstruction. modality_name here approximates
    build_model_from_config()'s own model.primary_modality (set from
    model_cfg['modality_name'] if given, else 'gene') -- true for every
    dataset config as of this writing (model_defaults.modality_name: gene
    in both domingo/config.yaml and morris/config.yaml).
    """
    model_cfg = cfg.get('model') or {}
    output_dir = os.path.join(model_cfg.get('output_dir', 'output'), model_cfg.get('label'))
    fit_args = normalize_stage_args((cfg.get('trans') or {}).get('fit'))
    modality_name = fit_args.get('modality_name') or model_cfg.get('modality_name') or 'gene'
    return output_dir, modality_name


def reconstruct_model(dataset_name: str, cis_gene: str, spec: DatasetSpec, cfg: dict,
                       *, device: Optional[str] = None):
    """Rebuild a fully-loaded bayesDREAM model (ntc+cis+trans posteriors) for
    (dataset_name, cis_gene) from its already-loaded real <label>_trans.yaml
    config (see _load_trans_config()) -- backfill_trans_summary.py's exact
    recipe -- without re-fitting anything.

    Builds DEFERRED (cis_gene omitted from the constructor, committed via
    add_cis_gene() below) even though the real rendered trans.yaml sets
    model.cis_gene eagerly. That's correct for the ORIGINAL fit -- fit_trans()
    itself never needed add_cis_gene()'s NTC-posterior extraction -- but
    replaying it here, on a model we're about to load_ntc_fit() onto, is
    lossy. Eager construction carves the cis gene out of the primary modality
    (and into the 'cis' modality) at __init__ time, BEFORE load_ntc_fit()
    below reloads the shared/per-gene NTC posterior file -- which DOES
    include the cis gene's row (it's the full untrimmed panel saved by the
    deferred fit_ntc stage; see publication_runs/common/run_ntc.py). With no
    'cis' feature slot to place that row into, feature-alignment-on-load
    silently drops it, so the cis gene's mu_ntc/o_x/alpha_x never make it
    into the reconstructed model even though they're sitting right there in
    the file. Building deferred and calling add_cis_gene() AFTER
    load_ntc_fit() -- the exact recipe publication_runs/common/
    run_cis_deferred.py used for the real fit_cis stage -- extracts that row
    into the 'cis' modality instead of discarding it.

    Confirmed 2026-08-27 as the real root cause of hill_eval.get_x_ntc()'s
    median-x_true fallback firing for every Domingo/Morris gene. It is NOT
    "Domingo's fit_ntc is deferred, Morris's is eager": both datasets' own
    fit_ntc/fit_cis stages already use deferred+add_cis_gene() (Morris's
    primary genes just get their OWN per-gene ntc fit instead of one shared
    across all genes -- see morris/generate_slurm.py's module docstring).
    What was eager, for both, was this reconstruction script's replay of the
    trans-stage config.

    Returns (model, output_dir, modality_name).
    """
    apply_device_override(cfg, device)

    cfg_for_build = dict(cfg)
    model_cfg = dict(cfg.get('model') or {})
    model_cfg.pop('cis_gene', None)
    cfg_for_build['model'] = model_cfg

    model = build_model_from_config(cfg_for_build)

    output_dir = os.path.join(model_cfg.get('output_dir', 'output'), model_cfg.get('label'))
    trans_cfg = cfg.get('trans') or {}
    ntc_cfg = cfg.get('ntc') or cfg.get('technical') or {}

    if 'technical_group_code' not in model.meta.columns:
        covariates = ntc_cfg.get('set_technical_groups')
        if covariates:
            model.set_technical_groups(covariates)

    if is_enabled(trans_cfg.get('load_ntc', trans_cfg.get('load_technical')), default=True):
        model.load_ntc_fit(**normalize_stage_args(trans_cfg.get('load_ntc') or trans_cfg.get('load_technical')))
    # Commits to cis_gene now (not at construction) -- see the docstring above.
    # Unconditional: even if load_ntc above was skipped/disabled, eager
    # construction would still have set cis_gene unconditionally, so this
    # must too (add_cis_gene() degrades gracefully to plain extraction with
    # no NTC posteriors to draw from when fit_ntc wasn't loaded).
    model.add_cis_gene(cis_gene)
    if is_enabled(trans_cfg.get('load_cis'), default=True):
        model.load_cis_fit(**normalize_stage_args(trans_cfg.get('load_cis')))

    apply_sum_factor_adjustments(model, cfg.get('sum_factor') or {})

    excl_cfg = cfg.get('exclude_trans_genes') or {}
    if is_enabled(excl_cfg, default=False):
        model.exclude_trans_genes(**normalize_stage_args(excl_cfg))

    fit_args = normalize_stage_args(trans_cfg.get('fit'))
    modality_name = fit_args.get('modality_name') or model.primary_modality

    # Loads the EXISTING saved posterior -- does not fit anything.
    model.load_trans_fit(modalities=[modality_name], subset_features=True)

    if spec.force_single_cell_line and 'cell_line' not in model.meta.columns:
        model.meta['cell_line'] = spec.force_single_cell_line

    return model, output_dir, modality_name


def reconstruct_and_export(
    dataset_name: str, cis_gene: str,
    *, label_prefix: Optional[str] = None, hill_log2fc_targets=HILL_LOG2FC_TARGETS,
    device: Optional[str] = None, force: bool = False,
) -> str:
    """Full pipeline for one (dataset, cis_gene):

    0. Checkpoint check: if trans_feature_summary_{modality}.csv already has
       every requested y_at_x_log2fc{...} column AND the save_model_for_plotting()
       export already exists, skip the whole reconstruction (this is a full
       model reload -- expensive for Morris/Replogle's transcriptome-wide
       panels) and return the existing export dir. Pass force=True to redo
       it anyway (e.g. after a fit was re-run with different results).
    1. Reconstruct the model from its real trans config (see reconstruct_model()).
    2. Regenerate trans_feature_summary_{modality}.csv, add hill_eval's
       y_at_x_log2fc{...} columns, and backfill it IN PLACE into the real
       production output_dir -- so comparative/trans_param_compare.py picks
       up the new columns automatically, no separate step needed.
    3. Call save_model_for_plotting() into this dataset's
       save_for_plotting_dir_fn() location (Comparative/input/<Dataset>_<gene>_GEX/),
       so comparative/dose_response_panels.py can reload it for full curve panels.

    Explicitly frees the reconstructed model (del + gc.collect()) before
    returning -- model.load_trans_fit() loads FULL posterior samples (not
    lean point estimates) for every trans gene, which for Morris/Replogle's
    ~20k-gene panels is large enough that leaving several of these alive
    across a reconstruct_and_export_all() loop risks real memory pressure.

    Returns the save_model_for_plotting() export directory.
    """
    spec = _SPEC_BY_NAME[dataset_name]
    cfg = _load_trans_config(dataset_name, cis_gene, label_prefix)
    output_dir, modality_name = _output_dir_and_modality(cfg)
    save_dir = spec.save_for_plotting_dir_fn(cis_gene)

    if not force and is_already_backfilled(output_dir, modality_name, save_dir, hill_log2fc_targets):
        print(f"[{dataset_name}/{cis_gene}] already backfilled + exported -- skipping (force=True to redo)")
        return save_dir

    print(f"[{dataset_name}/{cis_gene}] reconstructing model from real trans config...")
    model, output_dir, modality_name = reconstruct_model(dataset_name, cis_gene, spec, cfg, device=device)

    print(f"[{dataset_name}/{cis_gene}] computing summary + y_at_x_log2fc{{...}} columns...")
    # x_ntc=get_x_ntc(model): now that reconstruct_model() builds deferred and
    # calls add_cis_gene() (see its docstring), the cis modality's
    # posterior_samples_ntc['mu_ntc'] is legitimately populated -- get_x_ntc()
    # finds it on its primary (non-fallback) branch, same as Replogle. Passed
    # explicitly anyway for consistency/robustness rather than relying on
    # save_trans_summary()'s own internal auto-lookup to reach the same value.
    #
    # compute_derivative_roots/compute_inflection/compute_lfc_ci=False: none of
    # what this backfill actually needs -- add_log2fc_at_columns() only reads
    # is_dependent/y_ntc off `df` and does its own Hill evaluation from
    # posterior_samples_trans; observed_log2fc/EC50_a_log2fc/full_log2fc/n_a/
    # Vmax_a (DEFAULT_GRID_PARAMS) are all gated by compute_log2fc_params alone,
    # not these three. Confirmed 2026-08-27: for Domingo's 89-feature GFI1B
    # panel, u_roots (compute_derivative_roots) was 68% of loop1's time and
    # obs_lfc_ci+full_lfc_ci (compute_lfc_ci) another 30%, for columns nothing
    # downstream reads -- for Morris/Replogle's ~11-20k-feature panels this is
    # the difference between minutes and the better part of an hour. Losing
    # compute_lfc_ci just means observed_log2fc_lower/upper fall back to the
    # posterior mean instead of a real CI; trans_param_compare.py never reads
    # those anyway (PARAM_ALIASES['observed_log2fc'] has no _lower/_upper).
    df = model.save_trans_summary(
        output_dir=output_dir, modality_name=modality_name, x_ntc=get_x_ntc(model),
        compute_derivative_roots=False, compute_inflection=False, compute_lfc_ci=False,
    )
    df = add_log2fc_at_columns(model, df, modality_name=modality_name, targets=hill_log2fc_targets)
    csv_path = os.path.join(output_dir, f'trans_feature_summary_{modality_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"[{dataset_name}/{cis_gene}] backfilled {csv_path}")

    print(f"[{dataset_name}/{cis_gene}] exporting for plotting -> {save_dir}")
    save_model_for_plotting(model, save_dir=save_dir)

    del model, df
    gc.collect()

    return save_dir


def reconstruct_and_export_all(
    datasets: Optional[List[str]] = None, genes: Optional[Dict[str, List[str]]] = None,
    **kwargs,
) -> Dict[str, Dict[str, str]]:
    """Loop reconstruct_and_export() over every cis gene for each dataset
    (default: every gene in DatasetSpec.cis_genes for Domingo and Morris).

    Raises immediately (does not skip) if any (dataset, gene)'s rendered
    trans config, or any file/column it references, is missing --
    DatasetSpec.cis_genes is defined as "has a completed fit_trans run", so
    a gene listed there with missing data is a real inconsistency to fix,
    not an expected gap to silently paper over.

    Returns {dataset_name: {cis_gene: export_dir}}.
    """
    datasets = datasets or ['Domingo', 'Morris']
    genes = genes or {name: _SPEC_BY_NAME[name].cis_genes for name in datasets}

    results: Dict[str, Dict[str, str]] = {}
    for name in datasets:
        results[name] = {}
        for gene in genes.get(name, []):
            results[name][gene] = reconstruct_and_export(name, gene, **kwargs)
    return results
