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
from .hill_eval import add_log2fc_at_columns, HILL_LOG2FC_TARGETS  # noqa: E402

_SPEC_BY_NAME = {'Domingo': DOMINGO, 'Morris': MORRIS}
_DATASET_DIRNAME = {'Domingo': 'domingo', 'Morris': 'morris'}
_LABEL_PREFIX = {'Domingo': 'domingo_20260806', 'Morris': 'morris_20260806'}


def _trans_config_path(dataset_name: str, cis_gene: str, label_prefix: str) -> str:
    dirname = _DATASET_DIRNAME[dataset_name]
    return os.path.join(REPO_ROOT, 'publication_runs', dirname, 'slurm', 'configs',
                         f'{label_prefix}_{cis_gene}_trans.yaml')


def reconstruct_model(dataset_name: str, cis_gene: str, spec: DatasetSpec,
                       *, label_prefix: Optional[str] = None, device: Optional[str] = None):
    """Rebuild a fully-loaded bayesDREAM model (ntc+cis+trans posteriors) for
    (dataset_name, cis_gene) by replaying its real <label>_trans.yaml config
    -- backfill_trans_summary.py's exact recipe -- without re-fitting
    anything.

    Returns (model, output_dir, modality_name).
    """
    label_prefix = label_prefix or _LABEL_PREFIX[dataset_name]
    cfg_path = _trans_config_path(dataset_name, cis_gene, label_prefix)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"[{dataset_name}/{cis_gene}] no rendered trans config at {cfg_path!r} -- "
            f"has generate_slurm.py been run for this dataset/gene?"
        )
    cfg = load_bayesdream_yaml(Path(cfg_path))
    apply_device_override(cfg, device)

    model = build_model_from_config(cfg)

    model_cfg = cfg.get('model') or {}
    output_dir = os.path.join(model_cfg.get('output_dir', 'output'), model_cfg.get('label'))
    trans_cfg = cfg.get('trans') or {}
    ntc_cfg = cfg.get('ntc') or cfg.get('technical') or {}

    if is_enabled(trans_cfg.get('load_ntc', trans_cfg.get('load_technical')), default=True):
        model.load_ntc_fit(**normalize_stage_args(trans_cfg.get('load_ntc') or trans_cfg.get('load_technical')))
    if is_enabled(trans_cfg.get('load_cis'), default=True):
        model.load_cis_fit(**normalize_stage_args(trans_cfg.get('load_cis')))

    if 'technical_group_code' not in model.meta.columns:
        covariates = ntc_cfg.get('set_technical_groups')
        if covariates:
            model.set_technical_groups(covariates)

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
    device: Optional[str] = None,
) -> str:
    """Full pipeline for one (dataset, cis_gene):

    1. Reconstruct the model from its real trans config (see reconstruct_model()).
    2. Regenerate trans_feature_summary_{modality}.csv, add hill_eval's
       y_at_x_log2fc{...} columns, and backfill it IN PLACE into the real
       production output_dir -- so comparative/trans_param_compare.py picks
       up the new columns automatically, no separate step needed.
    3. Call save_model_for_plotting() into this dataset's
       save_for_plotting_dir_fn() location (Comparative/input/<Dataset>_<gene>_GEX/),
       so comparative/dose_response_panels.py can reload it for full curve panels.

    Returns the save_model_for_plotting() export directory.
    """
    spec = _SPEC_BY_NAME[dataset_name]

    print(f"[{dataset_name}/{cis_gene}] reconstructing model from real trans config...")
    model, output_dir, modality_name = reconstruct_model(
        dataset_name, cis_gene, spec, label_prefix=label_prefix, device=device,
    )

    print(f"[{dataset_name}/{cis_gene}] computing summary + y_at_x_log2fc{{...}} columns...")
    df = model.save_trans_summary(output_dir=output_dir, modality_name=modality_name)
    df = add_log2fc_at_columns(model, df, modality_name=modality_name, targets=hill_log2fc_targets)
    csv_path = os.path.join(output_dir, f'trans_feature_summary_{modality_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"[{dataset_name}/{cis_gene}] backfilled {csv_path}")

    save_dir = spec.save_for_plotting_dir_fn(cis_gene)
    print(f"[{dataset_name}/{cis_gene}] exporting for plotting -> {save_dir}")
    save_model_for_plotting(model, save_dir=save_dir)

    return save_dir


def reconstruct_and_export_all(
    datasets: Optional[List[str]] = None, genes: Optional[Dict[str, List[str]]] = None,
    **kwargs,
) -> Dict[str, Dict[str, str]]:
    """Loop reconstruct_and_export() over every cis gene for each dataset
    (default: every gene in DatasetSpec.cis_genes for Domingo and Morris).
    Skips (with a printed message) any (dataset, gene) whose rendered trans
    config isn't found, rather than aborting the whole run.

    Returns {dataset_name: {cis_gene: export_dir}}.
    """
    datasets = datasets or ['Domingo', 'Morris']
    genes = genes or {name: _SPEC_BY_NAME[name].cis_genes for name in datasets}

    results: Dict[str, Dict[str, str]] = {}
    for name in datasets:
        results[name] = {}
        for gene in genes.get(name, []):
            try:
                results[name][gene] = reconstruct_and_export(name, gene, **kwargs)
            except FileNotFoundError as e:
                print(f"[skip {name}/{gene}] {e}")
    return results
