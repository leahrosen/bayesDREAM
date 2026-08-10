"""
Shared config plumbing for publication_runs/.

Two layers of YAML, deliberately kept separate (see publication_runs/README.md):

1. Dataset orchestration config (``<dataset>/config.yaml``) -- read directly
   with ``load_yaml`` by each dataset's ``generate_slurm.py``. No fixed
   schema; each dataset defines its own.
2. Per-gene/per-stage "bayesdream config" -- the exact schema
   ``bayesDREAM/cli.py`` expects (``data:``/``model:``/``ntc:``/``cis:``/
   ``trans:``/``report:``). ``render_bayesdream_config`` builds these by
   deep-merging a dataset-wide base dict with per-gene overrides, and
   ``write_yaml`` writes the result to ``<output_dir>/<label>/configs/``.

``load_bayesdream_yaml``/``normalize_stage_args``/``is_enabled``/
``read_table``/``load_guide_assignment`` reuse ``bayesDREAM.cli``'s private
helpers of the same name (underscore-prefixed there) rather than
reimplementing them -- straightforward, config-shape-agnostic logic that's
already correct in cli.py.

``build_model_from_config`` is NOT a re-export of ``cli.py``'s
``_build_model``, though -- it duplicates that function's data-loading logic
but with an EXTENDED ``allowed_model_keys`` set. ``_build_model`` filters
``model:`` config keys through an allow-list that's missing ``exclude_guides``
and ``min_count`` (both needed by Morris on essentially every model
construction) -- see the ``_EXTENDED_MODEL_KEYS`` comment below. Editing
``bayesDREAM/cli.py`` itself would be a core-code change needing sign-off
(see CLAUDE.md); duplicating ~30 lines here instead keeps every dataset's
scripts working without one. If ``cli.py``'s own allow-list is ever extended
upstream, ``_EXTENDED_MODEL_KEYS`` should be reconciled with it.
"""

import copy
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from bayesDREAM import bayesDREAM
from bayesDREAM.cli import (
    _load_yaml as load_bayesdream_yaml,
    _normalize_stage_args as normalize_stage_args,
    _is_enabled as is_enabled,
    _read_table as read_table,
    _load_guide_assignment as load_guide_assignment,
)

# Superset of bayesDREAM.cli._build_model's allowed_model_keys (as of this
# writing) -- adds 'exclude_guides' and 'min_count', both accepted by
# bayesDREAM.__init__ but not forwarded by the CLI's own allow-list.
_EXTENDED_MODEL_KEYS = {
    "modality_name", "cis_gene", "cis_feature", "guide_covariates", "guide_covariates_ntc",
    "sum_factor_col", "output_dir", "label", "device", "random_seed", "cores",
    "exclude_targets", "require_ntc",
    "exclude_guides", "min_count",
}


def _read_counts(path, read_csv_kwargs):
    """Like bayesDREAM.cli's own counts loading, but also accepts a `.npz`
    sparse matrix (`scipy.sparse.load_npz`) -- needed for Morris, where a
    dense CSV of ~31k genes x ~94k cells is infeasible. Sparse `.npz` counts
    have NO row/column labels; bayesDREAM aligns them to `meta` PURELY BY
    POSITION in that case (confirmed: bayesDREAM/core.py:331-336 falls back
    to `self._cell_names = self.meta['cell'].tolist()` when counts isn't a
    DataFrame -- i.e. column i of counts is assumed to be cell i of meta,
    unchecked). The `.npz` file's columns MUST already be in the exact same
    order as the `data.meta` CSV's rows -- see morris/preprocess.py, which
    guarantees this by construction.

    read_csv_kwargs defaults to {"index_col": 0} only when the caller
    passes None (key absent from the rendered config entirely) -- NOT via
    `read_csv_kwargs or {"index_col": 0}`, which would also catch an
    EXPLICIT {} (a dataset deliberately opting out of index_col, e.g. a
    plain gene_id/gene_name CSV with no leading unnamed index column) and
    silently override it, since {} is falsy in Python. See
    build_model_from_config's feature_meta loading for the real bug this
    exact pattern caused (Morris's feature_meta_read_csv_kwargs: {} was
    silently ignored, index_col=0 applied anyway, turning feature_meta's
    index into gene_id strings instead of integer row positions).
    """
    path = str(path)
    if path.endswith(".npz"):
        from scipy import sparse
        return sparse.load_npz(path)
    if read_csv_kwargs is None:
        read_csv_kwargs = {"index_col": 0}
    return read_table(path, read_csv_kwargs)


def build_model_from_config(cfg: Dict[str, Any]) -> "bayesDREAM":
    """Construct a bayesDREAM model from a bayesdream-CLI-schema config dict.

    See module docstring: duplicates bayesDREAM.cli._build_model's
    data-loading logic with an extended model-kwarg allow-list
    (_EXTENDED_MODEL_KEYS) rather than reusing it directly.
    """
    data_cfg = cfg.get("data") or {}
    model_cfg = cfg.get("model") or {}

    if "meta" not in data_cfg:
        raise ValueError("Config missing required key: data.meta")
    if "counts" not in data_cfg:
        raise ValueError("Config missing required key: data.counts")

    meta = read_table(data_cfg["meta"], data_cfg.get("meta_read_csv_kwargs"))
    counts = _read_counts(data_cfg["counts"], data_cfg.get("counts_read_csv_kwargs"))

    feature_meta = None
    if data_cfg.get("feature_meta"):
        # NOT `data_cfg.get(...) or {"index_col": 0}` -- an explicit {} (a
        # dataset opting OUT of index_col, e.g. Morris's plain
        # gene_id/gene_name gene_meta.csv with no leading unnamed index
        # column) is falsy in Python and would be silently overridden by
        # that `or`, applying index_col=0 anyway. .get(key, default) only
        # substitutes when the key is truly ABSENT, which is what we want:
        # feature_meta.index becomes gene_id (a string) instead of an
        # integer row position, and _extract_cis_from_gene's
        # feature_meta[...].index[0] lookup (used for positional sparse-
        # matrix indexing) silently gets a gene-id string instead of an
        # int -- this is exactly what caused Morris's high-MOI
        # "IndexError: Index dimension must be 1 or 2" crash inside
        # bayesDREAM.__init__ for every stage using cis_gene-at-construction
        # (compensation/trans/permutation/recapitulation).
        feature_meta_kwargs = data_cfg.get("feature_meta_read_csv_kwargs", {"index_col": 0})
        feature_meta = read_table(data_cfg["feature_meta"], feature_meta_kwargs)

    guide_assignment = None
    if data_cfg.get("guide_assignment"):
        guide_assignment = load_guide_assignment(data_cfg["guide_assignment"])

    guide_meta = None
    if data_cfg.get("guide_meta"):
        guide_meta = read_table(data_cfg["guide_meta"], data_cfg.get("guide_meta_read_csv_kwargs"))

    guide_target = None
    if data_cfg.get("guide_target"):
        guide_target = read_table(data_cfg["guide_target"], data_cfg.get("guide_target_read_csv_kwargs"))

    model_kwargs = {k: v for k, v in model_cfg.items() if k in _EXTENDED_MODEL_KEYS}
    if feature_meta is not None:
        model_kwargs["feature_meta"] = feature_meta

    return bayesDREAM(
        meta=meta,
        counts=counts,
        guide_assignment=guide_assignment,
        guide_meta=guide_meta,
        guide_target=guide_target,
        **model_kwargs,
    )

__all__ = [
    "build_model_from_config",
    "load_bayesdream_yaml",
    "normalize_stage_args",
    "is_enabled",
    "read_table",
    "load_guide_assignment",
    "load_yaml",
    "write_yaml",
    "deep_merge",
    "render_bayesdream_config",
    "apply_sum_factor_adjustments",
    "ensure_dataset_dir_on_syspath",
    "resolve_paths",
    "load_ntc_for_stage",
]


def resolve_paths(paths: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve {placeholder} cross-references in a dataset's `paths:` block,
    allowing CHAINED references (e.g. data_dir: "{raw_data_dir}/preprocessed",
    meta: "{data_dir}/cell_meta.csv") by repeatedly formatting until nothing
    changes -- a single `.format(**paths)` pass only resolves one level deep,
    which silently leaves a literal unresolved "{raw_data_dir}/..." in `meta`
    if `data_dir` itself is still a template at that point. Originally lived
    only in morris/generate_slurm.py (Morris always needed 2-level chains);
    moved here once domingo/config.yaml grew its own raw_data_dir -> data_dir
    -> meta/counts chain (see domingo/README.md's "Preprocessing" section).
    """
    resolved = dict(paths)
    for _ in range(len(paths) + 1):
        changed = False
        for k, v in resolved.items():
            if isinstance(v, str) and "{" in v:
                new_v = v.format(**resolved)
                if new_v != v:
                    resolved[k] = new_v
                    changed = True
        if not changed:
            break
    return resolved


def ensure_dataset_dir_on_syspath(cfg: Dict[str, Any]) -> None:
    """Put a rendered config's ``_dataset_dir`` (set by generate_slurm.py --
    e.g. .../publication_runs/domingo -- NOT the same as the directory the
    rendered YAML itself lives in, which is .../slurm/configs/) onto
    sys.path, so dataset-specific modules referenced by dynamic-import
    config blocks (e.g. domingo/load_modalities.py, morris/
    compensation_exclude_cells.py) are actually importable at runtime.

    Without this, `import <dataset_module>` in a freshly-started SLURM job
    process has no reason to find a module living in domingo/ or morris/ --
    generate_slurm.py's own sys.path.insert() calls only affect the
    config-GENERATION process, not the separate process that later runs the
    rendered config. Call this before any `importlib.import_module(...)`
    driven by a config value.
    """
    dataset_dir = cfg.get("_dataset_dir")
    if dataset_dir and dataset_dir not in sys.path:
        sys.path.insert(0, dataset_dir)


def load_yaml(path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a YAML mapping: {path}")
    return cfg


def write_yaml(path, cfg: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into a deep copy of ``base``.

    Dicts are merged key-by-key; any other value (including lists) in
    ``overrides`` replaces the corresponding value in ``base`` outright.
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def render_bayesdream_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge a per-gene/per-stage override dict onto the dataset's base
    bayesdream-CLI-schema config. Thin wrapper over deep_merge kept as its
    own name so call sites read as "render a CLI config", not "merge dicts".
    """
    return deep_merge(base, overrides)


def apply_sum_factor_adjustments(model, section: Dict[str, Any], steps=("compute_scran", "adjust_ntc_sum_factor", "refit_sumfactor")) -> None:
    """Re-run compute_scran_sum_factor() / adjust_ntc_sum_factor() /
    refit_sumfactor() in THIS process, in that order (skipping whichever
    aren't both requested via `steps` and enabled in `section`).

    None of these survive a save/load round trip -- 'sum_factor_new' (Morris'
    per-cell-subset scran factor), 'sum_factor_adj', and 'sum_factor_refit'
    are computed into the in-memory sum_factors DataFrame and never written
    by save_cis_fit()/save_trans_fit() or restored by load_cis_fit() (see
    bayesDREAM/io/save.py, io/load.py -- only EXISTING sum_factors columns
    are subset on load, never recomputed). So every stage after fit_cis that
    references any of these columns (cis itself, trans, permutation,
    recapitulation -- NOT compensation, which always uses the raw
    'sum_factor' column per project convention) must call this itself, with
    the SAME covariates/args used everywhere else for this dataset --
    otherwise it silently gets a KeyError (column missing) or, worse, an
    inconsistent sum factor from an ad hoc partial recomputation.

    `steps` lets a caller select a PREFIX of the chain: the cis stage only
    needs ('compute_scran', 'adjust_ntc_sum_factor') (refit_sumfactor needs
    x_true, which doesn't exist until AFTER fit_cis); trans/permutation/
    recapitulation use the full default chain (with 'refit_sumfactor'
    typically disabled outright in `section` for datasets that don't use its
    output -- e.g. Morris, see morris/config.yaml).

    Also handles a specific known collision: some datasets' meta ships a
    precomputed 'adjustment_factor' column from upstream R preprocessing
    (Domingo's does, always -- see domingo/README.md). adjust_ntc_sum_factor()
    computes its own internal column of that same name; if the stale one is
    still there it can collide. Renamed (not dropped) to
    'adjustment_factor_old' automatically, once, right before
    adjust_ntc_sum_factor() runs, whenever present -- a no-op for datasets
    that don't have it.

    Expects a ``sum_factor:`` config block, reused verbatim across cis/trans/
    permutation/recapitulation stage configs so they all stay consistent::

        sum_factor:
          compute_scran:                          # Morris only
            enabled: true
            args: {batch_col: lane, sum_factor_col_out: sum_factor_new}
          adjust_ntc_sum_factor:
            enabled: true
            args: {covariates: [lane, cell_line]}
          refit_sumfactor:
            enabled: true
            args: {covariates: [lane, cell_line]}
    """
    if "compute_scran" in steps and is_enabled(section.get("compute_scran"), default=False):
        from compute_scran_sum_factor import compute_scran_sum_factor
        compute_scran_sum_factor(model, **normalize_stage_args(section.get("compute_scran")))

    if "adjust_ntc_sum_factor" in steps and is_enabled(section.get("adjust_ntc_sum_factor"), default=False):
        if "adjustment_factor" in model.meta.columns:
            model.meta["adjustment_factor_old"] = model.meta["adjustment_factor"].copy()
            del model.meta["adjustment_factor"]
        model.adjust_ntc_sum_factor(**normalize_stage_args(section.get("adjust_ntc_sum_factor")))

    if "refit_sumfactor" in steps and is_enabled(section.get("refit_sumfactor"), default=False):
        model.refit_sumfactor(**normalize_stage_args(section.get("refit_sumfactor")))


def load_ntc_for_stage(model, load_ntc_args: Dict[str, Any], modality_name: str) -> None:
    """Load fit_ntc() results for a permutation/recapitulation stage that may
    target a NON-primary modality (e.g. Domingo's binomial splicing
    modalities, attached via an ``attach_modality:`` config block before this
    is called).

    A custom modality's OWN fit_ntc() result was never saved into
    ``load_ntc_args['input_dir']`` (typically ``ntc_shared_dir``) -- that
    directory only ever holds the PRIMARY modality's shared fit. The custom
    modality's own ntc fit was instead saved by its
    ``07_modality_<gene>_<mod>.sh`` job (domingo/load_modalities.py's
    ``model.save_ntc_fit()`` call, no explicit output_dir -- defaults to
    ``<output_dir>/<label>``, i.e. this GENE's own directory) via a
    SEPARATE, later `fit_ntc(modality_name=...)` call, not the shared one.

    Calling ``model.load_ntc_fit(input_dir=ntc_shared_dir)`` alone (the old
    behavior) silently no-ops for that modality -- `os.path.exists()` guards
    every per-file read in `load_ntc_fit`, so a missing
    `posterior_samples_ntc_<modality>.pt` is skipped, not an error, right up
    until `fit_trans(modality_name=...)` later raises "has not been fit with
    fit_ntc()" because the modality's `alpha_y_prefit` is still `None`.

    So this makes TWO calls when `modality_name` isn't the primary modality:
    one restricted to the primary modality from the configured
    `input_dir` (for the primary modality's own alpha_x/alpha_y, needed by
    downstream `permute_x_true`/sum-factor plumbing), and one for JUST
    `modality_name`, from the default directory (`<output_dir>/<label>`) --
    exactly where its own fit_ntc job saved it. For the primary modality
    itself, this is a single unchanged call (the pre-existing behavior).
    """
    if modality_name == model.primary_modality:
        model.load_ntc_fit(**load_ntc_args)
        return

    primary_load_args = dict(load_ntc_args)
    primary_load_args.setdefault("modalities", [model.primary_modality])
    model.load_ntc_fit(**primary_load_args)
    model.load_ntc_fit(modalities=[modality_name], mask_features=load_ntc_args.get("mask_features", False))
