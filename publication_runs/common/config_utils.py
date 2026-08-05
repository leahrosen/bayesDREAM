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

``build_model_from_config`` / ``load_bayesdream_yaml`` intentionally reuse
``bayesDREAM.cli``'s private ``_build_model``/``_load_yaml``/
``_normalize_stage_args``/``_is_enabled`` helpers rather than reimplementing
model construction here -- that logic (reading data.meta/data.counts,
resolving guide_assignment formats, filtering allowed model kwargs, ...) is
non-trivial and already correct in cli.py. This is reuse, not a fork: if
cli.py's config schema changes, these scripts pick it up automatically.
"""

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

from bayesDREAM.cli import (
    _build_model as build_model_from_config,
    _load_yaml as load_bayesdream_yaml,
    _normalize_stage_args as normalize_stage_args,
    _is_enabled as is_enabled,
    _read_table as read_table,
    _load_guide_assignment as load_guide_assignment,
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
]


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


def apply_sum_factor_adjustments(model, section: Dict[str, Any]) -> None:
    """Re-run adjust_ntc_sum_factor() / refit_sumfactor() in THIS process.

    Both are pure, deterministic transforms of meta['sum_factor'] (and, for
    refit_sumfactor, self.x_true) -- neither 'sum_factor_adj' nor
    'sum_factor_refit' is written by save_cis_fit()/save_trans_fit() or
    restored by load_cis_fit() (see bayesDREAM/io/save.py, io/load.py --
    only the sum_factors DataFrame's existing columns are subset on load,
    never recomputed). So every stage after fit_cis that references either
    column (compensation, trans, permutation, recapitulation) must call this
    itself, right after load_cis_fit(), with the SAME covariates used during
    the original fit_cis/fit_trans run -- otherwise it silently gets a
    KeyError (column missing) or, worse, an inconsistent sum factor if a
    caller made its own ad hoc partial version.

    Expects a ``sum_factor:`` config block, reused verbatim across cis/
    compensation/trans/permutation/recapitulation stage configs so they all
    stay consistent::

        sum_factor:
          adjust_ntc_sum_factor:
            enabled: true
            args: {covariates: [lane, cell_line]}
          refit_sumfactor:
            enabled: true
            args: {covariates: [lane, cell_line], sum_factor_col_old: sum_factor_adj}
    """
    if is_enabled(section.get("adjust_ntc_sum_factor"), default=False):
        model.adjust_ntc_sum_factor(**normalize_stage_args(section.get("adjust_ntc_sum_factor")))
    if is_enabled(section.get("refit_sumfactor"), default=False):
        model.refit_sumfactor(**normalize_stage_args(section.get("refit_sumfactor")))
