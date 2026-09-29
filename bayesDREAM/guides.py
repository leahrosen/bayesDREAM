"""Guide mapping, classification, and assignment normalization."""

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd


def is_ntc_target_name(target_name):
    """Return whether a target name is a recognized non-targeting control."""
    return target_name is not None and str(target_name).strip().lower() in {
        "ntc", "non-targeting", "non-targeting-control", "non-targeting_control"
    }


def resolve_guide_target_mapping(guide_target=None, guide_meta=None):
    """Build a guide-to-targets mapping from an explicit map or guide metadata.

    ``guide_target`` must contain ``guide`` and ``target`` columns and may have
    multiple rows per guide. When it is absent, a one-target mapping is built
    from ``guide_meta['target']`` if available; otherwise ``None`` is returned.
    """
    if guide_target is None:
        if guide_meta is None or 'target' not in guide_meta.columns:
            return None
        return {row['guide']: [row['target']] for _, row in guide_meta.iterrows()}

    required = {'guide', 'target'}
    missing = required - set(guide_target.columns)
    if missing:
        raise ValueError(
            f"guide_target missing required columns: {missing}. "
            f"Available columns: {list(guide_target.columns)}"
        )
    mapping = {}
    for guide, target in guide_target[['guide', 'target']].itertuples(index=False, name=None):
        mapping.setdefault(guide, []).append(target)
    return mapping


def classify_target_from_guides(guide_names, guide_targets_dict, cis_gene=None, exclude_targets=None, exclude_guides=None):
    if exclude_guides and any(guide in exclude_guides for guide in guide_names):
        return "excluded"
    targets = [target for guide in guide_names for target in guide_targets_dict.get(guide, [])]
    if exclude_targets and any(target in exclude_targets for target in targets):
        return "excluded"
    if cis_gene is not None and cis_gene in targets:
        return cis_gene
    return "ntc" if any(is_ntc_target_name(target) for target in targets) else "other"


@dataclass
class GuideState:
    """Validated guide inputs in cells-by-guides orientation."""

    is_high_moi: bool
    guide_assignment: np.ndarray | None
    guide_meta: pd.DataFrame | None
    guide_targets_dict: dict | None
    n_cells_assignment: int | None


def build_guide_state(meta, guide_assignment=None, guide_meta=None, guide_target=None):
    """Validate guide inputs and normalize high-MOI assignment orientation.

    Returns a single-guide state when neither high-MOI argument is supplied.
    For high-MOI inputs, requires a two-dimensional assignment matrix aligned
    with ``meta`` rows and ``guide_meta`` rows, auto-transposing guide-by-cell
    input when its dimensions unambiguously indicate that orientation.
    """
    if guide_assignment is None and guide_meta is None:
        return GuideState(False, None, None, resolve_guide_target_mapping(guide_target), None)
    if guide_assignment is None or guide_meta is None:
        raise ValueError(
            "Both guide_assignment and guide_meta must be provided for high MOI mode. "
            "Got guide_assignment={}, guide_meta={}".format(
                type(guide_assignment).__name__ if guide_assignment is not None else None,
                type(guide_meta).__name__ if guide_meta is not None else None
            )
        )
    if guide_assignment.ndim != 2:
        raise ValueError(
            f"guide_assignment must be a 2D matrix (cells × guides), "
            f"but got shape {guide_assignment.shape} with {guide_assignment.ndim} dimensions"
        )

    n_cells, n_guides = guide_assignment.shape
    if (n_cells, n_guides) == (len(meta), len(guide_meta)):
        pass
    elif (n_guides, n_cells) == (len(meta), len(guide_meta)):
        warnings.warn(
            f"[HIGH MOI] guide_assignment appears to be transposed (shape {guide_assignment.shape} = guides × cells). "
            f"Expected (cells × guides). Auto-transposing to ({n_guides}, {n_cells}).",
            UserWarning
        )
        guide_assignment = guide_assignment.T
        n_cells, n_guides = guide_assignment.shape
    else:
        raise ValueError(
            f"guide_assignment shape {guide_assignment.shape} does not match expected dimensions:\n"
            f"  - guide_meta has {len(guide_meta)} guides\n"
            f"  - meta has {len(meta)} cells\n"
            f"Expected guide_assignment shape: ({len(meta)}, {len(guide_meta)}) [cells × guides]\n"
            f"Got: {guide_assignment.shape}\n"
            "Please check your guide_assignment matrix orientation."
        )
    if 'guide' not in guide_meta.columns:
        raise ValueError(
            f"guide_meta missing required column 'guide'. "
            f"Available columns: {list(guide_meta.columns)}"
        )
    normalized_meta = guide_meta.copy()
    normalized_meta['guide_code'] = range(n_guides)
    targets = resolve_guide_target_mapping(guide_target, normalized_meta)
    if targets is None:
        raise ValueError(
            "Either guide_target DataFrame or guide_meta['target'] column must be provided "
            "to specify guide-target relationships in high MOI mode."
        )
    print(f"[INFO] High MOI: {n_guides} guides, avg {guide_assignment.sum(axis=1).mean():.2f} per cell")
    return GuideState(True, guide_assignment.copy(), normalized_meta, targets, n_cells)


def expand_guide_assignment(assignment, guide_meta, targets, meta, guide_covariates, guide_covariates_ntc):
    """
    High-MOI analogue of single-guide mode's ``guide_used`` column (see the
        ``if not self.is_high_moi`` branch just above this method's call site).

        Splits each guide's column in ``self.guide_assignment``/``self.guide_meta``
        into one column per distinct combination of covariate values observed
        among the cells carrying that guide, so the same physical guide can have
        an independent ``x_eff_g`` effect per covariate level (e.g. per lane).
        NTC-classified guides (any target in the NTC variants) are split by
        ``guide_covariates_ntc``; all other guides (cis-targeting, or
        not-yet-classified 'other' when cis_gene is still deferred) are split by
        ``guide_covariates``. No-op if both lists are empty (preserves prior
        behavior exactly — this is the default).

        ``_model_x`` needs no changes for this: it already treats each
        ``guide_assignment`` column as an independent latent effect and sums
        per-cell via matmul, so this is purely a data-prep step. Requires
        ``self.meta`` to already be row-aligned (by position) with
        ``self.guide_assignment`` when called.

        ``guide_meta['guide']`` is preserved unchanged (original guide name,
        possibly now duplicated across the guide's split columns) so that
        ``self.guide_targets_dict`` lookups elsewhere (NTC-mask computation,
        ``add_cis_gene()``'s guide pruning, etc.) keep working without any
        changes — they iterate ``guide_meta`` rows and look up
        ``guide_targets_dict.get(row['guide'], [])``, which tolerates duplicate
        keys since it's never used as a name -> single-row mapping. Downstream
        per-guide plots/summaries that key off ``guide_meta['guide']`` will show
        one row per (guide, covariate) column rather than one row per guide —
        the correct behavior here, since each column now has its own effect.
    """
    if not guide_covariates and not guide_covariates_ntc:
        return assignment, guide_meta

    ntc_names = {'ntc', 'NTC', 'non-targeting', 'non-targeting-control', 'Non-Targeting'}
    key_ntc = meta[guide_covariates_ntc].astype(str).agg('|'.join, axis=1).values if guide_covariates_ntc else None
    key_other = meta[guide_covariates].astype(str).agg('|'.join, axis=1).values if guide_covariates else None
    columns, rows = [], []
    for index, (_, row) in enumerate(guide_meta.iterrows()):
        is_ntc = any(target in ntc_names for target in targets.get(row['guide'], []))
        keys, column = (key_ntc if is_ntc else key_other), assignment[:, index]
        mask = column.astype(bool)
        if keys is None or not mask.any():
            columns.append(column)
            row = row.copy()
            row['guide_covariate_key'] = ''
            rows.append(row)
            continue
        for key in sorted(set(keys[mask])):
            expanded = np.zeros_like(column)
            expanded[mask & (keys == key)] = 1
            columns.append(expanded)
            expanded_row = row.copy()
            expanded_row['guide_covariate_key'] = key
            rows.append(expanded_row)

    expanded_meta = pd.DataFrame(rows).reset_index(drop=True)
    expanded_meta['guide_code'] = range(len(expanded_meta))
    if len(columns) != assignment.shape[1]:
        print(f"[INFO] High MOI: expanded {assignment.shape[1]} guides to {len(columns)} "
              f"(guide, covariate)-columns (guide_covariates={guide_covariates}, "
              f"guide_covariates_ntc={guide_covariates_ntc})")
    return np.stack(columns, axis=1), expanded_meta
