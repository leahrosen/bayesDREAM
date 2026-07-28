"""
Cis-panel simulation for the single-Hill recovery study (docs/SIMULATION_STUDY_PLAN.md).

Simulates one "cell-design scenario": a synthetic cis gene + guide panel (mirrors
_model_x's generative process) plus a large panel of trans genes spanning the
null / single_hill fold-change grid, generated via ``simulate_from_trans_summary``
so the simulation and the real fitting code share the exact same A/V/K reconstruction
semantics (see plan §4.2).
"""

import numpy as np
import pandas as pd

from .simulation import simulate_from_trans_summary, _compute_AV_from_fc

# ---------------------------------------------------------------------------
# Trans-panel grid (plan §3.2) — identical across all cell-design scenarios
# ---------------------------------------------------------------------------

Y_NTC_LOG2_LEVELS = (-4, -1, 1, 4)
N_VALUES = (-5, -1, -0.5, 0.5, 1, 5)
K_LOG2FC_VALUES = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
FULL_LOG2FC_VALUES = (0.5, 1, 2, 4)

# Guide log2FC patterns (plan §3.1). Keys: (n_targeting_guides, shape).
GUIDE_PATTERNS = {
    (3, 'even'): (-3, -2, -1),
    (3, 'gap'): (-3, -2.5, -0.5),
    (3, 'small'): (-1.5, -1, -0.5),
    (5, 'even'): (-4, -3, -2, -1, 0),
    (5, 'gap'): (-4, -3.5, -3, -1, -0.5),
    (5, 'small'): (-1.5, -1.25, -1, -0.75, -0.5),
}


def _o_y_log2_levels(y_ntc_log2: float) -> tuple:
    """o_y grid depends on y_ntc level (plan §3.2): the y_ntc=-4 regime uses a
    different, higher pair to keep the negbinom draw informative at such low counts."""
    if y_ntc_log2 == -4:
        return (-0.3, 2.0)
    return (-1.5, 0.0)


def build_trans_panel_grid() -> pd.DataFrame:
    """Build the 1736-row synthetic trans_summary_df grid (8 y_ntc/o_y combos x 217
    effect scenarios: 1 null + 6 n x 9 K_log2FC x 4 full_log2FC).

    Returns a DataFrame with the columns ``simulate_from_trans_summary`` expects for
    the single_hill negbinom fold-change parameterization (``feature``,
    ``function_type``, ``distribution``, ``o_y_median``, ``y_ntc_median``,
    ``n_a_median``, ``K_log2FC_a_median``, ``full_log2FC_a_median``), plus ground-truth
    bookkeeping columns not consumed by that function (``y_ntc_log2``, ``o_y_log2``,
    ``effect_type``, ``n_true``, ``K_log2FC_true``, ``full_log2FC_true``).
    ``x_ntc_median`` is NOT included here — it depends on the cis-side scenario and
    must be set by the caller before simulation.
    """
    rows = []
    for y_ntc_log2 in Y_NTC_LOG2_LEVELS:
        y_ntc = 2.0 ** y_ntc_log2
        for o_y_log2 in _o_y_log2_levels(y_ntc_log2):
            o_y = 2.0 ** o_y_log2
            rows.append(dict(
                feature=f"trans_y{y_ntc_log2}_o{o_y_log2}_null",
                function_type='single_hill',
                distribution='negbinom',
                o_y_median=o_y,
                y_ntc_median=y_ntc,
                n_a_median=0.0,
                K_log2FC_a_median=0.0,
                full_log2FC_a_median=0.0,
                y_ntc_log2=y_ntc_log2,
                o_y_log2=o_y_log2,
                effect_type='no_effect',
                n_true=0.0,
                K_log2FC_true=np.nan,
                full_log2FC_true=0.0,
            ))
            for n in N_VALUES:
                for K_log2FC in K_LOG2FC_VALUES:
                    for full_log2FC in FULL_LOG2FC_VALUES:
                        rows.append(dict(
                            feature=(
                                f"trans_y{y_ntc_log2}_o{o_y_log2}"
                                f"_n{n}_K{K_log2FC}_F{full_log2FC}"
                            ),
                            function_type='single_hill',
                            distribution='negbinom',
                            o_y_median=o_y,
                            y_ntc_median=y_ntc,
                            n_a_median=float(n),
                            K_log2FC_a_median=float(K_log2FC),
                            full_log2FC_a_median=float(full_log2FC),
                            y_ntc_log2=y_ntc_log2,
                            o_y_log2=o_y_log2,
                            effect_type='single_hill',
                            n_true=float(n),
                            K_log2FC_true=float(K_log2FC),
                            full_log2FC_true=float(full_log2FC),
                        ))
    df = pd.DataFrame(rows)
    assert df['feature'].is_unique, "trans panel feature names collided"
    return df


def _sample_nb(mu: np.ndarray, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Sample NegBinom(mean=mu, dispersion=phi) via the Gamma-Poisson mixture
    (Var = mu + mu^2/phi), avoiding any float/int constraints on numpy's own
    negative_binomial(n, p)."""
    lam = rng.gamma(shape=phi, scale=mu / phi)
    return rng.poisson(lam)


def simulate_cis_panel(
    cells_per_gene: int,
    n_guides: int,
    guide_shape: str,
    sigma_eff: float,
    log2_X_NTC: float,
    log2_o_x: float,
    rng: np.random.Generator,
    cis_gene_name: str = 'CisGene',
):
    """Simulate the cis side of one cell-design scenario (plan §4.1, mirrors `_model_x`).

    Cells are split across `n_guides` targeting guides + 1 NTC guide via floor
    division, with the remainder added to the NTC group (plan §2 rounding rule).

    Returns
    -------
    meta : pd.DataFrame        cell, guide, target, cell_line, sum_factor
    x_true : np.ndarray         (n_cells,) true cis expression, linear scale
    x_obs : np.ndarray          (n_cells,) simulated NB counts for the cis gene
    sum_factor : np.ndarray     (n_cells,)
    cis_ground_truth : pd.DataFrame   per-cell latent x_true
    guide_ground_truth : pd.DataFrame per-guide true effect
    """
    guide_log2fc = GUIDE_PATTERNS[(n_guides, guide_shape)]
    per_target_guide = cells_per_gene // (n_guides + 1)
    n_ntc = cells_per_gene - per_target_guide * n_guides

    X_NTC = 2.0 ** log2_X_NTC
    o_x = 2.0 ** log2_o_x
    phi_x = 1.0 / (o_x ** 2)

    cells, guides, targets_col, log2_x_true_list = [], [], [], []
    guide_ground_truth_rows = []

    guide_plan = [('NTC', 'ntc', 0.0, n_ntc)]
    for i, fc in enumerate(guide_log2fc):
        guide_plan.append((f'g{i + 1}', cis_gene_name, float(fc), per_target_guide))

    idx = 0
    for gname, target, fc, n_cells_g in guide_plan:
        x_eff_g = X_NTC * (2.0 ** fc)
        guide_ground_truth_rows.append(dict(
            guide=gname, target=target, guide_log2FC=fc,
            x_eff_g_true=x_eff_g, sigma_eff=sigma_eff, n_cells=n_cells_g,
        ))
        for _ in range(n_cells_g):
            cells.append(f'cell_{idx:05d}')
            guides.append(gname)
            targets_col.append(target)
            log2_x_true_list.append(rng.normal(np.log2(x_eff_g), sigma_eff))
            idx += 1

    n_cells = idx
    log2_x_true = np.array(log2_x_true_list)
    x_true = 2.0 ** log2_x_true

    log2_sf = rng.normal(0.0, np.sqrt(0.5), size=n_cells)
    sum_factor = 2.0 ** log2_sf

    x_obs = _sample_nb(x_true * sum_factor, phi_x, rng)

    meta = pd.DataFrame({
        'cell': cells,
        'guide': guides,
        'target': targets_col,
        'cell_line': 'batch0',
        'sum_factor': sum_factor,
    })

    cis_ground_truth = pd.DataFrame({
        'cell': cells,
        'guide': guides,
        'log2_x_true': log2_x_true,
        'x_true': x_true,
    })
    guide_ground_truth = pd.DataFrame(guide_ground_truth_rows)

    return meta, x_true, x_obs, sum_factor, cis_ground_truth, guide_ground_truth


def simulate_scenario(
    cells_per_gene: int,
    n_guides: int,
    guide_shape: str,
    log2_X_NTC: float,
    log2_o_x: float,
    seed: int,
    sigma_eff: float = 0.7,
    cis_gene_name: str = 'CisGene',
):
    """Simulate one full cell-design scenario: cis panel + full 1736-feature trans
    panel (plan §4). `seed` drives every RNG stream used here (numpy for the cis
    side, and is passed through to `simulate_from_trans_summary`'s own RNG for the
    trans side) — see plan §6.

    Returns a dict with keys: meta, counts, cis_ground_truth, guide_ground_truth,
    trans_ground_truth, config.
    """
    rng = np.random.default_rng(seed)

    meta, x_true, x_obs, sum_factor, cis_ground_truth, guide_ground_truth = simulate_cis_panel(
        cells_per_gene=cells_per_gene, n_guides=n_guides, guide_shape=guide_shape,
        sigma_eff=sigma_eff, log2_X_NTC=log2_X_NTC, log2_o_x=log2_o_x, rng=rng,
        cis_gene_name=cis_gene_name,
    )

    panel_df = build_trans_panel_grid()
    X_NTC = 2.0 ** log2_X_NTC
    panel_df['x_ntc_median'] = X_NTC

    counts = simulate_from_trans_summary(
        trans_summary_df=panel_df,
        meta=meta,
        x_true=x_true,
        x_counts=x_obs,
        cis_gene=cis_gene_name,
        sim_sum_factor=sum_factor,
        fdr_threshold=None,
        seed=seed,
    )

    A_true, V_true, K_true = _compute_AV_from_fc(
        n=panel_df['n_true'].values,
        y_ntc=panel_df['y_ntc_median'].values,
        x_ntc=panel_df['x_ntc_median'].values,
        K_log2FC=panel_df['K_log2FC_true'].values,
        full_log2FC=panel_df['full_log2FC_true'].values,
    )

    trans_ground_truth = panel_df[[
        'feature', 'effect_type', 'y_ntc_log2', 'o_y_log2',
        'y_ntc_median', 'o_y_median', 'x_ntc_median',
        'n_true', 'K_log2FC_true', 'full_log2FC_true',
    ]].rename(columns={
        'y_ntc_median': 'y_ntc_true', 'o_y_median': 'o_y_true', 'x_ntc_median': 'x_ntc_true',
    }).copy()
    trans_ground_truth['A_true'] = A_true
    trans_ground_truth['Vmax_true'] = V_true
    trans_ground_truth['K_true'] = K_true

    config = dict(
        cells_per_gene=cells_per_gene, n_guides=n_guides, guide_shape=guide_shape,
        sigma_eff=sigma_eff, log2_X_NTC=log2_X_NTC, log2_o_x=log2_o_x, seed=seed,
        cis_gene_name=cis_gene_name, n_cells=len(meta), n_trans_features=len(panel_df),
    )

    return dict(
        meta=meta,
        counts=counts,
        cis_ground_truth=cis_ground_truth,
        guide_ground_truth=guide_ground_truth,
        trans_ground_truth=trans_ground_truth,
        config=config,
    )
