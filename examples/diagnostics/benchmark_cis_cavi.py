"""
Compare the SVI fit_cis with Polya-Gamma CAVI (bayesDREAM.fitting.cis_cavi) on a simulated
single-guide dataset with known per-guide log2 x_eff_g, per-cell log2 x_i, technical group
effects alpha_x and dispersion phi_x.

Both methods get the true alpha_x and phi_x (as if fit_ntc had estimated them exactly):
alpha_x_prefit is set on the model and o_x = phi_x^(-1/2) is placed in the cis modality's
posterior_samples_ntc, which is all fit_cis reads from the technical fit.

Reports RMSE and 90 % interval coverage for log2 x_eff_g and log2 x_i, and runtime.
SVI intervals are the 5 % and 95 % posterior sample quantiles; CAVI intervals are
mean +/- 1.645 SD.

Usage:
    python examples/diagnostics/benchmark_cis_cavi.py [svi_niters] [seed]

Torch runs single threaded (torch.set_num_threads(1)); on small tensors the thread pool
overhead otherwise dominates both methods on some machines.
"""

import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from bayesDREAM import bayesDREAM
from bayesDREAM.fitting.cis_cavi import fit_cis_negbinom_cavi

Z90 = 1.6449


def simulate(seed=0, n_cells=3000, n_ntc=6, n_targeting=14, baseline=8.0, phi_x=5.0,
             alpha_x=(1.0, 1.5, 0.7), n_other_genes=20):
    """Single-guide data; returns meta, counts and the truth per cell / guide."""
    rng = np.random.default_rng(seed)
    G = n_ntc + n_targeting
    guides = [f"ntc_{j}" for j in range(n_ntc)] + [f"cis_{j}" for j in range(n_targeting)]
    log2_eff = np.concatenate([np.full(n_ntc, np.log2(baseline)),
                               np.log2(baseline) + np.linspace(-0.25, -3.0, n_targeting)])
    sigma_eff = rng.uniform(0.2, 0.5, G)
    g = rng.integers(0, G, n_cells)
    lines = np.array(['A', 'B', 'C'][:len(alpha_x)])
    c = rng.integers(0, len(lines), n_cells)
    a = np.asarray(alpha_x)[c]
    sf = np.exp(rng.normal(0, 0.3, n_cells))
    log2_x = rng.normal(log2_eff[g], sigma_eff[g])
    mean = a * 2.0 ** log2_x * sf
    y = rng.negative_binomial(phi_x, phi_x / (phi_x + mean))
    other = rng.negative_binomial(5, 5 / (5 + 10 * sf[None]), (n_other_genes, n_cells))
    cells = [f"c{i}" for i in range(n_cells)]
    meta = pd.DataFrame({'cell': cells, 'guide': np.array(guides)[g],
                         'target': np.where(g < n_ntc, 'ntc', 'CIS'),
                         'cell_line': lines[c], 'sum_factor': sf})
    counts = pd.DataFrame(np.vstack([y[None], other]), columns=cells,
                          index=['CIS'] + [f"gene{j}" for j in range(n_other_genes)])
    truth = dict(log2_eff=dict(zip(guides, log2_eff)), log2_x=dict(zip(cells, log2_x)),
                 alpha_x=dict(zip(lines, alpha_x)), phi_x=phi_x)
    return meta, counts, truth


def score(est, lo, hi, truth):
    err = est - truth
    return {'rmse': round(float(np.sqrt(np.mean(err ** 2))), 4),
            'cov90': round(float(np.mean((truth >= lo) & (truth <= hi))), 3),
            'mean_width90': round(float(np.mean(hi - lo)), 4)}


def main(svi_niters=None, seed=0):
    torch.set_num_threads(1)
    meta, counts, truth = simulate(seed)
    model = bayesDREAM(meta=meta, counts=counts, cis_gene='CIS', label='cis_benchmark',
                       output_dir='./benchmark_output', device='cpu')
    model.set_technical_groups(['cell_line'])
    code_of = dict(model.meta[['cell_line', 'technical_group_code']].drop_duplicates().values)
    lines_by_code = sorted(code_of, key=code_of.get)
    ref = lines_by_code[0]
    # alpha_x relative to the reference group, and o_x, as fit_ntc would provide them
    model.alpha_x_prefit = torch.tensor([truth['alpha_x'][ln] / truth['alpha_x'][ref]
                                         for ln in lines_by_code], dtype=torch.float32)
    model.get_modality('cis').posterior_samples_ntc = {
        'o_x': torch.tensor([[truth['phi_x'] ** -0.5]], dtype=torch.float32)}
    # x in fit_cis is in units of the reference group
    shift = np.log2(truth['alpha_x'][ref])

    cells = model.meta['cell'].values
    guide_codes = model.meta['guide_code'].values
    G = int(guide_codes.max()) + 1
    guide_of_code = model.meta.groupby('guide_code')['guide'].first().reindex(range(G)).values
    true_eff = np.array([truth['log2_eff'][gd] for gd in guide_of_code]) + shift
    true_x = np.array([truth['log2_x'][cl] for cl in cells]) + shift
    results = {'seed': seed, 'n_cells': len(cells), 'n_guides': G}

    # ---- SVI (current fit_cis) ----
    kw = {} if svi_niters is None else {'niters': svi_niters}
    t0 = time.time()
    # force=True skips the low-expression check, which needs the full fit_ntc posterior
    model.fit_cis(sum_factor_col='sum_factor', force=True, **kw)
    t_svi = time.time() - t0
    ps = model.posterior_samples_cis
    S = ps['x_eff_g'].shape[0]
    l2eff = np.log2(ps['x_eff_g'].numpy().reshape(S, -1))
    l2x = ps['log_x_true'].numpy().reshape(S, -1)
    q = lambda a: (np.median(a, 0), np.quantile(a, 0.05, 0), np.quantile(a, 0.95, 0))
    results['svi'] = {'niters': svi_niters or 100_000,
                      'log2_x_eff': score(*q(l2eff), true_eff),
                      'log2_x_cell': score(*q(l2x), true_x),
                      'time_s': round(t_svi, 1)}

    # ---- CAVI ----
    sf = model.get_modality(model.primary_modality).sum_factors.loc[cells, 'sum_factor'].values
    a_cell = model.alpha_x_prefit.numpy()[model.meta['technical_group_code'].values]
    y = model.get_modality('cis').counts.loc[model.get_modality('cis').feature_meta.index[0]].values \
        if isinstance(model.get_modality('cis').counts, pd.DataFrame) \
        else np.asarray(model.get_modality('cis').counts).reshape(-1)
    r = fit_cis_negbinom_cavi(y, guide_codes, sf, a_cell, truth['phi_x'], verbose=False)
    ci = lambda m, sd: (m, m - Z90 * sd, m + Z90 * sd)
    results['cavi_laplace'] = {
        'log2_x_eff': score(*ci(r['log2_x_eff_mean'], r['log2_x_eff_sd']), true_eff),
        'log2_x_cell': score(*ci(r['log2_x_true_mean'], r['log2_x_true_sd']), true_x),
        'time_s': round(r['time_s'], 2), 'n_iter': r['n_iter'],
        'laplace_iter': r['laplace_rounds_run'], 'converged': bool(r['converged'] and r['laplace_converged'])}
    results['cavi_only'] = {
        'log2_x_eff': score(*ci(r['log2_x_eff_mean_cavi'], r['log2_x_eff_sd_cavi']), true_eff),
        'log2_x_cell': score(*ci(r['log2_x_true_mean_cavi'], r['log2_x_true_sd_cavi']), true_x)}
    results['meanfield_theta_cov90'] = score(
        *ci(r['log2_x_eff_mean_cavi'], r['log2_x_eff_sd_meanfield']), true_eff)['cov90']
    results['sd_ratio_cavi_over_laplace'] = {
        'log2_x_eff': round(float(np.mean(r['log2_x_eff_sd_cavi'] / r['log2_x_eff_sd'])), 3),
        'log2_x_cell': round(float(np.mean(r['log2_x_true_sd_cavi'] / r['log2_x_true_sd'])), 3),
        'log2_x_eff_meanfield': round(float(np.mean(r['log2_x_eff_sd_meanfield'] / r['log2_x_eff_sd'])), 3)}
    print("RESULT", json.dumps(results))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None,
         int(sys.argv[2]) if len(sys.argv) > 2 else 0)
