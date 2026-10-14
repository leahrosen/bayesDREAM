"""
Compare the SVI fit_ntc with Polya-Gamma CAVI (bayesDREAM.fitting.ntc_cavi) on simulated
NTC data with known baseline means mu_t, technical group effects a_ct and dispersions phi_t.

Reports RMSE and 90 % interval coverage for log2 a_ct, log mu_t and log phi_t, and runtime.

Usage:
    python examples/diagnostics/benchmark_ntc_cavi.py [svi_niters]
"""

import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from bayesDREAM import bayesDREAM
from bayesDREAM.fitting.ntc_cavi import fit_ntc_negbinom_cavi

Z90 = 1.6449


def simulate(seed=0, n_cells=2000, n_genes=200, n_groups=3):
    rng = np.random.default_rng(seed)
    lines = np.array(['A', 'B', 'C'][:n_groups])
    line = rng.choice(lines, n_cells)
    sf = np.exp(rng.normal(0, 0.3, n_cells))
    mu = np.exp(rng.uniform(np.log(0.1), np.log(100), n_genes))
    phi = np.exp(rng.uniform(np.log(1), np.log(50), n_genes))
    log2_a = {'A': np.zeros(n_genes)}
    for ln in lines[1:]:
        log2_a[ln] = rng.normal(0, 0.5, n_genes)
    l2a_cells = np.stack([log2_a[ln] for ln in line])
    mean = mu[None] * 2.0 ** l2a_cells * sf[:, None]
    y = rng.negative_binomial(phi[None], phi[None] / (phi[None] + mean))
    cells = [f"c{i}" for i in range(n_cells)]
    genes = [f"g{t}" for t in range(n_genes)]
    meta = pd.DataFrame({'cell': cells, 'guide': 'NTC_1', 'target': 'ntc',
                         'cell_line': line, 'sum_factor': sf})
    counts = pd.DataFrame(y.T, index=genes, columns=cells)
    return meta, counts, dict(mu=mu, phi=phi, log2_a=log2_a)


def score(est, sd, truth):
    err = est - truth
    return {'rmse': float(np.sqrt(np.mean(err ** 2))),
            'cov90': float(np.mean(np.abs(err) <= Z90 * sd))}


def main(svi_niters=None):
    meta, counts, truth = simulate()
    model = bayesDREAM(meta=meta, counts=counts, label='ntc_benchmark',
                       output_dir='./benchmark_output', device='cpu')
    model.set_technical_groups(['cell_line'])
    code_of = dict(model.meta[['cell_line', 'technical_group_code']].drop_duplicates().values)
    lines_by_code = sorted(code_of, key=code_of.get)
    ref = lines_by_code[0]

    # Truth relative to the reference group used by the model
    true_l2a = np.stack([truth['log2_a'][ln] - truth['log2_a'][ref] for ln in lines_by_code[1:]])
    true_logmu = np.log(truth['mu']) + np.log(2.0) * truth['log2_a'][ref]
    true_logphi = np.log(truth['phi'])

    results = {}

    # ---- SVI (current fit_ntc) ----
    t0 = time.time()
    model.fit_ntc(sum_factor_col='sum_factor', niters=svi_niters, nsamples=500)
    t_svi = time.time() - t0
    ps = model.get_modality('gene').posterior_samples_ntc
    l2a = ps['log2_alpha_y'].numpy().reshape(ps['log2_alpha_y'].shape[0], -1, counts.shape[0])
    logmu = np.log(ps['mu_ntc'].numpy().reshape(-1, counts.shape[0]))
    logphi = -2.0 * np.log(ps['o_y'].numpy().reshape(-1, counts.shape[0]))
    q = lambda a: (a.mean(0), (np.quantile(a, 0.95, 0) - np.quantile(a, 0.05, 0)) / (2 * Z90))
    results['svi'] = {'log2_alpha': score(*q(l2a), true_l2a), 'log_mu': score(*q(logmu), true_logmu),
                      'log_phi': score(*q(logphi), true_logphi), 'time_s': round(t_svi, 1)}

    # ---- CAVI ----
    groups = model.meta['technical_group_code'].values
    r = fit_ntc_negbinom_cavi(counts[model.meta['cell']].values.T, groups,
                              model.meta['sum_factor'].values, verbose=False)
    results['cavi'] = {'log2_alpha': score(r['log2_alpha_mean'], r['log2_alpha_sd'], true_l2a),
                       'log_mu': score(r['log_mu_mean'], r['log_mu_sd'], true_logmu),
                       'log_phi': score(r['log_phi_mean'], r['log_phi_sd'], true_logphi),
                       'time_s': round(r['time_s'], 1), 'n_iter': r['n_iter']}
    print("RESULT", json.dumps(results))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
