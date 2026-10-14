"""
Benchmark trans function types on simulated CRISPRi dose-response data.

Simulates 3000 cells (5 NTC guides, 15 guides with graded knockdown of a cis gene)
and 60 trans genes: 20 null, 15 increasing, 15 decreasing and 10 non-monotone
(up then down). The true cis expression is given to the model, so only the trans
step is evaluated. Reports, per gene class, the RMSE of the posterior-mean log2
fold-change curve over the observed x range and the coverage of 90 % pointwise
credible bands, plus the AUC for separating null from non-null genes.

Usage:
    python examples/diagnostics/benchmark_logistic_hill.py logistic_hill 20000
    python examples/diagnostics/benchmark_logistic_hill.py additive_hill 20000
    python examples/diagnostics/benchmark_logistic_hill.py logistic_hill 5000 '{"logistic_guide": "normal"}'
"""

import json
import sys
import time

import numpy as np
import pandas as pd
import pyro
import torch
from scipy.stats import mannwhitneyu

from bayesDREAM import bayesDREAM
from bayesDREAM.utils import Hill_based_positive, Hill_based_positive_logK


def simulate(seed=0, n_cells=3000, n_null=20, n_up=15, n_down=15, n_bump=10):
    rng = np.random.default_rng(seed)
    kd = np.concatenate([np.ones(5), np.geomspace(0.04, 0.9, 15)])
    guides = [f"NTC_{i}" for i in range(5)] + [f"g{i}" for i in range(15)]
    targets = ['ntc'] * 5 + ['CIS'] * 15
    gi = rng.integers(0, len(guides), n_cells)
    cell_line = rng.choice(['A', 'B'], n_cells)
    sf = np.exp(rng.normal(0, 0.3, n_cells))
    x_ntc = 40.0
    x_true = x_ntc * kd[gi] * np.exp(rng.normal(0, 0.25, n_cells))
    x_counts = rng.negative_binomial(10, 10 / (10 + x_true * sf * np.where(cell_line == 'B', 1.2, 1.0)))

    kind = np.array(['null'] * n_null + ['up'] * n_up + ['down'] * n_down + ['bump'] * n_bump)
    T = len(kind)
    base = np.exp(rng.uniform(np.log(2), np.log(80), T))
    l2x_ntc = np.log2(x_ntc)
    lo, hi = np.log2(x_ntc * 0.04), l2x_ntc

    def hill_l2(v, l2K, h):  # Hill function written in log2 x
        return 1 / (1 + 2.0 ** (-h * (v - l2K)))

    curves = []
    for t in range(T):
        if kind[t] == 'null':
            curves.append(lambda v: 0 * v)
        elif kind[t] in ('up', 'down'):
            amp = rng.uniform(1.0, 3.0) * (1 if kind[t] == 'up' else -1)
            l2K, h = rng.uniform(lo + 1, hi - 0.5), rng.uniform(1, 4)
            curves.append(lambda v, a=amp, K=l2K, h=h: a * (hill_l2(v, K, h) - hill_l2(l2x_ntc, K, h)))
        else:
            a = rng.uniform(1.5, 3.0)
            K1 = rng.uniform(lo + 0.5, lo + 2.0)
            K2 = K1 + rng.uniform(1.5, 2.5)
            curves.append(lambda v, a=a, K1=K1, K2=K2: a * (hill_l2(v, K1, 3) - hill_l2(l2x_ntc, K1, 3))
                          - a * (hill_l2(v, K2, 3) - hill_l2(l2x_ntc, K2, 3)))
    f = np.stack([c(np.log2(x_true)) for c in curves])
    cl_eff = np.exp(rng.normal(0, 0.2, T))
    mu = base[:, None] * 2.0 ** f * np.where(cell_line == 'B', cl_eff[:, None], 1.0) * sf[None, :]
    phi = rng.uniform(5, 20, T)[:, None]
    y = rng.negative_binomial(phi, phi / (phi + mu))

    cells = [f"c{i}" for i in range(n_cells)]
    genes = [f"{k}_{t}" for t, k in enumerate(kind)]
    counts = pd.DataFrame(np.vstack([x_counts[None, :], y]), index=['CIS'] + genes, columns=cells)
    meta = pd.DataFrame({'cell': cells, 'guide': np.array(guides)[gi], 'target': np.array(targets)[gi],
                         'cell_line': cell_line, 'sum_factor': sf})
    return meta, counts, dict(kind=kind, curves=curves, x_true=x_true, l2x_ntc=l2x_ntc)


def logistic_curves(ps, l2x_grid, l2x_ref):
    """log2FC relative to l2x_ref for logistic_hill samples, shape [S, G, T]."""
    def eta(l2x):
        u = torch.tensor((np.atleast_1d(l2x) - ps['u_center']) / ps['u_scale'], dtype=torch.float32).view(-1, 1, 1, 1)
        V, k, n = ps['V'], ps['k'], ps['n']
        return (V * (torch.sigmoid(n * (u - k)) - torch.sigmoid(-n * k))).sum(-2)  # [G, S, T]
    return (eta(l2x_grid) - eta(l2x_ref)).permute(1, 0, 2).numpy()


def hill_curves(ps, l2x_grid, l2x_ref, function_type):
    """log2FC relative to l2x_ref for single/additive Hill samples, shape [S, G, T]."""
    ps = {k: v.squeeze() if isinstance(v, torch.Tensor) else v for k, v in ps.items()}

    def component(x, s):
        if f'log_K_{s}' in ps:
            return Hill_based_positive_logK(x, Vmax=ps[f'Vmax_{s}'], A=0, logK=ps[f'log_K_{s}'], n=ps[f'n_{s}'])
        return Hill_based_positive(x, Vmax=ps[f'Vmax_{s}'], A=0, K=ps[f'K_{s}'], n=ps[f'n_{s}'])

    def log2y(l2x):
        x = torch.tensor(2.0 ** np.atleast_1d(l2x), dtype=torch.float32).view(-1, 1, 1)
        y = ps['A'] + ps['alpha'] * component(x, 'a')
        if function_type == 'additive_hill':
            y = y + ps['beta'] * component(x, 'b')
        return torch.log2(y.clamp_min(1e-8))
    return (log2y(l2x_grid) - log2y(l2x_ref)).permute(1, 0, 2).numpy()


def evaluate(model, truth, function_type):
    ps = model.posterior_samples_trans
    l2x = np.log2(truth['x_true'])
    grid = np.linspace(np.quantile(l2x, 0.01), np.quantile(l2x, 0.99), 50)
    true = np.stack([c(grid) for c in truth['curves']], axis=1)  # [G, T]
    if function_type == 'logistic_hill':
        est = logistic_curves(ps, grid, truth['l2x_ntc'])
    else:
        est = hill_curves(ps, grid, truth['l2x_ntc'], function_type)
    rmse = np.sqrt(((est.mean(0) - true) ** 2).mean(0))
    lo, hi = np.quantile(est, [0.05, 0.95], axis=0)
    cover = ((true >= lo) & (true <= hi)).mean(0)
    score = np.abs(est).max(1).mean(0)
    kind = truth['kind']
    nonnull = kind != 'null'
    out = {'auc': float(mannwhitneyu(score[nonnull], score[~nonnull]).statistic / (nonnull.sum() * (~nonnull).sum()))}
    for k in ['null', 'up', 'down', 'bump']:
        out[f'rmse_{k}'] = round(float(np.median(rmse[kind == k])), 3)
        out[f'cov90_{k}'] = round(float(cover[kind == k].mean()), 3)
    return out


if __name__ == '__main__':
    function_type = sys.argv[1] if len(sys.argv) > 1 else 'logistic_hill'
    niters = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}

    pyro.set_rng_seed(0)
    meta, counts, truth = simulate(seed=0)
    model = bayesDREAM(meta=meta, counts=counts, cis_gene='CIS', output_dir='./benchmark_output',
                       label='logistic_benchmark', device='cpu')
    model.set_technical_groups(['cell_line'])
    model.fit_technical(sum_factor_col='sum_factor', niters=3000, nsamples=200)
    x = torch.tensor(truth['x_true'], dtype=torch.float32)
    model.x_true, model.x_true_type = x, 'point'
    model.log2_x_true, model.log2_x_true_type = torch.log2(x), 'point'

    t0 = time.time()
    model.fit_trans(sum_factor_col='sum_factor', function_type=function_type, niters=niters,
                    nsamples=300, **extra)
    result = evaluate(model, truth, function_type)
    result.update(time_s=round(time.time() - t0, 1), n_steps=len(model.losses_trans))
    print("RESULT", function_type, extra, json.dumps(result))
