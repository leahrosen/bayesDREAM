"""
Compare fit_ntc(method='svi') and fit_ntc(method='cavi') on real counts without ground truth.

Calibration is checked by splitting the cells into two random halves (stratified by
technical group) and fitting each half separately. For every gene and parameter,
    z = (estimate_A - estimate_B) / sqrt(sd_A^2 + sd_B^2)
should be roughly N(0, 1) if the reported uncertainties are right, so |z| < 1.645 for
about 90 % of genes. Also reported: agreement between the methods on the full data,
runtime, and a posterior predictive check of the fraction of zero counts per gene.

Usage (one fit per process, so they can run in parallel):
    python compare_ntc_svi_cavi_realdata.py fit  --counts C.csv [--meta M.csv] --method svi  --part A --out DIR
    python compare_ntc_svi_cavi_realdata.py fit  ... --part full
    python compare_ntc_svi_cavi_realdata.py summarize --out DIR

Without --meta, all cells are treated as NTC and the technical group is the barcode prefix
before the first '_' (e.g. 'L1', 'L2'); sum factors are library size over median.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

Z90 = 1.6449


def load(counts_path, meta_path, n_cells, seed):
    counts = pd.read_csv(counts_path, index_col=0)
    if meta_path:
        meta = pd.read_csv(meta_path)
        meta = meta[meta['target'] == 'ntc']
    else:
        lib = counts.sum(0)
        meta = pd.DataFrame({'cell': counts.columns,
                             'cell_line': [c.split('_')[0] for c in counts.columns],
                             'target': 'ntc', 'guide': 'NTC',
                             'sum_factor': (lib / lib.median()).values})
        meta = meta[meta['sum_factor'] > 0]
    rng = np.random.default_rng(seed)
    if n_cells and len(meta) > n_cells:
        frac = n_cells / len(meta)
        keep = np.concatenate([rng.choice(idx, int(round(frac * len(idx))), replace=False)
                               for idx in meta.groupby('cell_line').indices.values()])
        meta = meta.iloc[np.sort(keep)]
    # random halves, stratified by technical group
    half = np.empty(len(meta), dtype=object)
    for _, idx in meta.groupby('cell_line').indices.items():
        perm = rng.permutation(idx)
        half[perm[: len(perm) // 2]] = 'A'
        half[perm[len(perm) // 2:]] = 'B'
    meta = meta.assign(half=half).reset_index(drop=True)
    return counts, meta


def fit(args):
    import torch
    from bayesDREAM import bayesDREAM
    torch.set_num_threads(args.threads)
    counts, meta = load(args.counts, args.meta, args.n_cells, args.seed)
    if args.part != 'full':
        meta = meta[meta['half'] == args.part].reset_index(drop=True)
    sub = counts[meta['cell']]
    model = bayesDREAM(meta=meta, counts=sub, label=f'{args.method}_{args.part}',
                       output_dir=os.path.join(args.out, 'models'), device='cpu')
    model.set_technical_groups(['cell_line'])
    t0 = time.time()
    model.fit_ntc(sum_factor_col='sum_factor', nsamples=500, method=args.method,
                  niters=args.svi_niters if args.method == 'svi' else None)
    dt = time.time() - t0
    mod = model.get_modality('gene')
    ps = mod.posterior_samples_ntc
    T = mod.dims['n_features']
    l2a = ps['log2_alpha_y'].numpy().reshape(ps['log2_alpha_y'].shape[0], -1, T)
    logmu = np.log(ps['mu_ntc'].numpy().reshape(-1, T))
    logphi = -2.0 * np.log(ps['o_y'].numpy().reshape(-1, T))
    codes = model.meta[['cell_line', 'technical_group_code']].drop_duplicates()
    np.savez(os.path.join(args.out, f'{args.method}_{args.part}.npz'),
             genes=np.asarray(mod.feature_ids, dtype=str),
             l2a_mean=l2a.mean(0), l2a_sd=l2a.std(0),
             logmu_mean=logmu.mean(0), logmu_sd=logmu.std(0),
             logphi_mean=logphi.mean(0), logphi_sd=logphi.std(0),
             group_order=codes.sort_values('technical_group_code')['cell_line'].values.astype(str),
             time_s=dt)
    # posterior predictive: fraction of zeros per gene at the posterior means
    y = sub.loc[list(mod.feature_ids)].values.T
    s = meta['sum_factor'].values
    g = model.meta['technical_group_code'].values
    a = np.vstack([np.zeros((1, T)), l2a.mean(0)])
    mu = np.exp(logmu.mean(0))[None] * 2.0 ** a[g] * s[:, None]
    phi = np.exp(logphi.mean(0))[None]
    p0 = (phi / (phi + mu)) ** phi
    np.savez(os.path.join(args.out, f'{args.method}_{args.part}_ppc.npz'),
             obs_zero=(y == 0).mean(0), pred_zero=p0.mean(0))
    print(f"done {args.method} {args.part}: {dt:.1f} s, T={T}")


def summarize(args):
    def ld(name):
        return dict(np.load(os.path.join(args.out, name + '.npz'), allow_pickle=True))
    out = {}
    for method in ('svi', 'cavi'):
        A, B = ld(f'{method}_A'), ld(f'{method}_B')
        assert list(A['group_order']) == list(B['group_order'])
        common = np.intersect1d(A['genes'], B['genes'])
        ia = np.array([np.where(A['genes'] == x)[0][0] for x in common])
        ib = np.array([np.where(B['genes'] == x)[0][0] for x in common])
        res = {'n_genes': int(len(common)), 'time_s': [float(A['time_s']), float(B['time_s'])]}
        for key in ('l2a', 'logmu', 'logphi'):
            ma, sa = A[key + '_mean'][..., ia], A[key + '_sd'][..., ia]
            mb, sb = B[key + '_mean'][..., ib], B[key + '_sd'][..., ib]
            z = (ma - mb) / np.sqrt(sa ** 2 + sb ** 2)
            ok = np.isfinite(z)  # genes absent from one half are NaN in both methods
            res[key] = {'frac_abs_z_lt_1.645': float(np.mean(np.abs(z[ok]) < Z90)),
                        'sd_z': float(np.std(z[ok])), 'n': int(ok.sum()),
                        'median_sd': float(np.nanmedian(np.concatenate([sa.ravel(), sb.ravel()])))}
        ppc = ld(f'{method}_full_ppc') if os.path.exists(os.path.join(args.out, f'{method}_full_ppc.npz')) else None
        if ppc is not None:
            res['ppc_zero_frac_mae'] = float(np.nanmean(np.abs(ppc['obs_zero'] - ppc['pred_zero'])))
        out[method] = res
    if all(os.path.exists(os.path.join(args.out, f'{m}_full.npz')) for m in ('svi', 'cavi')):
        S, Cv = ld('svi_full'), ld('cavi_full')
        common = np.intersect1d(S['genes'], Cv['genes'])
        i_s = np.array([np.where(S['genes'] == x)[0][0] for x in common])
        i_c = np.array([np.where(Cv['genes'] == x)[0][0] for x in common])
        agree = {}
        for key in ('l2a', 'logmu', 'logphi'):
            a, b = S[key + '_mean'][..., i_s].ravel(), Cv[key + '_mean'][..., i_c].ravel()
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            agree[key] = {'pearson_r': float(np.corrcoef(a, b)[0, 1]),
                          'median_abs_diff': float(np.median(np.abs(a - b)))}
        out['agreement_full'] = agree
        out['time_full_s'] = {'svi': float(S['time_s']), 'cavi': float(Cv['time_s'])}
    print(json.dumps(out, indent=1))
    with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['fit', 'summarize'])
    p.add_argument('--counts')
    p.add_argument('--meta', default=None)
    p.add_argument('--method', choices=['svi', 'cavi'])
    p.add_argument('--part', choices=['A', 'B', 'full'])
    p.add_argument('--out', required=True)
    p.add_argument('--n-cells', type=int, default=None)
    p.add_argument('--svi-niters', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=1)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fit(a) if a.mode == 'fit' else summarize(a)
