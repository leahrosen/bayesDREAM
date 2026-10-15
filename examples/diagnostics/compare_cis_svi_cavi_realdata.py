"""
Compare fit_cis(method='svi') and fit_cis(method='cavi') on real Perturb-seq counts.

Input: <prefix>_counts.csv and <prefix>_meta.csv as written by replogle_prepare.py cis
(one target gene plus non-targeting cells). The NTC step uses fit_ntc(method='cavi') for
both methods, so only the cis fit differs. Gem groups are coarsened into 4 technical
groups so that each group keeps enough NTC cells after splitting.

For each fit the script records, on the log2 scale, the level of the targeting guide
(x_eff_g; the guide with the most cells if a target has several) and its knockdown relative to the mean of the NTC guides, with posterior SDs.
Calibration is checked across targets by fitting two random halves of the cells
(stratified by technical group and by target vs NTC):
    z = (est_A - est_B) / sqrt(sd_A^2 + sd_B^2).

Usage:
    python compare_cis_svi_cavi_realdata.py fit --prefix DIR/cis_GENE --target GENE --method svi --part A --out OUT
    python compare_cis_svi_cavi_realdata.py summarize --out OUT
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

Z90 = 1.6449


def fit(a):
    import torch
    from bayesDREAM import bayesDREAM
    torch.set_num_threads(a.threads)
    counts = pd.read_csv(a.prefix + '_counts.csv', index_col=0)
    meta = pd.read_csv(a.prefix + '_meta.csv')
    gem = meta['cell_line'].str.replace('gem', '').astype(int)
    meta['cell_line'] = 'batch' + ((gem - 1) * 4 // 48).astype(str)
    if a.part != 'full':
        rng = np.random.default_rng(a.seed)
        half = np.empty(len(meta), dtype=object)
        for _, idx in meta.groupby(['cell_line', 'target']).indices.items():
            perm = rng.permutation(idx)
            half[perm[: len(perm) // 2]] = 'A'
            half[perm[len(perm) // 2:]] = 'B'
        meta = meta[half == a.part].reset_index(drop=True)
    counts = counts[meta['cell']]
    model = bayesDREAM(meta=meta, counts=counts, cis_gene=a.target, label=f'{a.target}_{a.method}_{a.part}',
                       output_dir=os.path.join(a.out, 'models'), device='cpu')
    model.set_technical_groups(['cell_line'])
    model.fit_ntc(sum_factor_col='sum_factor', nsamples=200, method='cavi')
    t0 = time.time()
    model.fit_cis(sum_factor_col='sum_factor', nsamples=1000, method=a.method)
    dt = time.time() - t0

    codes = model.meta[['guide', 'guide_code', 'target']].drop_duplicates()
    # Some targets have more than one guide construct; use the one with the most cells.
    t_cells = model.meta.loc[model.meta['target'] == a.target, 'guide_code'].value_counts()
    t_code = [int(t_cells.index[0])]
    n_codes = codes.loc[codes['target'] == 'ntc', 'guide_code'].values
    l2 = torch.log2(model.posterior_samples_cis['x_eff_g']).numpy()     # [S, G]
    target_l2 = l2[:, t_code[0]]
    kd = target_l2 - l2[:, n_codes].mean(1)
    res = {'target': a.target, 'method': a.method, 'part': a.part, 'time_s': dt,
           'n_cells': int(len(model.meta)), 'n_guides': int(len(codes)),
           'target_l2_mean': float(target_l2.mean()), 'target_l2_sd': float(target_l2.std()),
           'kd_mean': float(kd.mean()), 'kd_sd': float(kd.std())}
    with open(os.path.join(a.out, f'{a.target}_{a.method}_{a.part}.json'), 'w') as fh:
        json.dump(res, fh)
    print("done", json.dumps(res))


def summarize(a):
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(a.out, '*.json')) if not f.endswith('summary.json')]
    df = pd.DataFrame(rows)
    out = {}
    for method, d in df.groupby('method'):
        A = d[d.part == 'A'].set_index('target')
        B = d[d.part == 'B'].set_index('target')
        common = A.index.intersection(B.index)
        r = {'n_targets': int(len(common)),
             'median_time_s_half': float(pd.concat([A.time_s, B.time_s]).median())}
        for q in ('target_l2', 'kd'):
            z = (A.loc[common, q + '_mean'] - B.loc[common, q + '_mean']) / np.sqrt(
                A.loc[common, q + '_sd'] ** 2 + B.loc[common, q + '_sd'] ** 2)
            r[q] = {'z': [round(float(v), 2) for v in z], 'frac_abs_z_lt_1.645': float(np.mean(np.abs(z) < Z90)),
                    'median_sd': float(pd.concat([A[q + '_sd'], B[q + '_sd']]).median())}
        out[method] = r
    full = df[df.part == 'full'].pivot(index='target', columns='method', values=['kd_mean', 'kd_sd', 'time_s'])
    out['full'] = json.loads(full.round(4).to_json(orient='index'))
    print(json.dumps(out, indent=1))
    with open(os.path.join(a.out, 'summary.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['fit', 'summarize'])
    p.add_argument('--prefix')
    p.add_argument('--target')
    p.add_argument('--method', choices=['svi', 'cavi'])
    p.add_argument('--part', choices=['A', 'B', 'full'])
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=1)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fit(a) if a.mode == 'fit' else summarize(a)
