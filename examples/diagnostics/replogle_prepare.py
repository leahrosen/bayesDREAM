"""
Extract subsets of the Replogle et al. 2022 K562 essential-gene Perturb-seq raw counts
(K562_essential_raw_singlecell_01.h5ad, figshare+ 10.25452/figshare.plus.20029387) in the
layout bayesDREAM expects: counts (genes x cells, CSV) and cell metadata (CSV).

Two modes:
  ntc  -- non-targeting cells from a few gem groups and a random set of expressed genes,
          for the fit_ntc comparison (compare_ntc_svi_cavi_realdata.py --meta ...)
  cis  -- for each chosen target gene: its cells plus subsampled non-targeting cells,
          with the target gene included among the features, for the fit_cis comparison

Metadata columns: cell, guide (sgID_AB), target ('ntc' for non-targeting, else gene),
cell_line ('gem<gem_group>', used as technical group), sum_factor (UMI_count / median).

Usage:
  python replogle_prepare.py ntc --h5ad FILE --out DIR [--n-gem-groups 8] [--n-genes 300]
  python replogle_prepare.py cis --h5ad FILE --out DIR --targets T1,T2,... [--n-ntc 3000] [--n-genes 200]
"""

import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd


def _read_rows(adata, rows):
    rows = np.sort(np.asarray(rows))
    return rows, np.asarray(adata.X[rows, :])


def _meta(obs, rows, median_umi):
    o = obs.iloc[rows]
    is_ntc = (o['gene'].astype(str) == 'non-targeting').values
    return pd.DataFrame({
        'cell': o.index.astype(str),
        'guide': o['sgID_AB'].astype(str).values,
        'target': np.where(is_ntc, 'ntc', o['gene'].astype(str).values),
        'cell_line': 'gem' + o['gem_group'].astype(str).values,
        'sum_factor': (o['UMI_count'].values / median_umi).astype(float),
    })


def _write(out, counts, gene_names, meta, tag):
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame(counts.T.astype(np.int32), index=gene_names, columns=meta['cell'].values)
    df.to_csv(os.path.join(out, f'{tag}_counts.csv'))
    meta.to_csv(os.path.join(out, f'{tag}_meta.csv'), index=False)
    print(f"[{tag}] {df.shape[0]} genes x {df.shape[1]} cells -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['ntc', 'cis'])
    p.add_argument('--h5ad', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n-gem-groups', type=int, default=8)
    p.add_argument('--n-genes', type=int, default=300)
    p.add_argument('--n-ntc', type=int, default=3000)
    p.add_argument('--targets', default='')
    p.add_argument('--min-mean', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    adata = ad.read_h5ad(a.h5ad, backed='r')
    obs = adata.obs
    gene_names = adata.var['gene_name'].astype(str).values
    median_umi = float(np.median(obs['UMI_count'].values))
    ntc_rows = np.where(obs['gene'].astype(str).values == 'non-targeting')[0]

    if a.mode == 'ntc':
        gems = np.sort(obs['gem_group'].iloc[ntc_rows].unique())[: a.n_gem_groups]
        rows = ntc_rows[np.isin(obs['gem_group'].iloc[ntc_rows].values, gems)]
        rows, X = _read_rows(adata, rows)
        expressed = np.where(X.mean(0) >= a.min_mean)[0]
        genes = np.sort(rng.choice(expressed, min(a.n_genes, len(expressed)), replace=False))
        _write(a.out, X[:, genes], gene_names[genes], _meta(obs, rows, median_umi), 'ntc')
        return

    targets = [t for t in a.targets.split(',') if t]
    ntc_sub = np.sort(rng.choice(ntc_rows, min(a.n_ntc, len(ntc_rows)), replace=False))
    rows_ntc, X_ntc = _read_rows(adata, ntc_sub)
    expressed = np.where(X_ntc.mean(0) >= a.min_mean)[0]
    background = rng.choice(expressed, min(a.n_genes, len(expressed)), replace=False)
    for t in targets:
        t_rows = np.where(obs['gene'].astype(str).values == t)[0]
        t_col = np.where(gene_names == t)[0]
        if len(t_rows) == 0 or len(t_col) == 0:
            print(f"[skip] {t}: {len(t_rows)} cells, in matrix: {len(t_col) > 0}")
            continue
        rows_t, X_t = _read_rows(adata, t_rows)
        genes = np.unique(np.concatenate([t_col, background]))
        X = np.vstack([X_ntc[:, genes], X_t[:, genes]])
        meta = pd.concat([_meta(obs, rows_ntc, median_umi), _meta(obs, rows_t, median_umi)], ignore_index=True)
        _write(a.out, X, gene_names[genes], meta, f'cis_{t}')


if __name__ == '__main__':
    main()
