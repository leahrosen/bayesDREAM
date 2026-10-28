"""
Write one comparison notebook per Replogle cis gene (replogle_variants_<GENE>.ipynb)
from the template below. Re-run after editing the template:

    python comparative/notebooks/replogle_variants/make_notebooks.py
"""

import json
import os

GENES = ['GFI1B', 'HHEX', 'IKZF1', 'MYB', 'NFE2', 'RUNX1', 'TET2']
HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': text.strip('\n').splitlines(keepends=True)}


def cells_for(gene):
    return [
        md(f"""
# Replogle {gene}: fit-variant comparison

Compares the three publication-run fits of cis gene **{gene}** (`publication_runs/replogle/`, label prefix `replogle_20260908`):

| variant | NTC cells | `independent_mu_sigma` |
|---|---|---|
| `bm_indmu` | curated + batch-matched | True |
| `bm_noindmu` | curated + batch-matched | False |
| `all_indmu` | all NTC cells | True |

**a) cis:** per-cell `x_true` posterior median and tightness (95% CI width, posterior SD, both in log2 units), plus guide-level `x_eff_g`.
**b) trans:** Hill coefficient `n_a`, `observed_log2fc`, `full_log2fc`, and the fitted curve's y-log2FC at x log2FC = -1.

Helpers live in `comparative/replogle_variants.py`. Run on Dardel (outputs are under `/cfs/klemming`); the trans backfill reloads full models, so use a node with enough memory (`main`).
"""),
        code(f"""
%load_ext autotime

import os
import sys
import importlib

# Repo root: from the installed bayesDREAM package if this kernel's env has it
# (pip install -e .), else by walking up from the notebook's directory
# (comparative/notebooks/replogle_variants/) to the dir containing bayesDREAM/.
try:
    _bayesdream_pkg = importlib.import_module('bayesDREAM')
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(_bayesdream_pkg.__file__)))
except ModuleNotFoundError:
    REPO_ROOT = os.path.abspath(os.getcwd())
    while not os.path.isfile(os.path.join(REPO_ROOT, 'bayesDREAM', '__init__.py')):
        parent = os.path.dirname(REPO_ROOT)
        if parent == REPO_ROOT:
            raise ModuleNotFoundError(
                "bayesDREAM is not installed in this kernel's env and no bayesDREAM/ dir was found "
                "above the notebook's cwd -- pick the right kernel or `pip install -e .` the repo.")
        REPO_ROOT = parent
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
print('kernel:', sys.executable)
print('repo:  ', REPO_ROOT)

import matplotlib.pyplot as plt
import pandas as pd

from comparative.replogle_variants import (
    VARIANTS, CACHE_DIR,
    load_cis_cells, cis_variant_overview, cis_guide_summary, cis_guide_posterior, cis_cells_wide,
    plot_cis_pairwise, plot_cis_tightness, plot_guide_posteriors,
    backfill_variant, variant_specs, dependency_overlap,
)
from comparative.trans_param_compare import load_all, plot_pairwise_grid

GENE = '{gene}'
PLOT_DIR = os.path.join(CACHE_DIR, GENE, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(PLOT_DIR, f'{{GENE}}_{{name}}.png'), dpi=150, bbox_inches='tight')
"""),
        md("""
## a) cis fits

One row per cell per variant. The cis gene's target cells are in all three fits; `bm`'s NTC cells are a subset of `all_indmu`'s, so pairwise panels only use cells present in both fits. `log2fc_x_true` is relative to each variant's own NTC reference (median `x_true` over its NTC cells).
"""),
        code("""
cells = load_cis_cells(GENE)
cis_variant_overview(cells).round(3)
"""),
        md("### x_true posterior median"),
        code("""
fig = plot_cis_pairwise(cells, 'log2_x_true_median', title=f'{GENE}: ')
save(fig, 'cis_log2_x_true_median')
fig = plot_cis_pairwise(cells, 'log2fc_x_true', title=f'{GENE}: ', target_only=True)
save(fig, 'cis_log2fc_x_true_target')
"""),
        md("### x_true posterior tightness"),
        code("""
fig = plot_cis_tightness(cells, title=f'{GENE}: ')
save(fig, 'cis_tightness')
fig = plot_cis_pairwise(cells, 'ci_width', title=f'{GENE}: ', target_only=True)
save(fig, 'cis_ci_width_target')
"""),
        md("### Per guide"),
        code("""
guide_tab = cis_guide_summary(cells)
guide_tab[guide_tab['target'] == GENE].pivot(
    index='guide', columns='variant',
    values=['n_cells', 'log2fc_x_true', 'ci_width_median'],
).round(3)
"""),
        code("""
guide_post = pd.concat([cis_guide_posterior(GENE, v) for v in VARIANTS])
fig = plot_guide_posteriors(guide_post, title=f'{GENE}: ')
save(fig, 'cis_guide_x_eff')
"""),
        md("""
## b) trans fits

`backfill_variant()` reloads each variant's fitted model once (replaying its real `<label>_trans.yaml`, nothing re-fit), adds the y-log2FC at x log2FC = -1 to the production `trans_feature_summary_gene.csv`, and caches the result under `CACHE_DIR/<gene>/<variant>/`. Later runs reuse the cache; pass `force=True` to redo.
"""),
        code("""
for v in VARIANTS:
    backfill_variant(GENE, v)
"""),
        md("""
Loaded with `trans_param_compare.load_trans_summary()` (single-Hill or not-dependent genes only; Hill-B-only genes relabelled onto the `a` columns; `observed_log2fc`/`full_log2fc` signed by curve direction).

Dependent trans genes per variant (diagonal) and shared between pairs:
"""),
        code("""
dfs = load_all(variant_specs(), GENE)
dependency_overlap(dfs)
"""),
        md("""
Pairwise scatters, coloured by which variant(s) call the gene dependent. Genes dependent in neither are dropped (`exclude_categories`); pass `exclude_categories=()` to show them too.

- `n_a`: Hill coefficient
- `observed_log2fc`: log2FC over the observed x range
- `full_log2fc`: log2FC over x in [0, inf)
- `y_log2fc_at_xm1`: fitted y-log2FC at x log2FC = -1 (50% cis knockdown)
"""),
        code("""
for param in ['n_a', 'observed_log2fc', 'full_log2fc', 'y_log2fc_at_xm1']:
    fig = plot_pairwise_grid(dfs, param, GENE)
    save(fig, f'trans_{param}')
    plt.show()
"""),
    ]


def main():
    for gene in GENES:
        nb = {
            'cells': cells_for(gene),
            'metadata': {
                'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
                'language_info': {'name': 'python'},
            },
            'nbformat': 4, 'nbformat_minor': 5,
        }
        for i, c in enumerate(nb['cells']):
            c['id'] = f'{gene.lower()}-{i:02d}'
        path = os.path.join(HERE, f'replogle_variants_{gene}.ipynb')
        with open(path, 'w') as fh:
            json.dump(nb, fh, indent=1)
            fh.write('\n')
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
