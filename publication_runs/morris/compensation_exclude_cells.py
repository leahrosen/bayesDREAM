"""
Compensation exclude_cells rule for Morris: restrict check_systematic_shift()
to NTC cells + cells targeting the cis gene via a guide with padj<0.05 for
that gene (from Morris_gRNA2target_stats.csv), dropping cells whose only
cis-gene-targeting guide(s) are NOT significant. Replaces (does not
supplement) the SNP-499/500-specific exclusion from an earlier exploratory
version of the Morris GFI1B script -- that rule was specific to GFI1B and
is not part of the production pipeline.

Since this is high-MOI, a cell with guides targeting OTHER genes besides the
cis gene is untouched by this rule either way -- check_systematic_shift()
itself already restricts the comparison to NTC + cells whose `target_col`
equals the cis gene (see its docstring), so this function only needs to
decide, among cells that DO target the cis gene, which ones to drop.

Plugged in via run_compensation.py's dynamic exclude_cells resolver:

    compensation:
      args:
        exclude_cells:
          module: compensation_exclude_cells
          function: compute_padj_exclude_cells
          kwargs:
            stats_csv: /path/to/Morris_gRNA2target_stats.csv
            cis_gene_ensembl_id: ENSG00000169442   # baked in per-gene by generate_slurm.py
            padj_threshold: 0.05
"""

import numpy as np
import pandas as pd


def compute_padj_exclude_cells(
    model,
    cfg,
    stats_csv: str,
    cis_gene_ensembl_id: str,
    padj_threshold: float = 0.05,
    gene_use_col: str = "gene_use",
    grna_col: str = "gRNA ID new",
    padj_col: str = "padj",
):
    stats = pd.read_csv(stats_csv)
    good_guides = set(
        stats.loc[(stats[gene_use_col] == cis_gene_ensembl_id) & (stats[padj_col] < padj_threshold), grna_col]
    )

    guide_names = model.guide_meta["guide"].values
    cis_gene_name = model.cis_gene  # name form -- must match guide_targets_dict's values

    targets_this_gene = np.array([
        cis_gene_name in model.guide_targets_dict.get(g, []) for g in guide_names
    ])
    is_good_guide = np.isin(guide_names, list(good_guides))

    good_mask = targets_this_gene & is_good_guide
    bad_targeting_mask = targets_this_gene & ~is_good_guide

    ga = model.guide_assignment  # (cells, guides)
    n_cells = ga.shape[0]
    has_good = (ga[:, good_mask].sum(axis=1) > 0) if good_mask.any() else np.zeros(n_cells, dtype=bool)
    has_bad_only_targeting = (ga[:, bad_targeting_mask].sum(axis=1) > 0) if bad_targeting_mask.any() else np.zeros(n_cells, dtype=bool)

    exclude_mask = has_bad_only_targeting & ~has_good
    exclude_cells = model.meta.loc[exclude_mask, "cell"].tolist()

    print(f"[compensation_exclude_cells] {cis_gene_name} ({cis_gene_ensembl_id}): "
          f"{good_mask.sum()} significant guide(s), {bad_targeting_mask.sum()} non-significant "
          f"targeting guide(s), excluding {len(exclude_cells)} cell(s)")
    return exclude_cells
