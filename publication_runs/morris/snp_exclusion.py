"""
The 12 "worst offender" SNP-targeting guides that must be excluded from
every cis-gene run EXCEPT the one they themselves target -- these cause
extensive trans effects that would confound results for any OTHER cis gene.
This table is fixed reference data (not derivable from
Morris_gRNA2target_stats.csv -- confirmed with the dataset owner), used only
at config-generation time in generate_slurm.py, not by any per-job runtime
script.

Since this is a high-MOI dataset, cells with guides targeting OTHER genes
are otherwise kept (high-MOI models are supposed to see combinatorial
perturbations) -- this exclusion is deliberately narrow: only these 12
specific guides, not a blanket "exclude any non-cis-gene-targeting guide"
rule.

Matching is by SUBSTRING, not exact equality: SNP ids like "SNP-63" are
substrings of the actual guide names (e.g. "SNP-63-1", "SNP-63-2"), per the
`.str.contains()` pattern used in both reference preprocessing scripts. Do
not use `==`.

Note this substring approach means e.g. "SNP-63" would also match a
hypothetical "SNP-634-..." guide if one existed -- same risk already present
in both reference scripts' `.str.contains(pattern)` calls, not introduced
here. Spot-check `exclude_guides_for_cis_gene`'s output against your actual
guide_meta['guide'] values once if you're not already confident this can't
happen with your naming convention.
"""

from typing import Iterable, List

# (snp_id, feature) -- feature's leading token before " (CRE-N)" (if present)
# is the gene/locus this SNP belongs to.
SNP_TABLE = [
    ("SNP-63",  "GFI1B (CRE-1)"),
    ("SNP-498", "GFI1B (CRE-1)"),
    ("SNP-76",  "GFI1B (CRE-2)"),
    ("SNP-121", "NFE2 (CRE-1)"),
    ("SNP-83",  "NFE2 (CRE-1)"),
    ("SNP-120", "NFE2 (CRE-2)"),
    ("SNP-457", "IKZF1"),
    ("SNP-44",  "HHEX"),
    ("SNP-288", "RUNX1"),
    ("SNP-202", "miR-142"),
    ("SNP-59",  "miR-144/451 (CRE-1)"),
    ("SNP-200", "miR-144/451 (CRE-2)"),
]


def _feature_gene(feature: str) -> str:
    """"GFI1B (CRE-1)" -> "GFI1B"; "miR-142" -> "miR-142" (no CRE suffix to strip)."""
    return feature.split(" (")[0]


def exclude_guides_for_cis_gene(guide_names: Iterable[str], cis_gene: str, snp_table=SNP_TABLE) -> List[str]:
    """Guides (from `guide_names`) to exclude when fitting `cis_gene`: every
    guide containing one of the SNP_TABLE's SNP ids as a substring, EXCEPT
    the SNPs whose own associated gene/locus equals `cis_gene` (those are
    kept -- they're only "worst offenders" for OTHER cis genes).

    `cis_gene` should be a gene NAME/symbol (e.g. "GFI1B"), matching
    SNP_TABLE's feature-gene tokens -- not an Ensembl ID.
    """
    guide_names = list(guide_names)
    snps_to_exclude = [snp_id for snp_id, feature in snp_table if _feature_gene(feature) != cis_gene]
    if not snps_to_exclude:
        return []

    excluded = set()
    for snp_id in snps_to_exclude:
        excluded.update(g for g in guide_names if snp_id in g)
    return sorted(excluded)


def exclude_all_snp_guides(guide_names: Iterable[str], snp_table=SNP_TABLE) -> List[str]:
    """Every guide matching ANY SNP_TABLE entry, with no per-gene exception --
    for `ntc_shared`, which isn't "for" any one cis gene (cis_gene is
    deferred there), so there's no gene to be lenient toward the way
    `exclude_guides_for_cis_gene` is for a specific downstream cis-gene fit.
    Cells carrying these guides have extensive trans effects (see this
    module's docstring) and would otherwise confound the ONE shared
    alpha_y_prefit every cis gene's add_cis_gene() later extracts from --
    used because high-MOI's deferred-cis_gene fit_ntc() defaults to
    use_all_cells=True (see bayesDREAM/fitting/ntc.py), so without this,
    these cells are included in ntc_shared with no filtering at all.
    """
    guide_names = list(guide_names)
    excluded = set()
    for snp_id, _feature in snp_table:
        excluded.update(g for g in guide_names if snp_id in g)
    return sorted(excluded)
