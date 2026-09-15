"""
legacy_splicing.py
------------------
Monkey-patches add_splicing_modality back onto the bayesDREAM class for
legacy notebook use.  Not part of the bayesDREAM package proper.

Usage (at the top of your notebook, after importing bayesDREAM):

    import legacy_splicing  # noqa — side-effect only

Then call as normal:

    model.add_splicing_modality(
        sj_counts=sj_counts,
        sj_meta=sj_meta,
        splicing_types=['sj', 'donor', 'acceptor', 'exon_skip'],
        ...
    )
"""

import sys
import os
import types
import importlib.util

import bayesDREAM
import bayesDREAM.modality as _real_modality

# ---------------------------------------------------------------------------
# Load archive/splicing.py — work around its relative import (from .modality)
# by registering fake package entries in sys.modules before exec.
# ---------------------------------------------------------------------------
_archive_dir = os.path.join(os.path.dirname(bayesDREAM.__file__), "archive")
_archive_path = os.path.join(_archive_dir, "splicing.py")

# Register bayesDREAM.archive as a package so relative imports resolve
if 'bayesDREAM.archive' not in sys.modules:
    _archive_pkg = types.ModuleType('bayesDREAM.archive')
    _archive_pkg.__path__ = [_archive_dir]
    _archive_pkg.__package__ = 'bayesDREAM.archive'
    sys.modules['bayesDREAM.archive'] = _archive_pkg

# The relative `from .modality import Modality` resolves to
# bayesDREAM.archive.modality — point that at the real modality module.
sys.modules.setdefault('bayesDREAM.archive.modality', _real_modality)

# Now load splicing.py with the correct package context
_spec = importlib.util.spec_from_file_location(
    "bayesDREAM.archive.splicing",
    _archive_path,
)
_splicing_archive = importlib.util.module_from_spec(_spec)
_splicing_archive.__package__ = 'bayesDREAM.archive'
sys.modules['bayesDREAM.archive.splicing'] = _splicing_archive
_spec.loader.exec_module(_splicing_archive)

create_splicing_modality = _splicing_archive.create_splicing_modality


# ---------------------------------------------------------------------------
# The method itself (from archive/model_original.py)
# ---------------------------------------------------------------------------
def _add_splicing_modality(
    self,
    sj_counts,
    sj_meta,
    splicing_types=None,
    gene_counts=None,
    min_cell_total=1,
    min_total_exon=2,
):
    """
    Add splicing modalities (raw SJ counts, donor usage, acceptor usage, exon skipping).

    Parameters
    ----------
    sj_counts : pd.DataFrame
        Splice junction counts (junctions × cells).
    sj_meta : pd.DataFrame
        Junction metadata with required columns: coord.intron, chrom, intron_start,
        intron_end, strand, gene_name_start, gene_name_end.
        Optional: gene_id_start, gene_id_end (Ensembl ID support).
    splicing_types : str or list, default ['donor', 'acceptor', 'exon_skip']
        Which splicing metrics to compute: 'sj', 'donor', 'acceptor', 'exon_skip'.
    gene_counts : pd.DataFrame, optional
        Gene-level counts for SJ denominator (genes × cells).
        Defaults to model's primary counts (self.counts).
    min_cell_total : int
        Minimum reads for donor/acceptor features.
    min_total_exon : int
        Minimum reads for exon-skipping features.
    """
    if splicing_types is None:
        splicing_types = ['donor', 'acceptor', 'exon_skip']
    if isinstance(splicing_types, str):
        splicing_types = [splicing_types]

    if gene_counts is None:
        if hasattr(self, 'counts') and self.counts is not None:
            gene_counts_to_use = self.counts
        else:
            raise ValueError(
                "gene_counts must be provided or model must have been "
                "initialized with counts"
            )
    else:
        gene_counts_to_use = gene_counts

    valid_cells = self.meta['cell'].tolist()
    sj_cells = sj_counts.columns.tolist()
    common_sj_cells = [c for c in sj_cells if c in valid_cells]

    if len(common_sj_cells) == 0:
        raise ValueError("No overlapping cells between sj_counts and model cells")

    if len(common_sj_cells) < len(sj_cells):
        print(f"[INFO] Subsetting sj_counts from {len(sj_cells)} to "
              f"{len(common_sj_cells)} cells to match model")
        sj_counts_subset = sj_counts[common_sj_cells].copy()
    else:
        sj_counts_subset = sj_counts

    for stype in splicing_types:
        modality = create_splicing_modality(
            sj_counts=sj_counts_subset,
            sj_meta=sj_meta,
            splicing_type=stype,
            gene_counts=gene_counts_to_use,
            min_cell_total=min_cell_total,
            min_total_exon=min_total_exon,
        )
        self.add_modality(f'splicing_{stype}', modality)


# ---------------------------------------------------------------------------
# Patch onto the class
# ---------------------------------------------------------------------------
from bayesDREAM import bayesDREAM as _bayesDREAM_cls  # noqa

_bayesDREAM_cls.add_splicing_modality = _add_splicing_modality
print("[legacy_splicing] add_splicing_modality patched onto bayesDREAM class.")
