"""
Shared dataset configuration for the comparative/ tools.

Edit the paths/gene lists below as runs are added or re-labelled -- both
``trans_param_compare.py`` and ``dose_response_panels.py`` import their
dataset configuration from here, so there is exactly one place to update.

Everything here is driven by the real cluster layout of the three production
runs as of 2026-08-26:

- Domingo: low-MOI, CRISPRa+CRISPRi, 4 cis genes, ~91 shared trans genes
  (small enough for full per-gene dose-response panels).
- Morris: high-MOI, CRISPRi-only, transcriptome-wide trans, but fit_trans
  was only actually run to completion for 5 "primary" cis genes.
- Replogle: low-MOI (single-guide), CRISPRi-only, transcriptome-wide trans,
  fit with function_type='single_hill'; feature identifiers are Ensembl
  gene IDs (folder names + the 'feature'/'gene_name' column), with the
  real gene symbol carried separately in a 'gene_symbol' column.
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# ── Cell-line colour palettes for dose-response data points ──────────────────
# (passed as `color_palette=` to plot_xy_data's `color_by='cell_line'`)
PALETTE_A = {'CRISPRi': 'steelblue', 'CRISPRa': 'tomato'}   # Domingo (both arms)
PALETTE_B = {'CRISPRi': 'steelblue'}                        # Morris / Replogle (CRISPRi-only)


# ── Fit-curve colours (dataset identity), chosen to stay clear of the ────────
# cell-line data colours above (steelblue / tomato) and of each other.
# Colorblind-safe (ColorBrewer "Dark2" subset).
DATASET_COLORS = {
    'Domingo':  '#1b9e77',   # teal / bluish green
    'Morris':   '#7570b3',   # slate purple
    'Replogle': '#a6761d',   # ochre / brown
}


@dataclass
class DatasetSpec:
    """One fitted dataset, as needed by the comparative/ tools.

    Attributes
    ----------
    name : str
        Display name, also used as the DataFrame-suffix / legend label.
    color : str
        Fit-curve colour for dose-response overlays (DATASET_COLORS by default).
    cell_line_palette : dict
        Passed straight through to plot_xy_data(color_palette=...).
    cis_genes : list of str
        Cis genes with a *completed* fit_trans run (i.e. a
        trans_feature_summary_{modality}.csv actually on disk). Used to
        compute "shared cis genes" between two datasets -- keep this in
        sync with what's actually been run, not what's merely queued.
    symbol_col : str
        Column in trans_feature_summary_{modality}.csv holding the gene
        SYMBOL for that row (the cross-dataset join key). 'feature' for
        datasets whose feature identifiers already are gene symbols
        (Domingo, Morris); 'gene_symbol' for datasets indexed by a
        different identifier (Replogle uses Ensembl gene IDs).
    modality_name : str
        Which modality's trans summary to read (default 'gene').
    run_dir_fn : callable
        cis_gene (symbol) -> directory containing
        trans_feature_summary_{modality_name}.csv for that cis gene's run.
    save_for_plotting_dir_fn : callable, optional
        cis_gene (symbol) -> directory written by save_model_for_plotting()
        for that cis gene, if/when that preprocessing step has been run.
        Only needed for dose_response_panels.py's full per-gene curve
        reload -- leave as None for datasets where you haven't run it.
    init_sum_factor_col : str
        sum_factor_col used at model *construction* time (bayesDREAM.__init__),
        needed to reconstruct the model in dose_response_panels.py. This is
        NOT the sum_factor_col passed to plot_xy_data/fit_trans.
    plot_sum_factor_col : str, optional
        The sum_factor column dose_response_panels.py should plot against
        (passed to plot_xy_data's sum_factor_col). This is the *final*,
        most-adjusted column for this dataset's pipeline -- Domingo's own
        refit_sumfactor() writes 'sum_factor_new'; Morris/Replogle stop at
        adjust_ntc_sum_factor() and use 'sum_factor_adj'. If left None,
        dose_response_panels.pick_sum_factor_col() falls back to probing
        the reloaded model for the best available column -- set this
        explicitly once you know which column a given run actually has.
    force_single_cell_line : str, optional
        If set (e.g. 'CRISPRi'), overwrite model.meta['cell_line'] with this
        value after reload -- for CRISPRi-only datasets whose meta may be
        missing/inconsistent in that column (mirrors what the original
        GEX_comp_Doming_Morris.ipynb did for Morris).
    cis_gene_id_fn : callable, optional
        cis_gene (symbol) -> the identifier actually used in this dataset's
        saved counts/feature index, ONLY where that differs from the symbol
        (Replogle: Ensembl gene ID, via REPLOGLE_GENE_TO_ID.get -- see its
        module comment). None (default) for datasets where the feature index
        already IS the gene symbol (Domingo, Morris). dose_response_panels.py's
        load_model_for_plotting() uses this to translate before constructing
        bayesDREAM(cis_gene=...): counts_plot.npz's feature_names for Replogle
        are Ensembl IDs, so passing the bare symbol straight through raises
        "cis_gene 'GFI1B' not found in counts.index" against an
        ENSG00000... index.
    """
    name: str
    color: str
    cell_line_palette: Dict[str, str]
    cis_genes: List[str]
    symbol_col: str = 'feature'
    modality_name: str = 'gene'
    run_dir_fn: Optional[Callable[[str], str]] = None
    save_for_plotting_dir_fn: Optional[Callable[[str], str]] = None
    cis_gene_id_fn: Optional[Callable[[str], Optional[str]]] = None
    init_sum_factor_col: str = 'sum_factor'
    plot_sum_factor_col: Optional[str] = None
    force_single_cell_line: Optional[str] = None

    def trans_summary_path(self, cis_gene: str, modality_name: Optional[str] = None) -> str:
        modality_name = modality_name or self.modality_name
        if self.run_dir_fn is None:
            raise ValueError(f"[{self.name}] run_dir_fn is not configured.")
        d = self.run_dir_fn(cis_gene)
        path = os.path.join(d, f'trans_feature_summary_{modality_name}.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[{self.name}] no trans_feature_summary_{modality_name}.csv for cis gene "
                f"{cis_gene!r} at {path!r} -- has fit_trans()/save_trans_summary() been run "
                f"for this gene yet?"
            )
        return path

    def plotting_save_dir(self, cis_gene: str) -> str:
        if self.save_for_plotting_dir_fn is None:
            raise ValueError(
                f"[{self.name}] save_for_plotting_dir_fn is not configured for this dataset. "
                "Run save_model_for_plotting() (see save_for_plotting.py) in the fitting "
                "session for this (dataset, cis_gene) pair, then add its output directory "
                "to this DatasetSpec."
            )
        d = self.save_for_plotting_dir_fn(cis_gene)
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"[{self.name}] expected save_model_for_plotting() output at {d!r} for cis "
                f"gene {cis_gene!r}, but that directory doesn't exist. Run "
                "save_model_for_plotting(model, save_dir=...) in the fitting session first."
            )
        return d


# ── Replogle: gene symbol -> Ensembl gene ID ─────────────────────────────────
# Folder names under fit_trans/ are `papermill_<ensembl_id>`, and the
# 'feature'/'gene_name' column inside trans_feature_summary_gene.csv is also
# the Ensembl ID (see 10_bayesDREAM_fit_trans_MYB.ipynb: gene_meta_full sets
# gene_name := gene_id, keeping the real symbol in a separate 'gene_symbol'
# column). Extend this as more cis genes are fit.
REPLOGLE_GENE_TO_ID = {
    'MYB':   'ENSG00000118513',
    'NFE2':  'ENSG00000123405',
    'HHEX':  'ENSG00000152804',
    'RUNX1': 'ENSG00000159216',
    'GFI1B': 'ENSG00000165702',
    'TET2':  'ENSG00000168769',
    'IKZF1': 'ENSG00000185811',
}


# ── Concrete dataset specs ────────────────────────────────────────────────────

DOMINGO_OUTDIR = (
    '/cfs/klemming/projects/snic/lappalainen_lab1/users/Leah/data/Domingo2024/'
    'processed_Leah/BayesianModel_outs'
)
MORRIS_OUTDIR = (
    '/cfs/klemming/projects/snic/lappalainen_lab1/users/Leah/CRISPRmodelling/'
    'BayesianModelling/Morris_GEX/output'
)
REPLOGLE_OUTDIR = (
    '/cfs/klemming/projects/snic/lappalainen_lab1/users/lisetts/Replogle_data/'
    'bayesDREAM/output/fit_trans'
)
# Ad hoc save_model_for_plotting() export location used for the original
# Domingo/Morris GFI1B comparison -- reused here as the *convention* for any
# future gene, i.e. "{dataset}_{gene}_GEX". Nothing is created automatically;
# you still need to call save_model_for_plotting() yourself for each new
# (dataset, gene) pair before dose_response_panels.py can reload it.
COMPARATIVE_INPUT_DIR = (
    '/cfs/klemming/projects/snic/lappalainen_lab1/users/Leah/CRISPRmodelling/'
    'BayesianModelling/Comparative/input'
)

DOMINGO = DatasetSpec(
    name='Domingo',
    color=DATASET_COLORS['Domingo'],
    cell_line_palette=PALETTE_A,
    cis_genes=['GFI1B', 'NFE2', 'MYB', 'TET2'],
    symbol_col='feature',
    run_dir_fn=lambda g: os.path.join(DOMINGO_OUTDIR, f'domingo_20260806_{g}'),
    save_for_plotting_dir_fn=lambda g: os.path.join(COMPARATIVE_INPUT_DIR, f'Domingo_{g}_GEX'),
    init_sum_factor_col='sum_factor',
    plot_sum_factor_col='sum_factor_new',
)

MORRIS = DatasetSpec(
    name='Morris',
    color=DATASET_COLORS['Morris'],
    cell_line_palette=PALETTE_B,
    # Only these 5 "primary genes" got a completed transcriptome-wide fit_trans
    # run (see publication_runs/morris/config.yaml gene_selection.primary_genes).
    cis_genes=['GFI1B', 'NFE2', 'IKZF1', 'HHEX', 'RUNX1'],
    symbol_col='feature',
    run_dir_fn=lambda g: os.path.join(MORRIS_OUTDIR, f'morris_20260806_{g}'),
    save_for_plotting_dir_fn=lambda g: os.path.join(COMPARATIVE_INPUT_DIR, f'Morris_{g}_GEX'),
    init_sum_factor_col='sum_factor',
    plot_sum_factor_col='sum_factor_adj',
    force_single_cell_line='CRISPRi',
)

REPLOGLE = DatasetSpec(
    name='Replogle',
    color=DATASET_COLORS['Replogle'],
    cell_line_palette=PALETTE_B,
    cis_genes=sorted(REPLOGLE_GENE_TO_ID),
    symbol_col='gene_symbol',
    # NOT REPLOGLE_OUTDIR/papermill_<id>/ (that's your student's directory --
    # you have read access there, confirmed, but not write access, confirmed
    # 2026-08-26 via a real PermissionError backfilling trans_feature_summary_gene.csv
    # there). comparative/reconstruct_export_replogle.py writes the backfilled
    # summary CSV into save_for_plotting_dir_fn()'s directory instead (your own
    # Comparative/input/ tree, confirmed writable), so that's also where reads
    # need to look -- run_dir_fn and save_for_plotting_dir_fn are the same
    # directory for Replogle specifically (unlike Domingo/Morris, which back-
    # fill in place in their own production output_dir and additionally export
    # a copy here).
    run_dir_fn=lambda g: os.path.join(COMPARATIVE_INPUT_DIR, f'Replogle_{g}_GEX'),
    save_for_plotting_dir_fn=lambda g: os.path.join(COMPARATIVE_INPUT_DIR, f'Replogle_{g}_GEX'),
    cis_gene_id_fn=REPLOGLE_GENE_TO_ID.get,
    init_sum_factor_col='sum_factor',
    plot_sum_factor_col='sum_factor_adj',
    force_single_cell_line='CRISPRi',
)

ALL_DATASETS = {d.name: d for d in (DOMINGO, MORRIS, REPLOGLE)}
