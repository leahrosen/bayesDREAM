"""
bayesDREAM: Bayesian Dosage Response Effects Across Modalities

Core implementation for bayesDREAM - a three-step Bayesian framework for modeling
perturbation effects across multiple molecular modalities:

1. fit_ntc() - Fit NTC cells to estimate gene-specific overdispersion and batch effects
2. fit_cis() - Model direct effects on targeted genes
3. fit_trans() - Model downstream effects as dose-response functions

The bayesDREAM class combines:
- _BayesDREAMCore: Base fitting infrastructure
- Modality mixins: Methods for adding different data types (transcripts, splicing, ATAC, custom)
"""

import warnings
from typing import Dict, Optional
import numpy as np
import pandas as pd

from .core import _BayesDREAMCore
from .modality import Modality
from .modalities import (
    TranscriptModalityMixin,
    SplicingModalityMixin,
    ATACModalityMixin,
    CustomModalityMixin,
    PlottingMixin
)
from .plotting.colors import ColorScheme

warnings.simplefilter(action="ignore", category=FutureWarning)


class bayesDREAM(
    PlottingMixin,
    TranscriptModalityMixin,
    SplicingModalityMixin,
    ATACModalityMixin,
    CustomModalityMixin,
    _BayesDREAMCore
):
    """
    bayesDREAM: Bayesian Dosage Response Effects Across Modalities

    A three-step Bayesian framework for modeling perturbation effects across
    multiple molecular modalities including genes, transcripts, splicing, and
    custom measurements.

    Supports:
    - Gene counts (negbinom) - the default/primary modality
    - Transcript counts (negbinom or multinomial for isoform usage)
    - Splice junction counts with donor/acceptor/exon skipping (multinomial/binomial)
    - ATAC-seq peaks (negbinom or binomial)
    - Custom modalities with user-specified distributions

    Attributes
    ----------
    modalities : Dict[str, Modality]
        Dictionary of all modalities. 'gene' is the primary modality.
    primary_modality : str
        Name of primary modality used for cis/trans analysis (default: 'gene')
    """

    def __init__(
        self,
        meta: pd.DataFrame,
        counts: pd.DataFrame = None,
        modality_name: str = 'gene',
        primary_modality: str = None,
        feature_meta: pd.DataFrame = None,
        cis_gene: str = None,
        cis_feature: str = None,
        guide_assignment: np.ndarray = None,
        guide_meta: pd.DataFrame = None,
        guide_target: pd.DataFrame = None,
        guide_covariates: list = None,
        guide_covariates_ntc: list = None,
        sum_factor_col: str = 'sum_factor',
        output_dir: str = "./model_out",
        label: str = None,
        device: str = None,
        random_seed: int = 2402,
        cores: int = 1,
        exclude_targets: list = None,
        require_ntc: bool = True,
        min_count: int = 1,
        color_scheme: 'ColorScheme' = None,
    ):
        """
        Initialize bayesDREAM.

        Parameters
        ----------
        meta : pd.DataFrame
            Cell-level metadata with columns: cell, guide, target, sum_factor, etc.
        counts : pd.DataFrame, np.ndarray, or scipy.sparse matrix, optional
            Count matrix for the primary modality (features × cells). This will be used
            for trans modeling and must represent negbinom count data.

            Accepts:
            - pd.DataFrame: Features as index, cells as columns
            - np.ndarray or scipy.sparse matrix: Features as rows, cells as columns
              (requires feature_meta to map feature names)

            The modality_name parameter determines how this data is interpreted:
            - 'gene': Gene expression counts (most common)
            - 'atac': ATAC-seq peak counts
            - Custom name: User-defined count modality

            Additional modalities can be added after initialization using add_*_modality() methods.
        modality_name : str, default='gene'
            Name/type of the primary modality provided in counts parameter.

            Pre-set types:
            - 'gene': Gene expression (default, use with feature_meta for gene annotations)
            - 'atac': ATAC-seq peaks

            Custom types:
            - Any string: Creates a custom negbinom modality with that name

            The primary modality MUST be negative binomial (count data) for cis/trans modeling.
        feature_meta : pd.DataFrame, optional
            Feature-level metadata for the primary modality.

            For modality_name='gene':
            - Recommended columns: 'gene', 'gene_name', 'gene_id'
            - If not provided, minimal metadata created from counts.index

            For other modalities:
            - Should contain relevant feature annotations
            - If not provided, minimal metadata created from counts.index
        cis_gene : str, optional
            Feature to extract as 'cis' modality. When modality_name='gene', this is
            a gene name (e.g., 'GFI1B'). For other modality types, use cis_feature instead.

            The cis feature will be:
            1. Extracted as a separate 'cis' modality (1 feature)
            2. Removed from the primary modality (which becomes trans-only)
            3. Used for cis effect modeling in fit_cis()
        cis_feature : str, optional
            Alternative to cis_gene for non-gene modalities. Specifies which feature
            to extract as the 'cis' modality.
        guide_assignment : np.ndarray, optional
            Binary matrix (cells × guides) for high MOI mode. Each cell can have
            multiple guides. If provided, meta should NOT have 'guide' or 'target'
            columns. Must be provided together with guide_meta.
        guide_meta : pd.DataFrame, optional
            Guide metadata for high MOI mode. Required columns: 'guide', 'target'.
            Must be provided together with guide_assignment. Index should match
            guide order in guide_assignment matrix.
        guide_covariates : list, optional
            Covariates for guide grouping (e.g., ['cell_line', 'batch'])
        guide_covariates_ntc : list, optional
            Covariates for NTC guide grouping (if different from guide_covariates)
        sum_factor_col : str, default='sum_factor'
            Column name in meta containing size factors
        output_dir : str, default="./model_out"
            Output directory for saving results
        label : str, optional
            Run label for organizing outputs
        device : str, optional
            'cpu' or 'cuda'. If None, auto-detects.
        random_seed : int, default=2402
            Random seed for reproducibility
        cores : int, default=1
            Number of CPU cores for parallel operations
        color_scheme : ColorScheme, optional
            Color scheme for visualizations. If None, auto-built from model
            metadata via ``ColorScheme.from_model(self)`` after initialization.
            Can also be set later via ``set_color_scheme()``.

        Raises
        ------
        ValueError
            - If cis_feature is specified but counts is None
            - If cis_feature not found in counts
            - If cis_feature has zero variance
            - If primary modality would be empty after filtering

        Examples
        --------
        Basic gene expression analysis:
        >>> model = bayesDREAM(
        ...     meta=cell_metadata,
        ...     counts=gene_counts,
        ...     cis_gene='GFI1B'
        ... )

        With gene metadata:
        >>> model = bayesDREAM(
        ...     meta=cell_metadata,
        ...     counts=gene_counts,
        ...     feature_meta=gene_metadata,
        ...     cis_gene='GFI1B',
        ...     guide_covariates=['cell_line']
        ... )

        ATAC-seq analysis:
        >>> model = bayesDREAM(
        ...     meta=cell_metadata,
        ...     counts=atac_counts,
        ...     modality_name='atac',
        ...     feature_meta=peak_metadata,
        ...     cis_feature='chr9:132283881-132284881'
        ... )

        Custom modality:
        >>> model = bayesDREAM(
        ...     meta=cell_metadata,
        ...     counts=my_counts,
        ...     modality_name='my_custom_modality',
        ...     cis_feature='feature_123'
        ... )

        Notes
        -----
        - The 'cis' modality is ONLY extracted during initialization from the primary modality
        - When you call add_*_modality() methods later, NO cis extraction occurs
        - The primary modality MUST be negbinom (count data) for cis/trans modeling
        - Additional modalities (transcripts, splicing, etc.) are added via add_*_modality() methods
        """
        # Initialize modalities dict (always start empty, build from counts)
        self.modalities = {}

        # primary_modality is an alias for modality_name
        if primary_modality is not None:
            modality_name = primary_modality

        # Resolve cis_feature: cis_gene is an alias for cis_feature when modality_name='gene'
        if cis_gene is not None and cis_feature is not None:
            raise ValueError("Provide either cis_gene or cis_feature, not both")
        if cis_gene is not None:
            if modality_name != 'gene':
                warnings.warn(
                    f"cis_gene parameter is intended for modality_name='gene'. "
                    f"You have modality_name='{modality_name}'. Use cis_feature instead.",
                    UserWarning
                )
            cis_feature = cis_gene

        # Store min_count for use in modality creation
        self.min_count = min_count

        # Store original counts for base class initialization
        counts_for_base = counts

        # Validate that counts is provided if cis_feature is specified
        if cis_feature is not None and counts is None:
            raise ValueError("Cannot specify cis_feature without providing counts")

        # Extract 'cis' modality from primary modality (if both counts and cis_feature provided)
        # Keep the original gene name and track numeric position
        cis_numeric_idx = None
        if counts is not None and cis_feature is not None:
            if modality_name == 'gene':
                cis_feature, cis_numeric_idx = self._extract_cis_from_gene(counts, cis_feature, feature_meta, meta)
            else:
                # Generic cis extraction for any negbinom modality
                cis_feature, cis_numeric_idx = self._extract_cis_generic(counts, cis_feature, modality_name, feature_meta, meta)

        # Create primary modality
        if counts is not None:
            if modality_name == 'gene':
                # Use gene-specific creation (with gene_meta handling)
                # Pass both the name and numeric index for exclusion
                self._create_gene_modality(counts, cis_feature, cis_numeric_idx, gene_meta=feature_meta, meta=meta, min_count=min_count)
            else:
                # Generic negbinom modality creation
                self._create_negbinom_modality(counts, modality_name, cis_feature, cis_numeric_idx, feature_meta, meta, min_count=min_count)

        # Store primary modality name
        self.primary_modality = modality_name

        # ========================================================================
        # CRITICAL VALIDATION: Primary modality MUST be negative binomial
        # ========================================================================
        if modality_name in self.modalities:
            primary_distribution = self.modalities[modality_name].distribution
            if primary_distribution != 'negbinom':
                raise ValueError(
                    f"Primary modality '{modality_name}' has distribution '{primary_distribution}', "
                    f"but bayesDREAM requires primary modality to be 'negbinom' for cis/trans modeling. "
                    f"The primary modality must represent count data that follows a negative binomial distribution."
                )
            if cis_feature is not None:
                pass
        elif counts is None:
            # No counts provided - user will add modalities later
            warnings.warn(
                f"No counts provided during initialization. Primary modality '{modality_name}' must be added "
                f"via add_custom_modality() with distribution='negbinom' before fitting.",
                UserWarning
            )

        # Get counts for base class initialization
        # IMPORTANT: Pass counts matrix as-is to avoid densification
        # core.py will handle matrix/array/DataFrame uniformly
        if counts_for_base is None:
            if modality_name in self.modalities:
                # Get from modality - prefer original counts over count_df
                mod = self.modalities[modality_name]
                primary_counts = mod.counts  # This is the matrix/array, not densified
            else:
                # No counts and no primary modality - create placeholder
                pass
                # Create minimal placeholder (will be replaced)
                primary_counts = np.ones((1, len(meta)))
        else:
            # Use original counts as-is (matrix, array, or DataFrame)
            primary_counts = counts_for_base

        # Prepare gene_meta for base class
        # Only pass feature_meta as gene_meta if modality_name is 'gene', otherwise None
        gene_meta_for_base = feature_meta if modality_name == 'gene' else None

        # Initialize base bayesDREAM with original counts (including cis feature)
        super().__init__(
            meta=meta,
            counts=primary_counts,
            gene_meta=gene_meta_for_base,
            cis_gene=cis_feature,  # Pass resolved cis_feature as cis_gene
            guide_assignment=guide_assignment,
            guide_meta=guide_meta,
            guide_target=guide_target,
            guide_covariates=guide_covariates,
            guide_covariates_ntc=guide_covariates_ntc,
            sum_factor_col=sum_factor_col,
            output_dir=output_dir,
            label=label,
            device=device,
            random_seed=random_seed,
            cores=cores,
            exclude_targets=exclude_targets,
            require_ntc=require_ntc
        )

        # Subset all modalities to match filtered cells from base class
        # Base class (super().__init__) has filtered self.meta to valid cells
        valid_cells = self.meta['cell'].tolist()
        for mod_name in list(self.modalities.keys()):
            mod = self.modalities[mod_name]
            if mod.cell_names is not None:
                # Find indices of valid cells in this modality
                cell_indices = [i for i, c in enumerate(mod.cell_names)
                              if c in valid_cells]
                if len(cell_indices) < len(mod.cell_names):
                    self.modalities[mod_name] = mod.get_cell_subset(cell_indices)
                else:
                    # This shouldn't happen - means we have fewer cells in modality than in meta
                    print(f"[WARNING] Modality '{mod_name}' has {len(mod.cell_names)} cells but filtered meta has {len(valid_cells)}")

        print(f"bayesDREAM: label={self.label}, device={self.device}, {len(self.modalities)} modalities")
        for name, mod in self.modalities.items():
            print(f"  - {name}: {mod}")

        # Initialise sum_factors on all negbinom modalities from meta.
        # Must run AFTER super().__init__() and cell subsetting so self.meta is final.
        self._init_sum_factors(sum_factor_col)

        # Colour scheme — user-supplied or auto-built from model metadata.
        # Must run AFTER super().__init__() so self.meta / guide_meta are final.
        if color_scheme is not None:
            self.color_scheme = color_scheme
        else:
            self.color_scheme = ColorScheme.from_model(self)

    def set_color_scheme(self, color_scheme: 'ColorScheme'):
        """
        Replace the model's color scheme.

        Parameters
        ----------
        color_scheme : ColorScheme
            New color scheme to use for all visualizations.
        """
        self.color_scheme = color_scheme

    def _init_sum_factors(self, sum_factor_col: str = 'sum_factor'):
        """
        Initialise sum_factors DataFrame on negbinom modalities from self.meta.

        Copies every column whose name contains 'sum_factor' into a cell-indexed
        DataFrame stored on each negbinom modality.  The 'cis' modality (when
        present) receives a reference to the *same* DataFrame as the primary
        modality — not a copy — so that writes made via adjust_ntc_sum_factor()
        or refit_sumfactor() are immediately visible to both.

        Called once at the end of __init__, after super().__init__() and cell
        subsetting so self.meta is in its final, filtered state.
        """
        if self.primary_modality not in self.modalities:
            return

        primary_mod = self.modalities[self.primary_modality]
        if primary_mod.distribution != 'negbinom':
            return

        # Collect all sum-factor columns present in meta
        sf_cols = [c for c in self.meta.columns if 'sum_factor' in c.lower()]
        if sum_factor_col not in sf_cols and sum_factor_col in self.meta.columns:
            sf_cols.append(sum_factor_col)

        if not sf_cols:
            primary_mod.sum_factors = pd.DataFrame(index=self.meta['cell'].values)
        else:
            sf_df = self.meta.set_index('cell')[sf_cols].copy()
            primary_mod.sum_factors = sf_df

        # The 'cis' modality is a view of the primary — share the same DataFrame.
        if 'cis' in self.modalities:
            self.modalities['cis'].sum_factors = primary_mod.sum_factors

        print(f"[INIT] sum_factors initialised on '{self.primary_modality}' "
              f"with columns: {list(primary_mod.sum_factors.columns)}")

    @staticmethod
    def _fit_gmm_bic(X: np.ndarray, max_components: int = 6):
        """
        Fit GMMs for k=2..max_components, return best by BIC and the BIC curve.

        Returns (best_gmm, ks, bics).
        """
        from sklearn.mixture import GaussianMixture
        ks, bics, best_gmm, best_bic = [], [], None, np.inf
        for k in range(2, max_components + 1):
            gmm_k = GaussianMixture(n_components=k, random_state=42, n_init=5)
            gmm_k.fit(X)
            bic = gmm_k.bic(X)
            ks.append(k)
            bics.append(bic)
            if bic < best_bic:
                best_bic, best_gmm = bic, gmm_k
        return best_gmm, np.array(ks), np.array(bics)

    def _compute_ntc_log2_exprs_from_fit(self):
        """
        Per-gene log2 NTC expression using fit_ntc() posteriors.

        fit_ntc() appends the cis gene as the last row of the counts matrix,
        fits it together with all trans genes, and stores the result in
        cis_modality.posterior_samples_ntc.  Both trans and cis mu_ntc
        values are therefore available here — no raw-count computation needed.

        Requires fit_ntc() to have been run (populates posterior_samples_ntc
        on both the primary and cis modalities).

        Returns
        -------
        dict with keys: all_log2_expr, cis_log2_expr, trans_log2_expr,
        cis_ntc_counts (raw counts for plot annotation only), n_ntc_cells.
        Or None on failure (warnings emitted).
        """
        import torch as _torch

        primary_mod = self.modalities.get(self.primary_modality)
        cis_mod     = self.modalities.get('cis')

        if primary_mod is None:
            warnings.warn(f"Primary modality '{self.primary_modality}' not found.", UserWarning)
            return None

        ps_primary = primary_mod.posterior_samples_ntc
        if ps_primary is None or 'mu_ntc' not in ps_primary:
            raise ValueError(
                "fit_ntc() must be run before checking cis expression. "
                "Call model.fit_ntc() first."
            )

        if cis_mod is None:
            warnings.warn("No 'cis' modality found.", UserWarning)
            return None

        ps_cis = cis_mod.posterior_samples_ntc
        if ps_cis is None or 'mu_ntc' not in ps_cis:
            raise ValueError(
                "fit_ntc() did not produce mu_ntc for the cis modality. "
                "Ensure fit_ntc() completed successfully."
            )

        def _to_numpy(t):
            if isinstance(t, _torch.Tensor):
                return t.mean(dim=0).detach().cpu().numpy().flatten()
            return np.asarray(t).mean(axis=0).flatten()

        # Trans genes
        log2_trans = np.log2(_to_numpy(ps_primary['mu_ntc']))  # LogNormal → always > 0

        # Cis gene (shape [S, 1] → scalar after mean+flatten)
        cis_mu_ntc = float(_to_numpy(ps_cis['mu_ntc'])[0])
        cis_log2_expr = float(np.log2(cis_mu_ntc))

        all_log2_expr = np.concatenate([log2_trans, [cis_log2_expr]])

        # Raw NTC counts — used only for the plot title annotation, not for the GMM
        ntc_meta  = self.meta[self.meta.get('target', pd.Series()).eq('ntc')] \
                    if 'target' in self.meta.columns else self.meta.iloc[:0]
        ntc_cells = ntc_meta['cell'].values
        cell_to_col_cis = {c: i for i, c in enumerate(cis_mod.cell_names or [])}
        ntc_idx_cis = np.array([cell_to_col_cis[c] for c in ntc_cells if c in cell_to_col_cis], dtype=int)
        if len(ntc_idx_cis) > 0:
            cis_raw = cis_mod.counts
            if hasattr(cis_raw, 'toarray'):
                cis_ntc = np.asarray(cis_raw[0, ntc_idx_cis].toarray()).flatten().astype(float)
            elif isinstance(cis_raw, pd.DataFrame):
                cis_ntc = cis_raw.iloc[0, ntc_idx_cis].values.astype(float)
            else:
                cis_ntc = np.asarray(cis_raw)[0, ntc_idx_cis].astype(float)
        else:
            cis_ntc = np.array([])

        return {
            'all_log2_expr':   all_log2_expr,
            'cis_log2_expr':   cis_log2_expr,
            'trans_log2_expr': log2_trans,
            'cis_ntc_counts':  cis_ntc,
            'n_ntc_cells':     len(ntc_idx_cis),
        }

    def _check_cis_expression_in_ntc(self):
        """
        Check whether the cis gene is meaningfully expressed in NTC cells.

        Fits GMMs (k=2..6, BIC selection) to per-gene log2 NTC expression
        from the technical fit posterior.  Raises ValueError if the cis gene
        falls in the lowest-mean component (unexpressed cluster).

        Requires fit_ntc() to have been run first (uses mu_ntc posterior).
        Called automatically by fit_cis(); also callable standalone (use
        plot_ntc_expression_gmm() to see the full diagnostic figure).
        """
        data = self._compute_ntc_log2_exprs_from_fit()
        if data is None:
            return

        try:
            from sklearn.mixture import GaussianMixture  # noqa: F401
        except ImportError:
            warnings.warn(
                "scikit-learn not available. Cannot perform GMM check for unexpressed cis gene.",
                UserWarning
            )
            return

        X = data['all_log2_expr'].reshape(-1, 1)
        best_gmm, _, _ = self._fit_gmm_bic(X)

        n_components     = best_gmm.n_components
        component_means  = best_gmm.means_.flatten()
        lowest_component = int(np.argmin(component_means))
        cis_component    = int(best_gmm.predict([[data['cis_log2_expr']]])[0])

        if cis_component == lowest_component:
            raise ValueError(
                f"Cis gene '{self.cis_gene}' appears to be unexpressed in NTC cells "
                f"(log2 NTC expression = {data['cis_log2_expr']:.2f}; "
                f"lowest GMM component mean = {component_means[lowest_component]:.2f}; "
                f"BIC-selected k={n_components}). "
                f"fit_ntc() cannot reliably estimate overdispersion (o_x) for an unexpressed "
                f"gene, so fit_cis() results will be unreliable. "
                f"Call plot_ntc_expression_gmm() to inspect the expression distribution."
            )
        else:
            print(
                f"[INFO] Cis gene '{self.cis_gene}' expressed in NTC cells: "
                f"log2 NTC expr = {data['cis_log2_expr']:.2f} "
                f"(GMM component {cis_component + 1}/{n_components}, "
                f"component mean = {component_means[cis_component]:.2f}; BIC k={n_components})"
            )

    def plot_ntc_expression_gmm(
        self,
        max_components: int = 6,
        figsize: tuple = (11, 4),
        n_bins: int = 80,
    ):
        """
        Diagnostic plot for the NTC expression GMM check.

        Produces a two-panel figure:

        Left — Histogram of mean log2 sum-factor-normalized expression across
        NTC cells for every gene (trans + cis).  Each GMM component is drawn
        as a filled Gaussian curve (weighted by mixing proportion, scaled to
        the histogram density).  The cis gene is marked with a vertical line
        and labelled with its raw count distribution in the title annotation.
        Components are coloured from cold (low mean) to warm (high mean).
        The lowest-mean component — the "unexpressed" cluster — is always
        shown in grey.

        Right — BIC vs number of components k=2..max_components.  The
        selected k is highlighted with a star.

        Parameters
        ----------
        max_components : int, default=6
            Maximum k to consider during BIC model selection.
        figsize : tuple, default=(11, 4)
            Figure size (width, height) in inches.
        n_bins : int, default=80
            Number of histogram bins.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        from scipy.stats import norm as scipy_norm

        try:
            from sklearn.mixture import GaussianMixture  # noqa: F401
        except ImportError:
            raise ImportError(
                "scikit-learn is required for plot_ntc_expression_gmm(). "
                "Install with: pip install scikit-learn"
            )

        data = self._compute_ntc_log2_exprs_from_fit()
        if data is None:
            raise ValueError(
                "Cannot compute NTC expression data. "
                "Ensure fit_ntc() has been run and the model has 'target' column, "
                "NTC cells, and sum_factors."
            )

        X = data['all_log2_expr'].reshape(-1, 1)
        best_gmm, ks, bics = self._fit_gmm_bic(X, max_components=max_components)

        n_k             = best_gmm.n_components
        component_means = best_gmm.means_.flatten()
        sort_order      = np.argsort(component_means)  # coldest → warmest
        lowest_label    = sort_order[0]                # "unexpressed" component
        cis_component   = int(best_gmm.predict([[data['cis_log2_expr']]])[0])
        assignments     = best_gmm.predict(X)

        # Colours: grey for lowest, then blue→orange for the rest
        palette = ['#999999'] + ['#4e9ede', '#e08c3a', '#5ab55a', '#c75f5f', '#9966cc']
        comp_colors = {}
        for rank, label in enumerate(sort_order):
            comp_colors[label] = palette[rank] if rank < len(palette) else f'C{rank}'

        fig, (ax_hist, ax_bic) = plt.subplots(1, 2, figsize=figsize)

        # ---- Left panel: histogram + GMM curves ----
        all_expr = data['all_log2_expr']
        counts_hist, bin_edges = np.histogram(all_expr, bins=n_bins, density=True)
        bin_width = bin_edges[1] - bin_edges[0]

        # Colour each bar by its dominant component
        bar_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bar_labels  = best_gmm.predict(bar_centres.reshape(-1, 1))
        bar_colors  = [comp_colors[lbl] for lbl in bar_labels]

        ax_hist.bar(
            bar_centres, counts_hist, width=bin_width,
            color=bar_colors, alpha=0.55, edgecolor='none'
        )

        # GMM component curves
        x_plot = np.linspace(all_expr.min() - 0.5, all_expr.max() + 0.5, 500)
        for k_label in range(n_k):
            mu     = float(best_gmm.means_[k_label])
            sigma  = float(np.sqrt(best_gmm.covariances_[k_label, 0, 0]))
            weight = float(best_gmm.weights_[k_label])
            y      = weight * scipy_norm.pdf(x_plot, mu, sigma)
            lw     = 2.0
            ls     = '-'
            rank   = int(np.where(sort_order == k_label)[0])
            label  = f'k={rank + 1} (μ={mu:.2f}, w={weight:.2f})'
            if k_label == lowest_label:
                label = f'[unexpressed] {label}'
            ax_hist.plot(x_plot, y, color=comp_colors[k_label], lw=lw, ls=ls, label=label)

        # Cis gene vertical line
        cis_val   = data['cis_log2_expr']
        cis_color = comp_colors[cis_component]
        is_unexpressed = (cis_component == lowest_label)
        ax_hist.axvline(
            cis_val, color=cis_color, lw=2, ls='--',
            label=f'{self.cis_gene} (log2 expr={cis_val:.2f})'
        )

        # Annotation: raw count distribution for cis gene in NTC cells
        raw_counts  = data['cis_ntc_counts'].astype(int)
        unique, cnt = np.unique(raw_counts, return_counts=True)
        count_str   = ', '.join(f'{u}:{c}' for u, c in zip(unique[:6], cnt[:6]))
        if len(unique) > 6:
            count_str += ', …'
        verdict = '⚠ UNEXPRESSED' if is_unexpressed else '✓ expressed'
        ax_hist.set_title(
            f'NTC expression distribution — {self.cis_gene} {verdict}\n'
            f'NTC raw counts: {count_str}  (n={data["n_ntc_cells"]} cells)',
            fontsize=9
        )
        ax_hist.set_xlabel('log$_2$(NTC mean expression) from technical fit', fontsize=10)
        ax_hist.set_ylabel('Density', fontsize=10)
        ax_hist.legend(fontsize=7, loc='upper left')

        # ---- Right panel: BIC curve ----
        best_k = best_gmm.n_components
        ax_bic.plot(ks, bics, 'o-', color='steelblue', lw=1.5, ms=5)
        ax_bic.scatter(
            [best_k], [bics[ks.tolist().index(best_k)]],
            marker='*', s=200, color='red', zorder=5, label=f'selected k={best_k}'
        )
        ax_bic.set_xlabel('Number of components k', fontsize=10)
        ax_bic.set_ylabel('BIC', fontsize=10)
        ax_bic.set_title('GMM model selection (BIC)', fontsize=10)
        ax_bic.set_xticks(ks)
        ax_bic.legend(fontsize=8)

        fig.tight_layout()
        return fig

    def plot_ntc_overdispersion(
        self,
        modality_name: str = None,
        ax=None,
        figsize: tuple = (6, 5),
    ):
        """
        Scatter plot of log2(mu_ntc) vs log2(o_y) from the technical fit.

        Highlights the cis gene in red.  Useful for diagnosing whether the
        cis gene falls in an unusual region of the mu_ntc–o_y relationship.

        Requires fit_ntc() to have been run first.

        Parameters
        ----------
        modality_name : str, optional
            Modality to plot (default: primary modality).
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.  Creates a new figure if None.
        figsize : tuple, default=(6, 5)
            Figure size (width, height) in inches.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if modality_name is None:
            modality_name = self.primary_modality

        modality = self.get_modality(modality_name)
        ps = modality.posterior_samples_ntc
        if ps is None or 'mu_ntc' not in ps or 'o_y' not in ps:
            raise ValueError(
                f"No technical fit found for modality '{modality_name}'. "
                "Run fit_ntc() first."
            )

        # Extract posterior means — tensors may be [S, T] or [S, C, T]
        mu_ntc_raw = ps['mu_ntc'].mean(dim=0).detach().numpy()
        o_y_raw    = ps['o_y'].mean(dim=0).detach().numpy()

        # Flatten multi-group tensors to 1D (take first group = NTC reference)
        mu_ntc = mu_ntc_raw.reshape(-1) if mu_ntc_raw.ndim == 1 else mu_ntc_raw[0]
        o_y    = o_y_raw.reshape(-1)    if o_y_raw.ndim == 1    else o_y_raw[0]

        log2_mu = np.log2(np.clip(mu_ntc, 1e-8, None))
        log2_oy = np.log2(np.clip(o_y,    1e-8, None))

        new_fig = ax is None
        if new_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.scatter(log2_mu, log2_oy, alpha=0.25, s=4, c='steelblue', rasterized=True,
                   label='trans genes')

        # Highlight cis gene if available
        if 'cis' in self.modalities:
            cis_ps = self.modalities['cis'].posterior_samples_ntc
            if cis_ps is not None and 'mu_ntc' in cis_ps and 'o_x' in cis_ps:
                cis_mu = float(cis_ps['mu_ntc'].mean().item())
                cis_ox = float(cis_ps['o_x'].mean().item())
                ax.scatter(
                    np.log2(max(cis_mu, 1e-8)),
                    np.log2(max(cis_ox, 1e-8)),
                    s=120, c='red', zorder=10, marker='*',
                    label=f'{self.cis_gene}'
                )

        ax.set_xlabel('log$_2$(mu_ntc)', fontsize=11)
        ax.set_ylabel('log$_2$(o_y)', fontsize=11)
        ax.set_title('Technical fit: NTC mean expression vs overdispersion', fontsize=11)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.legend(fontsize=9, markerscale=1.5)

        if new_fig:
            fig.tight_layout()

        return fig

    def _filter_features(self, counts, feature_meta, features_to_keep):
        """
        Filter features (rows) from counts matrix and feature_meta.

        Works uniformly with DataFrame, numpy array, sparse matrix.
        Returns filtered counts and feature_meta with reset numeric indices.

        Parameters
        ----------
        counts : pd.DataFrame, np.ndarray, or scipy.sparse matrix
            Count matrix (features × cells)
        feature_meta : pd.DataFrame
            Feature metadata (must have same number of rows as counts)
        features_to_keep : array-like of int
            Integer positions (iloc) of features to keep

        Returns
        -------
        counts_filtered : same type as input
            Filtered counts matrix
        feature_meta_filtered : pd.DataFrame
            Filtered feature_meta with index reset to range(n_kept_features)
        """
        if isinstance(counts, pd.DataFrame):
            # DataFrame: use iloc
            counts_filtered = counts.iloc[features_to_keep].copy()
            # Reset index to numeric
            counts_filtered.index = range(len(features_to_keep))
        else:
            # Array or sparse matrix: use numpy indexing
            if hasattr(counts, 'toarray'):
                # Sparse matrix
                counts_filtered = counts[features_to_keep, :]
            else:
                # Dense array
                counts_filtered = counts[features_to_keep, :]

        # Filter and reset feature_meta index
        feature_meta_filtered = feature_meta.iloc[features_to_keep].copy()
        feature_meta_filtered.index = range(len(features_to_keep))

        return counts_filtered, feature_meta_filtered

    def _counts_to_dataframe(self, counts, feature_meta: Optional[pd.DataFrame] = None, cell_names=None, keep_sparse=False):
        """
        Convert counts (DataFrame, array, or sparse matrix) to DataFrame.

        WARNING: This method densifies sparse matrices by default. Use keep_sparse=True to
        return a dict with sparse matrix + metadata instead.

        Parameters
        ----------
        counts : pd.DataFrame, np.ndarray, or scipy.sparse matrix
            Count matrix (features × cells)
        feature_meta : pd.DataFrame, optional
            Feature metadata to use as index. If None, uses integer indices.
        cell_names : list, optional
            Cell names to use as columns. If None, uses integer indices.
        keep_sparse : bool, default=False
            If True and counts is sparse, returns dict with keys:
            {'counts': sparse_matrix, 'gene_names': list, 'cell_names': list}
            instead of densifying to DataFrame.

        Returns
        -------
        pd.DataFrame or dict
            If keep_sparse=False: Counts as DataFrame
            If keep_sparse=True and input is sparse: dict with sparse data + metadata
        """
        if isinstance(counts, pd.DataFrame):
            if keep_sparse:
                # Already dense DataFrame, return as-is (no conversion to dict)
                return counts
            else:
                return counts

        # Check if sparse
        from scipy import sparse
        is_sparse = sparse.issparse(counts)

        if is_sparse and keep_sparse:
            # Return sparse matrix with metadata (do NOT densify)
            # Get feature names from feature_meta
            if feature_meta is not None:
                feature_names = feature_meta.index.tolist()
            else:
                feature_names = list(range(counts.shape[0]))

            # Get cell names
            if cell_names is None:
                cell_names = list(range(counts.shape[1]))

            return {
                'counts': counts,  # Keep sparse!
                'gene_names': feature_names,
                'cell_names': cell_names,
                'is_sparse': True
            }

        # Convert to dense array if sparse (legacy behavior)
        if is_sparse:
            print(f"[WARNING] Densifying sparse matrix of shape {counts.shape} - this may use significant memory!")
            counts_array = counts.toarray()
        else:
            counts_array = counts

        # Get cell names
        if cell_names is None:
            cell_names = list(range(counts_array.shape[1]))

        # Use numeric index for features (not feature_meta.index)
        # This enables compatibility when Ensembl IDs or gene names have duplicates
        # Row i in DataFrame corresponds to row i in feature_meta
        feature_index = list(range(counts_array.shape[0]))

        return pd.DataFrame(counts_array, index=feature_index, columns=cell_names)

    def _extract_cis_from_gene(self, counts, cis_gene: str, feature_meta: Optional[pd.DataFrame] = None, meta: Optional[pd.DataFrame] = None) -> tuple:
        """
        Extract 'cis' modality from gene counts.

        Parameters
        ----------
        counts : pd.DataFrame, np.ndarray, or scipy.sparse matrix
            Gene counts (genes × cells)
        cis_gene : str
            Gene name to extract as cis modality
        feature_meta : pd.DataFrame, optional
            Feature metadata with gene names. Required if counts is not a DataFrame.
        meta : pd.DataFrame, optional
            Cell metadata with 'cell' column. Required if counts is not a DataFrame
            (used to get cell names for proper subsetting later).

        Returns
        -------
        tuple (str, int)
            (original_gene_name, numeric_row_index)
            Returns the original gene name and the numeric row position
        """
        # Handle different count formats
        numeric_idx = None

        if isinstance(counts, pd.DataFrame):
            # DataFrame: check if index is numeric or string-based
            if pd.api.types.is_integer_dtype(counts.index):
                # Numeric index: use feature_meta to find gene
                if feature_meta is None:
                    raise ValueError(
                        f"counts has numeric index but no feature_meta provided to locate '{cis_gene}'"
                    )
                # Find gene in feature_meta columns (same order as core.py)
                gene_col = None
                for col in ['gene_name', 'gene', 'gene_id', 'feature_id']:
                    if col in feature_meta.columns and cis_gene in feature_meta[col].values:
                        gene_col = col
                        break

                if gene_col is None:
                    raise ValueError(
                        f"cis_gene '{cis_gene}' not found in feature_meta columns: {list(feature_meta.columns)}"
                    )

                # Get numeric position (iloc)
                numeric_idx = feature_meta[feature_meta[gene_col] == cis_gene].index[0]
                cis_counts = counts.iloc[[numeric_idx]].values
            else:
                # String-based index: use index directly
                if cis_gene not in counts.index:
                    raise ValueError(
                        f"cis_gene '{cis_gene}' not found in counts.index.\n"
                        f"Available genes: {counts.index[:10].tolist()}..."
                    )
                numeric_idx = counts.index.get_loc(cis_gene)
                cis_counts = counts.loc[[cis_gene]].values
        else:
            # Array or sparse matrix: use feature_meta
            if feature_meta is None:
                raise ValueError(
                    "When counts is not a DataFrame, feature_meta must be provided to locate cis_gene"
                )

            # Find gene in feature_meta columns (same order as core.py)
            gene_col = None
            for col in ['gene_name', 'gene', 'gene_id', 'feature_id']:
                if col in feature_meta.columns and cis_gene in feature_meta[col].values:
                    gene_col = col
                    break

            if gene_col is None:
                available_cols = list(feature_meta.columns)
                raise ValueError(
                    f"cis_gene '{cis_gene}' not found in feature_meta.\n"
                    f"Tried columns: gene_name, gene, gene_id, feature_id.\n"
                    f"Available columns: {available_cols}"
                )

            # Get numeric position (iloc, which is the row number)
            numeric_idx = feature_meta[feature_meta[gene_col] == cis_gene].index[0]

            # Extract row from counts (works for numpy arrays and sparse matrices)
            if hasattr(counts, 'toarray'):
                # Sparse matrix
                cis_counts = counts[numeric_idx, :].toarray()
            else:
                # Dense matrix or numpy array
                cis_counts = counts[numeric_idx:numeric_idx+1, :]

        # Check if cis gene has zero variance (critical for cis modeling)
        cis_gene_std = np.std(cis_counts)
        if cis_gene_std == 0:
            raise ValueError(
                f"Cis gene '{cis_gene}' has zero standard deviation across all cells. "
                f"Cannot model cis effects for a gene with constant expression."
            )

        print(f"[INFO] Extracting 'cis' modality: {cis_gene}")

        # Use numeric index and keep original gene name
        cis_feature_meta = pd.DataFrame({
            'gene': [cis_gene],
            'gene_name': [cis_gene]  # Store original name
        }, index=[numeric_idx])

        # Add Ensembl ID if available in feature_meta
        if feature_meta is not None and 'ens_id' in feature_meta.columns:
            cis_feature_meta['ens_id'] = feature_meta.loc[numeric_idx, 'ens_id']

        # Extract cell names from counts or meta
        if isinstance(counts, pd.DataFrame):
            cell_names = counts.columns.tolist()
        elif meta is not None and 'cell' in meta.columns:
            # Array or sparse matrix - get cell names from meta
            # Ensure correct number of cells
            if len(meta) != counts.shape[1]:
                raise ValueError(
                    f"Cell count mismatch: meta has {len(meta)} cells but counts has {counts.shape[1]} cells"
                )
            cell_names = meta['cell'].tolist()
        else:
            # No meta provided - will be None and subsetting will be skipped
            # This should not happen in normal usage
            warnings.warn(
                "Creating cis modality without cell_names. Modality will not be subsetted to match filtered cells.",
                UserWarning
            )
            cell_names = None

        self.modalities['cis'] = Modality(
            name='cis',
            counts=cis_counts,
            feature_meta=cis_feature_meta,
            cell_names=cell_names,
            distribution='negbinom',
            cells_axis=1,
            min_count=self.min_count,
        )

        return cis_gene, numeric_idx

    def _extract_cis_generic(self, counts, cis_feature: str, modality_name: str, feature_meta: Optional[pd.DataFrame] = None, meta: Optional[pd.DataFrame] = None) -> tuple:
        """
        Extract 'cis' modality from a generic negbinom modality.

        Parameters
        ----------
        counts : pd.DataFrame, np.ndarray, or scipy.sparse matrix
            Feature counts (features × cells)
        cis_feature : str
            Feature name to extract as cis modality
        modality_name : str
            Name of the modality type (for error messages)
        feature_meta : pd.DataFrame, optional
            Feature metadata. Required if counts is not a DataFrame.
        meta : pd.DataFrame, optional
            Cell metadata with 'cell' column. Required if counts is not a DataFrame
            (used to get cell names for proper subsetting later).

        Returns
        -------
        tuple (str, int)
            (original_feature_name, numeric_row_index)
            Returns the original feature name and the numeric row position
        """
        # Handle different count formats
        numeric_idx = None

        if isinstance(counts, pd.DataFrame):
            # DataFrame: check if index is numeric or string-based
            if pd.api.types.is_integer_dtype(counts.index):
                # Numeric index: use feature_meta to find feature
                if feature_meta is None:
                    raise ValueError(
                        f"counts has numeric index but no feature_meta provided to locate '{cis_feature}'"
                    )
                # Find feature in feature_meta columns (consistent with core.py)
                feature_col = None
                for col in ['gene_name', 'gene', 'gene_id', 'feature_id', 'feature', 'feature_name']:
                    if col in feature_meta.columns and cis_feature in feature_meta[col].values:
                        feature_col = col
                        break

                if feature_col is None:
                    raise ValueError(
                        f"cis_feature '{cis_feature}' not found in feature_meta columns: {list(feature_meta.columns)}"
                    )

                # Get numeric position (iloc)
                numeric_idx = feature_meta[feature_meta[feature_col] == cis_feature].index[0]
                cis_counts = counts.iloc[[numeric_idx]].values
            else:
                # String-based index: use index directly
                if cis_feature not in counts.index:
                    raise ValueError(
                        f"cis_feature '{cis_feature}' not found in counts.index.\n"
                        f"Available features: {counts.index[:10].tolist()}..."
                    )
                numeric_idx = counts.index.get_loc(cis_feature)
                cis_counts = counts.loc[[cis_feature]].values
        else:
            # Array or sparse matrix: use feature_meta
            if feature_meta is None:
                raise ValueError(
                    "When counts is not a DataFrame, feature_meta must be provided to locate cis_feature"
                )

            # Find feature in feature_meta columns (consistent with core.py)
            feature_col = None
            for col in ['gene_name', 'gene', 'gene_id', 'feature_id', 'feature', 'feature_name']:
                if col in feature_meta.columns and cis_feature in feature_meta[col].values:
                    feature_col = col
                    break

            if feature_col is None:
                available_cols = list(feature_meta.columns)
                raise ValueError(
                    f"cis_feature '{cis_feature}' not found in feature_meta.\n"
                    f"Tried columns: gene_name, gene, gene_id, feature_id, feature, feature_name.\n"
                    f"Available columns: {available_cols}"
                )

            # Get numeric position (iloc, which is the row number)
            numeric_idx = feature_meta[feature_meta[feature_col] == cis_feature].index[0]

            # Extract row from counts (works for numpy arrays and sparse matrices)
            if hasattr(counts, 'toarray'):
                # Sparse matrix
                cis_counts = counts[numeric_idx, :].toarray()
            else:
                # Dense matrix or numpy array
                cis_counts = counts[numeric_idx:numeric_idx+1, :]

        # Check if cis feature has zero variance (critical for cis modeling)
        cis_feature_std = np.std(cis_counts)
        if cis_feature_std == 0:
            raise ValueError(
                f"Cis feature '{cis_feature}' has zero standard deviation across all cells. "
                f"Cannot model cis effects for a feature with constant values."
            )

        print(f"[INFO] Extracting 'cis' modality: {cis_feature} (from '{modality_name}')")

        # Use numeric index and keep original feature name
        cis_feature_meta = pd.DataFrame({
            'feature': [cis_feature],
            'modality_type': [modality_name]
        }, index=[numeric_idx])

        # Extract cell names from counts or meta
        if isinstance(counts, pd.DataFrame):
            cell_names = counts.columns.tolist()
        elif meta is not None and 'cell' in meta.columns:
            # Array or sparse matrix - get cell names from meta
            if len(meta) != counts.shape[1]:
                raise ValueError(
                    f"Cell count mismatch: meta has {len(meta)} cells but counts has {counts.shape[1]} cells"
                )
            cell_names = meta['cell'].tolist()
        else:
            warnings.warn(
                "Creating cis modality without cell_names. Modality will not be subsetted to match filtered cells.",
                UserWarning
            )
            cell_names = None

        self.modalities['cis'] = Modality(
            name='cis',
            counts=cis_counts,
            feature_meta=cis_feature_meta,
            cell_names=cell_names,
            distribution='negbinom',
            cells_axis=1,
            min_count=self.min_count,
        )

        return cis_feature, numeric_idx

    def _create_negbinom_modality(
        self,
        counts,
        modality_name: str,
        cis_feature: Optional[str] = None,
        cis_feature_idx: Optional[int] = None,
        feature_meta: Optional[pd.DataFrame] = None,
        meta: Optional[pd.DataFrame] = None,
        min_count: int = 1,
    ):
        """
        Create a generic negbinom modality (excluding cis feature if specified).

        Parameters
        ----------
        counts : pd.DataFrame, np.ndarray, or scipy.sparse matrix
            Feature counts (features × cells)
        modality_name : str
            Name for this modality
        cis_feature : str, optional
            If provided, this feature will be excluded from the modality
        cis_feature_idx : int, optional
            Numeric row index of cis feature (for exclusion when counts has numeric index)
        feature_meta : pd.DataFrame, optional
            Feature metadata. If None, creates minimal metadata from counts.index
        meta : pd.DataFrame, optional
            Cell metadata with 'cell' column. Used to get cell names if counts is not a DataFrame.
        """
        if modality_name in self.modalities:
            warnings.warn(f"Modality '{modality_name}' already exists. Overwriting.")

        # Prepare feature metadata
        if feature_meta is None:
            # Auto-create from DataFrame index if available, else require it for arrays
            if isinstance(counts, pd.DataFrame):
                # Check if DataFrame has meaningful feature names as index
                if pd.api.types.is_integer_dtype(counts.index):
                    # Numeric index - no feature names available
                    raise ValueError(
                        "counts has numeric index but no feature_meta provided. "
                        "Either:\n"
                        "  1. Provide feature_meta with feature names/IDs, OR\n"
                        "  2. Set feature names as counts.index before initialization"
                    )
                else:
                    # String-based index - extract feature names
                    print(f"[INFO] No feature_meta provided - creating from counts.index")
                    feature_names = counts.index.tolist()
                    feature_meta = pd.DataFrame({
                        'feature': feature_names,
                        'feature_name': feature_names  # Also store in 'feature_name' column
                    }, index=range(len(feature_names)))
            else:
                # Array or sparse matrix - feature_meta is REQUIRED
                raise ValueError(
                    "When counts is not a DataFrame, feature_meta must be provided. "
                    "feature_meta should contain feature names/IDs to enable plotting and analysis."
                )
        else:
            # feature_meta provided - ensure it has numeric index and required columns
            if len(feature_meta) != counts.shape[0]:
                raise ValueError(
                    f"feature_meta has {len(feature_meta)} rows but counts has {counts.shape[0]} rows. "
                    f"They must match (row i in counts = row i in feature_meta)."
                )
            feature_meta = feature_meta.copy()

            # Ensure at least one feature identifier column exists
            feature_id_cols = ['feature_name', 'feature', 'gene_name', 'gene', 'feature_id']
            has_feature_col = any(col in feature_meta.columns for col in feature_id_cols)

            if not has_feature_col:
                # If DataFrame with string index, try to extract from there
                if isinstance(counts, pd.DataFrame) and not pd.api.types.is_integer_dtype(counts.index):
                    print(f"[INFO] feature_meta has no feature identifier columns - adding 'feature_name' from counts.index")
                    feature_meta['feature_name'] = counts.index.tolist()
                    feature_meta['feature'] = counts.index.tolist()
                else:
                    warnings.warn(
                        "feature_meta has no feature identifier columns (feature_name, feature, gene_name, gene, feature_id). "
                        "Plotting by feature name will not work.",
                        UserWarning
                    )

            feature_meta.index = range(len(feature_meta))  # Reset to numeric

        # Build list of features to keep
        features_to_keep = []

        for i in range(len(feature_meta)):
            # Skip cis feature
            if i == cis_feature_idx:
                continue

            # Check for zero std
            if isinstance(counts, pd.DataFrame):
                feature_std = counts.iloc[i].std()
            else:
                # Array or sparse matrix
                if hasattr(counts, 'toarray'):
                    feature_std = np.std(counts[i, :].toarray())
                else:
                    feature_std = np.std(counts[i, :])

            if feature_std == 0:
                continue  # Skip zero-std features

            features_to_keep.append(i)

        if len(features_to_keep) == 0:
            raise ValueError(f"No features left in '{modality_name}' after filtering!")

        # Report what was filtered
        n_total = len(feature_meta)
        n_cis = 1 if cis_feature_idx is not None else 0
        n_zero_std = n_total - len(features_to_keep) - n_cis

        pass  # Modality.__init__ reports feature counts and filtering

        # Filter counts and metadata
        counts_trans, mod_feature_meta = self._filter_features(counts, feature_meta, features_to_keep)

        # Extract cell names from counts or meta
        if isinstance(counts, pd.DataFrame):
            cell_names = counts.columns.tolist()
        elif meta is not None and 'cell' in meta.columns:
            if len(meta) != counts.shape[1]:
                raise ValueError(
                    f"Cell count mismatch: meta has {len(meta)} cells but counts has {counts.shape[1]} cells"
                )
            cell_names = meta['cell'].tolist()
        else:
            warnings.warn(
                f"Creating '{modality_name}' modality without cell_names. Modality will not be subsetted to match filtered cells.",
                UserWarning
            )
            cell_names = None

        # Extract feature names for feature_names (prefer feature_name, then feature, then gene_name, then gene)
        feature_names_for_modality = None
        for col in ['feature_name', 'feature', 'gene_name', 'gene', 'gene_id', 'feature_id']:
            if col in mod_feature_meta.columns:
                feature_names_for_modality = mod_feature_meta[col].tolist()
                break

        self.modalities[modality_name] = Modality(
            name=modality_name,
            counts=counts_trans,
            feature_meta=mod_feature_meta,
            cell_names=cell_names,
            distribution='negbinom',
            cells_axis=1,
            feature_names=feature_names_for_modality,
            min_count=min_count,
        )

    def _create_gene_modality(
        self,
        counts,
        cis_gene: Optional[str] = None,
        cis_gene_idx: Optional[int] = None,
        gene_meta: Optional[pd.DataFrame] = None,
        meta: Optional[pd.DataFrame] = None,
        min_count: int = 1,
    ):
        """
        Create 'gene' modality from gene counts (excluding cis gene if specified).

        Parameters
        ----------
        counts : pd.DataFrame, np.ndarray, or scipy.sparse matrix
            Gene counts (genes × cells)
        cis_gene : str, optional
            If provided, this gene will be excluded from the gene modality
        cis_gene_idx : int, optional
            Numeric row index of cis gene (for exclusion when counts has numeric index)
        gene_meta : pd.DataFrame, optional
            Gene metadata. If None, creates minimal metadata from counts.index
        meta : pd.DataFrame, optional
            Cell metadata with 'cell' column. Used to get cell names if counts is not a DataFrame.
        """
        if 'gene' in self.modalities:
            warnings.warn("Gene modality already exists. Overwriting.")

        # Prepare gene metadata
        if gene_meta is None:
            # Auto-create from DataFrame index if available, else require it for arrays
            if isinstance(counts, pd.DataFrame):
                # Check if DataFrame has meaningful gene names as index
                if pd.api.types.is_integer_dtype(counts.index):
                    # Numeric index - no gene names available
                    raise ValueError(
                        "counts has numeric index but no gene_meta provided. "
                        "Either:\n"
                        "  1. Provide gene_meta with gene names/IDs, OR\n"
                        "  2. Set gene names as counts.index before initialization"
                    )
                else:
                    # String-based index - extract gene names
                    print(f"[INFO] No gene_meta provided - creating from counts.index")
                    gene_names = counts.index.tolist()
                    gene_meta = pd.DataFrame({
                        'gene_name': gene_names,
                        'gene': gene_names  # Also store in 'gene' column for compatibility
                    }, index=range(len(gene_names)))
            else:
                # Array or sparse matrix - gene_meta is REQUIRED
                raise ValueError(
                    "When counts is not a DataFrame, gene_meta must be provided. "
                    "gene_meta should contain gene names/IDs to enable plotting and analysis."
                )
        else:
            # gene_meta provided - ensure it has numeric index and required columns
            if len(gene_meta) != counts.shape[0]:
                raise ValueError(
                    f"gene_meta has {len(gene_meta)} rows but counts has {counts.shape[0]} rows. "
                    f"They must match (row i in counts = row i in gene_meta)."
                )
            gene_meta = gene_meta.copy()

            # Ensure at least one gene identifier column exists
            gene_id_cols = ['gene_name', 'gene', 'gene_id', 'ens_id']
            has_gene_col = any(col in gene_meta.columns for col in gene_id_cols)

            if not has_gene_col:
                # If DataFrame with string index, try to extract from there
                if isinstance(counts, pd.DataFrame) and not pd.api.types.is_integer_dtype(counts.index):
                    print(f"[INFO] gene_meta has no gene identifier columns - adding 'gene_name' from counts.index")
                    gene_meta['gene_name'] = counts.index.tolist()
                    gene_meta['gene'] = counts.index.tolist()
                else:
                    warnings.warn(
                        "gene_meta has no gene identifier columns (gene_name, gene, gene_id, ens_id). "
                        "Plotting by gene name will not work.",
                        UserWarning
                    )

            gene_meta.index = range(len(gene_meta))  # Reset to numeric

        # Pre-compute row standard deviations for all formats (dense, scipy sparse, pandas sparse).
        # Vectorised: avoids 31K per-row conversions in the loop below.
        if isinstance(counts, pd.DataFrame):
            if isinstance(counts.dtypes.iloc[0], pd.SparseDtype):
                # Pandas sparse DataFrame — convert to scipy sparse for vectorised std
                _sp = counts.sparse.to_coo().tocsr().astype(float)
                _mean = np.array(_sp.mean(axis=1)).flatten()
                _sp2 = _sp.copy()
                _sp2.data **= 2
                _sq_mean = np.array(_sp2.mean(axis=1)).flatten()
                row_stds = np.sqrt(np.maximum(_sq_mean - _mean ** 2, 0))
            else:
                # Regular dense DataFrame
                row_stds = counts.std(axis=1).values
        elif hasattr(counts, 'toarray'):
            # Scipy sparse matrix
            _sp = counts.astype(float)
            _mean = np.array(_sp.mean(axis=1)).flatten()
            _sp2 = _sp.copy()
            _sp2.data **= 2
            _sq_mean = np.array(_sp2.mean(axis=1)).flatten()
            row_stds = np.sqrt(np.maximum(_sq_mean - _mean ** 2, 0))
        else:
            # Dense numpy array
            row_stds = counts.std(axis=1)

        # Build list of features to keep
        features_to_keep = []

        for i in range(len(gene_meta)):
            # Skip cis gene
            if i == cis_gene_idx:
                continue

            # Skip zero-std genes
            if row_stds[i] == 0:
                continue

            features_to_keep.append(i)

        if len(features_to_keep) == 0:
            raise ValueError("No genes left after filtering!")

        # Report what was filtered
        n_total = len(gene_meta)
        n_cis = 1 if cis_gene_idx is not None else 0
        n_zero_std = n_total - len(features_to_keep) - n_cis

        pass  # Modality.__init__ reports feature counts and filtering

        # Filter counts and metadata
        counts_trans, gene_feature_meta = self._filter_features(counts, gene_meta, features_to_keep)

        # Extract cell names from counts or meta
        if isinstance(counts, pd.DataFrame):
            cell_names = counts.columns.tolist()
        elif meta is not None and 'cell' in meta.columns:
            if len(meta) != counts.shape[1]:
                raise ValueError(
                    f"Cell count mismatch: meta has {len(meta)} cells but counts has {counts.shape[1]} cells"
                )
            cell_names = meta['cell'].tolist()
        else:
            warnings.warn(
                "Creating 'gene' modality without cell_names. Modality will not be subsetted to match filtered cells.",
                UserWarning
            )
            cell_names = None

        # Extract gene names for feature_names (prefer gene_name, then gene column)
        gene_names_for_modality = None
        for col in ['gene_name', 'gene', 'gene_id']:
            if col in gene_feature_meta.columns:
                gene_names_for_modality = gene_feature_meta[col].tolist()
                break

        self.modalities['gene'] = Modality(
            name='gene',
            counts=counts_trans,
            feature_meta=gene_feature_meta,
            cell_names=cell_names,
            distribution='negbinom',
            cells_axis=1,
            feature_names=gene_names_for_modality,
            min_count=min_count,
        )

    def add_modality(
        self,
        name: str,
        modality: Modality,
        overwrite: bool = False
    ):
        """
        Add a new modality.

        Parameters
        ----------
        name : str
            Modality name
        modality : Modality
            Modality object
        overwrite : bool
            Whether to overwrite existing modality with same name
        """
        if name in self.modalities and not overwrite:
            raise ValueError(f"Modality '{name}' already exists. Use overwrite=True to replace it.")

        # Negbinom modalities require their own sum_factors for normalisation.
        # Primary and cis share the same DataFrame (handled by _init_sum_factors),
        # but all other negbinom modalities must have sum_factors set explicitly
        # by the caller before fit_trans() is called on them.
        if (modality.distribution == 'negbinom'
                and modality.sum_factors is None
                and name not in (self.primary_modality, 'cis')):
            import warnings
            warnings.warn(
                f"Negbinom modality '{name}' has no sum_factors set. "
                f"You must assign modality.sum_factors (a cell-indexed DataFrame) "
                f"before calling fit_trans() on this modality.",
                UserWarning,
                stacklevel=2,
            )

        self.modalities[name] = modality
        print(f"[INFO] Added modality '{name}': {modality}")

    def get_modality(self, name: str) -> Modality:
        """Get a modality by name."""
        if name not in self.modalities:
            raise KeyError(f"Modality '{name}' not found. Available: {list(self.modalities.keys())}")
        return self.modalities[name]

    def list_modalities(self) -> pd.DataFrame:
        """
        List all modalities with fitting status.

        Returns
        -------
        pd.DataFrame
            Summary of modalities with columns:
            - name: Modality name
            - distribution: Distribution type
            - n_features: Number of features
            - n_cells: Number of cells
            - fit_ntc: ✓ if fitted, ✗ if not fitted, - if not configured
            - fit_cis: ✓ if fitted, ✗ if not fitted, - if not applicable
            - fit_trans: ✓ if fitted, ✗ if not fitted

        Notes
        -----
        fit_ntc shows:
        - '-' if technical_group_code not set (no technical covariates configured)
        - '✗' if technical_group_code set but fit_ntc() not run
        - '✓' if fit_ntc() completed
        """
        if not self.modalities:
            return pd.DataFrame(columns=['name', 'distribution', 'n_features', 'n_cells',
                                        'fit_ntc', 'fit_cis', 'fit_trans'])

        # Check if technical groups are configured
        has_technical_groups = 'technical_group_code' in self.meta.columns

        rows = []
        for name, mod in self.modalities.items():
            # Check fitting status for technical fit
            # Only show as needed (✓/✗) if technical groups are configured
            if not has_technical_groups:
                has_technical = '-'  # Not configured/not needed
            else:
                # For 'cis' modality: check primary modality's technical fit status
                # (cis gene was included in primary modality's technical fit)
                if name == 'cis' and self.primary_modality in self.modalities:
                    primary_mod = self.modalities[self.primary_modality]
                    has_technical = '✓' if primary_mod.alpha_y_prefit is not None else '✗'
                else:
                    has_technical = '✓' if mod.alpha_y_prefit is not None else '✗'

            # fit_cis only applies to 'cis' modality
            if name == 'cis':
                # x_true is stored on the main model, not on the modality
                has_cis = '✓' if getattr(self, 'x_true', None) is not None else '✗'
            else:
                has_cis = '-'  # Not applicable

            # fit_trans only applies to trans modalities, not to 'cis'
            if name == 'cis':
                has_trans = '-'  # Not applicable - cis is the predictor, not a trans response
            else:
                has_trans = '✓' if mod.posterior_samples_trans is not None else '✗'

            rows.append({
                'name': name,
                'distribution': mod.distribution,
                'n_features': mod.dims['n_features'],
                'n_cells': mod.dims['n_cells'],
                'fit_ntc': has_technical,
                'fit_cis': has_cis,
                'fit_trans': has_trans
            })
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        """String representation."""
        n_cells = len(self.meta)
        n_modalities = len(self.modalities)
        modality_names = ', '.join(self.modalities.keys())

        return (
            f"bayesDREAM(\n"
            f"  cis_gene='{self.cis_gene}',\n"
            f"  n_cells={n_cells},\n"
            f"  n_modalities={n_modalities},\n"
            f"  modalities=[{modality_names}],\n"
            f"  primary_modality='{self.primary_modality}',\n"
            f"  device={self.device}\n"
            f")"
        )
