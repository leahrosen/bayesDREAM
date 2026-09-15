"""
Cross-dataset comparison tools for bayesDREAM fits (Domingo / Morris / Replogle, ...).

Two independent workflows live here, matching two different scales of comparison:

- ``trans_param_compare.py`` -- genome/transcriptome-wide comparisons, built
  purely from each run's ``trans_feature_summary_{modality}.csv`` (already
  written by ``save_trans_summary()`` as part of the normal fit_trans stage,
  see ``publication_runs/common/run_trans.py``). No model reload needed, so
  this scales to Morris/Replogle's thousands of trans genes. Use this for
  fitted-parameter scatter/correlation plots.

- ``dose_response_panels.py`` -- per-gene dose-response curve panels (data +
  fitted Hill curve, optionally overlaid across datasets). Requires a full
  model reload via ``load_model_for_plotting()``, which in turn requires
  ``save_model_for_plotting()`` (see ``save_for_plotting.py`` at the repo
  root) to have been run once, in the original fitting session, for that
  (dataset, cis_gene) pair. Only feasible for a bounded number of genes
  (e.g. Domingo's 91 shared trans genes, or a handful of cherry-picked
  Morris/Replogle genes) -- not transcriptome-wide.

``datasets.py`` holds the shared configuration (paths, colors, palettes,
which cis genes have completed fits) used by both.

Two supporting modules feed both workflows:

- ``hill_eval.py`` -- evaluates a fitted Hill curve at a given cis-gene
  log2FC (e.g. -1.0 = 50% knockdown), with full posterior 95% CIs. Needs a
  real model reload (posterior_samples_trans), so it's meant to be run once
  per (dataset, cis_gene) and its result backfilled into that gene's
  trans_feature_summary CSV -- not recomputed on every comparison. Column
  naming matches ``examples/vignette_trans_fit_crispri.py`` exactly.

- ``reconstruct_export.py`` -- automates save_model_for_plotting() +
  hill_eval backfilling for Domingo and Morris, by replaying each cis
  gene's real rendered ``<label>_trans.yaml`` config (the same recipe
  ``publication_runs/common/backfill_trans_summary.py`` uses) rather than
  re-deriving the sum-factor/data-loading logic by hand. Does not cover
  Replogle (no rendered config there -- see its module docstring).
"""
