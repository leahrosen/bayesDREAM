#!/usr/bin/env Rscript
# Extract a tidy long-format table joining trans_ground_truth.csv + fitted
# trans_feature_summary_gene.csv + design_matrix.csv scenario variables, across the
# full simulation study, for evaluation plotting (docs/SIMULATION_STUDY_PLAN.md §8).
# See plot_evaluation.R for the plots that consume this output.
#
# Pure R, no torch needed -- trans_ground_truth.csv/trans_feature_summary_gene.csv are
# plain CSVs. For the fit_cis MSE evaluation (which needs the .pt posterior files), see
# extract_cis_mse.py instead.
#
# Column semantics assumed here (verified against source, 2026-08-03):
#   - trans_ground_truth.csv: feature, effect_type ('no_effect'/'single_hill'),
#     y_ntc_log2, o_y_log2, y_ntc_true, o_y_true, x_ntc_true, n_true, K_log2FC_true,
#     full_log2FC_true, A_true, Vmax_true, K_true. Only x_ntc_true/K_true vary by
#     scenario_id (via log2_X_NTC); everything else is identical across all 144
#     scenarios (bayesDREAM/simulation/cis_panel_simulation.py's build_trans_panel_grid()
#     takes no scenario-specific arguments).
#   - trans_feature_summary_gene.csv (additive_hill fit): feature, fdr_alpha, fdr_beta,
#     is_dependent, fit_type ('additive_hill'/'single_hill'/'not_dependent'),
#     which_active ('a'/'b'/'both'/NA), classification, n_a_median, n_b_median,
#     EC50_a_log2fc, EC50_b_log2fc (K in the SAME log2FC-relative-to-x_ntc units as
#     ground truth's K_log2FC_true -- do not compare against K_a_median/K_b_median
#     directly, those are in absolute cis-expression units), Vmax_a_median,
#     Vmax_b_median, full_log2fc_median, observed_log2fc, y_ntc, x_ntc.
#     No scenario_id/replicate_id column -- identity comes from the directory path.
#
# Derived here:
#   - prop_log2FC_observed: fraction of a feature's *theoretical* dynamic range
#     (full_log2FC_true) actually spanned by the guides' achieved cis-expression range
#     for that scenario/replicate (from guide_ground_truth.csv's x_eff_g_true, min/max
#     across guides including the NTC row). Formula replicated from
#     bayesDREAM/simulation/simulation.py's _compute_AV_from_fc / single-Hill curve:
#       y(x) = A_true + Vmax_true * x^n_true / (x^n_true + K_true^n_true)
#       prop_log2FC_observed = |log2(y(x_max)) - log2(y(x_min))| / full_log2FC_true
#     NA for effect_type=='no_effect' (K_true/n_true/full_log2FC_true aren't meaningful
#     for null features).
#   - guide_log2_range: log2(x_max) - log2(x_min) of the guides' achieved cis-expression
#     range for that scenario/replicate (same x_min/x_max used for prop_log2FC_observed
#     above) -- a scenario-level quantity (same value for every feature row within a
#     scenario/replicate), for plots that want to see effects vs. raw guide dynamic
#     range rather than the per-feature fraction-of-curve-observed.
#   - estimated_n / estimated_K_log2FC / estimated_Vmax: pulled from whichever fitted
#     component (a or b) matches `which_active`, NA if which_active is 'both' or NA
#     (ambiguous -- no single recovered curve to compare against a single-Hill truth).
#     This assumption (use the *matching-active* component, not e.g. always component a)
#     follows docs/SIMULATION_STUDY_PLAN.md §8's 2026-07-29 update.
#   - is_positive: fdr-significance call, TRUE if is_dependent, else FALSE (recomputed
#     from fdr_alpha/fdr_beta < fdr_threshold if is_dependent is missing, as a fallback).
#
# Usage:
#   Rscript extract_trans_eval.R --design_matrix $OUT/design_matrix.csv \
#       --data_root $DATA --outfile $OUT/trans_eval_long.csv

suppressPackageStartupMessages({
  library(data.table)
})

# Minimal --flag value parser (base R only -- avoids depending on optparse, which
# isn't guaranteed to be installed alongside data.table).
parse_flags <- function(argv, defaults) {
  args <- defaults
  i <- 1
  while (i <= length(argv)) {
    flag <- sub("^--", "", argv[i])
    if (!flag %in% names(defaults)) stop(sprintf("Unknown flag: --%s", flag))
    args[[flag]] <- argv[i + 1]
    i <- i + 2
  }
  args
}
args <- parse_flags(
  commandArgs(trailingOnly = TRUE),
  list(design_matrix = NULL, data_root = NULL, outfile = NULL, fdr_threshold = "0.05")
)
stopifnot(!is.null(args$design_matrix), !is.null(args$data_root), !is.null(args$outfile))
args$fdr_threshold <- as.numeric(args$fdr_threshold)

design_matrix <- fread(args$design_matrix)

hill_y <- function(x, A, Vmax, K, n) {
  A + Vmax * x^n / (x^n + K^n)
}

scenario_dir <- function(data_root, scenario_id, replicate_id) {
  file.path(data_root, paste0("scenario_", scenario_id), paste0("rep_", replicate_id))
}

# Columns to keep from trans_feature_summary_gene.csv -- kept minimal and defensive
# (fread's fill=TRUE + explicit column-existence checks below) since exact column
# availability can depend on fit_trans()/save_trans_summary() call options.
fit_cols_wanted <- c(
  "feature", "fdr_alpha", "fdr_beta", "is_dependent", "fit_type", "which_active",
  "classification", "n_a_median", "n_b_median", "EC50_a_log2fc", "EC50_b_log2fc",
  "Vmax_a_median", "Vmax_b_median", "full_log2fc_median", "observed_log2fc",
  "y_ntc", "x_ntc"
)

process_one <- function(design_row) {
  sdir <- scenario_dir(args$data_root, design_row$scenario_id, design_row$replicate_id)
  gt_path <- file.path(sdir, "trans_ground_truth.csv")
  guide_path <- file.path(sdir, "guide_ground_truth.csv")
  fit_path <- file.path(sdir, "fit", "recovery", "trans_feature_summary_gene.csv")

  if (!file.exists(gt_path) || !file.exists(guide_path) || !file.exists(fit_path)) {
    return(NULL)
  }

  gt <- fread(gt_path)
  guide_gt <- fread(guide_path)
  if (!"x_eff_g_true" %in% names(guide_gt)) {
    warning(sprintf("%s: guide_ground_truth.csv missing x_eff_g_true, skipping", sdir))
    return(NULL)
  }
  x_min <- min(guide_gt$x_eff_g_true, na.rm = TRUE)
  x_max <- max(guide_gt$x_eff_g_true, na.rm = TRUE)
  guide_log2_range <- log2(x_max) - log2(x_min)

  fit <- fread(fit_path, fill = TRUE)
  present_cols <- intersect(fit_cols_wanted, names(fit))
  missing_cols <- setdiff(fit_cols_wanted, names(fit))
  fit <- fit[, ..present_cols]
  for (col in missing_cols) fit[[col]] <- NA

  merged <- merge(gt, fit, by = "feature", all.x = TRUE)
  merged[, guide_log2_range := guide_log2_range]

  # prop_log2FC_observed: only meaningful for single_hill-truth rows.
  merged[, prop_log2FC_observed := NA_real_]
  is_sh <- merged$effect_type == "single_hill"
  if (any(is_sh)) {
    y_max <- hill_y(x_max, merged$A_true[is_sh], merged$Vmax_true[is_sh],
                     merged$K_true[is_sh], merged$n_true[is_sh])
    y_min <- hill_y(x_min, merged$A_true[is_sh], merged$Vmax_true[is_sh],
                     merged$K_true[is_sh], merged$n_true[is_sh])
    eps <- 1e-12
    merged[is_sh, prop_log2FC_observed :=
             abs(log2(pmax(y_max, eps)) - log2(pmax(y_min, eps))) / full_log2FC_true]
  }

  # Active-component estimate, matched to which_active ('a'/'b'); NA if 'both' or NA.
  merged[, `:=`(estimated_n = NA_real_, estimated_K_log2FC = NA_real_,
                estimated_Vmax = NA_real_)]
  is_a <- !is.na(merged$which_active) & merged$which_active == "a"
  is_b <- !is.na(merged$which_active) & merged$which_active == "b"
  merged[is_a, `:=`(estimated_n = n_a_median, estimated_K_log2FC = EC50_a_log2fc,
                     estimated_Vmax = Vmax_a_median)]
  merged[is_b, `:=`(estimated_n = n_b_median, estimated_K_log2FC = EC50_b_log2fc,
                     estimated_Vmax = Vmax_b_median)]

  # Significance call, with a fallback if is_dependent wasn't present in the CSV.
  if (all(is.na(merged$is_dependent))) {
    merged[, is_positive := (!is.na(fdr_alpha) & fdr_alpha < args$fdr_threshold) |
             (!is.na(fdr_beta) & fdr_beta < args$fdr_threshold)]
  } else {
    merged[, is_positive := as.logical(is_dependent)]
  }

  design_cols <- intersect(
    c("scenario_id", "replicate_id", "cells_per_gene", "n_guides", "guide_shape",
      "sigma_eff", "log2_X_NTC", "log2_o_x", "seed"),
    names(design_row)
  )
  for (col in design_cols) merged[[col]] <- design_row[[col]]

  merged
}

n_rows <- nrow(design_matrix)
results <- vector("list", n_rows)
n_ok <- 0L
n_skipped <- 0L
for (i in seq_len(n_rows)) {
  res <- process_one(design_matrix[i])
  if (is.null(res)) {
    n_skipped <- n_skipped + 1L
  } else {
    n_ok <- n_ok + 1L
    results[[i]] <- res
  }
  if (i %% 50 == 0) cat(sprintf("[%d/%d] scenarios processed (%d ok, %d skipped)\n",
                                 i, n_rows, n_ok, n_skipped))
}

combined <- rbindlist(results, fill = TRUE)
fwrite(combined, args$outfile)
cat(sprintf(
  "Wrote %d rows (%d scenario/replicates x up to 1736 features; %d scenario/replicates skipped -- fit not complete yet or missing files) to %s\n",
  nrow(combined), n_ok, n_skipped, args$outfile
))
