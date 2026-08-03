#!/usr/bin/env Rscript
# Evaluation plots for the simulation study (docs/SIMULATION_STUDY_PLAN.md §8), reading
# the tidy outputs of extract_trans_eval.R (trans_eval_long.csv) and extract_cis_mse.py
# (cis_eval.csv). Writes one PDF per plot to --outdir.
#
# ASSUMPTIONS made where the request was ambiguous -- check these against the real output
# and adjust if wrong, they're isolated to the helper functions / facet formulas below:
#   - TPR/prop-additive-hill plots: "facets for the different y_ntc values" +
#     "additionally facetted by X/Y" is implemented as nested facet_grid rows
#     (e.g. cells_per_gene + y_ntc_log2 ~ n_guides), not a separate figure per y_ntc.
#   - TPR's continuous x-axis (prop_log2FC_observed) is binned (8 equal-width bins,
#     0-1) to compute a rate per bin; full_log2FC_true is used directly as a discrete
#     fill (it only takes 4 values in the design grid: 0.5/1/2/4, no binning needed).
#   - "prop additive hill fit" plot: produced with both facet schemes (mirroring the
#     paired FPR/TPR plots), using the same x/fill structure as the TPR plots.
#   - Plot 3d (full log2FC estimated-vs-true) includes both null- and single_hill-truth
#     features (nulls have prop_log2FC_observed = NA -> shown as a distinct grey fill);
#     3a/3b (n, EC50) are restricted to single_hill-truth features with a resolved
#     active-component estimate (estimated_* not NA), since n_true/K_log2FC_true aren't
#     meaningful for null features.
#   - fit_cis MSE plot: facet_grid(log2_X_NTC ~ log2_o_x) is used inside each
#     guide_shape-specific figure. Note log2_X_NTC has 4 grid levels (-1,0,1,2), not 3 --
#     this produces 4 rows x 2 columns, not "3 rows" as stated; the grid is genuinely
#     4-level (bayesDREAM/simulation/cis_panel_simulation.py's LOG2_X_NTC_VALUES), so if
#     3 rows was intentional (e.g. excluding one level), filter before plotting.
#
# Usage:
#   Rscript plot_evaluation.R --trans_eval trans_eval_long.csv --cis_eval cis_eval.csv \
#       --outdir ./eval_plots

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

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
  list(trans_eval = NULL, cis_eval = NULL, outdir = "./eval_plots")
)
stopifnot(!is.null(args$trans_eval), !is.null(args$cis_eval))
dir.create(args$outdir, showWarnings = FALSE, recursive = TRUE)

save_plot <- function(p, name, width = 10, height = 8) {
  path <- file.path(args$outdir, paste0(name, ".pdf"))
  ggsave(path, p, width = width, height = height, limitsize = FALSE)
  cat(sprintf("Wrote %s\n", path))
}

trans_eval <- fread(args$trans_eval)
cis_eval <- fread(args$cis_eval)

# ---------------------------------------------------------------------------
# 1. False positive rate
# ---------------------------------------------------------------------------
# FPR per (y_ntc_log2, o_y_log2, <facet vars>) = fraction of no_effect-truth features
# called positive (is_positive), pooling across all matching scenarios/replicates.
fpr_data <- trans_eval[effect_type == "no_effect"]

make_fpr_plot <- function(dt, facet_rows, facet_cols, title_suffix) {
  agg <- dt[, .(FPR = mean(is_positive, na.rm = TRUE), n = .N),
            by = c("y_ntc_log2", "o_y_log2", facet_rows, facet_cols)]
  p <- ggplot(agg, aes(x = factor(y_ntc_log2), y = FPR, fill = factor(o_y_log2))) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    geom_hline(yintercept = 0.05, linetype = "dashed", color = "red") +
    facet_grid(reformulate(facet_cols, paste(facet_rows, collapse = "+"))) +
    labs(x = "y_ntc (log2)", y = "False positive rate", fill = "y overdispersion (log2)",
         title = paste("FPR --", title_suffix)) +
    theme_bw()
  p
}

save_plot(make_fpr_plot(fpr_data, "cells_per_gene", "n_guides",
                         "faceted by n_guides (x) x cells_per_gene (y)"),
          "1a_fpr_by_guides_cells")
save_plot(make_fpr_plot(fpr_data, "log2_X_NTC", "log2_o_x",
                         "faceted by cis o_x (x) x cis log2_X_NTC (y)"),
          "1b_fpr_by_cis_ox_xntc")

# ---------------------------------------------------------------------------
# 2. True positive rate (+ prop. additive_hill fit)
# ---------------------------------------------------------------------------
# Restricted to single_hill-truth features whose FIT came out single_hill or
# not_dependent ("flat") -- excludes additive_hill fits, reported separately below.
tpr_base <- trans_eval[effect_type == "single_hill" &
                          fit_type %in% c("single_hill", "not_dependent")]
tpr_base[, prop_bin := cut(prop_log2FC_observed, breaks = seq(0, 1, by = 0.125),
                            include.lowest = TRUE)]
tpr_base[, prop_bin_mid := {
  b <- as.character(prop_bin)
  nums <- regmatches(b, gregexpr("[0-9.]+", b))
  sapply(nums, function(x) mean(as.numeric(x)))
}]

make_tpr_plot <- function(dt, facet_rows, facet_cols, title_suffix) {
  agg <- dt[!is.na(prop_bin_mid), .(TPR = mean(is_positive, na.rm = TRUE), n = .N),
            by = c("prop_bin_mid", "full_log2FC_true", "y_ntc_log2", facet_rows, facet_cols)]
  p <- ggplot(agg, aes(x = prop_bin_mid, y = TPR, color = factor(full_log2FC_true))) +
    geom_point(aes(size = n), alpha = 0.7) +
    geom_line(aes(group = factor(full_log2FC_true))) +
    facet_grid(reformulate(facet_cols, paste(c(facet_rows, "y_ntc_log2"), collapse = "+"))) +
    labs(x = "Proportion of full log2FC observed", y = "True positive rate",
         color = "True full log2FC", size = "n features",
         title = paste("TPR (single_hill/flat fits only) --", title_suffix)) +
    theme_bw()
  p
}

save_plot(make_tpr_plot(tpr_base, "cells_per_gene", "n_guides",
                         "faceted by n_guides (x) x cells_per_gene+y_ntc (y)"),
          "2a_tpr_by_guides_cells", height = 14)
save_plot(make_tpr_plot(tpr_base, "log2_X_NTC", "log2_o_x",
                         "faceted by cis o_x (x) x cis log2_X_NTC+y_ntc (y)"),
          "2b_tpr_by_cis_ox_xntc", height = 14)

# prop. additive_hill fit: among single_hill-truth features, fraction fit as
# 'additive_hill' (i.e. NOT single_hill/not_dependent) -- same x/fill/facet structure.
prop_add_base <- trans_eval[effect_type == "single_hill"]
prop_add_base[, prop_bin := cut(prop_log2FC_observed, breaks = seq(0, 1, by = 0.125),
                                 include.lowest = TRUE)]
prop_add_base[, prop_bin_mid := {
  b <- as.character(prop_bin)
  nums <- regmatches(b, gregexpr("[0-9.]+", b))
  sapply(nums, function(x) mean(as.numeric(x)))
}]
prop_add_base[, is_additive := fit_type == "additive_hill"]

make_prop_additive_plot <- function(dt, facet_rows, facet_cols, title_suffix) {
  agg <- dt[!is.na(prop_bin_mid), .(prop_additive = mean(is_additive, na.rm = TRUE), n = .N),
            by = c("prop_bin_mid", "full_log2FC_true", "y_ntc_log2", facet_rows, facet_cols)]
  p <- ggplot(agg, aes(x = prop_bin_mid, y = prop_additive, color = factor(full_log2FC_true))) +
    geom_point(aes(size = n), alpha = 0.7) +
    geom_line(aes(group = factor(full_log2FC_true))) +
    facet_grid(reformulate(facet_cols, paste(c(facet_rows, "y_ntc_log2"), collapse = "+"))) +
    labs(x = "Proportion of full log2FC observed", y = "Proportion fit as additive_hill",
         color = "True full log2FC", size = "n features",
         title = paste("Prop. additive_hill fit (single_hill truth) --", title_suffix)) +
    theme_bw()
  p
}

save_plot(make_prop_additive_plot(prop_add_base, "cells_per_gene", "n_guides",
                                   "faceted by n_guides (x) x cells_per_gene+y_ntc (y)"),
          "2c_prop_additive_by_guides_cells", height = 14)
save_plot(make_prop_additive_plot(prop_add_base, "log2_X_NTC", "log2_o_x",
                                   "faceted by cis o_x (x) x cis log2_X_NTC+y_ntc (y)"),
          "2d_prop_additive_by_cis_ox_xntc", height = 14)

# ---------------------------------------------------------------------------
# 3. Estimated vs true
# ---------------------------------------------------------------------------
make_est_vs_true_plot <- function(dt, x_col, y_col, fill_col, facet_rows, facet_cols,
                                   x_lab, y_lab, fill_lab, title_suffix) {
  dt <- dt[!is.na(get(x_col)) & !is.na(get(y_col))]
  p <- ggplot(dt, aes(x = .data[[x_col]], y = .data[[y_col]], color = .data[[fill_col]])) +
    geom_point(alpha = 0.5, size = 0.8) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
    facet_grid(reformulate(facet_cols, facet_rows)) +
    scale_color_viridis_c(na.value = "grey70") +
    labs(x = x_lab, y = y_lab, color = fill_lab, title = paste(y_lab, "vs true --", title_suffix)) +
    theme_bw()
  p
}

sh_active <- trans_eval[effect_type == "single_hill" & !is.na(estimated_n)]
all_features_c <- copy(trans_eval)
all_features_c[, y_ntc_log2_fit := log2(y_ntc)]  # fitted y_ntc (natural scale) -> log2
all_features_d <- copy(trans_eval)

facet_schemes <- list(
  guides_cells = list(rows = "cells_per_gene", cols = "n_guides",
                       label = "faceted by n_guides (x) x cells_per_gene (y)"),
  cis_ox_xntc = list(rows = "log2_X_NTC", cols = "log2_o_x",
                      label = "faceted by cis o_x (x) x cis log2_X_NTC (y)")
)

for (scheme_name in names(facet_schemes)) {
  s <- facet_schemes[[scheme_name]]

  save_plot(make_est_vs_true_plot(sh_active, "n_true", "estimated_n", "prop_log2FC_observed",
                                   s$rows, s$cols, "True n", "Estimated n",
                                   "Prop. log2FC observed", s$label),
            sprintf("3a_n_est_vs_true_%s", scheme_name))

  save_plot(make_est_vs_true_plot(sh_active, "K_log2FC_true", "estimated_K_log2FC",
                                   "prop_log2FC_observed", s$rows, s$cols,
                                   "True EC50 (log2FC)", "Estimated EC50 (log2FC)",
                                   "Prop. log2FC observed", s$label),
            sprintf("3b_EC50_est_vs_true_%s", scheme_name))

  save_plot(make_est_vs_true_plot(all_features_c, "y_ntc_log2", "y_ntc_log2_fit", "o_y_log2",
                                   s$rows, s$cols, "True y_ntc (log2)",
                                   "Estimated y_ntc (log2)", "y overdispersion (log2)", s$label),
            sprintf("3c_yntc_est_vs_true_%s", scheme_name))

  save_plot(make_est_vs_true_plot(all_features_d, "full_log2FC_true", "full_log2fc_median",
                                   "prop_log2FC_observed", s$rows, s$cols,
                                   "True full log2FC", "Estimated full log2FC",
                                   "Prop. log2FC observed (NA = null truth)", s$label),
            sprintf("3d_fulllog2fc_est_vs_true_%s", scheme_name))
}

# ---------------------------------------------------------------------------
# 4. fit_cis accuracy: MSE of log2(x_true)
# ---------------------------------------------------------------------------
cis_plot_data <- cis_eval[!is.na(mse_log2_x_true)]

for (shape in unique(cis_plot_data$guide_shape)) {
  d <- cis_plot_data[guide_shape == shape]
  p <- ggplot(d, aes(x = factor(cells_per_gene), y = mse_log2_x_true, fill = factor(n_guides))) +
    geom_point(position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.7),
               shape = 21, size = 2, alpha = 0.7) +
    stat_summary(aes(group = factor(n_guides)), fun = mean, geom = "line",
                 position = position_dodge(width = 0.7), color = "black") +
    facet_grid(log2_X_NTC ~ log2_o_x,
               labeller = label_both) +
    labs(x = "Cells per gene", y = "MSE of log2(x_true)", fill = "n_guides",
         title = sprintf("fit_cis accuracy -- guide_shape = %s", shape)) +
    theme_bw()
  save_plot(p, sprintf("4_cis_mse_%s", shape), height = 12)
}

cat("Done.\n")
