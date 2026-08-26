"""
Evaluate a fitted single/additive-Hill trans function at a given cis-gene
log2FC (e.g. -1.0 = 50% knockdown vs. NTC), with full posterior 95% CIs.

Ported from examples/vignette_trans_fit_crispri.py's HILL_LOG2FC_TARGETS /
hill_value_at_log2fc() / hill_value_at_log2fc_all_genes() (kept name- and
column-convention-compatible so summaries stay comparable across every
place this is used -- the vignette, and comparative/reconstruct_export.py's
backfill of production trans_feature_summary_{modality}.csv files).

Unlike trans_param_compare.py's predict_hill_from_summary_row()-based
overlays (which only need the lightweight summary CSV), this needs the
model's full posterior_samples_trans -- i.e. a real model reload. Use
add_log2fc_at_columns() once per (dataset, cis_gene) right after
load_trans_fit(), then save/backfill the resulting summary CSV; downstream
genome-wide comparisons (trans_param_compare.py) then just read the
resulting y_at_..._median/lower/upper columns like any other parameter.
"""

import os
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch

# Same default as the vignette -- x log2FC value(s) to evaluate the fitted
# Hill curve at. -1.0 = the cis gene at 50% of its NTC expression.
HILL_LOG2FC_TARGETS = (-1.0,)


def tag_for_log2fc(x_log2fc: float) -> str:
    """The column-name tag add_log2fc_at_columns() uses for a given
    x_log2fc target, e.g. -1.0 -> 'x_log2fcm1'. Shared here (not just
    inlined in add_log2fc_at_columns()) so callers checking whether a
    summary CSV has already been backfilled -- see is_already_backfilled(),
    used by comparative/reconstruct_export*.py for checkpointing -- use the
    exact same naming and can't drift out of sync.
    """
    return f"x_log2fc{x_log2fc:+.0f}".replace("+", "p").replace("-", "m")


def is_already_backfilled(output_dir: str, modality_name: str, save_dir: str,
                           targets: Sequence[float] = HILL_LOG2FC_TARGETS) -> bool:
    """True if this (dataset, cis_gene) doesn't need reconstruct_and_export()
    run again: `output_dir`'s trans_feature_summary_{modality_name}.csv
    already has every requested y_at_{tag}_median column from
    add_log2fc_at_columns(), AND `save_dir` looks like a complete
    save_model_for_plotting() export.

    Only checks column presence (cheap: pandas' nrows=0 reads just the
    header) and a couple of expected export filenames -- not that the
    VALUES are correct. If a fit was re-run with different results, pass
    force=True to comparative/reconstruct_export*.py's reconstruct_and_export()
    rather than relying on this to detect that.
    """
    csv_path = os.path.join(output_dir, f'trans_feature_summary_{modality_name}.csv')
    if not os.path.exists(csv_path):
        return False
    try:
        cols = set(pd.read_csv(csv_path, nrows=0).columns)
    except Exception:
        return False
    needed = {f'y_at_{tag_for_log2fc(t)}_median' for t in targets}
    if not needed.issubset(cols):
        return False

    if not os.path.isdir(save_dir):
        return False
    required_files = ['meta_plot.csv', 'counts_plot.npz']
    return all(os.path.exists(os.path.join(save_dir, f)) for f in required_files)


def get_x_ntc(model) -> float:
    """The cis gene's NTC reference expression (linear space).

    Prefers the cis modality's own fit_ntc-derived mu_ntc (requires
    add_cis_gene() to have extracted it from the shared panel); falls back
    to the median *fitted* x_true among this model's own NTC cells if that
    extraction never happened (e.g. cis_gene was set eagerly instead) --
    less precise but keeps this usable either way.
    """
    cis_mod = model.get_modality("cis")
    ps_ntc = cis_mod.posterior_samples_ntc
    if ps_ntc is not None and "mu_ntc" in ps_ntc:
        mu_ntc_cis = ps_ntc["mu_ntc"]
        return float(mu_ntc_cis.mean().item() if isinstance(mu_ntc_cis, torch.Tensor) else np.mean(mu_ntc_cis))
    x_true = model.x_true
    x_true = x_true.detach().cpu().numpy() if isinstance(x_true, torch.Tensor) else np.asarray(x_true)
    ntc_mask = (model.meta["target"].values == "ntc")
    return float(np.median(x_true[ntc_mask]))


def hill_value_at_log2fc(model, modality_name: str, x_log2fc: float,
                          is_dependent: Optional[np.ndarray], y_ntc: np.ndarray):
    """Median + 95% CI of the fitted single-Hill y at a given cis-gene log2FC.

    Uses the exact same formula as fit_trans()'s own single_hill model
    (bayesDREAM/utils.py's Hill_based_positive_logK) and io/summary.py's
    _hill_value: y = A + alpha * Vmax_a * x^n / (K_a^n + x^n).

    Also returns the same value as a trans-gene log2FC, log2(y) - log2(y_ntc)
    -- same reference/definition as save_trans_summary()'s own
    full_log2fc/observed_log2fc. Pass `y_ntc` as e.g. df['y_ntc'].values from
    the same save_trans_summary() call so both use an identical reference.

    is_dependent : array of bool, or None
        If given, values for genes where `is_dependent` is False are
        replaced with NaN. Pass None to skip masking (every trans gene,
        dependent or not) -- see hill_value_at_log2fc_all_genes().

    Returns
    -------
    (y_median, y_lower, y_upper, y_log2fc_median, y_log2fc_lower, y_log2fc_upper)
    """
    mod = model.get_modality(modality_name)
    ps = mod.posterior_samples_trans

    def full(key, default=None):
        if key not in ps:
            return default
        v = ps[key]
        v = v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        if v.ndim == 3 and v.shape[1] == 1:
            v = v[:, 0, :]
        return v  # [S, T]

    A = full("A")
    Vmax = full("Vmax_a")
    K = full("K_a")
    n = full("n_a")
    alpha = full("alpha", default=np.ones_like(A))

    x_ntc = get_x_ntc(model)
    x_target = x_ntc * (2.0 ** x_log2fc)

    eps = 1e-12
    x_n = np.exp(n * np.log(max(x_target, eps)))
    K_n = np.exp(n * np.log(np.clip(K, eps, None)))
    hill = x_n / (K_n + x_n)
    y_samples = A + alpha * Vmax * hill  # [S, T]

    y_median = np.median(y_samples, axis=0)
    y_lower = np.quantile(y_samples, 0.025, axis=0)
    y_upper = np.quantile(y_samples, 0.975, axis=0)

    # log2FC(y) = log2(y) - log2(y_ntc), per-sample before collapsing to
    # median/CI (not log2 of the already-collapsed y_median/lower/upper --
    # those don't commute, and per-sample matches io/summary.py).
    y_ntc = np.asarray(y_ntc, dtype=float)
    log2fc_samples = np.log2(np.clip(y_samples, eps, None)) - np.log2(np.clip(y_ntc, eps, None))
    y_log2fc_median = np.median(log2fc_samples, axis=0)
    y_log2fc_lower = np.quantile(log2fc_samples, 0.025, axis=0)
    y_log2fc_upper = np.quantile(log2fc_samples, 0.975, axis=0)

    if is_dependent is not None:
        y_median = np.where(is_dependent, y_median, np.nan)
        y_lower = np.where(is_dependent, y_lower, np.nan)
        y_upper = np.where(is_dependent, y_upper, np.nan)
        y_log2fc_median = np.where(is_dependent, y_log2fc_median, np.nan)
        y_log2fc_lower = np.where(is_dependent, y_log2fc_lower, np.nan)
        y_log2fc_upper = np.where(is_dependent, y_log2fc_upper, np.nan)

    return y_median, y_lower, y_upper, y_log2fc_median, y_log2fc_lower, y_log2fc_upper


def hill_value_at_log2fc_all_genes(model, modality_name: str, x_log2fc: float, y_ntc: np.ndarray):
    """Same as hill_value_at_log2fc() but for every trans gene, dependent or
    not (no is_dependent masking) -- the fitted curve exists regardless of
    whether a gene clears the FDR gate; useful for sanity-checking
    borderline/non-dependent calls.
    """
    return hill_value_at_log2fc(model, modality_name, x_log2fc, None, y_ntc)


def add_log2fc_at_columns(
    model, df: pd.DataFrame, *, modality_name: str = "gene",
    targets: Sequence[float] = HILL_LOG2FC_TARGETS,
) -> pd.DataFrame:
    """Add y_at_x_log2fc{tag}_{median,lower,upper} (dependent-gene-only, NaN
    elsewhere) and the same _allgenes / _log2fc_* variants, for each value in
    `targets`, to `df` (typically the output of model.save_trans_summary()).
    Mutates and returns `df`. Column names match
    examples/vignette_trans_fit_crispri.py exactly, e.g. for x_log2fc=-1.0:

        y_at_x_log2fcm1_median / _lower / _upper                  (dependent-only)
        y_at_x_log2fcm1_log2fc_median / _lower / _upper           (dependent-only, log2FC(y) space)
        y_at_x_log2fcm1_median_allgenes / ..._log2fc_median_allgenes  (every gene)

    Requires `model` to already have fit_trans's posterior loaded
    (load_trans_fit()) and `df` to have 'is_dependent' and 'y_ntc' columns
    (both written by save_trans_summary()).
    """
    is_dep = df["is_dependent"].fillna(False).astype(bool).values
    y_ntc = df["y_ntc"].values

    for x_log2fc in targets:
        tag = tag_for_log2fc(x_log2fc)

        med, lo, hi, lfc_med, lfc_lo, lfc_hi = hill_value_at_log2fc(
            model, modality_name, x_log2fc, is_dep, y_ntc
        )
        df[f"y_at_{tag}_median"] = med
        df[f"y_at_{tag}_lower"] = lo
        df[f"y_at_{tag}_upper"] = hi
        df[f"y_at_{tag}_log2fc_median"] = lfc_med
        df[f"y_at_{tag}_log2fc_lower"] = lfc_lo
        df[f"y_at_{tag}_log2fc_upper"] = lfc_hi

        med_a, lo_a, hi_a, lfc_med_a, lfc_lo_a, lfc_hi_a = hill_value_at_log2fc_all_genes(
            model, modality_name, x_log2fc, y_ntc
        )
        df[f"y_at_{tag}_median_allgenes"] = med_a
        df[f"y_at_{tag}_lower_allgenes"] = lo_a
        df[f"y_at_{tag}_upper_allgenes"] = hi_a
        df[f"y_at_{tag}_log2fc_median_allgenes"] = lfc_med_a
        df[f"y_at_{tag}_log2fc_lower_allgenes"] = lfc_lo_a
        df[f"y_at_{tag}_log2fc_upper_allgenes"] = lfc_hi_a

    return df
