"""
Temporary: plot_hill_with_ntc_sf with K-prior density underlay on x-axis.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def _add_x_prior_density(
    ax,
    prior_dist,         # scipy.stats frozen dist evaluated in x-axis units
    xlim,
    ylim,
    margin_frac=0.10,   # fraction of y-axis height for the density shape
    alpha=0.25,
    color="mediumpurple",
    label=None,
    n_pts=300,
):
    """
    Draw a bottom-margin density shape (fill_between) showing the prior
    on the x-axis variable (EC50 / inflection in log2 or linear space).
    """
    x_vals  = np.linspace(xlim[0], xlim[1], n_pts)
    density = prior_dist.pdf(x_vals)
    if density.max() == 0:
        return

    density_norm = density / density.max()

    y0 = ylim[0]
    dy = (ylim[1] - ylim[0]) * margin_frac
    y_fill = y0 + density_norm * dy      # shape rises from bottom edge

    ax.fill_between(
        x_vals, y0, y_fill,
        color=color, alpha=alpha, zorder=0, linewidth=0, label=label,
    )
    ax.plot(
        x_vals, y_fill,
        color=color, lw=0.8, alpha=min(alpha * 2, 1.0), zorder=1,
    )


def plot_hill_with_ntc_sf(
    df,
    sf_df,
    x_base            = "EC50",
    x_log2            = True,
    xlim              = None,
    separate_by_count = True,
    sf_cols           = ("sum_factor", "sum_factor_adj", "sum_factor_refit"),
    target_col        = "target",
    ntc_label         = "ntc",
    count_col         = "cis_count_bin",
    xtrue_col         = "log2_x_true",
    y_a_col           = "n_a_median",
    y_b_col           = "n_b_median",
    min_n             = 50,
    smooth_window_max = 400,
    min_periods       = 20,
    figsize           = (16, 5),
    outfile           = None,
    # ── prior density underlay on x-axis ─────────────────────────────────
    trans_prior_params = None,   # dict from model.trans_prior_params (preferred)
    prior_K_log_mu     = None,   # alternative: pass scalar directly (natural log units)
    prior_K_log_sigma  = None,   # alternative: pass scalar directly (natural log units)
    prior_margin_frac  = 0.10,   # fraction of y-height occupied by the density strip
    prior_alpha        = 0.22,
    prior_color        = "mediumpurple",
):
    """
    Plot Hill coefficients vs EC50/inflection with NTC sum-factor smooths.

    Optionally draws the prior distribution on the x-axis (EC50 / inflection)
    as a density strip at the bottom of each panel.

    Prior arguments
    ---------------
    trans_prior_params : dict, optional
        Pass ``model.trans_prior_params`` directly.  The function reads
        ``K_log_mu`` and ``K_log_sigma`` from it (natural-log parameterisation
        of the Log-Normal K prior).  When ``x_log2=True`` these are converted
        to log2 units automatically.

    prior_K_log_mu, prior_K_log_sigma : float, optional
        Direct override -- only used when ``trans_prior_params`` is None.
        Must be in natural-log units (matching the model's convention).

    Example::

        plot_hill_with_ntc_sf(df, sf_df,
                              trans_prior_params=model.trans_prior_params,
                              xlim=(-6, 2))
    """
    # ── resolve prior ─────────────────────────────────────────────────────
    _prior_dist = None
    if trans_prior_params is not None:
        _mu_ln  = trans_prior_params.get("K_log_mu")
        _sig_ln = trans_prior_params.get("K_log_sigma")
    else:
        _mu_ln  = prior_K_log_mu
        _sig_ln = prior_K_log_sigma

    if _mu_ln is not None and _sig_ln is not None:
        if x_log2:
            # log2(K) = log(K) / ln(2)  =>  Normal(K_log_mu/ln2, K_log_sigma/ln2)
            _mu_plot  = _mu_ln  / np.log(2)
            _sig_plot = _sig_ln / np.log(2)
            _prior_dist = st.norm(loc=_mu_plot, scale=_sig_plot)
        else:
            # Linear K ~ LogNormal(K_log_mu, K_log_sigma)
            _prior_dist = st.lognorm(s=_sig_ln, scale=np.exp(_mu_ln))

    # ── build plot ────────────────────────────────────────────────────────
    sf_ntc = sf_df[sf_df[target_col].eq(ntc_label)].copy()

    x_a_col = f"{x_base}_a_mean"
    x_b_col = f"{x_base}_b_mean"

    if x_log2:
        x_a     = np.log2(df[x_a_col])
        x_b     = np.log2(df[x_b_col])
        x_label = f"log2({x_base})"
    else:
        x_a     = df[x_a_col]
        x_b     = df[x_b_col]
        x_label = x_base

    colors = ["navy", "darkorange", "green", "red", "purple"]

    fig, axes = plt.subplots(1, len(sf_cols), figsize=figsize, sharey=True)
    if len(sf_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, sf_cols):
        if col not in sf_ntc.columns:
            ax.set_title(f"{col}: not found")
            continue

        ax.scatter(x_a, df[y_a_col], s=2, alpha=0.4, label="Hill function a", zorder=3)
        ax.scatter(x_b, df[y_b_col], s=2, alpha=0.4, label="Hill function b", zorder=3)

        ax.set_xlabel(x_label)
        ax.set_ylabel("Hill coefficient")
        ax.set_title(col)

        if xlim is not None:
            ax.set_xlim(xlim)

        # fix ylim before drawing underlay so margin_frac is meaningful
        all_n = np.concatenate([df[y_a_col].dropna().values, df[y_b_col].dropna().values])
        y_lo  = np.nanpercentile(all_n, 1)
        y_hi  = np.nanpercentile(all_n, 99)
        pad   = 0.05 * (y_hi - y_lo)
        ylim_n = (y_lo - pad, y_hi + pad)
        ax.set_ylim(ylim_n)

        # ── prior density strip ───────────────────────────────────────────
        if _prior_dist is not None:
            _xlim_now = xlim if xlim is not None else ax.get_xlim()
            _add_x_prior_density(
                ax, _prior_dist, _xlim_now, ylim_n,
                margin_frac=prior_margin_frac,
                alpha=prior_alpha,
                color=prior_color,
                label="K prior",
            )

        # ── secondary axis: NTC sum-factor smooths ────────────────────────
        ax2 = ax.twinx()

        valid = (
            sf_ntc[col].notna() &
            sf_ntc[xtrue_col].notna() &
            np.isfinite(sf_ntc[col]) &
            (sf_ntc[col] > 0)
        )

        if separate_by_count:
            for i in range(5):
                mask = valid & (sf_ntc[count_col] == i)
                if mask.sum() < min_n:
                    continue
                sub = sf_ntc.loc[mask].sort_values(xtrue_col)
                w   = min(smooth_window_max, max(min_periods, mask.sum() // 3))
                x_smooth = sub[xtrue_col].rolling(w, center=True, min_periods=min_periods).mean()
                y_smooth = np.log2(sub[col]).rolling(w, center=True, min_periods=min_periods).mean()
                ax2.plot(x_smooth, y_smooth, color=colors[i], lw=2,
                         label=f"NTC count={i}{'+' if i == 4 else ''}")
        else:
            sub = sf_ntc.loc[valid].sort_values(xtrue_col)
            if len(sub) >= min_n:
                w = min(smooth_window_max, max(min_periods, len(sub) // 3))
                x_smooth = sub[xtrue_col].rolling(w, center=True, min_periods=min_periods).mean()
                y_smooth = np.log2(sub[col]).rolling(w, center=True, min_periods=min_periods).mean()
                ax2.plot(x_smooth, y_smooth, color="black", lw=2, label="NTC sum-factor smooth")

        ax2.set_ylabel(f"log2({col})")
        if valid.sum() > 0:
            ax2.set_ylim(np.nanpercentile(np.log2(sf_ntc.loc[valid, col]), [1, 99]))
        if xlim is not None:
            ax2.set_xlim(xlim)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, markerscale=3)

    by_count_txt = "by count bin" if separate_by_count else "overall"
    plt.suptitle(
        f"Hill coefficients vs {x_label} with NTC sum-factor smooths ({by_count_txt})",
        y=1.02,
    )
    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight")

    plt.show()
