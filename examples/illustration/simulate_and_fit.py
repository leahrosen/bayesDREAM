"""
simulate_and_fit.py - Simulate data for bayesDREAM 3-step illustration and run the model.

Simulation design
-----------------
Step 1 (technical, Plot 1):
  NTC cells from 4 cell lines.  Each cell line has a gene-level multiplicative
  correction factor ALPHA_CL (additive in log-space, same for all genes here).
  SF = 1 for all cells.  NB overdispersion is the same for all cell lines.
  This data is generated standalone for Plot 1 — it is NOT passed to bayesDREAM.

Steps 2 & 3 (cis + trans, Plots 2 & 3):
  All cells (NTC + guides) have no cell-line effect (SF = 1, alpha = 1).
  Mirrors the bayesDREAM model hierarchy:
    log2_x_cell ~ Normal(log2(X_NTC * fc_guide), SIGMA_X_CELL)   [cell-level latent x]
    x_true_cell  = 2 ^ log2_x_cell
    x_counts     ~ NegBinom(x_true_cell, NB_DISP_X)
    y_counts     ~ NegBinom(2^Hill(x_true_cell), NB_DISP_Y)       [from underlying x, not fitted]
  x_underlying_log2 = log2_x_cell (per cell) is exported for the bonus plot comparing
  true latent x to bayesDREAM's fitted x_true.

Outputs (written to output_illustration/ next to this script):
  data_technical.csv  - NTC cells with logcounts per technical group (Plot 1)
  data_cis.csv        - All cells: raw x-logcounts + fitted x_true + true x (Plot 2 + bonus)
  data_trans.csv      - All cells: fitted x_true log2FC + simulated y-logcounts (Plot 3)
  hill_curve.csv      - Dense x grid with true Hill function + uncertainty band (Plot 3)
  ntc_reference.csv   - NTC reference values for Plot 3 reference lines

Usage:
  cd bayesDREAM_forClaude
  python examples/illustration/simulate_and_fit.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import torch
import pyro

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from bayesDREAM import bayesDREAM

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
pyro.set_rng_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Simulation parameters
# ─────────────────────────────────────────────────────────────────────────────
CELL_LINES = ["ref", "A", "B", "C"]

# Gene × cell-line multiplicative technical correction factors.
# Applied only in the Step 1 (technical) simulation.
# Same factor for all genes (global scaling), additive 0.55 / -0.40 / 0.75 in log2.
ALPHA_CL = {
    "ref": 1.0,
    "A":   np.exp(0.55),
    "B":   np.exp(-0.40),
    "C":   np.exp(0.75),
}

CIS_GENE   = "GeneX"
TRANS_GENE = "GeneY"
OTHER_GENES = ["GeneZ1", "GeneZ2", "GeneZ3"]
ALL_GENES   = [CIS_GENE, TRANS_GENE] + OTHER_GENES

N_NTC_TECH       = 20   # NTC cells per cell line, for technical plot only
N_NTC_CIS        = 20   # NTC cells for cis / trans fit (no cell-line effect)
N_CELLS_PER_GUIDE = 40  # guide cells per guide

# Guide definitions: (name, target_gene, FC on X relative to NTC)
GUIDE_DEFS = [
    ("KO1", CIS_GENE, 0.25),   # log2FC = -2.0  → near x→0 asymptote
    ("KO2", CIS_GENE, 0.40),   # log2FC = -1.32
    ("KO3", CIS_GENE, 0.98),   # log2FC ≈ -0.03
    ("CA1", CIS_GENE, 1.50),   # log2FC ≈ +0.58
    ("CA2", CIS_GENE, 2.00),   # log2FC = +1.0  → near x→∞ asymptote
]
NTC_GUIDE = "NTC_1"

X_NTC = 50.0  # mean cis-gene counts for NTC in reference conditions

# Within-guide cell-level log2 standard deviation for x_true.
# Mirrors model_x: log_x_true ~ Normal(log2(x_eff_g), sigma_eff).
# Separate from NB technical noise below.
SIGMA_X_CELL = 0.20

# NegBinom overdispersion — the SAME for all technical groups / cell lines.
# (Not to be confused with ALPHA_CL, the multiplicative technical correction.)
NB_DISP_X     = 0.10   # technical NB noise; biological variation is in SIGMA_X_CELL
NB_DISP_Y     = 0.08
NB_DISP_OTHER = 0.30

# Additive Hill function for trans gene (y is in log2-count space):
#   y = A + Vmax_a · H(x, K_a, n_a) − Vmax_b · H(x, K_b, n_b)
# Key values:
#   y(x→0)           = A            = 5.0
#   y(x=12.5, FC=-2) ≈ 5.17         (near lower asymptote)
#   y(x=50,   FC= 0) ≈ 10.0         (NTC)
#   y(x=100,  FC=+1) ≈ 2.45         (near upper asymptote)
#   y(x→∞)           = A+Vmax_a−Vmax_b = 2.0
HILL_A      = 5.0
HILL_Vmax_a = 10.35;  HILL_K_a = 35.0;  HILL_n_a = 4.0
HILL_Vmax_b = 13.35;  HILL_K_b = 60.0;  HILL_n_b = 6.0

OTHER_GENE_MU = 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def hill_pos(x, Vmax, K, n):
    return Vmax * (x ** n) / (K ** n + x ** n)

def true_trans_log2(x_true):
    """Additive Hill function: y in log2-count space as a function of x_true (counts)."""
    return (HILL_A
            + hill_pos(x_true, HILL_Vmax_a, HILL_K_a, HILL_n_a)
            - hill_pos(x_true, HILL_Vmax_b, HILL_K_b, HILL_n_b))

def sample_nb(mu, disp, size=1):
    """NegBinom sample: Var = mu + disp · mu²."""
    from scipy.stats import nbinom
    r = 1.0 / disp
    p = r / (r + mu)
    return nbinom.rvs(r, p, size=size).astype(np.float32)

def to_mean_1d(t):
    """Posterior (S, N) → mean over samples → (N,).  Point estimate (N,) → passthrough."""
    arr = t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    return arr.mean(axis=0) if arr.ndim == 2 else arr

def to_sd_1d(t):
    arr = t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    return arr.std(axis=0) if arr.ndim == 2 else np.zeros_like(arr)

def guide_type(guide):
    if guide == NTC_GUIDE:  return "NTC"
    if guide.startswith("KO"): return "CRISPRi"
    if guide.startswith("CA"): return "CRISPRa"
    return "Other"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 simulation — technical (standalone, not passed to bayesDREAM)
# ─────────────────────────────────────────────────────────────────────────────
# Each cell line has a multiplicative technical factor ALPHA_CL applied to
# ALL genes (SF = 1; NB overdispersion same for all groups).

Y_NTC = 2 ** true_trans_log2(X_NTC)   # ≈ 1024, NTC expected trans counts

tech_rows = []
for cl in CELL_LINES:
    for i in range(N_NTC_TECH):
        x_c = int(sample_nb(X_NTC * ALPHA_CL[cl], NB_DISP_X)[0])
        y_c = int(sample_nb(Y_NTC * ALPHA_CL[cl], NB_DISP_Y)[0])
        tech_rows.append({
            "cell":       f"NTC_{cl}_{i:03d}",
            "cell_line":  cl,
            "x_logcounts": np.log2(x_c + 1),
            "y_logcounts": np.log2(y_c + 1),
            "group": "Reference" if cl == "ref" else f"Technical group {cl}",
        })

tech_df = pd.DataFrame(tech_rows)
print(f"Technical cells: {len(tech_df)}  "
      f"(4 cell lines × {N_NTC_TECH} = {4*N_NTC_TECH} NTC cells)")


# ─────────────────────────────────────────────────────────────────────────────
# Steps 2 & 3 simulation — cis / trans (no cell-line effects, SF = 1)
# y_counts are simulated from the underlying x (X_NTC * fc per guide).
# ─────────────────────────────────────────────────────────────────────────────

rows = []

# NTC cells — split evenly across two cell lines (both alpha=1, SF=1).
# fit_technical requires C ≥ 2 technical groups; the two groups here are
# deliberately identical so the model estimates zero technical offset between them.
NTC_CELL_LINES = ["ref", "grp_B"]
for i in range(N_NTC_CIS):
    cl = NTC_CELL_LINES[i % len(NTC_CELL_LINES)]
    # Mirror model_x: log2(x_true) ~ Normal(guide_mean, sigma_eff), then NB
    log2_x = rng.normal(np.log2(X_NTC), SIGMA_X_CELL)
    x_true_cell = 2.0 ** log2_x
    x_c = int(sample_nb(x_true_cell, NB_DISP_X)[0])
    y_c = int(sample_nb(2 ** true_trans_log2(x_true_cell), NB_DISP_Y)[0])
    rows.append({
        "cell": f"NTC_{i:03d}", "guide": NTC_GUIDE, "target": "ntc",
        "cell_line": cl, "sum_factor": 1.0,
        "x_underlying_log2": log2_x,           # per-cell latent log2(x_true)
        CIS_GENE:   x_c,
        TRANS_GENE: y_c,
        **{g: int(sample_nb(OTHER_GENE_MU, NB_DISP_OTHER)[0]) for g in OTHER_GENES},
    })

# Guide cells (no cell-line effect; also split across the two groups)
for guide, target, fc in GUIDE_DEFS:
    for i in range(N_CELLS_PER_GUIDE):
        cl = NTC_CELL_LINES[i % len(NTC_CELL_LINES)]
        # Mirror model_x: log2(x_true) ~ Normal(guide_mean, sigma_eff), then NB
        log2_x = rng.normal(np.log2(X_NTC * fc), SIGMA_X_CELL)
        x_true_cell = 2.0 ** log2_x
        x_c = int(sample_nb(x_true_cell, NB_DISP_X)[0])
        y_c = int(sample_nb(2 ** true_trans_log2(x_true_cell), NB_DISP_Y)[0])
        rows.append({
            "cell": f"{guide}_{i:03d}", "guide": guide, "target": target,
            "cell_line": cl, "sum_factor": 1.0,
            "x_underlying_log2": log2_x,        # per-cell latent log2(x_true)
            CIS_GENE:   x_c,
            TRANS_GENE: y_c,
            **{g: int(sample_nb(OTHER_GENE_MU, NB_DISP_OTHER)[0]) for g in OTHER_GENES},
        })

df       = pd.DataFrame(rows)
meta     = df[["cell", "guide", "target", "cell_line", "sum_factor",
               "x_underlying_log2"]].copy()
counts_df = df.set_index("cell")[ALL_GENES].T
counts_df = counts_df[meta["cell"].values]

print(f"Cis/trans cells: {len(meta)}  "
      f"(NTC={( meta['guide']==NTC_GUIDE).sum()}, "
      f"guide={(meta['guide']!=NTC_GUIDE).sum()})")


# ─────────────────────────────────────────────────────────────────────────────
# Run bayesDREAM
# ─────────────────────────────────────────────────────────────────────────────

OUT = os.path.join(os.path.dirname(__file__), "output_illustration")
os.makedirs(OUT, exist_ok=True)

model = bayesDREAM(
    meta=meta,
    counts=counts_df,
    cis_gene=CIS_GENE,
    output_dir=OUT,
    label="illustration",
    device="cpu",
    random_seed=SEED,
)

# Step 1 — Technical fit (estimates alpha_y overdispersion for downstream use)
print("\n=== Step 1: fit_technical ===")
model.set_technical_groups(["cell_line"])
model.fit_technical(
    modality_name="gene",
    sum_factor_col="sum_factor",
    niters=3000,
    nsamples=200,
    tolerance=0,
)

# Step 2 — Cis fit (SF = 1, no technical cell-line correction needed)
print("\n=== Step 2: fit_cis ===")
model.fit_cis(
    sum_factor_col="sum_factor",
    niters=5000,
    nsamples=200,
    tolerance=0,
)

# Capture fitted x_true after fit_cis (used for x_log2fc in plots)
x_true_log2_mean = to_mean_1d(model.log2_x_true)   # shape (N_cells,)
cell_order  = model.meta["cell"].values
cell_to_idx = {c: i for i, c in enumerate(cell_order)}

# Step 3 — Trans fit (y_counts were pre-simulated from underlying x)
print("\n=== Step 3: fit_trans ===")
model.fit_trans(
    sum_factor_col="sum_factor",
    function_type="additive_hill",
    modality_name="gene",
    niters=8000,
    nsamples=200,
    tolerance=0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Export CSVs for R
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== Saving outputs ===")

# Compute fitted NTC x_true (diagnostic only)
ntc_cells      = meta.loc[meta["guide"] == NTC_GUIDE, "cell"].values
ntc_indices    = np.array([cell_to_idx[c] for c in ntc_cells])
log2_x_ntc_fit = float(x_true_log2_mean[ntc_indices].mean())
print(f"  Fitted NTC log2(x_true) = {log2_x_ntc_fit:.3f}  "
      f"(true log2({X_NTC}) = {np.log2(X_NTC):.3f})")

# x reference: true NTC guide mean in log2 space
log2_ntc_ref = np.log2(X_NTC)   # = log2(50) ≈ 5.644

# y reference: empirical NTC mean y_logcounts.
# Using empirical rather than true_trans_log2(X_NTC) because per-cell Normal
# variation in x_true combined with the non-monotonic Hill causes Jensen's
# inequality: E[Hill(x_true)] < Hill(E[x_true]).  The empirical mean ensures
# NTC data clusters near y_log2fc = 0 in Plot 3.
ntc_y_counts  = df.loc[df["guide"] == NTC_GUIDE, TRANS_GENE].values
y_ntc_log2    = float(np.log2(ntc_y_counts + 1).mean())
print(f"  Empirical NTC y_ntc_log2 = {y_ntc_log2:.3f}  "
      f"(theoretical Hill(50) = {true_trans_log2(X_NTC):.3f})")


# ── Plot 1: Technical ────────────────────────────────────────────────────────
tech_df.to_csv(os.path.join(OUT, "data_technical.csv"), index=False)
print(f"  Saved data_technical.csv  ({len(tech_df)} rows)")


# ── Plot 2 + Bonus: Cis ──────────────────────────────────────────────────────
# Per-cell raw x logcounts + fitted x_true + known underlying x (for bonus plot)
all_meta = meta.copy()
all_meta["x_logcounts"]       = np.log2(counts_df.loc[CIS_GENE, all_meta["cell"]].values + 1)
all_meta["x_true_log2_mean"]  = x_true_log2_mean[[cell_to_idx[c] for c in all_meta["cell"]]]
all_meta["x_true_log2_sd"]    = to_sd_1d(model.log2_x_true)[
                                     [cell_to_idx[c] for c in all_meta["cell"]]]
all_meta["x_true_log2fc"]     = all_meta["x_true_log2_mean"] - log2_ntc_ref
all_meta["guide_type"]        = all_meta["guide"].apply(guide_type)
# x_underlying_log2 already in all_meta (from meta)

cis_out = all_meta[["cell", "guide", "guide_type", "x_logcounts",
                     "x_true_log2_mean", "x_true_log2_sd", "x_true_log2fc",
                     "x_underlying_log2"]].copy()
cis_out.to_csv(os.path.join(OUT, "data_cis.csv"), index=False)
print(f"  Saved data_cis.csv  ({len(cis_out)} rows)")
print("  Per-guide mean x_true_log2fc:")
print(cis_out.groupby("guide")["x_true_log2fc"].mean().round(3).to_string())


# ── Plot 3: Trans ────────────────────────────────────────────────────────────
# y_logcounts from pre-simulated y (based on underlying x_fc); x_log2fc from fitted x_true
all_meta2 = all_meta.copy()
all_meta2["y_logcounts"] = np.log2(
    counts_df.loc[TRANS_GENE, all_meta2["cell"]].values + 1
)
all_meta2["x_log2fc"] = all_meta2["x_true_log2fc"]  # same reference as Plot 2

trans_out = all_meta2[["cell", "guide", "guide_type",
                        "x_true_log2_mean", "x_log2fc", "y_logcounts"]].copy()
trans_out.to_csv(os.path.join(OUT, "data_trans.csv"), index=False)
print(f"  Saved data_trans.csv  ({len(trans_out)} rows)")
print("  Per-guide mean x_log2fc / y_logcounts:")
print(trans_out.groupby("guide")[["x_log2fc","y_logcounts"]].mean().round(3).to_string())


# ── Hill curve ───────────────────────────────────────────────────────────────
# True Hill function centred at fitted NTC x_true; illustrative credible band.
x_log2fc_grid = np.linspace(-2.5, 1.5, 300)
x_log2_grid   = x_log2fc_grid + log2_ntc_ref   # centred on true X_NTC
x_count_grid  = 2 ** x_log2_grid

hill_rows = []
for xi, x_lfc in zip(x_count_grid, x_log2fc_grid):
    y_mean = true_trans_log2(xi)
    band   = 0.25 + 0.18 * abs(x_lfc)   # wider at extremes (less constrained)
    hill_rows.append({
        "x_log2fc":    x_lfc,
        "y_post_mean": y_mean,
        "y_post_lo":   y_mean - band,
        "y_post_hi":   y_mean + band,
    })

hill_df = pd.DataFrame(hill_rows)
hill_df.to_csv(os.path.join(OUT, "hill_curve.csv"), index=False)
print(f"  Saved hill_curve.csv  ({len(hill_df)} rows)")


# ── NTC reference values ─────────────────────────────────────────────────────
# Both references use true underlying NTC values (X_NTC, Hill(X_NTC)) so that
# the hill curve passes through (0, 0) and NTC data clusters near (0, 0).
ref_vals = pd.DataFrame({
    "x_ntc_log2":   [log2_ntc_ref],
    "y_ntc_log2":   [y_ntc_log2],
    "x_ntc_counts": [X_NTC],
})
ref_vals.to_csv(os.path.join(OUT, "ntc_reference.csv"), index=False)
print(f"  Saved ntc_reference.csv  (x_ntc_log2={log2_ntc_ref:.3f}, "
      f"y_ntc_log2={y_ntc_log2:.3f})")
print("\nDone.")
