"""
analyze_cis_variability.py
Run fit_technical + fit_cis 100 times with different seeds and report
per-guide estimated log2FC vs true log2FC.

Usage:
  cd bayesDREAM_forClaude
  python examples/illustration/analyze_cis_variability.py
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
import torch
import pyro
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from bayesDREAM import bayesDREAM

# ── Simulation parameters (must match simulate_and_fit.py) ────────────────
CIS_GENE   = "GeneX"
TRANS_GENE = "GeneY"
OTHER_GENES = ["GeneZ1", "GeneZ2", "GeneZ3"]
ALL_GENES   = [CIS_GENE, TRANS_GENE] + OTHER_GENES

N_NTC_CIS        = 20
N_CELLS_PER_GUIDE = 40
NTC_CELL_LINES   = ["ref", "grp_B"]

GUIDE_DEFS = [
    ("KO1", CIS_GENE, 0.25),
    ("KO2", CIS_GENE, 0.40),
    ("KO3", CIS_GENE, 0.98),   # near-null: log2FC ≈ -0.03
    ("CA1", CIS_GENE, 1.50),
    ("CA2", CIS_GENE, 2.00),
]
NTC_GUIDE = "NTC_1"
X_NTC     = 50.0
SIGMA_X_CELL = 0.20
NB_DISP_X    = 0.10
NB_DISP_OTHER = 0.30

TRUE_LOG2FC = {g: np.log2(fc) for g, _, fc in GUIDE_DEFS}
TRUE_LOG2FC[NTC_GUIDE] = 0.0

N_SEEDS = 100
OUT_DIR = os.path.join(os.path.dirname(__file__), "output_illustration")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────
def sample_nb(rng_np, mu, disp, size=1):
    from scipy.stats import nbinom
    r = 1.0 / disp
    p = r / (r + mu)
    return nbinom.rvs(r, p, size=size, random_state=rng_np).astype(np.float32)

def to_mean_1d(t):
    arr = t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    return arr.mean(axis=0) if arr.ndim == 2 else arr

def simulate_cells(rng):
    rows = []
    for i in range(N_NTC_CIS):
        cl = NTC_CELL_LINES[i % len(NTC_CELL_LINES)]
        log2_x = rng.normal(np.log2(X_NTC), SIGMA_X_CELL)
        x_c = int(sample_nb(None, 2.0**log2_x, NB_DISP_X)[0])
        rows.append({
            "cell": f"NTC_{i:03d}", "guide": NTC_GUIDE, "target": "ntc",
            "cell_line": cl, "sum_factor": 1.0,
            "x_underlying_log2": log2_x,
            CIS_GENE: x_c,
            TRANS_GENE: 1,
            **{g: int(sample_nb(None, 100.0, NB_DISP_OTHER)[0]) for g in OTHER_GENES},
        })
    for guide, target, fc in GUIDE_DEFS:
        for i in range(N_CELLS_PER_GUIDE):
            cl = NTC_CELL_LINES[i % len(NTC_CELL_LINES)]
            log2_x = rng.normal(np.log2(X_NTC * fc), SIGMA_X_CELL)
            x_c = int(sample_nb(None, 2.0**log2_x, NB_DISP_X)[0])
            rows.append({
                "cell": f"{guide}_{i:03d}", "guide": guide, "target": target,
                "cell_line": cl, "sum_factor": 1.0,
                "x_underlying_log2": log2_x,
                CIS_GENE: x_c,
                TRANS_GENE: 1,
                **{g: int(sample_nb(None, 100.0, NB_DISP_OTHER)[0]) for g in OTHER_GENES},
            })
    df = pd.DataFrame(rows)
    meta = df[["cell", "guide", "target", "cell_line", "sum_factor"]].copy()
    x_underlying = df[["cell", "guide", "x_underlying_log2"]].copy()
    counts_df = df.set_index("cell")[ALL_GENES].T
    counts_df = counts_df[meta["cell"].values]
    return meta, counts_df, x_underlying


# ── Main loop ─────────────────────────────────────────────────────────────
records = []

for seed_i in range(N_SEEDS):
    seed = seed_i * 17 + 42
    rng  = np.random.default_rng(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    np.random.seed(seed)

    meta, counts_df, x_underlying = simulate_cells(rng)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bayesDREAM(
            meta=meta, counts=counts_df, cis_gene=CIS_GENE,
            output_dir=OUT_DIR, label=f"var_{seed_i}",
            device="cpu", random_seed=seed,
        )
        model.set_technical_groups(["cell_line"])
        model.fit_technical(
            modality_name="gene", sum_factor_col="sum_factor",
            niters=3000, nsamples=100, tolerance=1e-4,
        )
        model.fit_cis(
            sum_factor_col="sum_factor",
            niters=20000, nsamples=100, tolerance=0,
        )

    x_true_log2_mean = to_mean_1d(model.log2_x_true)
    cell_to_idx = {c: i for i, c in enumerate(model.meta["cell"].values)}
    log2_ntc_ref = np.log2(X_NTC)

    for guide in [NTC_GUIDE] + [g for g, _, _ in GUIDE_DEFS]:
        guide_cells   = meta.loc[meta["guide"] == guide, "cell"].values
        guide_indices = [cell_to_idx[c] for c in guide_cells]
        xt_underlying = x_underlying.loc[x_underlying["guide"] == guide, "x_underlying_log2"].values
        xt_est        = x_true_log2_mean[guide_indices]
        est = float(xt_est.mean()) - log2_ntc_ref
        r_p = float(pearsonr(xt_underlying, xt_est)[0])
        r_s = float(spearmanr(xt_underlying, xt_est)[0])
        records.append({
            "seed":       seed_i,
            "guide":      guide,
            "true_fc":    TRUE_LOG2FC[guide],
            "est_fc":     est,
            "bias":       est - TRUE_LOG2FC[guide],
            "r_pearson":  r_p,
            "r_spearman": r_s,
        })

    pyro.clear_param_store()
    if (seed_i + 1) % 10 == 0:
        print(f"  Completed {seed_i + 1}/{N_SEEDS} seeds")

# ── Summary ───────────────────────────────────────────────────────────────
df = pd.DataFrame(records)

print("\n=== Per-guide summary across 100 seeds ===")
print(f"{'Guide':<8} {'True FC':>8} {'Mean est':>9} {'Bias':>7} {'SD':>6} {'|bias|>0.3':>11}")
for guide in [NTC_GUIDE] + [g for g, _, _ in GUIDE_DEFS]:
    sub = df[df["guide"] == guide]
    true_fc  = sub["true_fc"].iloc[0]
    mean_est = sub["est_fc"].mean()
    bias     = sub["bias"].mean()
    sd       = sub["bias"].std()
    frac_large = (sub["bias"].abs() > 0.3).mean()
    print(f"{guide:<8} {true_fc:>8.3f} {mean_est:>9.3f} {bias:>7.3f} {sd:>6.3f} {frac_large:>10.0%}")

# Save full results
out_path = os.path.join(OUT_DIR, "cis_variability_100seeds.csv")
df.to_csv(out_path, index=False)
print(f"\nFull results saved to {out_path}")
