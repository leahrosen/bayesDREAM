# %% [markdown]
# # bayesDREAM vignette: fit_trans (single_hill) for a CRISPRi screen
#
# Builds on `vignette_ntc_fit_crispri.py` / `vignette_cis_fit_crispri.py`.
# Same memory-lean, per-gene-file philosophy as the `fit_cis` vignette,
# extended to the **third stage** — `fit_trans()`:
#
# - Still the minimal file per gene: NTC-cells-only counts + that gene's own
#   guide-cell counts, concatenated — **plus** an extra trim, as early as
#   possible, dropping trans genes below `log2(mu_ntc) < -4` (the lab's
#   standing convention) *before* the model is even constructed.
# - CPU memory profiling (same fresh-subprocess-per-gene pattern as the
#   `fit_cis` vignette).
# - **New**: wall-clock time estimation from a short per-iteration timing
#   probe, extrapolated to production `niters` — feeds directly into the
#   Dardel array job's `--time`.
# - A Dardel array job, reusing the repo's real `publication_runs/common/
#   run_trans.py` + `sbatch_blocks.py` infrastructure.
# - `function_type='single_hill'` throughout (no `additive_hill` warmup
#   phase — see below).
# - Manual runs, with `save_trans_summary()` **plus a hand-added column**
#   (median + 95% CI of the fitted Hill curve at `x_log2FC = -1`, i.e. 50%
#   cis knockdown), for the 7 named genes: `GFI1B`, `NFE2`, `MYB`, `TET2`,
#   `IKZF1`, `HHEX`, `RUNX1`.
# - Example plots for those 7 genes: `plot_xy_data`, proportion
#   positive/negative/not-dependent (vs. `y_ntc` and `full_log2FC`),
#   EC50 (log2FC) vs. Hill coefficient `n`, and observed vs. full log2FC
#   ("how much of the dose-response curve did we actually see?").
#
# Run top to bottom as a script, or cell-by-cell (the `# %%` markers).

# %%
import os
import sys
import math
import time
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pyro
import matplotlib.pyplot as plt
import yaml

deviceno = 0
device = torch.device(f'cuda:{deviceno}' if torch.cuda.is_available() else 'cpu')

from bayesDREAM import bayesDREAM
from bayesDREAM.plotting.xy_plots import plot_xy_data

# %% [markdown]
# ## Reproducibility

# %%
SEED = 20260817
rng = np.random.default_rng(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
pyro.set_rng_seed(SEED)


def find_repo_root(start: str = ".") -> str:
    """Walk upward from `start` until a directory containing `publication_runs/`
    is found — robust to whatever the notebook's/script's cwd happens to be."""
    p = os.path.abspath(start)
    while not os.path.isdir(os.path.join(p, "publication_runs")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("Could not find repo root (no publication_runs/ found above cwd)")
        p = parent
    return p


REPO_ROOT = find_repo_root()
OUTDIR = os.path.join(REPO_ROOT, "examples", "output_vignette_trans")
DATADIR = os.path.join(OUTDIR, "data")
os.makedirs(DATADIR, exist_ok=True)
print(f"REPO_ROOT = {REPO_ROOT}")
print(f"OUTDIR    = {OUTDIR}")

# %% [markdown]
# ## Simulate the cohort, including real trans dose-response
#
# Same shape as the `fit_cis` vignette (12 batches, memory-lean per-gene
# files) but two additions needed to make `fit_trans()` interesting:
#
# 1. A **continuous per-cell** knock-down level (`u_cell`, log2FC of the
#    targeted gene vs. its own NTC mean) rather than a single guide-level
#    constant — cell-to-cell noise around each guide's mean potency.
# 2. Among the 300 background ("trans") genes, **30 genes with a real
#    single-Hill dependence** on `u_cell` (15 responding in the "positive"
#    direction — Hill exponent `n > 0` — and 15 "negative", `n < 0`), so the
#    summary plots later have real signal instead of pure noise. The other
#    270 background genes stay flat (no dependence at all) — some
#    deliberately very lowly expressed, to give the `log2(mu_ntc) < -4`
#    trans-gene filter something real to do.
#
# The 7 candidate cis genes are all comfortably expressed (no `force=True`
# needed here — that was the `fit_cis` vignette's lesson; this one assumes
# you already know it).

# %%
N_BATCHES = 12
N_NTC_PER_BATCH = 25
N_BACKGROUND_GENES = 300
N_RESPONSIVE_POS = 15
N_RESPONSIVE_NEG = 15
N_GUIDES_PER_TARGET = 3
N_CELLS_PER_GUIDE = 20
SIGMA_CELL = 0.30  # cell-to-cell noise (log2FC units) around each guide's mean potency

BATCHES = [f"batch{i + 1}" for i in range(N_BATCHES)]
batch_effect = {b: float(np.exp(rng.normal(0, 0.25))) for b in BATCHES}

NAMED_GENES = ["GFI1B", "NFE2", "MYB", "TET2", "IKZF1", "HHEX", "RUNX1"]
CANDIDATE_GENES = NAMED_GENES  # no force=True demo needed here -- see fit_cis vignette

CANDIDATE_MU = {
    "GFI1B": 3.00, "NFE2": 2.20, "MYB": 1.80, "TET2": 1.50,
    "IKZF1": 1.30, "HHEX": 1.20, "RUNX1": 1.10,
}

background_genes = [f"Gene{i:04d}" for i in range(N_BACKGROUND_GENES)]
responsive_pos_genes = [f"RespPos{i:02d}" for i in range(N_RESPONSIVE_POS)]
responsive_neg_genes = [f"RespNeg{i:02d}" for i in range(N_RESPONSIVE_NEG)]
responsive_genes = responsive_pos_genes + responsive_neg_genes

ALL_GENES = CANDIDATE_GENES + background_genes + responsive_genes
gene_index = {g: i for i, g in enumerate(ALL_GENES)}

# Dedicated RNG stream for gene-level ground truth, decoupled from `rng`
# (used for cell/guide assignment) -- see fit_cis vignette for why.
rng_bg = np.random.default_rng(SEED + 1)
mu_background = np.exp(rng_bg.normal(loc=0.3, scale=1.4, size=N_BACKGROUND_GENES))  # wide range, some < -4
mu_responsive = np.exp(rng_bg.normal(loc=1.0, scale=0.5, size=len(responsive_genes)))  # comfortably expressed

mu_gene = dict(zip(ALL_GENES,
                    [CANDIDATE_MU[g] for g in CANDIDATE_GENES]
                    + list(mu_background) + list(mu_responsive)))
o_y_gene = {g: float(rng_bg.uniform(0.15, 0.5)) for g in ALL_GENES}

# Ground-truth single-Hill shape per responsive gene, in log2FC(x) space:
# y_log2fc(u) = Vmax_log2 * Hill(2**u; K=2**K_log2, n)  -- same formula fit_trans()
# itself uses (bayesDREAM/utils.py's Hill_based_positive_logK), so the fitted
# n_a/K_a_log2fc/Vmax_a should recover these numbers (up to fit noise).
hill_truth = {}
for g in responsive_pos_genes:
    hill_truth[g] = dict(n=float(rng_bg.uniform(1.5, 4.0)),
                          K_log2=float(rng_bg.uniform(-2.0, -0.5)),
                          Vmax_log2=float(rng_bg.uniform(1.0, 2.5)))
for g in responsive_neg_genes:
    hill_truth[g] = dict(n=float(rng_bg.uniform(-4.0, -1.5)),
                          K_log2=float(rng_bg.uniform(-2.0, -0.5)),
                          Vmax_log2=float(rng_bg.uniform(1.0, 2.5)))

# --- NTC cells ---
ntc_rows = [
    {"cell": f"ntc_{b}_{i}", "guide": f"sgNTC_{(i % 3) + 1}", "target": "ntc", "batch": b}
    for b in BATCHES for i in range(N_NTC_PER_BATCH)
]

# --- guide cells, with a continuous per-cell knock-down level u_cell ---
guide_rows = []
cell_u = {}
for gene in CANDIDATE_GENES:
    mean_fc = float(rng.uniform(0.20, 0.40))  # residual fraction of NTC mean
    mean_log2fc = float(np.log2(mean_fc))
    for g_idx in range(N_GUIDES_PER_TARGET):
        guide_name = f"sg{gene}_{g_idx + 1}"
        guide_mean_log2fc = mean_log2fc + float(rng.normal(0, 0.20))
        for i in range(N_CELLS_PER_GUIDE):
            b = rng.choice(BATCHES)
            cell_id = f"{guide_name}_{i}"
            cell_u[cell_id] = float(guide_mean_log2fc + rng.normal(0, SIGMA_CELL))
            guide_rows.append({"cell": cell_id, "guide": guide_name, "target": gene, "batch": b})

for row in ntc_rows:
    cell_u[row["cell"]] = float(rng.normal(0, SIGMA_CELL))

meta_full = pd.DataFrame(ntc_rows + guide_rows)
meta_full["sum_factor"] = rng.uniform(0.7, 1.3, size=len(meta_full))
u_vec = np.array([cell_u[c] for c in meta_full["cell"]])

# --- simulate counts ---
mu_vec = np.array([mu_gene[g] for g in ALL_GENES])
o_vec = np.array([o_y_gene[g] for g in ALL_GENES])
batch_vec = meta_full["batch"].map(batch_effect).values.astype(float)
sf_vec = meta_full["sum_factor"].values.astype(float)
mean_mat = np.outer(mu_vec, batch_vec * sf_vec)  # [genes, cells]

# cis genes: u_cell only applies to cells targeting that specific gene
for gene in CANDIDATE_GENES:
    gi = gene_index[gene]
    cell_mask = (meta_full["target"].values == gene)
    mean_mat[gi, cell_mask] *= 2.0 ** u_vec[cell_mask]

# responsive trans genes: respond to u_cell regardless of which gene it came
# from (simplification -- these 30 genes react the same way to knock-down of
# any of the 7 TFs; real biology would differ per cis gene, but this keeps
# the toy dataset small while giving every one of the 7 per-gene fits real,
# non-degenerate dose-response signal to recover)
for gene in responsive_genes:
    gi = gene_index[gene]
    t = hill_truth[gene]
    x_for_hill = 2.0 ** u_vec
    K_for_hill = 2.0 ** t["K_log2"]
    x_n = x_for_hill ** abs(t["n"])
    K_n = K_for_hill ** abs(t["n"])
    frac = (K_n / (K_n + x_n)) if t["n"] < 0 else (x_n / (K_n + x_n))
    mean_mat[gi, :] *= 2.0 ** (t["Vmax_log2"] * frac)

# background (null) genes: mean_mat left untouched -- no u dependence

phi_vec = 1.0 / (o_vec ** 2)
phi_mat = np.tile(phi_vec[:, None], (1, mean_mat.shape[1]))
p_mat = phi_mat / (phi_mat + mean_mat)
counts_full = pd.DataFrame(
    rng.negative_binomial(phi_mat, p_mat), index=ALL_GENES, columns=meta_full["cell"].values
)
gene_meta = pd.DataFrame({"gene": ALL_GENES, "gene_name": ALL_GENES})

print(f"meta_full: {meta_full.shape}, counts_full: {counts_full.shape} (in-memory only, for simulation)")
print(f"{len(responsive_genes)} genes with real single-Hill ground truth "
      f"({len(responsive_pos_genes)} positive, {len(responsive_neg_genes)} negative direction)")

# %% [markdown]
# ### Write the memory-lean files to disk (same layout as `fit_cis`)

# %%
meta_full.to_csv(os.path.join(DATADIR, "meta_full.csv"), index=False)

ntc_cells = meta_full.loc[meta_full["target"] == "ntc", "cell"].tolist()
counts_full.loc[:, ntc_cells].to_csv(os.path.join(DATADIR, "ntc_counts.csv"))

for gene in CANDIDATE_GENES:
    gene_cells = meta_full.loc[meta_full["target"] == gene, "cell"].tolist()
    counts_full.loc[:, gene_cells].to_csv(os.path.join(DATADIR, f"counts_{gene}.csv"))

gene_meta.to_csv(os.path.join(DATADIR, "gene_meta.csv"), index=False)

TARGETING_MODE = "guide_target"  # see fit_cis vignette for the target_column alternative
guide_target_table = meta_full[["guide", "target"]].drop_duplicates().reset_index(drop=True)
guide_target_table.to_csv(os.path.join(DATADIR, "guide_target_table.csv"), index=False)

del counts_full
print("Files written:")
for f in sorted(os.listdir(DATADIR)):
    print(f"  {f}")


def load_gene_model_inputs(gene: str, keep_genes=None):
    """NTC + this gene's own cells, all genes as rows -- unless `keep_genes`
    is given, in which case rows are trimmed to `keep_genes | {gene}` right
    after reading, before anything else touches the DataFrame. Returns
    (meta, counts, feature_meta) -- feature_meta is row-aligned to counts,
    which matters once `keep_genes` has trimmed rows: passing the full,
    untrimmed `gene_meta` to `bayesDREAM(...)` alongside trimmed `counts`
    raises ("gene_meta has N rows but counts has M rows")."""
    ntc_counts = pd.read_csv(os.path.join(DATADIR, "ntc_counts.csv"), index_col=0)
    gene_counts = pd.read_csv(os.path.join(DATADIR, f"counts_{gene}.csv"), index_col=0)
    counts = pd.concat([ntc_counts, gene_counts], axis=1)

    if keep_genes is not None:
        rows = [g for g in counts.index if g in keep_genes or g == gene]
        counts = counts.loc[rows]

    feature_meta = gene_meta.set_index("gene").loc[counts.index].reset_index()

    keep_cells = set(counts.columns)
    meta = meta_full[meta_full["cell"].isin(keep_cells)].copy()
    del meta["target"]  # derived from guide_target_table instead (TARGETING_MODE)
    return meta, counts, feature_meta

# %% [markdown]
# ## Shared `fit_ntc()` — recap from the technical-fit vignette
#
# Reruns quickly here so this notebook is self-contained; in practice you'd
# `load_ntc_fit()` the output already saved by the technical-fit vignette.

# %%
ntc_counts = pd.read_csv(os.path.join(DATADIR, "ntc_counts.csv"), index_col=0)
meta_ntc = meta_full[meta_full["cell"].isin(ntc_counts.columns)].copy()

NTC_SHARED_LABEL = "vignette_ntc_shared"
ntc_model = bayesDREAM(
    meta=meta_ntc, counts=ntc_counts, feature_meta=gene_meta,
    output_dir=OUTDIR, label=NTC_SHARED_LABEL, device=str(device),
)
ntc_model.set_technical_groups(["batch"])

NITERS_NTC = 3000
ntc_fit_path = os.path.join(OUTDIR, NTC_SHARED_LABEL, "posterior_samples_ntc_gene.pt")
if os.path.exists(ntc_fit_path):
    print("[INFO] Loading existing shared ntc fit...")
    ntc_model.load_ntc_fit(os.path.dirname(os.path.realpath(ntc_fit_path)))
else:
    print("[INFO] Running shared ntc fit...")
    ntc_model.fit_ntc(tolerance=0, niters=NITERS_NTC)
    ntc_model.save_ntc_fit()

NTC_SHARED_DIR = os.path.join(OUTDIR, NTC_SHARED_LABEL)

# %% [markdown]
# ## Subset trans genes to `log2(mu_ntc) >= -4` — as early as possible
#
# This is the lab's standing convention for every dataset (see
# `publication_runs/README.md`'s "exclude_trans_genes" note): genes too
# lowly expressed in NTC cells to estimate overdispersion reliably are
# dropped before `fit_trans()` ever sees them. We determine the keep-list
# **once**, against the shared `fit_ntc()` posterior — the same `mu_ntc`
# read pattern used throughout this series of vignettes — *before* reading
# any per-gene counts file's full row set. `load_gene_model_inputs()` above
# then trims rows to this list immediately after `pd.read_csv`, so the
# excluded genes' Torch tensors (SVI parameters, gradients, posterior
# samples) never get built at all — not just filtered out of the summary
# afterwards.
#
# (`model.exclude_trans_genes(min_log2_mu_ntc=...)` is the library's own
# method for this same filter, applied to an already-constructed model's
# modality — we still call it below too, once per gene, as a defensive
# second pass; with the pre-filtered load it should find nothing new to do.)

# %%
MIN_LOG2_MU_NTC_TRANS = -4.0

gene_mod = ntc_model.get_modality("gene")
mu_ntc_all = gene_mod.posterior_samples_ntc["mu_ntc"]
if isinstance(mu_ntc_all, torch.Tensor):
    mu_ntc_all = mu_ntc_all.mean(dim=0).detach().cpu().numpy().flatten()
else:
    mu_ntc_all = np.asarray(mu_ntc_all).mean(axis=0).flatten()
log2_mu_by_gene = dict(zip(gene_mod.feature_names, np.log2(mu_ntc_all)))

trans_candidates = background_genes + responsive_genes
kept_trans_genes = {g for g in trans_candidates if log2_mu_by_gene.get(g, -np.inf) >= MIN_LOG2_MU_NTC_TRANS}
print(f"{len(kept_trans_genes)}/{len(trans_candidates)} trans genes kept at "
      f"log2(mu_ntc) >= {MIN_LOG2_MU_NTC_TRANS} "
      f"({len(trans_candidates) - len(kept_trans_genes)} excluded)")
print(f"Responsive genes retained: "
      f"{sum(g in kept_trans_genes for g in responsive_genes)}/{len(responsive_genes)}")

# %% [markdown]
# ## Recap: `fit_cis()` per gene (deferred workflow, see previous vignette)
#
# Compressed here (no `force=True` demo, no memory/Dardel walkthrough — see
# `vignette_cis_fit_crispri.py` for that) purely so this notebook has real
# `x_true`/`posterior_samples_cis` to load from for the `fit_trans()` section
# below, which is this vignette's actual subject. One label per gene
# (`vignette_<gene>`), reused for **both** the cis and trans stages — matches
# the real per-dataset scripts (see `publication_runs/domingo/
# generate_slurm.py`), and is what lets `load_cis_fit()` find the right
# files with no explicit `input_dir=` later.

# %%
def fit_one_cis_gene(gene: str, niters: int):
    meta, counts, feature_meta = load_gene_model_inputs(gene, keep_genes=kept_trans_genes)
    model = bayesDREAM(
        meta=meta, counts=counts, feature_meta=feature_meta,
        output_dir=OUTDIR, label=f"vignette_{gene}", device=str(device),
        guide_target=guide_target_table,
    )
    model.set_technical_groups(["batch"])
    # lean=True: nothing here reads a per-draw CI out of posterior_samples_ntc
    # (mu_ntc is only ever averaged, via add_cis_gene()'s internal extraction
    # and this vignette's own x_ntc lookups) -- see fit_cis vignette for the
    # full reasoning. Cuts the dominant cost of posterior_samples_ntc_gene.pt
    # (full [samples, groups, features] tensors) to medians + CI siblings.
    model.load_ntc_fit(input_dir=NTC_SHARED_DIR, mask_features=True, lean=True)
    model.add_cis_gene(gene)
    model.adjust_ntc_sum_factor(covariates=["batch"])
    model.fit_cis(sum_factor_col="sum_factor_adj", tolerance=0, niters=niters)
    model.save_cis_fit()
    return model


NITERS_CIS = 1500
cis_models = {}
for gene in CANDIDATE_GENES:
    cis_models[gene] = fit_one_cis_gene(gene, niters=NITERS_CIS)
    print(f"[fit_one_cis_gene] {gene}: done (niters={NITERS_CIS})")

# %% [markdown]
# ## `fit_trans()`, one gene at a time
#
# `cis_gene` stays **deferred** here (omitted at construction, committed via
# `add_cis_gene()` after `load_ntc_fit()`) — the same pattern as the cis-fit
# recap above, and for the same reason: `load_ntc_fit()` only ever restores a
# modality's `posterior_samples_ntc` from a `posterior_samples_ntc_<name>.pt`
# file that actually exists on disk. The shared fit never had a `'cis'`
# modality (it ran on the whole trans panel, transcriptome-wide), so no
# `posterior_samples_ntc_cis.pt` exists to load — only `add_cis_gene()`
# knows how to pull *this* gene's slice out of the *already-loaded* `'gene'`
# modality's full-panel posterior into a fresh `'cis'` modality
# (`_extract_cis_alpha_from_ntc_posteriors`, same as the cis-fit stage).
# Setting `cis_gene` eagerly at construction instead skips that extraction
# entirely — `cis_mod.posterior_samples_ntc` stays `None`, and every
# downstream `x_ntc`-derived quantity (`K_a_log2fc`, `full_log2fc`,
# `observed_log2fc`, and this vignette's own manual Hill-at-`x_log2FC`
# column) silently comes out `NaN`. Found by hitting exactly that with an
# eager first draft of this function — worth knowing if you ever reach for
# eager `cis_gene` on a model that's meant to reuse a *shared* `fit_ntc()`.
#
# Sequence: init (deferred) -> `set_technical_groups()` ->
# `load_ntc_fit(mask_features=True)` -> `add_cis_gene()` -> `load_cis_fit()`
# (reads the `x_true`/`posterior_samples_cis` saved above — **not** re-fitting
# cis) -> `adjust_ntc_sum_factor()` + `refit_sumfactor()` (neither survives a
# save/load round trip, so both are recomputed fresh here, in that order —
# same as `publication_runs/common/config_utils.py`'s
# `apply_sum_factor_adjustments` docstring explains) -> defensive
# `exclude_trans_genes()` pass -> `fit_trans(function_type='single_hill')`.
#
# **Why `single_hill` has no warm-up phase to account for**: `fit_trans()`'s
# `warmup` curriculum (a cheaper single-Hill phase before the real fit) only
# applies to `additive_hill`/`nested_hill` — for `function_type='single_hill'`
# `_do_warmup` is always `False`, so `niters` **is** the total step count,
# with no extra phase-1 steps to add on top (unlike `additive_hill`, where
# total steps = `niters` + a computed `warmup_steps`; see
# `docs/SIMULATION_STUDY_PLAN.md` §7b for a real benchmark of that case).
#
# **Caveat for the Dardel array job further below**: it calls the repo's
# real `publication_runs/common/run_trans.py`, which only supports *eager*
# `cis_gene` (`config_utils.build_model_from_config` passes it straight to
# the constructor — there's no `add_cis_gene()` call anywhere in that
# script). Submitted through that path, the same gap applies: `x_ntc`-derived
# columns will come out `NaN`. That's a real limitation of `run_trans.py`
# combined with this memory-lean, pre-trimmed-panel layout, not something
# this vignette works around for you — flag it upstream if you hit it for a
# real dataset.

# %%
def build_trans_model(gene: str, niters: int, function_type: str = "single_hill", label_suffix: str = ""):
    meta, counts, feature_meta = load_gene_model_inputs(gene, keep_genes=kept_trans_genes)
    label = f"vignette_{gene}{label_suffix}"
    model = bayesDREAM(
        meta=meta, counts=counts, feature_meta=feature_meta,
        output_dir=OUTDIR, label=label, device=str(device),
        guide_target=guide_target_table,
    )
    model.set_technical_groups(["batch"])
    model.load_ntc_fit(input_dir=NTC_SHARED_DIR, mask_features=True, lean=True)
    model.add_cis_gene(gene)
    # load_cis_fit() is deliberately left lean=False (the default): this
    # model is later handed to plot_xy_data(), which may want genuine
    # per-cell/guide uncertainty from posterior_samples_cis -- unlike
    # posterior_samples_ntc above, nothing here has verified that only
    # point estimates are ever read from it downstream.
    model.load_cis_fit(input_dir=os.path.join(OUTDIR, f"vignette_{gene}"))
    model.adjust_ntc_sum_factor(covariates=["batch"])
    model.refit_sumfactor(covariates=["batch"])
    model.exclude_trans_genes(min_log2_mu_ntc=MIN_LOG2_MU_NTC_TRANS)
    return model, label


HILL_LOG2FC_TARGETS = [-1.0]  # x log2FC value(s) to evaluate the fitted Hill curve at


def hill_value_at_log2fc(model, modality_name: str, x_log2fc: float, is_dependent: np.ndarray):
    """Median + 95% CI of the fitted single-Hill y at a given cis-gene log2FC
    (e.g. -1.0 = 50% knock-down vs. NTC), for is_dependent genes only (NaN
    otherwise). Uses the exact same formula as fit_trans()'s own single_hill
    model (bayesDREAM/utils.py's Hill_based_positive_logK) and
    io/summary.py's _hill_value: y = A + alpha * Vmax_a * x^n / (K_a^n + x^n).
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

    # x_ntc: prefer the cis modality's own fit_ntc-derived mu_ntc (requires
    # add_cis_gene() to have extracted it from the shared panel -- see the
    # markdown note above `build_trans_model()`). Falls back to the median
    # *fitted* x_true among this model's own NTC cells if that extraction
    # never happened (e.g. cis_gene was set eagerly instead) -- less precise
    # (a point estimate from the cis fit itself, not the technical fit's own
    # o_x-aware estimate) but keeps this function usable either way.
    cis_mod = model.get_modality("cis")
    ps_ntc = cis_mod.posterior_samples_ntc
    if ps_ntc is not None and "mu_ntc" in ps_ntc:
        mu_ntc_cis = ps_ntc["mu_ntc"]
        x_ntc = float(mu_ntc_cis.mean().item() if isinstance(mu_ntc_cis, torch.Tensor) else np.mean(mu_ntc_cis))
    else:
        x_true = model.x_true
        x_true = x_true.detach().cpu().numpy() if isinstance(x_true, torch.Tensor) else np.asarray(x_true)
        ntc_mask = (model.meta["target"].values == "ntc")
        x_ntc = float(np.median(x_true[ntc_mask]))
    x_target = x_ntc * (2.0 ** x_log2fc)

    eps = 1e-12
    x_n = np.exp(n * np.log(max(x_target, eps)))
    K_n = np.exp(n * np.log(np.clip(K, eps, None)))
    hill = x_n / (K_n + x_n)
    y_samples = A + alpha * Vmax * hill  # [S, T]

    y_median = np.median(y_samples, axis=0)
    y_lower = np.quantile(y_samples, 0.025, axis=0)
    y_upper = np.quantile(y_samples, 0.975, axis=0)

    y_median = np.where(is_dependent, y_median, np.nan)
    y_lower = np.where(is_dependent, y_lower, np.nan)
    y_upper = np.where(is_dependent, y_upper, np.nan)
    return y_median, y_lower, y_upper


def fit_and_summarise_trans(gene: str, niters: int, function_type: str = "single_hill"):
    model, label = build_trans_model(gene, niters=niters, function_type=function_type)
    model.fit_trans(sum_factor_col="sum_factor_refit", function_type=function_type,
                     tolerance=0, niters=niters)
    model.save_trans_fit()

    out_dir = os.path.join(OUTDIR, label)
    df = model.save_trans_summary(output_dir=out_dir, modality_name="gene")

    is_dep = df["is_dependent"].fillna(False).astype(bool).values
    for x_log2fc in HILL_LOG2FC_TARGETS:
        med, lo, hi = hill_value_at_log2fc(model, "gene", x_log2fc, is_dep)
        tag = f"x_log2fc{x_log2fc:+.0f}".replace("+", "p").replace("-", "m")
        df[f"y_at_{tag}_median"] = med
        df[f"y_at_{tag}_lower"] = lo
        df[f"y_at_{tag}_upper"] = hi

    csv_path = os.path.join(out_dir, "trans_feature_summary_gene.csv")
    df.to_csv(csv_path, index=False)
    print(f"[fit_and_summarise_trans] {gene}: {len(df)} trans features, "
          f"{int(is_dep.sum())} dependent, summary + manual column -> {csv_path}")
    return model, df

# %% [markdown]
# ## Measure real memory *and* wall-clock time, convert to Dardel resources
#
# Same fresh-subprocess-per-gene isolation as the `fit_cis` vignette (`ru_maxrss`
# is a whole-process high-water mark, so profiling several genes in one long
# process would report each gene's peak as the *cumulative* peak across all
# of them). This time we also **time** the fit itself: `fit_trans()`'s cost
# is genuinely per-iteration (unlike `fit_ntc()`/`fit_cis()`, whose cost is
# dominated by fixed setup — see `docs/SIMULATION_STUDY_PLAN.md` §7b's real
# Dardel benchmark: `fit_trans` throughput measurably scales with core count,
# 4-10 steps/s across 2-16 cores in that benchmark), so a short probe at a
# small `niters_probe` gives a real steps/s rate you can extrapolate from —
# you don't need to run anywhere near the full `niters` to see it.
#
# `single_hill` has no warm-up phase (see above), so
# `estimated_seconds = niters_production / (steps_measured / seconds_measured)`
# — no extra warmup-step accounting needed (contrast with `additive_hill`,
# where you'd need to add `warmup_steps` too).

# %%
DARDEL_MB_PER_CORE = 888.0
NITERS_PRODUCTION = 100_000  # fit_trans()'s own default niters
TIME_SAFETY_MARGIN = 1.3     # matches the lab convention of "raw estimate + margin"
NITERS_PROBE = 300           # enough steps that fixed setup cost is a small fraction of elapsed

_profile_worker_src = textwrap.dedent(f"""
    import sys, time, resource
    sys.path.insert(0, {REPO_ROOT!r})
    import pandas as pd
    from bayesDREAM import bayesDREAM

    gene, niters = sys.argv[1], int(sys.argv[2])

    DATADIR = {DATADIR!r}
    meta_full = pd.read_csv(f"{{DATADIR}}/meta_full.csv")
    gene_meta = pd.read_csv(f"{{DATADIR}}/gene_meta.csv")
    guide_target_table = pd.read_csv(f"{{DATADIR}}/guide_target_table.csv")
    kept_trans_genes = {sorted(kept_trans_genes)!r}

    ntc_counts = pd.read_csv(f"{{DATADIR}}/ntc_counts.csv", index_col=0)
    gene_counts = pd.read_csv(f"{{DATADIR}}/counts_{{gene}}.csv", index_col=0)
    counts = pd.concat([ntc_counts, gene_counts], axis=1)
    keep_rows = [g for g in counts.index if g in kept_trans_genes or g == gene]
    counts = counts.loc[keep_rows]
    feature_meta = gene_meta.set_index("gene").loc[counts.index].reset_index()
    meta = meta_full[meta_full["cell"].isin(counts.columns)].copy()
    del meta["target"]

    model = bayesDREAM(meta=meta, counts=counts, feature_meta=feature_meta,
                        output_dir={OUTDIR!r}, label=f"vignette_trans_profile_{{gene}}",
                        device="cpu", guide_target=guide_target_table)
    model.set_technical_groups(["batch"])
    # lean=True on both loads here -- this script only ever fits and times,
    # never plots, so there's no downstream need for per-draw posterior
    # samples from either the NTC or the cis fit (contrast with
    # build_trans_model() above, kept lean=False for load_cis_fit() because
    # its output gets handed to plot_xy_data()).
    model.load_ntc_fit(input_dir={NTC_SHARED_DIR!r}, mask_features=True, lean=True)
    model.add_cis_gene(gene)
    model.load_cis_fit(input_dir={OUTDIR!r} + f"/vignette_{{gene}}", lean=True)
    model.adjust_ntc_sum_factor(covariates=["batch"])
    model.refit_sumfactor(covariates=["batch"])
    model.exclude_trans_genes(min_log2_mu_ntc={MIN_LOG2_MU_NTC_TRANS})

    t0 = time.perf_counter()
    model.fit_trans(sum_factor_col="sum_factor_refit", function_type="single_hill",
                     tolerance=0, niters=niters)
    elapsed = time.perf_counter() - t0

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = raw / (1024.0 * 1024.0) if sys.platform == "darwin" else raw / 1024.0
    print(f"PEAK_RSS_MB={{peak_mb:.1f}}")
    print(f"FIT_TRANS_ELAPSED_SEC={{elapsed:.3f}}")
    print(f"FIT_TRANS_NITERS={{niters}}")
""")
profile_worker_path = os.path.join(OUTDIR, "_profile_one_trans_gene.py")
with open(profile_worker_path, "w") as f:
    f.write(_profile_worker_src)

profile_genes = ["GFI1B", "TET2", "RUNX1"]
profile_rows = []
for gene in profile_genes:
    result = subprocess.run(
        [sys.executable, profile_worker_path, gene, str(NITERS_PROBE)],
        capture_output=True, text=True, check=True,
    )
    out = {l.split("=")[0]: l.split("=")[1] for l in result.stdout.splitlines() if "=" in l}
    peak_mb = float(out["PEAK_RSS_MB"])
    elapsed_sec = float(out["FIT_TRANS_ELAPSED_SEC"])
    niters_probe = int(out["FIT_TRANS_NITERS"])

    cores_needed = math.ceil(peak_mb / DARDEL_MB_PER_CORE)
    steps_per_sec = niters_probe / elapsed_sec
    est_hours_raw = NITERS_PRODUCTION / steps_per_sec / 3600.0
    est_hours = est_hours_raw * TIME_SAFETY_MARGIN

    profile_rows.append({
        "gene": gene, "peak_rss_mb": peak_mb, "cores_needed": cores_needed,
        "steps_per_sec": steps_per_sec, "est_hours_raw": est_hours_raw, "est_hours": est_hours,
    })
    print(f"[profile] {gene}: peak RSS {peak_mb:.0f} MB -> {cores_needed} core(s); "
          f"{steps_per_sec:.2f} steps/s -> {est_hours_raw:.2f}h raw, {est_hours:.2f}h with "
          f"{TIME_SAFETY_MARGIN}x margin for niters={NITERS_PRODUCTION}")

profile_df = pd.DataFrame(profile_rows)
TRANS_CORES = max(1, int(profile_df["cores_needed"].max()))
TRANS_TIME_HOURS = float(np.ceil(profile_df["est_hours"].max()))
print(f"\nRequesting {TRANS_CORES} core(s), --time={TRANS_TIME_HOURS:.0f}h per array task "
      f"(max across {len(profile_genes)} profiled genes). Re-profile more genes before a real "
      f"submission if your candidate list is large or expression levels vary a lot — this is a "
      f"quick estimate, not a guarantee.")

# %% [markdown]
# ## Generate the Dardel array job
#
# Reuses `publication_runs/common/run_trans.py` (adjust -> fit_cis-already-
# loaded -> refit -> exclude_trans_genes -> `fit_trans()` -> save, exactly
# `build_trans_model()`/`fit_and_summarise_trans()` above) and
# `sbatch_blocks.py`'s `SbatchArray`, same pattern as the `fit_cis` vignette.
# Unlike the cis stage, `run_trans.py` expects `model.cis_gene` set directly
# in the config's `model:` block (eager, matching `build_trans_model()`
# above) — no `add_cis_gene()`/`ntc_shared_dir:` top-level keys needed here.

# %%
sys.path.insert(0, str(Path(REPO_ROOT) / "publication_runs" / "common" / "slurm"))
from sbatch_blocks import SbatchArray  # noqa: E402

PER_GENE_DIR = os.path.join(DATADIR, "per_gene_trans")
CONFIGS_DIR = os.path.join(OUTDIR, "slurm_dardel", "configs")
LOGS_DIR = os.path.join(OUTDIR, "slurm_dardel", "logs")
os.makedirs(PER_GENE_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

config_paths = []
for gene in CANDIDATE_GENES:
    meta_g, counts_g, _ = load_gene_model_inputs(gene, keep_genes=kept_trans_genes)
    meta_path = os.path.join(PER_GENE_DIR, f"{gene}_meta.csv")
    counts_path = os.path.join(PER_GENE_DIR, f"{gene}_counts.csv")
    meta_g.to_csv(meta_path, index=False)
    counts_g.to_csv(counts_path)

    cfg = {
        "data": {
            "meta": meta_path, "counts": counts_path,
            "guide_target": os.path.join(DATADIR, "guide_target_table.csv"),
        },
        "model": {
            "cis_gene": gene,  # eager -- already committed by this stage
            "output_dir": OUTDIR,
            "label": f"vignette_{gene}",
            "device": "cpu",
        },
        "ntc": {"set_technical_groups": ["batch"]},
        "sum_factor": {
            "adjust_ntc_sum_factor": {"enabled": True, "args": {"covariates": ["batch"]}},
            "refit_sumfactor": {"enabled": True, "args": {"covariates": ["batch"]}},
        },
        "exclude_trans_genes": {"enabled": True, "args": {"min_log2_mu_ntc": MIN_LOG2_MU_NTC_TRANS}},
        "trans": {
            # lean=True on both loads -- run_trans.py never plots, so there's
            # no downstream need for per-draw posterior samples from either
            # load (see build_trans_model()'s docstring note above for why
            # the *interactive* path above keeps load_cis_fit() lean=False
            # instead, ahead of its own plot_xy_data() call).
            "load_ntc": {"args": {"input_dir": NTC_SHARED_DIR, "mask_features": True, "lean": True}},
            "load_cis": {"enabled": True, "args": {"lean": True}},  # default input_dir=output_dir/label -- same label as cis stage
            "fit": {
                "sum_factor_col": "sum_factor_refit",
                "function_type": "single_hill",
                "tolerance": 0,
                "niters": NITERS_PRODUCTION,
            },
            "save": True,
        },
    }
    cfg_path = os.path.join(CONFIGS_DIR, f"{gene}_trans.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    config_paths.append(cfg_path)

configs_list_path = os.path.join(CONFIGS_DIR, "trans_sweep_configs.txt")
with open(configs_list_path, "w") as f:
    f.write("\n".join(config_paths) + "\n")
print(f"Wrote {len(config_paths)} per-gene config(s) + {configs_list_path}")

# %%
REPO_DIR_ON_CLUSTER = "/proj/<project>/users/<you>/bayesDREAM_forClaude"   # <-- fill in
PYTHON_ENV_ON_CLUSTER = "/proj/<project>/users/<you>/envs/bayesdream/bin/python"  # <-- fill in
DARDEL_ACCOUNT = "<dardel-account>"  # <-- fill in

array_commands = [
    f'CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{configs_list_path}")',
    f'"{PYTHON_ENV_ON_CLUSTER}" "{REPO_DIR_ON_CLUSTER}/publication_runs/common/run_trans.py" --config "$CONFIG"',
]
sweep_step = SbatchArray(
    job_name="vignette_trans_sweep",
    account=DARDEL_ACCOUNT,
    log_dir=LOGS_DIR,
    time_hours=TRANS_TIME_HOURS,
    cpus=TRANS_CORES,
    max_index=len(CANDIDATE_GENES) - 1,
    max_concurrent=min(len(CANDIDATE_GENES), 50),
    partition="shared",
    repo_dir=REPO_DIR_ON_CLUSTER,
    commands=array_commands,
    auto_requeue_on_timeout=True,  # fit_trans() has its own checkpoint/resume -- see sbatch_blocks.py docstring
)
sbatch_script_path = os.path.join(OUTDIR, "slurm_dardel", "01_trans_sweep.sh")
with open(sbatch_script_path, "w") as f:
    f.write(sweep_step.render())
os.chmod(sbatch_script_path, 0o755)
print(f"Wrote {sbatch_script_path}:\n")
print(sweep_step.render())

# %% [markdown]
# `auto_requeue_on_timeout=True` is deliberate here (unlike the `fit_cis`
# array, which left it `False`): `fit_trans()` checkpoints every
# `checkpoint_interval` steps (default 10,000) and resumes automatically —
# see `publication_runs/README.md`'s "Restart policy" — so a Dardel timeout
# mid-fit isn't a wasted run the way it would be for `fit_ntc()`/`fit_cis()`.
#
# Fill in the placeholders above, transfer `slurm_dardel/` and
# `data/per_gene_trans/` to Dardel, then the same `sbatch`/`squeue`/`sacct`
# flow as the `fit_cis` vignette.

# %% [markdown]
# ## Manual runs: `GFI1B`, `NFE2`, `MYB`, `TET2`, `IKZF1`, `HHEX`, `RUNX1`
#
# For this vignette's small simulated dataset, fit every named gene directly
# — in real usage this is what the array job above does for you, at the real
# `niters=100,000`.

# %%
NITERS_TRANS = 4000  # vignette default; production default is 100,000 (see NITERS_PRODUCTION above)

trans_models = {}
trans_summaries = {}
for gene in NAMED_GENES:
    model, df = fit_and_summarise_trans(gene, niters=NITERS_TRANS)
    trans_models[gene] = model
    trans_summaries[gene] = df

# %% [markdown]
# ## `plot_xy_data`: raw x-y relationship + fitted Hill curve
#
# `bayesDREAM.plotting.xy_plots.plot_xy_data` plots a trans feature's raw
# counts against `x_true`, with the fitted dose-response curve overlaid —
# requires `x_true` (from `load_cis_fit()`, already set on each model above).
# For `GFI1B`, picking one of the 30 genes with real simulated ground truth
# (rather than just "whichever gene has the lowest `fdr_alpha`" — at this
# vignette's reduced `NITERS_TRANS` the fit hasn't converged nearly as well
# as a real `niters=100,000` run would, so `fdr_alpha` alone is noisy enough
# to sometimes rank a background/null gene above a genuinely-responsive one;
# see the caveat below the proportion plots).

# %%
gfi1b_df = trans_summaries["GFI1B"]
_known_responsive_dependent = (
    gfi1b_df.loc[
        gfi1b_df["feature"].isin(responsive_genes) & gfi1b_df["is_dependent"].fillna(False),
        ["feature", "fdr_alpha"],
    ]
    .sort_values("fdr_alpha")
)
example_feature = (
    _known_responsive_dependent["feature"].iloc[0]
    if len(_known_responsive_dependent) else responsive_genes[0]
)

plot_xy_data(
    trans_models["GFI1B"],
    feature=example_feature,
    modality_name="gene",
    show_correction="both",   # side-by-side uncorrected vs. alpha_y-corrected
    show_hill_function=True,
    log2fc=True,
    color_by=None,  # one line, one colour -- no split by technical group or NTC/targeting
)

# %% [markdown]
# ## Proportion positive / negative / not-dependent
#
# `single_hill` has no `classification` column (that's only computed for
# `additive_hill` — see `bayesDREAM/plotting/diagnostics.py`'s
# `plot_trans_hits_by_gene`, which explicitly requires it and won't work
# here). Building the equivalent 3-way split ourselves is direct: `n_a`'s
# sign gives the direction (positive Hill coefficient = increases with cis
# gene expression; negative = decreases), gated by `is_dependent`.
#
# **Take the "dependent" fraction below with a grain of salt**: only 30 of
# the ~310 trans genes here have real simulated signal (~10%), but at this
# vignette's reduced `NITERS_TRANS` (a small fraction of `fit_trans()`'s own
# `niters=100,000` default) the posterior hasn't converged nearly as tightly
# as a real run would, and `is_dependent`'s FDR gate is more permissive on
# wide, noisy posteriors — so the observed "dependent" proportion below is
# inflated by false positives relative to what a fully-converged fit would
# call. Re-run with a much larger `NITERS_TRANS` (or the real
# `niters=100,000` via the Dardel array job above) before trusting these
# proportions on real data.
#
# **The false positives also skew heavily "negative"** rather than splitting
# evenly between the two directions (visible in the bar chart below: ~30-50%
# "negative" vs only ~5% "positive" per cis gene). Checking against the known
# ground truth confirms this isn't a detection-power problem — true
# `RespPos`/`RespNeg` genes are recovered at comparable rates in both
# directions (13/15 and 15/15 in one run) — the imbalance comes almost
# entirely from background (null) genes being spuriously called "negative"
# far more often than "positive".
#
# **The following explanation for *why* is Claude's guess, offered during
# the same session that built this vignette, and has not been independently
# verified**: this is CRISPRi-only data, so `x_true` never rises above the
# NTC reference (not even at the 99th percentile of cells). That leaves a
# dense, well-populated cluster of knocked-down cells that a spurious
# "negative" Hill fit (ceiling at low x, floor near NTC) can cheaply latch
# onto, while a spurious "positive" fit would need supporting data *above*
# NTC that basically doesn't exist in a knockdown-only design — so noise in
# a null gene has much less to work with in that direction. If this is
# right, the skew should shrink but not fully disappear at production
# `niters=100,000` (FDR calibration lowers the false-positive rate overall,
# but the underlying identifiability asymmetry is a property of the data,
# not just of under-convergence). Take this as a hypothesis to check against
# a real run, not an established fact.

# %%
DIRECTION_COLORS = {"positive": "#0072B2", "negative": "#D55E00", "not dependent": "#999999"}
DIRECTION_ORDER = ["positive", "negative", "not dependent"]


def classify_direction(df: pd.DataFrame) -> pd.Series:
    is_dep = df["is_dependent"].fillna(False).astype(bool)
    direction = pd.Series("not dependent", index=df.index)
    direction[is_dep & (df["n_a_median"] > 0)] = "positive"
    direction[is_dep & (df["n_a_median"] < 0)] = "negative"
    return pd.Categorical(direction, categories=DIRECTION_ORDER, ordered=True)


all_trans = pd.concat(
    [df.assign(cis_gene=gene) for gene, df in trans_summaries.items()], ignore_index=True
)
all_trans["direction"] = classify_direction(all_trans)

# --- overall proportion per cis gene ---
counts = pd.crosstab(all_trans["cis_gene"], all_trans["direction"]).reindex(columns=DIRECTION_ORDER)
proportions = counts.div(counts.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(8, 4.5))
bottom = np.zeros(len(proportions))
x = np.arange(len(proportions))
for d in DIRECTION_ORDER:
    vals = proportions[d].values
    ax.bar(x, vals, bottom=bottom, width=0.7, color=DIRECTION_COLORS[d], alpha=0.85,
           label=d, edgecolor="white", linewidth=1)
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels(proportions.index, rotation=45, ha="right")
ax.set_ylabel("proportion of trans genes")
ax.set_title("Positive / negative / not-dependent, per cis gene")
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.25, 1.0))
fig.tight_layout()
plt.show()

# %% [markdown]
# ### ... vs. `y_ntc` (NTC baseline expression)
#
# Bin trans genes by their own NTC expression level and look at how the
# positive/negative/not-dependent split shifts — e.g. whether dependence is
# harder to detect for lowly-expressed genes (more Poisson noise relative to
# signal) shows up here as a lower "dependent" fraction in the lowest bin.

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
gfi1b_df = trans_summaries["GFI1B"].assign(direction=classify_direction(trans_summaries["GFI1B"]))
gfi1b_df["log2_y_ntc_bin"] = pd.qcut(np.log2(gfi1b_df["y_ntc"].clip(lower=1e-3)), q=5, duplicates="drop")

counts_yntc = pd.crosstab(gfi1b_df["log2_y_ntc_bin"], gfi1b_df["direction"]).reindex(columns=DIRECTION_ORDER)
proportions_yntc = counts_yntc.div(counts_yntc.sum(axis=1), axis=0)

bottom = np.zeros(len(proportions_yntc))
x = np.arange(len(proportions_yntc))
for d in DIRECTION_ORDER:
    vals = proportions_yntc[d].values
    ax.bar(x, vals, bottom=bottom, width=0.7, color=DIRECTION_COLORS[d], alpha=0.85,
           label=d, edgecolor="white", linewidth=1)
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels([str(iv) for iv in proportions_yntc.index], rotation=45, ha="right", fontsize=8)
ax.set_xlabel("log2(y_ntc) quintile")
ax.set_ylabel("proportion of trans genes")
ax.set_title("GFI1B: dependence direction vs. NTC baseline expression")
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.25, 1.0))
fig.tight_layout()
plt.show()

# %% [markdown]
# ### ... vs. `full_log2FC` (effect size)

# %%
fig, ax = plt.subplots(figsize=(7, 5))
for d in ["positive", "negative"]:
    vals = gfi1b_df.loc[gfi1b_df["direction"] == d, "full_log2fc_median"].dropna()
    if len(vals) < 2:
        continue
    ax.hist(vals, bins=20, color=DIRECTION_COLORS[d], alpha=0.5, label=d, edgecolor="none")
ax.axvline(0, color="#666666", lw=1, ls=":")
ax.set_xlabel("full_log2FC (theoretical x-range, 0 to inf)")
ax.set_ylabel("number of trans genes")
ax.set_title("GFI1B: effect size by direction")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## EC50 (log2FC) vs. Hill coefficient `n`
#
# `K_a_log2fc` (`= log2(K_a) - log2(x_ntc)`, i.e. the half-max point
# expressed as a cis-gene log2FC rather than a raw expression level) against
# `n_a_median` (cooperativity/steepness, sign = direction) — one panel per
# named gene, dependent trans genes only.

# %%
fig, axes = plt.subplots(1, len(NAMED_GENES), figsize=(3.2 * len(NAMED_GENES), 3.6), sharey=True)
for ax, gene in zip(axes, NAMED_GENES):
    df = trans_summaries[gene]
    df = df.assign(direction=classify_direction(df))
    for d in ["positive", "negative"]:
        sub = df.loc[df["direction"] == d]
        ax.scatter(sub["K_a_log2fc"], sub["n_a_median"], s=14, alpha=0.6,
                   color=DIRECTION_COLORS[d], label=d, edgecolor="none")
    ax.axhline(0, color="#cccccc", lw=1, zorder=0)
    ax.axvline(-1.0, color="#666666", lw=1, ls="--", zorder=0)  # the same -1 threshold used earlier
    ax.set_title(gene, fontsize=10)
    ax.set_xlabel("EC50, log2FC(x)")
axes[0].set_ylabel("Hill coefficient n")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("EC50 (log2FC space) vs. Hill coefficient, dependent trans genes only", fontsize=11)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Proportion observed: `full_log2FC` vs. `observed_log2FC`
#
# `full_log2fc` is the theoretical dynamic range (x from 0 to infinity);
# `observed_log2fc` is the same curve evaluated only over `[x_obs_min,
# x_obs_max]` — the x-range your guides actually achieved. Points well below
# the diagonal are genes where the fitted curve implies a much bigger range
# than your knock-down potency let you actually see (compare against the
# `fit_cis` vignette's "minimum x_eff_g log2FC per gene" plot — weak guides
# there predict exactly this pattern here).

# %%
fig, axes = plt.subplots(1, len(NAMED_GENES), figsize=(3.2 * len(NAMED_GENES), 3.6), sharex=True, sharey=True)
for ax, gene in zip(axes, NAMED_GENES):
    df = trans_summaries[gene]
    df = df.assign(direction=classify_direction(df))
    dep = df.loc[df["direction"] != "not dependent"]
    lim = float(np.nanmax([dep["full_log2fc_median"].abs().max(), dep["observed_log2fc"].abs().max(), 1.0])) \
        if len(dep) else 1.0
    ax.plot([0, lim], [0, lim], color="#999999", lw=1, ls="--", zorder=0)
    for d in ["positive", "negative"]:
        sub = dep.loc[dep["direction"] == d]
        ax.scatter(sub["full_log2fc_median"].abs(), sub["observed_log2fc"].abs(), s=14, alpha=0.6,
                   color=DIRECTION_COLORS[d], label=d, edgecolor="none")
    ax.set_title(gene, fontsize=10)
    ax.set_xlabel("|full_log2FC|")
axes[0].set_ylabel("|observed_log2FC|")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("How much of the fitted dose-response curve was actually observed?", fontsize=11)
fig.tight_layout()
plt.show()
