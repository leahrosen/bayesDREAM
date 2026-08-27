# %% [markdown]
# # bayesDREAM vignette: fit_cis for a CRISPRi screen, memory-lean per-gene files
#
# Builds on `vignette_ntc_fit_crispri.py` (technical/NTC fit). This vignette
# covers the **second stage** — `fit_cis()` — for a scenario where:
#
# - You already ran `fit_ntc()` **once**, transcriptome-wide, on NTC cells
#   only, and saved it (previous vignette).
# - You do **not** keep one giant genes x all-cells counts matrix around.
#   Instead you have, on disk: one **NTC-cells-only** counts file (all genes)
#   and, **separately, one small counts file per candidate cis gene** (all
#   genes, but only *that gene's own* guide-targeting cells) — so fitting
#   gene G never requires loading any other gene's guide cells into memory.
# - You have the **full cohort metadata** in one place and subset it, per
#   gene, to whichever cells that gene's counts file actually contains.
# - Two equally-valid ways to tell bayesDREAM which guide targets which
#   gene are covered: a `guide_target` DataFrame, or a precomputed `target`
#   column in `meta`.
# - You test on **GFI1B** first, measure real memory usage, then generate a
#   SLURM **array job** for the rest of the candidate genes on **Dardel**.
#
# Run top to bottom as a script, or cell-by-cell (the `# %%` markers).

# %%
import os
import sys
import math
import resource
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pyro
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yaml

deviceno = 0
device = torch.device(f'cuda:{deviceno}' if torch.cuda.is_available() else 'cpu')

from bayesDREAM import bayesDREAM

# %% [markdown]
# ## Reproducibility

# %%
SEED = 20260807
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
OUTDIR = os.path.join(REPO_ROOT, "examples", "output_vignette_cis")
DATADIR = os.path.join(OUTDIR, "data")
os.makedirs(DATADIR, exist_ok=True)
print(f"REPO_ROOT = {REPO_ROOT}")
print(f"OUTDIR    = {OUTDIR}")

# %% [markdown]
# ## Simulate the memory-lean file layout
#
# One realistic cohort: 12 technical batches, 300 untargeted background
# genes, and 9 *candidate* cis genes — the 7 named genes we'll use later for
# the guide-level density plots (`GFI1B`, `NFE2`, `MYB`, `TET2`, `IKZF1`,
# `HHEX`, `RUNX1`), plus 3 filler candidates (`FillerA/B/D`) to make the
# "minimum log2FC per gene" panel more interesting. NTC mean expression is
# hand-set per candidate gene so the panel spans well-expressed ->
# borderline -> too-low-to-attempt.
#
# `HHEX` is deliberately set low enough that, after `fit_ntc()`'s shrinkage
# for sparse/lowly-expressed genes, its *fitted* `log2(mu_ntc)` lands in
# `[-1.5, -1)` — kept, but needs `force=True` (`fit_cis()`'s own built-in
# gate is at -1; our lab convention here is more lenient, at -1.5). `FillerD`
# is set low enough to land clearly below -1.5 — excluded entirely, never
# submitted. Exactly which other candidates land in the force band vs. get
# excluded is a real, data-dependent outcome of the fit below (`FillerA` in
# particular is deliberately borderline too) — the printed table after
# fitting is the actual source of truth, not this comment.
#
# We then simulate the **whole** cohort once (for internal consistency), but
# only ever *write to disk* — and later *read back* — the NTC-only block and
# each gene's own block, exactly mirroring what a real memory-lean layout
# would give you. Nothing downstream reads "some other gene's guide cells"
# out of this in-memory array; that would defeat the point of the exercise.

# %%
N_BATCHES = 12
N_NTC_PER_BATCH = 25
N_BACKGROUND_GENES = 300
N_GUIDES_PER_TARGET = 3
N_CELLS_PER_GUIDE = 20

BATCHES = [f"batch{i + 1}" for i in range(N_BATCHES)]
batch_effect = {b: float(np.exp(rng.normal(0, 0.25))) for b in BATCHES}

NAMED_GENES = ["GFI1B", "NFE2", "MYB", "TET2", "IKZF1", "HHEX", "RUNX1"]
FILLER_GENES = ["FillerA", "FillerB", "FillerD"]
CANDIDATE_GENES = NAMED_GENES + FILLER_GENES

# hand-set NTC mean expression (mu) per candidate gene -- log2(mu) shown alongside
CANDIDATE_MU = {
    "GFI1B":   3.00,   # log2 =  1.58
    "NFE2":    2.00,   # log2 =  1.00
    "MYB":     1.50,   # log2 =  0.58
    "TET2":    1.00,   # log2 =  0.00
    "IKZF1":   1.30,   # log2 =  0.38
    "HHEX":    0.44,   # log2 = -1.18  -- fit_ntc's shrinkage for sparse/lowly-expressed
                       #                  genes pulls the *fitted* log2(mu_ntc) well below
                       #                  this true value at our small NTC-per-batch count;
                       #                  lands in [-1.5, -1) after fitting -- kept, needs force=True
    "RUNX1":   0.90,   # log2 = -0.15
    "FillerA": 0.35,   # log2 = -1.51  -- borderline like HHEX; see printed table for outcome
    "FillerB": 2.20,   # log2 =  1.14
    "FillerD": 0.22,   # log2 = -2.18  -- comfortably below -1.5: excluded
}
for g, mu in CANDIDATE_MU.items():
    print(f"  {g:10s} true log2(mu_ntc) = {np.log2(mu):+.2f}")

background_genes = [f"Gene{i:04d}" for i in range(N_BACKGROUND_GENES)]
ALL_GENES = CANDIDATE_GENES + background_genes
gene_index = {g: i for i, g in enumerate(ALL_GENES)}

# Dedicated RNG stream for background-gene mu/dispersion, decoupled from the
# `rng` used for cell/guide simulation below -- so hand-tuning a candidate
# gene's true mu (to land its *fitted* log2(mu_ntc) in a specific band after
# fit_ntc's shrinkage) doesn't ripple into every other gene's simulated draw.
rng_bg = np.random.default_rng(SEED + 1)
mu_background = np.exp(rng_bg.normal(loc=0.3, scale=1.4, size=N_BACKGROUND_GENES))
mu_gene = dict(zip(ALL_GENES, [CANDIDATE_MU[g] for g in CANDIDATE_GENES] + list(mu_background)))
o_y_gene = {g: float(rng_bg.uniform(0.15, 0.5)) for g in ALL_GENES}

# --- NTC cells ---
ntc_rows = [
    {"cell": f"ntc_{b}_{i}", "guide": f"sgNTC_{(i % 3) + 1}", "target": "ntc", "batch": b}
    for b in BATCHES for i in range(N_NTC_PER_BATCH)
]

# --- per-candidate-gene guide cells, with per-guide (not just per-gene)
#     knockdown potency, so "minimum log2FC across a gene's own guides" is
#     a meaningful, non-degenerate quantity ---
guide_rows = []
guide_fc = {}  # (gene, guide) -> residual fraction of NTC mean
for gene in CANDIDATE_GENES:
    mean_fc = float(rng.uniform(0.20, 0.40))
    for g_idx in range(N_GUIDES_PER_TARGET):
        guide_name = f"sg{gene}_{g_idx + 1}"
        fc = float(np.clip(mean_fc * np.exp(rng.normal(0, 0.20)), 0.05, 0.9))
        guide_fc[(gene, guide_name)] = fc
        for i in range(N_CELLS_PER_GUIDE):
            b = rng.choice(BATCHES)
            guide_rows.append({"cell": f"{guide_name}_{i}", "guide": guide_name, "target": gene, "batch": b})

meta_full = pd.DataFrame(ntc_rows + guide_rows)
meta_full["sum_factor"] = rng.uniform(0.7, 1.3, size=len(meta_full))

# --- simulate counts (vectorised NegBinom draw), full cohort ---
mu_vec = np.array([mu_gene[g] for g in ALL_GENES])
o_vec = np.array([o_y_gene[g] for g in ALL_GENES])
batch_vec = meta_full["batch"].map(batch_effect).values.astype(float)
sf_vec = meta_full["sum_factor"].values.astype(float)
mean_mat = np.outer(mu_vec, batch_vec * sf_vec)

for (gene, guide_name), fc in guide_fc.items():
    gi = gene_index[gene]
    cell_mask = (meta_full["guide"].values == guide_name)
    mean_mat[gi, cell_mask] *= fc

phi_vec = 1.0 / (o_vec ** 2)
phi_mat = np.tile(phi_vec[:, None], (1, mean_mat.shape[1]))
p_mat = phi_mat / (phi_mat + mean_mat)
counts_full = pd.DataFrame(
    rng.negative_binomial(phi_mat, p_mat), index=ALL_GENES, columns=meta_full["cell"].values
)
gene_meta = pd.DataFrame({"gene": ALL_GENES, "gene_name": ALL_GENES})

print(f"\nmeta_full: {meta_full.shape}, counts_full: {counts_full.shape} (in-memory only, for simulation)")

# %% [markdown]
# ### Write the memory-lean files to disk
#
# `meta_full.csv` (the whole cohort's metadata — always cheap, it's just
# strings/small numbers), `ntc_counts.csv` (all genes x NTC cells only —
# this is what `fit_ntc()` alone ever needs), and one `counts_<gene>.csv`
# per candidate gene (all genes x *that gene's own* guide cells only).
# Everything downstream reads these back from disk exactly like a real
# pipeline would — we never again touch `counts_full` after this cell.

# %%
meta_full.to_csv(os.path.join(DATADIR, "meta_full.csv"), index=False)

ntc_cells = meta_full.loc[meta_full["target"] == "ntc", "cell"].tolist()
counts_full.loc[:, ntc_cells].to_csv(os.path.join(DATADIR, "ntc_counts.csv"))

for gene in CANDIDATE_GENES:
    gene_cells = meta_full.loc[meta_full["target"] == gene, "cell"].tolist()
    counts_full.loc[:, gene_cells].to_csv(os.path.join(DATADIR, f"counts_{gene}.csv"))

gene_meta.to_csv(os.path.join(DATADIR, "gene_meta.csv"), index=False)

del counts_full  # gone -- everything below reads the files back from disk
print("Files written:")
for f in sorted(os.listdir(DATADIR)):
    print(f"  {f}")

# %% [markdown]
# ## Two ways to specify guide -> gene targeting
#
# Pick **one** for your dataset (`TARGETING_MODE` below); both give bayesDREAM
# the exact same information.
#
# **Option 1 — `target` column in `meta`.** Simplest if you (or an upstream
# pipeline) already resolved guide -> target per cell, with no ambiguity.
# Our `meta_full['target']` above already IS this — nothing more to do.
#
# **Option 2 — a `guide_target` DataFrame.** Pass `{'guide', 'target'}` rows
# (multiple rows per guide allowed, for guides with more than one plausible
# target, e.g. predicted off-target effects) straight into `bayesDREAM(...)`
# and let it derive `target` per cell for you. `meta` does **not** need a
# `target` column in this case. This is the natural fit when you have a
# gRNA library annotation table (guide ID -> intended target gene) rather
# than a pre-resolved per-cell column — and it's what lets an *ambiguous*
# guide resolve differently depending on which cis gene is currently being
# fit (see CLAUDE.md's "Single-Guide Mode with guide_target").
#
# Since each per-gene file here already contains *only* that gene's own
# guides (plus NTC), there's no ambiguity to resolve in this dataset either
# way — but Option 2 is what you'd reach for if your guide library has
# guides with multiple plausible targets.

# %%
TARGETING_MODE = "guide_target"  # "target_column" or "guide_target"

# The guide -> target mapping table itself (this is your gRNA library
# annotation file in practice) -- built once, reused for every gene.
guide_target_table = (
    meta_full[["guide", "target"]]
    .drop_duplicates()
    .rename(columns={"target": "target"})
    .reset_index(drop=True)
)
print(guide_target_table.head())

# %% [markdown]
# ## Load a gene's small, concatenated model input
#
# `meta_full` is subset to whichever cells this gene's counts file actually
# has (NTC + this gene's own guides); `ntc_counts.csv` and `counts_<gene>.csv`
# are concatenated on columns. Both files carry the FULL gene panel as rows
# (needed so `add_cis_gene()` can align this gene's slot in the shared
# `fit_ntc()` posterior) — only the cell (column) count is small.

# %%
def load_gene_model_inputs(gene: str):
    ntc_counts = pd.read_csv(os.path.join(DATADIR, "ntc_counts.csv"), index_col=0)
    gene_counts = pd.read_csv(os.path.join(DATADIR, f"counts_{gene}.csv"), index_col=0)
    counts = pd.concat([ntc_counts, gene_counts], axis=1)

    keep_cells = set(counts.columns)
    meta = meta_full[meta_full["cell"].isin(keep_cells)].copy()

    if TARGETING_MODE == "target_column":
        pass  # meta['target'] is already correct: 'ntc' or `gene`, nothing ambiguous in this file
    elif TARGETING_MODE == "guide_target":
        del meta["target"]  # let bayesDREAM derive it from guide_target_table instead
    else:
        raise ValueError(TARGETING_MODE)

    return meta, counts


meta_gfi1b, counts_gfi1b = load_gene_model_inputs("GFI1B")
print(f"GFI1B model inputs: meta={meta_gfi1b.shape}, counts={counts_gfi1b.shape} "
      f"(vs. full cohort meta={meta_full.shape})")

# %% [markdown]
# ## Shared `fit_ntc()` — recap from the previous vignette
#
# `fit_ntc()` only ever touches NTC cells (`use_all_cells=False`, the
# default), so it only needs `ntc_counts.csv` — no guide cells at all. This
# reruns it quickly here so this vignette is self-contained; in practice you
# would `load_ntc_fit()` the output already saved by the technical-fit
# vignette instead of refitting.

# %%
ntc_counts = pd.read_csv(os.path.join(DATADIR, "ntc_counts.csv"), index_col=0)
meta_ntc = meta_full[meta_full["cell"].isin(ntc_counts.columns)].copy()

NTC_SHARED_LABEL = "vignette_cis_ntc_shared"
ntc_model = bayesDREAM(
    meta=meta_ntc,
    counts=ntc_counts,
    feature_meta=gene_meta,
    output_dir=OUTDIR,
    label=NTC_SHARED_LABEL,
    device=str(device),
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
# ## Pre-filter candidate cis genes at `log2(mu_ntc) >= -1.5`
#
# `fit_cis()` has its own built-in gate at `-1` (raises unless `force=True`);
# our lab convention here is more lenient, at `-1.5`. We check every
# candidate **once**, against the shared fit_ntc's transcriptome-wide
# posterior, *before* spending any per-gene compute — reusing the same
# `mu_ntc` read pattern as the technical-fit vignette's histogram, just
# targeted at specific genes instead of plotted as a distribution.

# %%
MIN_LOG2_MU_NTC = -1.5
FORCE_THRESHOLD = -1.0  # fit_cis()'s own hardcoded gate

gene_mod = ntc_model.get_modality("gene")
mu_ntc_all = gene_mod.posterior_samples_ntc["mu_ntc"]
if isinstance(mu_ntc_all, torch.Tensor):
    mu_ntc_all = mu_ntc_all.mean(dim=0).detach().cpu().numpy().flatten()
else:
    mu_ntc_all = np.asarray(mu_ntc_all).mean(axis=0).flatten()
log2_mu_by_gene = dict(zip(gene_mod.feature_names, np.log2(mu_ntc_all)))

candidate_table = pd.DataFrame({
    "gene": CANDIDATE_GENES,
    "log2_mu_ntc": [log2_mu_by_gene.get(g, np.nan) for g in CANDIDATE_GENES],
})
candidate_table["keep"] = candidate_table["log2_mu_ntc"] >= MIN_LOG2_MU_NTC
candidate_table["needs_force"] = candidate_table["keep"] & (candidate_table["log2_mu_ntc"] < FORCE_THRESHOLD)
print(candidate_table.to_string(index=False))

genes_to_run = candidate_table.loc[candidate_table["keep"], "gene"].tolist()
force_genes = set(candidate_table.loc[candidate_table["needs_force"], "gene"])
print(f"\n{len(genes_to_run)}/{len(CANDIDATE_GENES)} candidates kept "
      f"(dropped: {sorted(set(CANDIDATE_GENES) - set(genes_to_run))})")
print(f"needs force=True: {sorted(force_genes)}")

# %% [markdown]
# ## One gene, start to finish: `fit_cis()` for GFI1B
#
# The deferred-cis-gene sequence (CLAUDE.md's "Deferred Cis-Gene Workflow"):
# init *without* `cis_gene` -> `set_technical_groups()` -> `load_ntc_fit()`
# (loading the SHARED fit, with `mask_features=True` so a gene/feature that
# happened to get filtered out of *this* small per-gene subset — e.g. by
# zero-count filtering after cell subsetting — doesn't hard-error, just
# falls back to a neutral `alpha_y`; and `lean=True`, since nothing in this
# vignette ever reads a per-draw credible interval out of
# `posterior_samples_ntc` — every use, here and inside the library's own
# `plot_xy_data`, is a plain `.mean(dim=0)` point estimate. `lean=True`
# collapses that dict to medians + `_lower`/`_upper` siblings instead of
# keeping the full `[samples, technical_groups, features]` tensors — the
# dominant cost of `posterior_samples_ntc_gene.pt` at real dataset sizes.
# **Not** used below for `load_cis_fit()` — see the `fit_trans` vignette,
# which loads a saved cis fit and calls `plot_xy_data()` on it; that plot
# may want genuine per-cell/guide uncertainty from `posterior_samples_cis`,
# unlike the NTC case here) -> `add_cis_gene()` -> sum-factor adjustment ->
# `fit_cis()`.
#
# **Caution**: `set_technical_groups()` renumbers batches via
# `groupby(covariates).ngroup()` on *whatever's in this subset's `meta`*.
# That only stays aligned with the shared fit's own numbering because every
# per-gene file's NTC block spans every one of the 12 batches (guaranteed by
# construction here) — if your own NTC file ever failed to cover every batch
# present at the original `fit_ntc()` run, group numbering would silently
# renumber and misalign against the loaded `alpha_x_prefit`/`alpha_y_prefit`.

# %%
def fit_one_cis_gene(gene: str, niters: int, force: bool = False, verbose: bool = True):
    meta, counts = load_gene_model_inputs(gene)

    kwargs = dict(
        meta=meta,
        counts=counts,
        feature_meta=gene_meta,
        output_dir=OUTDIR,
        label=f"vignette_cis_{gene}",
        device=str(device),
    )
    if TARGETING_MODE == "guide_target":
        kwargs["guide_target"] = guide_target_table

    model = bayesDREAM(**kwargs)
    model.set_technical_groups(["batch"])
    model.load_ntc_fit(input_dir=NTC_SHARED_DIR, mask_features=True, lean=True)
    model.add_cis_gene(gene)
    model.adjust_ntc_sum_factor(covariates=["batch"])
    model.fit_cis(sum_factor_col="sum_factor_adj", tolerance=0, niters=niters, force=force)

    model.save_cis_fit()
    model.save_cis_summary()
    if verbose:
        print(f"[fit_one_cis_gene] {gene}: done "
              f"({'force=True, ' if force else ''}niters={niters})")
    return model


NITERS_CIS = 1500
gfi1b_model = fit_one_cis_gene("GFI1B", niters=NITERS_CIS, force=("GFI1B" in force_genes))

# %% [markdown]
# ## Measure real memory, convert to Dardel cores
#
# Dardel's `shared` CPU partition hands out a **fixed 888 MB per core**
# (`DefMemPerCPU=MaxMemPerCPU=888`, confirmed via
# `scontrol show partition shared`) — there's no separate `--mem`; cores
# *are* your memory budget. `cores_needed = ceil(peak_MB / 888)`.
#
# Peak memory is set by tensor **shapes**, not convergence, so a handful of
# `niters` is enough to see the real peak — no need to wait for a real fit.
# This mirrors `publication_runs/common/profile_memory.py` (the real tool —
# use it directly against your own per-gene YAML configs once you have them;
# reproduced inline here so this vignette stays self-contained). Model
# construction alone is a **lower bound**: `fit_cis()` allocates more on top
# (Adam optimiser state, posterior sample draws, gradient buffers) — always
# profile the fit call itself, not just the constructor.
#
# Each gene is profiled in its **own fresh subprocess**, not in a loop inside
# this notebook's process — `ru_maxrss` is the peak *since the process
# started*, so measuring several genes back-to-back in one process would
# report the cumulative peak across all of them (inflating every number
# after the first), not any single gene's own footprint. A real Dardel array
# task is a fresh process per gene, so this is also the fitting comparison.

# %%
import subprocess  # noqa: E402
import textwrap  # noqa: E402

DARDEL_MB_PER_CORE = 888.0

_profile_worker_src = textwrap.dedent(f"""
    import sys, resource
    sys.path.insert(0, {REPO_ROOT!r})
    import pandas as pd
    from bayesDREAM import bayesDREAM

    gene, niters, force = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"

    meta_full = pd.read_csv({os.path.join(DATADIR, "meta_full.csv")!r})
    gene_meta = pd.read_csv({os.path.join(DATADIR, "gene_meta.csv")!r})
    guide_target_table = meta_full[["guide", "target"]].drop_duplicates()

    ntc_counts = pd.read_csv({os.path.join(DATADIR, "ntc_counts.csv")!r}, index_col=0)
    gene_counts = pd.read_csv({DATADIR!r} + f"/counts_{{gene}}.csv", index_col=0)
    counts = pd.concat([ntc_counts, gene_counts], axis=1)
    meta = meta_full[meta_full["cell"].isin(counts.columns)].copy()
    targeting_mode = {TARGETING_MODE!r}
    if targeting_mode == "guide_target":
        del meta["target"]

    kwargs = dict(meta=meta, counts=counts, feature_meta=gene_meta,
                  output_dir={OUTDIR!r}, label=f"vignette_cis_profile_{{gene}}", device="cpu")
    if targeting_mode == "guide_target":
        kwargs["guide_target"] = guide_target_table

    model = bayesDREAM(**kwargs)
    model.set_technical_groups(["batch"])
    model.load_ntc_fit(input_dir={NTC_SHARED_DIR!r}, mask_features=True, lean=True)
    model.add_cis_gene(gene)
    model.adjust_ntc_sum_factor(covariates=["batch"])
    model.fit_cis(sum_factor_col="sum_factor_adj", tolerance=0, niters=niters, force=force)

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = raw / (1024.0 * 1024.0) if sys.platform == "darwin" else raw / 1024.0
    print(f"PEAK_RSS_MB={{peak_mb:.1f}}")
""")
profile_worker_path = os.path.join(OUTDIR, "_profile_one_gene.py")
with open(profile_worker_path, "w") as f:
    f.write(_profile_worker_src)

profile_genes = ["GFI1B", "HHEX", "FillerB"]  # a well-expressed, a force=True, and a filler gene
profile_rows = []
for gene in profile_genes:
    result = subprocess.run(
        [sys.executable, profile_worker_path, gene, "10", "1" if gene in force_genes else "0"],
        capture_output=True, text=True, check=True,
    )
    peak_mb = float([l for l in result.stdout.splitlines() if l.startswith("PEAK_RSS_MB=")][-1].split("=")[1])
    cores_needed = math.ceil(peak_mb / DARDEL_MB_PER_CORE)
    profile_rows.append({"gene": gene, "peak_rss_mb": peak_mb, "cores_needed": cores_needed})
    print(f"[profile] {gene}: peak RSS {peak_mb:.0f} MB -> {cores_needed} core(s) on Dardel `shared`")

profile_df = pd.DataFrame(profile_rows)
CIS_CORES = max(1, int(profile_df["cores_needed"].max()))
print(f"\nRequesting {CIS_CORES} core(s) per array task "
      f"(max across profiled genes, no extra safety margin added here --\n"
      f"add one yourself for real submissions, e.g. +20%, since this profile "
      f"only covers 3 of your {len(genes_to_run)} genes).")

# %% [markdown]
# ## Generate the Dardel array job
#
# Reuses the repo's real Dardel infrastructure rather than a one-off script:
#
# - `publication_runs/common/run_cis_deferred.py` — the per-gene driver
#   (init without `cis_gene` -> `load_ntc_fit(mask_features=True)` ->
#   `add_cis_gene()` -> sum-factor adjustment -> `fit_cis()` -> save; exactly
#   the sequence `fit_one_cis_gene()` above just ran inline). Its
#   `load_ntc_fit()` call is hard-coded (`mask_features=True`, no `lean=`
#   passthrough) — unlike this notebook's own `fit_one_cis_gene()`, which
#   passes `lean=True` (see its docstring above), the array job submitted
#   through this script doesn't get that memory saving; not something this
#   vignette can change without editing shared pipeline code.
# - `publication_runs/common/slurm/sbatch_blocks.py`'s `SbatchArray` — the
#   SLURM header builder (same Dardel conventions: `shared` partition sized
#   by `--cpus-per-task` alone, `%N` throttles concurrency).
# - The array-over-many-genes pattern (one config path per array index, via
#   `sed -n` on `$SLURM_ARRAY_TASK_ID`) mirrors
#   `publication_runs/morris/generate_slurm.py`'s `cis_sweep` block — built
#   for exactly this "fit_cis only, across many genes, share one fit_ntc"
#   scenario.
# - For your **own** dataset's *full* pipeline (ntc -> cis -> compensation ->
#   trans -> permutation/recapitulation), copy `publication_runs/
#   template_dataset/` rather than hand-rolling this — see its README and
#   `publication_runs/README.md`. Read `publication_runs/VERIFICATION.md`
#   before your first real submission.
#
# First, assemble each kept gene's small input files on disk (this is the
# one thing `run_cis_deferred.py` doesn't do for you — it expects
# already-concatenated `data.meta`/`data.counts` paths, same as every other
# `bayesDREAM` config).

# %%
sys.path.insert(0, str(Path(REPO_ROOT) / "publication_runs" / "common" / "slurm"))
from sbatch_blocks import SbatchArray  # noqa: E402

PER_GENE_DIR = os.path.join(DATADIR, "per_gene")
CONFIGS_DIR = os.path.join(OUTDIR, "slurm_dardel", "configs")
LOGS_DIR = os.path.join(OUTDIR, "slurm_dardel", "logs")
os.makedirs(PER_GENE_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

config_paths = []
for gene in genes_to_run:
    meta_g, counts_g = load_gene_model_inputs(gene)
    meta_path = os.path.join(PER_GENE_DIR, f"{gene}_meta.csv")
    counts_path = os.path.join(PER_GENE_DIR, f"{gene}_counts.csv")
    meta_g.to_csv(meta_path, index=False)
    counts_g.to_csv(counts_path)

    cfg = {
        "data": {"meta": meta_path, "counts": counts_path},
        "model": {
            # cis_gene intentionally absent -- deferred, committed via add_cis_gene() below
            "output_dir": OUTDIR,
            "label": f"vignette_cis_{gene}",
            "device": "cpu",
        },
        "cis_gene": gene,
        "ntc_shared_dir": NTC_SHARED_DIR,
        "ntc": {"set_technical_groups": ["batch"]},
        "sum_factor": {
            "adjust_ntc_sum_factor": {"enabled": True, "args": {"covariates": ["batch"]}},
        },
        "cis": {
            "fit": {
                "sum_factor_col": "sum_factor_adj",
                "tolerance": 0,
                "niters": 100_000,
                **({"force": True} if gene in force_genes else {}),
            },
            "save": True,
        },
    }
    if TARGETING_MODE == "guide_target":
        cfg["data"]["guide_target"] = os.path.join(DATADIR, "guide_target_table.csv")

    cfg_path = os.path.join(CONFIGS_DIR, f"{gene}_cis.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    config_paths.append(cfg_path)

if TARGETING_MODE == "guide_target":
    guide_target_table.to_csv(os.path.join(DATADIR, "guide_target_table.csv"), index=False)

configs_list_path = os.path.join(CONFIGS_DIR, "cis_sweep_configs.txt")
with open(configs_list_path, "w") as f:
    f.write("\n".join(config_paths) + "\n")

print(f"Wrote {len(config_paths)} per-gene config(s) + {configs_list_path}")

# %%
REPO_DIR_ON_CLUSTER = "/proj/<project>/users/<you>/bayesDREAM_forClaude"   # <-- fill in
PYTHON_ENV_ON_CLUSTER = "/proj/<project>/users/<you>/envs/bayesdream/bin/python"  # <-- fill in
DARDEL_ACCOUNT = "<dardel-account>"  # <-- fill in, see `sacctmgr show assoc user=$USER format=account`

array_commands = [
    f'CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{configs_list_path}")',
    f'"{PYTHON_ENV_ON_CLUSTER}" "{REPO_DIR_ON_CLUSTER}/publication_runs/common/run_cis_deferred.py" --config "$CONFIG"',
]
sweep_step = SbatchArray(
    job_name="vignette_cis_sweep",
    account=DARDEL_ACCOUNT,
    log_dir=LOGS_DIR,
    time_hours=6.0,
    cpus=CIS_CORES,
    max_index=len(genes_to_run) - 1,
    max_concurrent=min(len(genes_to_run), 50),
    partition="shared",
    repo_dir=REPO_DIR_ON_CLUSTER,
    commands=array_commands,
)
sbatch_script_path = os.path.join(OUTDIR, "slurm_dardel", "01_cis_sweep.sh")
with open(sbatch_script_path, "w") as f:
    f.write(sweep_step.render())
os.chmod(sbatch_script_path, 0o755)

print(f"Wrote {sbatch_script_path}:\n")
print(sweep_step.render())

# %% [markdown]
# Fill in `REPO_DIR_ON_CLUSTER`/`PYTHON_ENV_ON_CLUSTER`/`DARDEL_ACCOUNT` above
# for your own account, `scp`/`rsync` `slurm_dardel/` and `data/per_gene/` to
# Dardel, then:
#
# ```bash
# sacctmgr show assoc user=$USER format=MaxSubmitJobs   # confirm quota headroom first
# sbatch 01_cis_sweep.sh
# squeue -u $USER
# sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS
# ```
#
# `fit_cis()` has no checkpoint support (unlike `fit_trans()`) — a timeout
# means starting that gene's fit completely over; review failures manually
# (`publication_runs/common/slurm/list_job_status.py`) rather than
# auto-resubmitting.

# %% [markdown]
# ## Run all kept genes (standing in for "the array job already finished")
#
# For this vignette's simulated, tiny dataset we just fit every kept gene
# directly, interactively, at the same modest `niters` used for GFI1B above
# — in real usage this is what the array job on Dardel does for you, at the
# real `niters=100,000` from the generated configs.

# %%
cis_models = {"GFI1B": gfi1b_model}
for gene in genes_to_run:
    if gene in cis_models:
        continue
    cis_models[gene] = fit_one_cis_gene(gene, niters=NITERS_CIS, force=(gene in force_genes), verbose=False)
print(f"Fit {len(cis_models)} genes: {sorted(cis_models)}")

# %% [markdown]
# ## Minimum `x_eff_g` log2FC per cis gene
#
# `x_eff_g` is the model's own per-guide effective-expression latent
# (`_model_x`'s `pyro.deterministic("x_eff_g", ...)`), available in
# `model.posterior_samples_cis['x_eff_g']` regardless of mode — distinct
# from `cis_guide_summary.csv`'s `x_true_mean` column, which is a *cell-level
# average* per guide rather than this model latent. For each gene we take
# the posterior mean of `x_eff_g` per guide, convert to log2FC relative to
# that gene's own NTC reference (`mu_ntc` from the shared technical fit —
# the same `x_ntc` convention `save_trans_summary()` uses elsewhere in the
# codebase), and keep the **minimum** across that gene's own targeting
# guides (i.e. its strongest observed knock-down).

# %%
def min_xeffg_log2fc(model, gene: str) -> float:
    cis_ps = model.posterior_samples_cis
    x_eff_g = cis_ps["x_eff_g"]
    if isinstance(x_eff_g, torch.Tensor):
        x_eff_g = x_eff_g.mean(dim=0).detach().cpu().numpy()
    else:
        x_eff_g = np.asarray(x_eff_g).mean(axis=0)

    guide_lookup = model.meta[["guide", "guide_code", "target"]].drop_duplicates().set_index("guide_code")
    cis_mod = model.get_modality("cis")
    mu_ntc_cis = cis_mod.posterior_samples_ntc["mu_ntc"]
    x_ntc = float(mu_ntc_cis.mean().item() if isinstance(mu_ntc_cis, torch.Tensor) else np.mean(mu_ntc_cis))

    log2fc_by_code = np.log2(x_eff_g) - np.log2(x_ntc)
    own_guide_codes = guide_lookup.index[guide_lookup["target"] == gene]
    return float(np.min(log2fc_by_code[own_guide_codes]))


min_log2fc_df = pd.DataFrame({
    "gene": list(cis_models.keys()),
    "min_log2fc": [min_xeffg_log2fc(m, g) for g, m in cis_models.items()],
}).sort_values("min_log2fc").reset_index(drop=True)
print(min_log2fc_df.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(min_log2fc_df))
ax.scatter(x, min_log2fc_df["min_log2fc"], color="#0072B2", s=60, zorder=3)
ax.axhline(-1.0, color="#666666", lw=1.5, ls="--", label="log2FC = -1")
ax.set_xticks(x)
ax.set_xticklabels(min_log2fc_df["gene"], rotation=45, ha="right")
ax.set_ylabel("min(x_eff_g log2FC) across a gene's own guides", fontsize=10)
ax.set_title("Strongest observed knock-down per cis gene", fontsize=11)
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Density plots: x_true in log2FC space, coloured by guide
#
# One row per named gene. Per-cell `x_true` (already a point estimate,
# `model.x_true`) converted to log2FC against that gene's own NTC reference
# (`log2fc = log2(x_true) - log2(x_ntc)`), grouped by guide — **every** guide
# targeting that gene gets its own curve (`guides = sorted(df["guide"].unique())`,
# not just the first/mean guide; each row's label includes the guide count so
# this is visible directly in the figure). The grey "NTC" curve is **gene-
# specific too**: it's that row's own gene's `log2fc` for cells with
# `target == 'ntc'` in *that gene's own model* — i.e. the GFI1B row's NTC
# curve is GFI1B's own expression in NTC cells, not some blend of all 7
# genes' NTC expression. Pooling NTC across genes would be meaningless here:
# `x_ntc` is gene-specific by construction, so "log2FC" means something
# different in every row, and mixing them would combine unrelated
# quantities. Each gene gets one fixed hue (Okabe-Ito, colourblind-safe),
# with its own guides as light-to-dark shades of that hue — a categorical
# family per gene, sequential within it, never a cycled/rainbow assignment.
# Curves are **filled** (`alpha=0.4`) with a thin matching outline on top for
# a crisp edge, not bare outlines.
#
# Each curve is **peak-normalised** (divided by its own max) rather than
# plotted as a true density. A strong, low-variance guide effect (e.g. a
# guide that collapses `x_true` to a narrow range around its knock-down
# level) produces a tall, narrow true-density spike that would otherwise
# visually swamp every wider, noisier curve on the same axes — peak
# normalisation trades that absolute-area information (which you can already
# get from `cis_guide_summary.csv`) for the thing this plot is actually for:
# comparing *where* each guide's/gene's distribution sits and how tight it
# is, at a common visual scale.

# %%
GENE_HUES = {
    "GFI1B": "#0072B2",  # blue
    "NFE2":  "#D55E00",  # vermillion
    "MYB":   "#009E73",  # bluish green
    "TET2":  "#CC79A7",  # reddish purple
    "IKZF1": "#E69F00",  # orange
    "HHEX":  "#56B4E9",  # sky blue
    "RUNX1": "#8B6F00",  # dark gold (Okabe-Ito's yellow, darkened for line visibility on white)
}
NTC_GREY = "#999999"
FILL_ALPHA = 0.4


def gene_shades(base_hex: str, n: int):
    base_rgb = np.array(mcolors.to_rgb(base_hex))
    white = np.array([1.0, 1.0, 1.0])
    fracs = np.linspace(0.70, 0.05, n)  # lightest guide first, darkest (near base hue) last
    return [tuple(white * f + base_rgb * (1 - f)) for f in fracs]


def cell_log2fc_table(model, gene: str) -> pd.DataFrame:
    x_true = model.x_true
    x_true = x_true.detach().cpu().numpy() if isinstance(x_true, torch.Tensor) else np.asarray(x_true)
    cis_mod = model.get_modality("cis")
    mu_ntc_cis = cis_mod.posterior_samples_ntc["mu_ntc"]
    x_ntc = float(mu_ntc_cis.mean().item() if isinstance(mu_ntc_cis, torch.Tensor) else np.mean(mu_ntc_cis))
    return pd.DataFrame({
        "gene": gene,
        "guide": model.meta["guide"].values,
        "target": model.meta["target"].values,
        "log2fc": np.log2(x_true) - np.log2(x_ntc),
    })


from scipy.stats import gaussian_kde


def _peak_normalised(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    y = gaussian_kde(values)(grid)
    peak = y.max()
    return y / peak if peak > 0 else y


def _filled_curve(ax, grid, curve, color, alpha=FILL_ALPHA, lw=1.4, **kwargs):
    ax.fill_between(grid, curve, color=color, alpha=alpha, linewidth=0)
    ax.plot(grid, curve, color=color, lw=lw, **kwargs)


grid = np.linspace(-6, 6, 400)

fig, axes = plt.subplots(
    len(NAMED_GENES), 1, figsize=(9, 1.7 * len(NAMED_GENES)), sharex=True
)

for row, (gene, ax) in enumerate(zip(NAMED_GENES, axes)):
    full_df = cell_log2fc_table(cis_models[gene], gene)
    ntc_vals = full_df.loc[full_df["target"] == "ntc", "log2fc"].values
    df = full_df.loc[full_df["target"] == gene]
    guides = sorted(df["guide"].unique())
    shades = gene_shades(GENE_HUES[gene], len(guides))

    _filled_curve(ax, grid, _peak_normalised(ntc_vals, grid), NTC_GREY, label="NTC" if row == 0 else None)
    for guide, color in zip(guides, shades):
        vals = df.loc[df["guide"] == guide, "log2fc"].values
        if len(vals) < 2 or np.std(vals) == 0:
            continue
        _filled_curve(ax, grid, _peak_normalised(vals, grid), color)

    ax.set_ylabel(f"{gene}\n(n={len(guides)} guides)", rotation=0, ha="right",
                   va="center", fontsize=9, color=GENE_HUES[gene])
    ax.set_yticks([])
    ax.set_ylim(0, 1.15)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

axes[0].legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("x_true log2FC (vs. that gene's own NTC mean)", fontsize=10)
fig.suptitle("Per-cell x_true, log2FC space — one row per gene, shaded by guide", fontsize=11)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Next steps
#
# - `model.check_systematic_shift()` (compensation stage) before trusting
#   `sum_factor_adj` on cells with unusually large shifts — see
#   `publication_runs/common/run_compensation.py` for the reference pattern.
# - `model.refit_sumfactor(covariates=[...])` after `fit_cis()`, to produce
#   `sum_factor_refit` for `fit_trans()` — needs `x_true`, which is why it
#   runs *after* this stage, not alongside `adjust_ntc_sum_factor()`.
# - `fit_trans()` on whichever genes cleared the density/log2FC sanity
#   checks above — see `docs/FIT_TRANS_GUIDE.md` and this repo's
#   `publication_runs/domingo/` / `publication_runs/morris/` for full worked
#   pipelines (ntc -> cis -> compensation -> trans -> permutation ->
#   recapitulation) to copy `publication_runs/template_dataset/` from.
