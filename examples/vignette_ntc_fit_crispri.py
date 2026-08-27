# %% [markdown]
# # bayesDREAM vignette: NTC (technical) fit for a CRISPRi screen
#
# This vignette walks through the **first stage** of the bayesDREAM pipeline —
# `fit_ntc()` — on a simulated CRISPRi dataset that mirrors a typical
# real-world setup:
#
# - **CRISPRi only** (single perturbation modality, single guide per cell)
# - Only the **primary ('gene') modality** — no transcript/splicing/ATAC data
# - **NTC (non-targeting control) cells** are present, required for `fit_ntc()`
# - **No guide-level covariates** (`guide_covariates` is left empty)
# - **Many technical batches** — this is the whole point of `fit_ntc()`: it
#   learns a per-batch, per-gene correction (`alpha_y`) from the NTC cells so
#   that batch effects don't get mistaken for perturbation effects downstream
# - `cis_gene` is **not set at initialisation** — we run `fit_ntc()` once,
#   across the whole transcriptome, and only commit to a specific cis gene
#   afterwards via `add_cis_gene()` (not covered in this vignette — see the
#   "Next steps" section at the end).
#
# Run this file top to bottom as a script, or cell-by-cell in VS Code /
# Jupyter (the `# %%` markers define cells).

# %%
import os

import numpy as np
import pandas as pd
import torch
import pyro
import matplotlib.pyplot as plt

deviceno = 0
device = torch.device(f'cuda:{deviceno}' if torch.cuda.is_available() else 'cpu')

from bayesDREAM import bayesDREAM

# %% [markdown]
# ## Reproducibility
#
# Fix every RNG bayesDREAM touches: numpy (data simulation + internal use),
# torch (model parameters), and pyro (SVI sampling).

# %%
SEED = 20260730
rng = np.random.default_rng(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
pyro.set_rng_seed(SEED)

# %% [markdown]
# ## What `meta` needs to contain
#
# `bayesDREAM(meta=..., counts=..., ...)` validates `meta` up front. In
# **single-guide mode** (one guide per cell — the CRISPRi case here, as
# opposed to high-MOI mode where `guide_assignment`/`guide_meta` are passed
# instead), `meta` must have one row per cell and these columns:
#
# | column        | meaning                                                             |
# |---------------|----------------------------------------------------------------------|
# | `cell`        | cell barcode/ID — must match `counts.columns` 1:1                   |
# | `guide`       | guide identifier (any string, unique per guide)                     |
# | `target`      | gene the guide targets; **NTC cells must use the literal string `"ntc"`** (lowercase) — `fit_ntc()`/init raise an error if no `"ntc"` cells are found, unless `require_ntc=False` |
# | `sum_factor`  | per-cell size factor, strictly > 0 (e.g. from `scran::calculateSumFactors`, as in `run_pipeline/prepare_inputs.py`) |
#
# Plus, for every technical covariate you plan to correct for (batch, lane,
# 10x chip, processing date, ...) — one column per covariate, e.g. `batch`.
# These are *not* required by `__init__` itself, but `set_technical_groups()`
# needs them before `fit_ntc()` can run.
#
# We are **not** using `guide_covariates`/`guide_covariates_ntc` here (no
# per-guide covariate columns needed) — CRISPRi with a plain guide -> target
# mapping is the simplest case.
#
# `counts` is a genes x cells `pd.DataFrame` (or ndarray/sparse matrix +
# `feature_meta`) for the **primary modality** (`modality_name='gene'` by
# default). `feature_meta` is optional gene annotation — recommended columns
# `gene`/`gene_name`/`gene_id`; if omitted, minimal metadata is built from
# `counts.index`.

# %% [markdown]
# ## Simulate a CRISPRi dataset
#
# - `N_BACKGROUND_GENES` genes with no guides targeting them ("trans"
#   candidates) plus 3 named genes (`GeneA`, `GeneB`, `GeneC`) that *do* have
#   guides — these stand in for genes you might later commit to as the cis
#   gene via `add_cis_gene()`.
# - `N_BATCHES` technical batches, each with its own multiplicative
#   library-depth-like effect (`batch_effect`) applied to every gene equally
#   — this is exactly the kind of nuisance effect `fit_ntc()`/`alpha_y` is
#   meant to soak up.
# - Every gene's NTC counts are drawn `NegBinom(mu_gene * batch_effect *
#   sum_factor, o_y_gene)`. Guide-targeting cells additionally get a
#   knock-down multiplier applied to *their own target gene only* — this
#   doesn't matter for `fit_ntc()` (which only ever sees NTC cells) but keeps
#   the dataset realistic for the (later) cis/trans steps.
# - `GeneC` is deliberately given very low NTC expression, to illustrate the
#   pre-flight check in the histogram below.

# %%
N_BATCHES = 12
N_NTC_PER_BATCH = 30
N_BACKGROUND_GENES = 300
N_GUIDES_PER_TARGET = 3
N_CELLS_PER_GUIDE = 25

BATCHES = [f"batch{i + 1}" for i in range(N_BATCHES)]
# one multiplicative nuisance factor per batch, shared across all genes
batch_effect = {b: float(np.exp(rng.normal(0, 0.25))) for b in BATCHES}

CANDIDATE_CIS_GENES = ["GeneA", "GeneB", "GeneC"]
background_genes = [f"Gene{i:04d}" for i in range(N_BACKGROUND_GENES)]
ALL_GENES = CANDIDATE_CIS_GENES + background_genes

# NTC mean expression (mu) per gene: candidates are hand-set to span
# well-expressed -> borderline -> too-low-to-use; background genes get a
# realistic wide dynamic range.
mu_candidates = np.array([15.0, 4.0, 0.12])  # GeneA, GeneB, GeneC
mu_background = np.exp(rng.normal(loc=0.3, scale=1.4, size=N_BACKGROUND_GENES))
mu_gene = dict(zip(ALL_GENES, np.concatenate([mu_candidates, mu_background])))

# per-gene negative-binomial overdispersion (o_y in the cell2location-style
# parameterisation fit_ntc() uses: Var = mu + o_y^2 * mu^2)
o_y_gene = {g: float(rng.uniform(0.15, 0.5)) for g in ALL_GENES}

# a curated "gene panel" of interest (e.g. a marker panel a student might
# care about) -- just a random subset here for illustration
GENE_PANEL = list(rng.choice(background_genes, size=20, replace=False))
# genes to highlight individually -- our three cis-gene candidates
HIGHLIGHT_GENES = CANDIDATE_CIS_GENES

# --- NTC cells: N_NTC_PER_BATCH per batch ---
ntc_rows = [
    {"cell": f"ntc_{b}_{i}", "guide": f"sgNTC_{(i % 3) + 1}", "target": "ntc", "batch": b}
    for b in BATCHES
    for i in range(N_NTC_PER_BATCH)
]

# --- guide-targeting cells for the 3 candidate cis genes ---
knockdown_fc = {"GeneA": 0.25, "GeneB": 0.30, "GeneC": 0.35}  # residual fraction of NTC mean
guide_rows = []
for gene in CANDIDATE_CIS_GENES:
    for g_idx in range(N_GUIDES_PER_TARGET):
        guide_name = f"sg{gene}_{g_idx + 1}"
        for i in range(N_CELLS_PER_GUIDE):
            b = rng.choice(BATCHES)
            guide_rows.append({"cell": f"{guide_name}_{i}", "guide": guide_name, "target": gene, "batch": b})

meta = pd.DataFrame(ntc_rows + guide_rows)
meta["sum_factor"] = rng.uniform(0.7, 1.3, size=len(meta))

# --- simulate counts (vectorised NegBinom draw) ---
gene_index = {g: i for i, g in enumerate(ALL_GENES)}
mu_vec = np.array([mu_gene[g] for g in ALL_GENES])
o_vec = np.array([o_y_gene[g] for g in ALL_GENES])

batch_vec = meta["batch"].map(batch_effect).values.astype(float)
sf_vec = meta["sum_factor"].values.astype(float)
mean_mat = np.outer(mu_vec, batch_vec * sf_vec)  # (genes, cells)

for gene, fc in knockdown_fc.items():
    gi = gene_index[gene]
    cell_mask = (meta["target"].values == gene)
    mean_mat[gi, cell_mask] *= fc

phi_vec = 1.0 / (o_vec ** 2)
phi_mat = np.tile(phi_vec[:, None], (1, mean_mat.shape[1]))
p_mat = phi_mat / (phi_mat + mean_mat)
counts_arr = rng.negative_binomial(phi_mat, p_mat)

counts = pd.DataFrame(counts_arr, index=ALL_GENES, columns=meta["cell"].values)
gene_meta = pd.DataFrame({"gene": ALL_GENES, "gene_name": ALL_GENES})

print(f"meta: {meta.shape}, counts: {counts.shape}")
print(meta.head())

# %% [markdown]
# ## The `bayesDREAM` class structure
#
# `bayesDREAM` stores every data type ("modality") in one place:
#
# - `model.modalities` — `dict[str, Modality]`. At init (no `cis_gene`
#   passed) this contains just `{'gene': Modality(...)}`. Once you call
#   `add_cis_gene()` later, a second `'cis'` modality appears, holding the
#   single committed cis gene, and the primary `'gene'` modality loses that
#   one gene (see "Next steps" below).
# - `model.primary_modality` — name of the modality that drives cis/trans
#   modelling; `'gene'` here (the default).
# - `model.get_modality(name)` / `model.list_modalities()` — access/inspect
#   modalities by name.
#
# Each `Modality` object (`bayesDREAM/modality.py`) carries:
#
# - `.counts` — the count matrix itself (features x cells here, since
#   `cells_axis=1` is the default for a `pd.DataFrame` with cells as columns)
# - `.feature_meta` — per-feature annotation `DataFrame` (from our
#   `gene_meta` above, plus any columns bayesDREAM adds internally)
# - `.feature_names` — canonical, order-matched list of feature identifiers
#   (aligned with the feature axis of `.counts` and the rows of
#   `.feature_meta` — always use this instead of re-deriving names from
#   `feature_meta` yourself)
# - `.cell_names` — column names of `.counts` in order
# - `.sum_factors` — cell-indexed `DataFrame` of every `*sum_factor*` column
#   copied from `meta` (auto-populated at init; the `'cis'` modality later
#   shares the *same* `DataFrame` object as the primary modality, so
#   `adjust_ntc_sum_factor()`/`refit_sumfactor()` updates are visible to both)
# - `.distribution` — `'negbinom'` for gene counts
# - `.dims` — `{'n_features': ..., 'n_cells': ...}`
# - `.posterior_samples_ntc` — populated by `fit_ntc()`; a dict of tensors
#   including `'mu_ntc'` (per-gene fitted NTC mean) and `'o_y'`
#   (overdispersion), used below to preview NTC expression
#
# **Primary vs. cis modality**: the primary modality (`'gene'`) holds every
# gene that will be treated as a *trans* readout (an outcome modelled as a
# dose-response function of the cis gene in `fit_trans()`). The `'cis'`
# modality holds exactly one feature — the gene whose expression is the
# *input* ("dose") to that dose-response curve, estimated by `fit_cis()`.
# Until `add_cis_gene()` is called, there is no `'cis'` modality yet, so all
# of `GeneA`/`GeneB`/`GeneC` (our future cis-gene candidates) still live
# together with every background gene inside `model.modalities['gene']`.

# %%
model = bayesDREAM(
    meta=meta,
    counts=counts,
    feature_meta=gene_meta,
    output_dir="./output",
    label="vignette_crispri_ntc",  # required explicitly since cis_gene is not set
    device=str(device),
)

print(model.list_modalities())

# %% [markdown]
# ## Technical groups and the NTC fit
#
# `set_technical_groups()` must run before `fit_ntc()` — it assigns every
# cell a `technical_group_code` from the covariate combination you give it
# (here, just `batch`; pass a list for multiple covariates, e.g.
# `['batch', 'lane']`). Cells in a technical group with *no* NTC
# representation are dropped automatically (with a warning), since
# `alpha_y` for that group couldn't be estimated otherwise.

# %%
model.set_technical_groups(["batch"])

# %% [markdown]
# `NITERS` is a plain variable (not hard-coded inline) so it's easy to bump
# up for a real run — `fit_ntc()`'s own default is 50,000 iterations
# (100,000 for multinomial modalities). We use a smaller number here purely
# so the vignette runs quickly; `tolerance=0` (disabling early stopping on
# ELBO convergence) is intentionally hard-coded, matching the lab's usual
# fitting convention.

# %%
NITERS = 5000

# --- TECHNICAL FIT: Load if exists, otherwise fit and save ---
#ntc_fit_path = os.path.join(model.output_dir, model.label, 'posterior_samples_ntc_gene.pt')
ntc_fit_path = os.path.join('./output/vignette_crispri_ntc/posterior_samples_ntc_gene.pt')
if os.path.exists(ntc_fit_path):
    print("[INFO] Loading existing ntc fit...")
    model.load_ntc_fit(os.path.dirname(os.path.realpath(ntc_fit_path)))
else:
    print("[INFO] Running ntc fit (this may take a while)...")
    model.fit_ntc(tolerance=0, niters=NITERS)
    model.save_ntc_fit()

# %% [markdown]
# ## Adjusting NTC sum factors
#
# `adjust_ntc_sum_factor()` renormalises each guide's sum factors against the
# NTC mean *within* each technical group, so guide-level differences in
# sequencing depth don't get conflated with real perturbation effects before
# `fit_cis()`. Pass the same covariates used for `set_technical_groups()`.

# %%
model.adjust_ntc_sum_factor(covariates=["batch"])

# %% [markdown]
# ## Preview NTC expression *before* committing to a cis gene
#
# `model.plot_ntc_expression()` (the built-in helper) requires a `'cis'`
# modality to already exist, since it reads `mu_ntc` off both the `'cis'`
# and primary modalities. We haven't called `add_cis_gene()` yet, so instead
# we read `mu_ntc` straight off the primary `'gene'` modality's
# `posterior_samples_ntc` — every candidate cis gene is still in there. This
# is useful for eyeballing several candidate genes side by side (and against
# a gene panel of interest) before deciding which one to commit to.
#
# `fit_cis()` will later raise a `ValueError` if the committed cis gene's
# `log2(mu_ntc)` falls below `-1` (counts too sparse in NTC cells for
# reliable overdispersion estimation) — the dotted grey line below shows
# that same cutoff.

# %%
def plot_ntc_log2_mu(model, gene_panel=None, highlight_genes=None,
                      modality_name='gene', n_bins=60, threshold=-1.0,
                      figsize=(8, 5)):
    """Histogram of log2(mu_ntc) for all features in `modality_name`,
    with an optional overlay histogram for `gene_panel` and vertical
    highlight lines for `highlight_genes`. Requires fit_ntc() to have
    already been run on this modality."""
    mod = model.get_modality(modality_name)
    ps = mod.posterior_samples_ntc
    if ps is None or 'mu_ntc' not in ps:
        raise ValueError(f"Run fit_ntc() on modality '{modality_name}' first.")

    mu_ntc = ps['mu_ntc']
    if isinstance(mu_ntc, torch.Tensor):
        mu_ntc = mu_ntc.mean(dim=0).detach().cpu().numpy().flatten()
    else:
        mu_ntc = np.asarray(mu_ntc).mean(axis=0).flatten()

    feature_names = mod.feature_names
    log2_mu = np.log2(mu_ntc)
    name_to_log2 = dict(zip(feature_names, log2_mu))

    fig, ax = plt.subplots(figsize=figsize)

    finite = np.isfinite(log2_mu)
    ax.hist(log2_mu[finite], bins=n_bins, color='steelblue', alpha=0.55,
            edgecolor='none', label=f'all genes (n={int(finite.sum())})')

    if gene_panel:
        panel_vals = np.array([
            name_to_log2[g] for g in gene_panel
            if g in name_to_log2 and np.isfinite(name_to_log2[g])
        ])
        ax.hist(panel_vals, bins=n_bins, color='darkorange', alpha=0.7,
                edgecolor='none', label=f'gene panel (n={len(panel_vals)})')

    colors = plt.cm.tab10.colors
    if highlight_genes:
        for i, g in enumerate(highlight_genes):
            if g not in name_to_log2:
                print(f"[WARN] '{g}' not found in modality '{modality_name}' feature_names — skipping")
                continue
            val = name_to_log2[g]
            ax.axvline(val, color=colors[i % len(colors)], lw=2, ls='--',
                        label=f'{g}  (log2 = {val:.2f})')

    ax.axvline(threshold, color='grey', lw=1.5, ls=':',
               label=f'fit_cis threshold ({threshold})')

    ax.set_xlabel('log2(mu_ntc) — mean NTC expression from fit_ntc()', fontsize=10)
    ax.set_ylabel('number of genes', fontsize=10)
    ax.set_title(f"NTC expression — '{modality_name}' modality (before add_cis_gene)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


fig = plot_ntc_log2_mu(
    model,
    gene_panel=GENE_PANEL,
    highlight_genes=HIGHLIGHT_GENES,
)
plt.show()

# %% [markdown]
# `GeneC` should land left of the grey threshold line — this is exactly the
# check `fit_cis()` performs automatically for whichever gene you commit to
# via `add_cis_gene()`; seeing it here first means you can rule it out (or
# pass `force=True` to `fit_cis()` deliberately) before spending compute on
# a cis fit that will fail its own sanity check.

# %% [markdown]
# ## Next steps (not covered in this vignette)
#
# With `fit_ntc()` done once across the whole transcriptome, you can now fork
# off a separate model per candidate cis gene without refitting NTC each
# time:
#
# ```python
# import copy
# for gene in ["GeneA", "GeneB"]:            # skip GeneC — too lowly expressed
#     m = copy.deepcopy(model)
#     m.add_cis_gene(gene)                    # extracts the 'cis' modality
#     m.fit_cis(sum_factor_col="sum_factor_adj")
#     m.fit_trans(sum_factor_col="sum_factor_adj", function_type="additive_hill")
# ```
#
# See the "Deferred Cis-Gene Workflow (`add_cis_gene`)" section of
# `CLAUDE.md` for the full details of what `add_cis_gene()` does internally.
