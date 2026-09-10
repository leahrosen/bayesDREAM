# Adding Replogle to publication_runs/ — strategy & open decisions

Status: **design only, nothing implemented yet.** This document explains what
the existing ad hoc Replogle notebooks do, answers the "how are NTCs
currently subset" question, proposes how to fold Replogle into
`publication_runs/` (like `domingo/`/`morris/`), and lists every place a
choice needs to be made before writing `preprocess.py`/`config.yaml`/
`generate_slurm.py`.

## 1. What currently exists for Replogle

Replogle has **no `publication_runs/` entry today**. It's run as a set of
hand-edited/papermill-parameterized notebooks directly against pre-built
parquet files on Dardel, and `comparative/reconstruct_export_replogle.py`
(used only for cross-dataset plotting comparisons, not fitting) explicitly
documents this:

> "there is no rendered YAML config to replay -- Replogle's pipeline is a
> hand-written papermill notebook (`10_bayesDREAM_fit_trans_<GENE>.ipynb`),
> not the `publication_runs/` config system."

Two notebooks were reviewed (both under `tmp/`):

### `06_bayesDREAM_fit_ntc_combined.ipynb` — the shared `fit_ntc()` step

- Loads `pr_data/for_bayesDREAM/K562_combined/NTC/` — a **pre-built,
  NTC-only** subset (`cell_meta.csv`, `gene_meta.csv`, sparse
  `gene_counts.npz`), 8202 genes × 85711 NTC cells. `meta` already has a
  `sum_factor` column and `batch`/`experiment` columns.
- Builds a **deferred-cis_gene** `bayesDREAM` model (`cis_gene` omitted,
  `label="combined_ntc"` given explicitly) over **all 85711 NTC cells** —
  no per-gene subsetting at this stage at all, by construction (the file it
  reads only ever contained NTC cells).
- `model.set_technical_groups(["batch", "experiment"])` → 315 groups.
- `model.fit_ntc(tolerance=0)` (default `niters` escalates to 100,000 for
  negbinom + AutoNormal), then `model.save_ntc_fit()`.
- `model.adjust_ntc_sum_factor(covariates=["batch"])` (NOT `experiment`).
- A plotting helper (`plot_ntc_log2_mu`) that histograms `log2(mu_ntc)`
  against an "essential gene" panel and the 7 named cis genes, with a
  `threshold=-1.0` reference line (informal EDA, not used downstream).
- The notebook, as saved, is **28 cells and ends here** — it does not
  contain the code that later produces `technical_group_code_mapping.csv`
  or `mu_ntc_per_gene.csv` (see §3 below; both files are read by the trans
  notebook and by `comparative/reconstruct_export_replogle.py`, so they
  exist on Dardel, just not in this notebook's saved cells).

### `10_bayesDREAM_fit_trans_MYB.ipynb` — one gene's `fit_trans()` (papermill template, 7 copies: GFI1B/MYB/NFE2/TET2/IKZF1/HHEX/RUNX1)

This notebook does **not** call `fit_cis()` — it *loads* an already-completed
cis fit from `output/fit_cis/cis_<gene_id>/` via `model.load_cis_fit(...)`.
The actual `fit_cis()` call therefore happened in some other, unsynced
notebook — I could not find it locally (checked all `.ipynb` files under
`Postdoc/Code`; `grep -r independent_mu_sigma` across the whole repo only
hits `bayesDREAM/fitting/cis.py` itself, not any Replogle-specific script).
**I'm inferring the cis-stage input construction from the trans notebook's
own `load_gene_model_inputs()` function**, which is reused verbatim by
`comparative/reconstruct_export_replogle.py` (`_load_gene_model_inputs`) —
i.e. it's treated elsewhere in this codebase as the canonical, still-current
way Replogle builds a per-gene model. See §2 for why I believe it's also
how the cis stage was built, and please correct me if the actual `fit_cis`
notebook did something different.

Steps, in order:

1. Reads `cell_meta_full.parquet` / `gene_meta_full.parquet` (the full,
   not NTC-only, combined panel — includes every cis-gene's guide cells
   plus all NTC cells; `target` column has values `{gene_id, "ntc", ...}`
   for every gene ever screened, not just the 7 chosen ones).
2. **Manually re-attaches `technical_group_code`** by reading
   `combined_ntc`'s own `technical_group_code_mapping.csv` and joining on a
   `f"{experiment}-{batch}"` key — i.e. it does **not** call
   `set_technical_groups()` again (that would renumber groups and desync
   from the shared ntc fit's `alpha_y_prefit`, which is indexed by the
   *original* group numbering). This file must be something
   `save_ntc_fit()`/`save_ntc_summary()` or a manual export cell writes out
   in the (unsynced) full version of `06_...ipynb`.
3. Loads `mu_ntc_per_gene.csv` (also from the ntc fit's output dir) and
   builds `kept_trans_genes = {g : log2(mu_ntc) >= -4.0}` — this is the
   *trans gene panel* filter, applied **before** any per-gene model is even
   built (as an early row-trim on `counts`, not via bayesDREAM's
   `exclude_trans_genes()` method, which does the same numeric thing but
   post-construction).
4. `load_gene_model_inputs(gene_id, ...)`, the function that answers your
   "how are NTCs currently subset" question — see §2.
5. Builds the model **eagerly-deferred**: `bayesDREAM(meta=meta,
   counts=counts, feature_meta=feature_meta, ...)` with `cis_gene` omitted,
   then `model.load_ntc_fit(input_dir=NTC_FIT, mask_features=True,
   lean=True)`, then `model.add_cis_gene(gene_id)`.
6. `model.load_cis_fit(input_dir=CIS_FIT/cis_<gene_id>)` — **not** re-fit.
7. `model.adjust_ntc_sum_factor(covariates=["batch"])`.
8. `model.exclude_trans_genes(min_log2_mu_ntc=-4.0)` — belt-and-suspenders
   with step 3's pre-trim (harmless: everything that survives step 3 also
   survives this).
9. `model.fit_trans(sum_factor_col="sum_factor_adj", function_type=
   "single_hill", tolerance=0, niters=100_000)`, then `save_trans_fit()` +
   `save_trans_summary()`.

**Function type**: `single_hill`, not `additive_hill` (unlike Domingo/
Morris's default). Confirmed from the actual run log in the notebook
(`fit_type: additive_hill=0  single_hill=2317  not_dependent=5877`).

## 2. How NTCs are currently subset (your question)

`load_gene_model_inputs()`, verbatim logic:

```python
meta = cell_meta_full[cell_meta_full["target"].isin([gene_id, "ntc"])].copy()

is_tgt = meta["target"] != "ntc"                    # this gene's own cells
keys = set(meta.loc[is_tgt, "batch"])                # which batches contain them
meta = meta[is_tgt | meta["batch"].isin(keys)]       # keep target cells + NTCs from THOSE batches only
```

In words: start from every cell targeting this gene, or an NTC guide.
Then **throw away NTC cells that come from a `batch` where this gene was
never actually targeted.** Every cis-gene-targeting cell is kept
regardless of batch; only the NTC pool is batch-restricted.

Why this matters for Replogle specifically: `K562_combined` pools together
multiple original Replogle screens ("essential" + presumably others) that
ran across many `batch` values, but any single gene's guides only appear in
a handful of those batches. Comparing a gene's knockdown cells against NTC
cells drawn from batches that never contained that gene's guides would mix
in whatever batch-to-batch technical variation exists beyond what
`sum_factor`/`alpha_y` corrects for — i.e. it's a **within-experiment NTC
matching** convention, analogous in spirit to why Domingo's scran step uses
`clusters=guide_crispr` (not global) and why Morris blocks `quickCluster`
by `lane`. It is **not** the same thing as `set_technical_groups`/`alpha_y`
correction — those correct the mean level per technical group
multiplicatively; batch-matching instead changes *which cells are in the
model at all*.

Two consequences worth flagging:
- **`experiment` is not part of the match** — only `batch`. If a gene's
  cells span multiple `experiment` values that share the same `batch`
  numbering space, this could over- or under-restrict depending on how
  `batch` is actually coded (I haven't seen the raw `batch`/`experiment`
  value semantics, so I can't confirm this is intentional vs. an oversight
  in the original notebook).
- This subsetting is applied identically for the **trans**-stage model
  build shown above. I'm assuming (not confirmed — the fit_cis notebook is
  missing) the **cis**-stage model was built the same way, since (a) it's
  the only per-gene subsetting logic that exists anywhere in this codebase
  for Replogle, and (b) the trans notebook's `add_cis_gene()` call only
  works correctly if the cis fit it loads was computed on a
  cell population consistent with — at minimum, a superset of — the trans
  model's own NTC+cis-gene cells (add_cis_gene's alpha-transfer step doesn't
  care about cell composition, but `load_cis_fit()`'s `x_true` **is**
  indexed per-cell, so if the cis fit had a different NTC population than
  the trans-stage model rebuilds, cells would misalign). **Please confirm**
  this assumption before I build anything around it.

## 3. What `independent_mu_sigma` does (your question, re: fit_cis)

From `bayesDREAM/fitting/cis.py` (`fit_cis(independent_mu_sigma: bool =
False, ...)`): the cis model's per-guide effect `x_eff_g` is drawn as
`log2(x_eff_g) ~ Normal(mu, sigma) * StudentT-noise`, where `mu`/`sigma` are
hyperparameters **shared across every guide** by default. With
`independent_mu_sigma=True`, `mu`/`sigma` are instead fit **separately per
distinct value of `meta['target']`** (e.g. one `mu`/`sigma` for
`target=='ntc'` guides, a separate one for `target==<cis_gene>` guides) —
requires `>= 2` unique `target` values, which is always true here (NTC +
the cis gene). Practically: pooling NTC and on-target guides into one
`mu`/`sigma` assumes their log2-effect distributions are exchangeable
before you've told the model which is which; letting them differ lets the
NTC guides' (near-zero-effect) distribution stop pulling on the cis gene's
(real-knockdown) guides' hyperprior, and vice versa. This is a real,
substantive modeling choice, not a technical/plumbing one — hence wanting
to compare with vs. without.

## 4. Proposed shape for `publication_runs/replogle/`

Mirrors `domingo/`/`morris/`'s low-MOI, single-guide, `add_cis_gene()`-deferred
pattern (Replogle IS single-guide/low-MOI — confirmed via the notebooks'
`guide_code`/single `guide` column and no `guide_assignment` matrix), reusing
`common/subset_per_gene.py` + `common/run_ntc.py` + `common/run_cis_deferred.py`
+ `common/run_trans.py` rather than hand-porting the notebook code, because:

- The manual `technical_group_code_mapping.csv` re-join in §1 step 2 exists
  **only** because the notebook rebuilds each gene's model from
  independently-loaded parquet files rather than deriving it from the SAME
  in-memory model object that ran `fit_ntc()`. `subset_per_gene.py` avoids
  this by construction — it classifies each gene's cells from the shared
  model itself, so `technical_group_code` numbering never needs
  reconciling by hand. Same for the `mu_ntc_per_gene.csv` pre-trim (step 3)
  — `model.exclude_trans_genes(min_log2_mu_ntc=-4.0)` (already used by both
  Domingo and Morris) does the same thing through bayesDREAM's own public
  API instead of a hand-exported CSV.
- **This is a design substitution, not a literal port** — I want to flag it
  explicitly rather than silently swap it in. The *statistical* result
  should be identical (same cells, same technical grouping, same trans-gene
  filter); the *mechanism* is cleaner/less error-prone. If there's a reason
  the notebook did it the hand-rolled way (e.g. some Dardel/memory
  constraint I'm not aware of), let me know and I'll adjust.

Sketch:

```
publication_runs/replogle/
├── config.yaml           # cis gene list, paths, cluster settings, per-stage resources
├── preprocess.py         # NEW — see §5, does not exist today even in ad hoc form
├── generate_slurm.py     # ntc_shared -> subset_per_gene (x2 variants, see §6) -> cis (x3) -> trans (x3)
└── README.md
```

Stage graph (per cis gene, × 7 genes):

```
ntc_shared (fit_ntc, ALL NTC cells, cis_gene deferred, set_technical_groups(["batch","experiment"]))
   │
   ├── subset_per_gene(gene, ntc_batch_match=True)  -> full_bm/, cis_only_bm/
   │     ├── cis(gene, subset=bm, independent_mu_sigma=True)   -> trans(gene, subset=bm, ind_mu_sigma=True)
   │     └── cis(gene, subset=bm, independent_mu_sigma=False)  -> trans(gene, subset=bm, ind_mu_sigma=False)
   │
   └── subset_per_gene(gene, ntc_batch_match=False) -> full_all/, cis_only_all/
         └── cis(gene, subset=all, independent_mu_sigma=True)  -> trans(gene, subset=all, ind_mu_sigma=True)
```

i.e. **2 data subsets** per gene (batch-matched vs. all-NTC), **3 cis fits**
per gene (2 on the batch-matched subset differing only in
`independent_mu_sigma`, 1 on the all-NTC subset with
`independent_mu_sigma=True`), **3 trans fits** per gene (one per cis fit,
each reading that cis fit's own matching data subset). You said you won't
necessarily submit all 3 trans runs together — that's fine, they're
independent `generate_slurm.py` outputs/dependency chains, nothing forces
them to run together.

`independent_mu_sigma=False` on the all-NTC subset was not requested (your
list has only 3 combinations, not the 4th) — confirmed I'm not missing one.

## 5. `preprocess.py` — sum_factor scope (needs your input, see §7)

You said: *"preprocess to run on all chosen cis genes + all NTC cells,
meaning the sum factors are calculated for all of these data at once."*

This does not exist anywhere today, even in the ad hoc notebooks — the
notebooks *read* an already-built `sum_factor` column from
`for_bayesDREAM/K562_combined/`, whose own computation I have not seen (not
in this repo; presumably a separate script on Dardel or in the original
Replogle-processing pipeline, scope unknown — possibly computed across
*every* screened gene genome-wide, not just the 7 chosen ones). Building a
new `preprocess.py` that computes `sum_factor` **once**, restricted to
[NTC cells] ∪ [cells targeting one of the 7 chosen cis genes], mirrors
Domingo/Morris's "scran computed once upstream, shared everywhere" rule
(see `publication_runs/README.md`'s "sum_factor recomputation" section) —
but needs three things only you can specify (§7 Q3/Q4): where the raw
counts/guide metadata actually live on Dardel, and which scran clustering
convention to use (Domingo: `clusters=guide identity, ref.clust='NTC'`;
Morris: `quickCluster` blocked by `lane`; something else).

## 6. New mechanism needed: NTC batch-matching in `subset_per_gene.py`

Today `subset_per_gene.py` calls `add_cis_gene(gene)` and writes whatever
cells that leaves (**all** NTC + that gene's cells, no batch filter) — this
is exactly your option 3 ("using all NTCs"), for free, already. Options 1/2
need the extra row-filter from §2 applied **after** `add_cis_gene()` and
**before** writing `full/`/`cis_only/` to disk. This filter doesn't exist
in `common/` yet. Two ways to add it (§7 Q2):

- **(a) Generic addition to `common/subset_per_gene.py`**: a new config
  flag (e.g. `cis_gene_batch_match_ntc: {enabled: true, batch_col: batch}`)
  that any future dataset could also use. Touches shared code but is purely
  additive/opt-in (default off, so Domingo/Morris's generated configs are
  unaffected).
- **(b) Replogle-local hook**: a small `replogle/batch_match_ntc.py`
  function, invoked from `replogle/generate_slurm.py`'s own subsetting step
  (or a thin Replogle-specific wrapper script around
  `common/subset_per_gene.py`), keeping `common/` untouched. Matches the
  existing "dynamic dataset-module hook" pattern Morris already uses for
  `compensation_exclude_cells.py`.

I lean towards (a) since the concept ("don't compare a gene's cells against
NTCs from batches that never saw that gene") isn't Replogle-specific and
Morris/Domingo could plausibly want it later, but it's your call.

## 7. Decisions (confirmed 2026-09-08)

1. **Cis gene list**: GFI1B, MYB, NFE2, TET2, IKZF1, HHEX, RUNX1 (matches
   `comparative/datasets.py`'s `REPLOGLE_GENE_TO_ID`).
2. **Batch-matching mechanism**: generic opt-in flag added to
   `common/subset_per_gene.py` (§6a) — default off, Domingo/Morris
   unaffected.
3. **§2's assumption** (cis-stage subsetting == trans-stage's
   `load_gene_model_inputs()` batch-matched-NTC logic): **confirmed**.
4. **sum_factor method**: `quickCluster` (Morris-style, unsupervised),
   blocked by `batch` only (not `experiment`) — matches
   `adjust_ntc_sum_factor(covariates=["batch"])`'s own choice in the
   existing notebooks, computed once on [NTC cells] ∪ [cells targeting one
   of the 7 chosen cis genes].
5. **fit_cis/fit_trans hyperparameters**: carry over the notebook's own
   values as the publication defaults — `niters=100_000`, `tolerance=0`,
   `function_type="single_hill"` for trans (not Domingo/Morris's
   `additive_hill` default).

## 8. Raw data location (confirmed 2026-09-08)

`for_bayesDREAM/K562_combined/` itself is `preprocess.py`'s starting point
(read-only, confirmed no write access:
`/cfs/klemming/projects/snic/lappalainen_lab1/users/lisetts/Replogle_data/pr_data/for_bayesDREAM`)
-- reassemble the [NTC] ∪ [7 chosen genes] cell/count union from its
already-split files (`cell_meta_full.parquet`/`gene_meta_full.parquet`/
`NTC_subset/counts_subset.parquet`/`<gene_id>/counts.parquet`) and
recompute `sum_factor` on top, discarding whatever `sum_factor` values are
already in there. All written output goes to Leah's own directory instead:
`/cfs/klemming/projects/snic/lappalainen_lab1/users/Leah/data/Replogle2022/processed_Leah`.

## 9. Implemented

Everything in §4-6 has now been written:

- `common/subset_per_gene.py` — new opt-in `batch_match_ntc` config flag
  (§6a), tested against toydata (both the no-op case and a forced-mismatch
  case that confirms it actually drops the right NTC cells).
- `replogle/preprocess.py` — reassembles [NTC] ∪ [7 genes] from the
  read-only parquet tree, recomputes `sum_factor` via `quickCluster`
  blocked by `batch`, writes to Leah's `processed_Leah/` directory.
- `replogle/config.yaml`, `replogle/generate_slurm.py`, `replogle/README.md`
  — full `ntc_shared` -> (2 data variants) -> (3 fit_cis variants) ->
  (3 fit_trans) pipeline, dry-run verified (`generate_slurm.py --no-tag`
  renders 57 sbatch scripts + a correctly dependency-chained
  `submit_all.sh`, spot-checked several rendered per-gene/per-variant YAML
  configs by hand).

Not yet runnable end-to-end against real data (no access to Dardel from
here) — see `replogle/README.md`'s "Known gaps" section for what's still a
placeholder (cluster account, resource sizing) versus what's a real
modeling assumption worth double-checking (`guide_covariates`, the missing
fit_cis notebook's other hyperparameters).

## 10. Resources + NTC-source correction (2026-09-09)

**Resources** (per user request): `cluster.account` set to
`naiss2026-3-681`. `ntc_shared` and `trans`'s `all_indmu` cis_variant now
run on a single GPU node each (`partition_gpu`/`python_env_gpu`/
`gpu_single_sbatch_lines`), time fixed at 24h; `cis` (all 3 variants) and
`trans`'s `bm_*` variants stay on CPU, also 24h, `cores:` left as an
unprofiled placeholder pending a real `common/profile_memory.py` run
(instructions in `replogle/README.md`'s new "Resources" section — I can't
run this myself without Dardel access). Answered along the way: `fit_cis`
only loads the cis gene's own single-feature `cis_only/` subset for the
actual SVI fit, but `run_cis_deferred.py`'s `add_cis_gene()` setup step
transiently loads the shared `ntc_shared` fit's FULL ~8195-feature
posterior from disk to extract this one gene's alpha — so real peak memory
is likely dominated by that transient load, not the tiny 1-gene fit.

**NTC-source correction — supersedes §1/§2/§8's references to
`NTC_subset/counts_subset.parquet`.** The user clarified (2026-09-09) that
`NTC_subset/counts_subset.parquet` (what `preprocess.py` originally read as
"all NTC cells") is itself ALREADY subsetted to a chosen set of NTC guides
— not the full NTC population, contrary to what I'd assumed in §1/§8. Per
the user's explicit instruction:

1. `preprocess.py` now reads `<indir>/NTC/` instead (the SAME directory
   `tmp/06_bayesDREAM_fit_ntc_combined.ipynb` — literally named
   "fit_ntc_combined" — itself reads: 85711 cells, every NTC guide) as the
   true full-NTC source. This feeds BOTH `meta_ntc.csv`/`gene_counts_ntc.npz`
   (→ `ntc_shared`) and the combined `meta.csv`/`gene_counts.npz` (→ the
   `all_indmu` cis_variant, whose own `subset_per_gene.py` step never
   narrows the NTC pool further). `NTC_subset/` is no longer read at all.
2. Each `<gene_id>/counts.parquet` is no longer trusted to contain only
   that gene's own target cells — its columns are now intersected against
   `cell_meta_full`'s authoritative `target` column, dropping any cell
   (e.g. a stray NTC cell) not labelled as truly targeting that gene,
   before it's combined with the NTC block. This also prevents any
   overlap/duplicate-cell collision between the `NTC/` block and a
   per-gene block.
3. A future, explicit guide-level NTC restriction (a specific list of NTC
   guides — distinct from `bm`'s batch-based restriction) was requested but
   deliberately left **unimplemented** pending that list from the user
   ("please leave this blank for now"). **Since implemented** (2026-09-09,
   same day) once the user supplied the 5 guides — see
   `bm_selected_ntc_guides` in `config.yaml` and `common/subset_per_gene.py`'s
   new `select_ntc_guides` flag (composes with `batch_match_ntc` via AND for
   the NTC portion; both together replicate the original pipeline's actual
   NTC subsetting for the `bm` variant). Verified against toydata: combining
   both flags correctly intersected down from 4281 to 3807 cells (3292
   target cells + only the two selected-guide NTC cells' 515).

Verified via a synthetic fixture (not real data — still no Dardel access):
built a fake `NTC/` + `cell_meta_full`/`gene_meta_full` + two per-gene
`counts.parquet` files, one of which had a stray NTC cell mixed into its
columns. Confirmed `preprocess.py` (a) assembled the combined output with
no duplicate cells, (b) correctly dropped the stray NTC cell from the
per-gene file (logged: "dropped 1/3 cell(s) ... not labelled target=... in
cell_meta_full"), and (c) `meta_ntc.csv`/`gene_counts_ntc.npz` exactly
reproduced the full `NTC/` input, unchanged except for the recomputed
`sum_factor`.

`replogle/config.yaml`/`generate_slurm.py` needed NO changes for this --
`bm`/`all` cis_variants already read from `subset_per_gene.py`'s
`add_cis_gene()` output over whatever NTC population `preprocess.py`'s
combined `meta.csv` contains; fixing that source was sufficient to correct
both variants (and `ntc_shared`) at once.

## 11. Feature identity: `feature_name_col` (2026-09-10)

Prompted by checking recent bayesDREAM commits (`24bc2f5`, "Unify feature
identity resolution") against every dataset in `publication_runs/`, not
something Replogle-specific that came up first here. Full mechanism: see
`domingo/README.md`/`morris/README.md`'s own new "Feature identity"
sections (Morris's needed a REAL fix -- `gene_id` now outranks `gene_name`
in the identifier-column cascade, which would have silently switched
Morris's whole pipeline's identity convention; Domingo's didn't, protected
automatically by a DataFrame-index special case, but got the same explicit
override anyway for consistency).

**Replogle needed the SAME explicit pin**, `model_defaults.feature_name_col:
gene_id`, threaded through `base_cfg["model"]` in `generate_slurm.py` (§4's
propagation pattern) -- Replogle's counts are a sparse `.npz` with no row
labels, same situation as Morris, so without this override every stage
would fall through to `gene_meta.csv`'s column cascade, which (as of the
same commit) already ranks `gene_id` above `gene_name` anyway -- so this
override doesn't change Replogle's *resulting* identity (already gene_id by
design, confirmed §1), but makes correctness independent of cascade
ordering rather than incidental to it.

**This let `preprocess.py` be simplified**: the earlier version deliberately
overwrote `gene_meta_out['gene_name'] = gene_meta_out['gene_id']` (ported
from the reference notebooks' own identical hack) specifically to win
whatever column-priority resolution was in effect, at the cost of throwing
away the real gene symbol entirely. With `feature_name_col='gene_id'` now
pinning identity explicitly (an unconditional override, step 1 of
`resolve_feature_ids`, outranking any column-priority reasoning), that hack
is no longer needed -- `gene_meta.csv`'s `gene_name` column now holds the
REAL symbol, with `gene_id` (Ensembl) as the separate, pinned identity
column.

**A real, separate bug found and fixed along the way** (unrelated to the
bayesDREAM commits, pre-existing in this pipeline's own
`generate_slurm.py`): `base_cfg["data"]` never included `feature_meta` at
all -- every stage except `ntc_shared` (which set it via its own override)
relied on this. Harmless before `feature_name_col` was set (well, not
exactly -- see below), but with `feature_name_col='gene_id'` requiring
`feature_meta` to be non-None (resolve_feature_ids' step 1 raises
otherwise), `01b_subset_<gene>_<variant>.sh`'s own model construction (the
ONE stage that doesn't override `data.feature_meta` via
`subset_data_block()`) would have raised
`"feature_name_col='gene_id' was given but feature_meta was not provided"`
the first time it actually ran against real data -- never caught by the
dry-run config-generation tests already done for this pipeline (those only
render YAML, they don't construct a real model). Actually, this construction
was ALREADY broken even before `feature_name_col` existed: with sparse
`.npz` counts and no `feature_meta`, `bayesDREAM.__init__` has no source at
all for gene identity and raises immediately regardless (confirmed with a
standalone negative-control test: `ValueError: When counts is not a
DataFrame, gene_meta must be provided`) -- so this was a real gap in the
subset step specifically, present since Replogle's `generate_slurm.py` was
first written, simply never exercised locally (no Dardel access to run it
for real). Fixed by adding `feature_meta`/`feature_meta_read_csv_kwargs` to
`base_cfg["data"]` directly, so every stage inherits it.

Verified end-to-end with a synthetic sparse-`.npz` + `gene_meta.csv`
fixture (20 genes, 75 cells, one named `ENSG00000165702`/`GFI1B`) through
`common/subset_per_gene.py` with `feature_name_col: gene_id` set:
`add_cis_gene('ENSG00000165702')` located the gene correctly, and both the
`full/` and `cis_only/` output `gene_meta.csv` files came out gene_id-indexed
with the real symbol preserved in `gene_name` -- confirming both the
`feature_name_col` pin and the `feature_meta` fix work together correctly.

## 12. Small `batch` values break scran's `quickCluster` (2026-09-10)

A real Dardel run of `preprocess.py` hit `RRuntimeError: ... fewer cells
than the minimum cluster size` inside `quickCluster(sce, block=batch)`.
The traceback's `1 remote errors, element index: 27` means exactly ONE
`batch` value in the combined [full NTC + 7 genes] population has fewer
cells than `quickCluster`'s own default `min.size=100` -- the "499
unevaluated" alongside it is `BiocParallel`'s cascade from that one worker
dying, not 499 separate failures.

Considered dropping `batch` from scran's blocking entirely (i.e.
`clusters=NULL`) vs. pooling just the too-small batch(es). Chose pooling:
dropping blocking would change the normalization for every one of the
~300 batches to fix a problem specific to one of them, for no reason (the
other batches all cluster fine on their own).

**Fix**: `_bin_small_batches()` in `preprocess.py` -- any `batch` value
with fewer than `--min-block-size` cells (default 100, matching
`quickCluster`'s own default) gets merged into one
`'_pooled_small_batches'` bucket, used ONLY as scran's blocking variable
(a separate temp column/frame, never written to `meta.csv` or used
anywhere else in the pipeline -- `set_technical_groups`,
`adjust_ntc_sum_factor`, `bm`'s `batch_match_ntc` all still see the real,
unmodified `batch` column). If pooling ALL rare batches together still
doesn't reach `min_size` (i.e. scran's minimum genuinely can't be
satisfied), raises with the offending batch/counts rather than silently
proceeding.

Verified via two synthetic fixtures: (a) one 5-cell rare batch with
nothing to pool it with -- correctly raises
`"pooling ALL of them together still only totals 5 cells (< 100)"`; (b)
three small batches (40+40+30=110 cells) that pool to a valid size --
correctly succeeds, scran called with `{'b1': 152, '_pooled_small_batches':
110}`, and the real `batch` column in the written `meta.csv` came back
with all four original distinct values, unmutated.
