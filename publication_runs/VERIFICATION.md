# Verification checklist

How to confirm a dataset's pipeline is actually correct before trusting it
at real scale, and how to run it once you have. Read this before your first
real submission for a dataset, and again after any change to `common/` or a
dataset's `generate_slurm.py`/`config.yaml`.

## 1. Static checks

```bash
cd publication_runs
find . -name "*.py" | xargs python -m py_compile
```

Then generate for real (see step 2) and:

```bash
find <dataset>/slurm -name "*.sh" | xargs -n1 bash -n
for f in <dataset>/slurm/configs/*.yaml; do python -c "import yaml; yaml.safe_load(open('$f'))" || echo "FAILED: $f"; done
```

These catch syntax errors and malformed config rendering, not logic errors —
necessary but not sufficient.

## 2. Dry-run the generator

No data, no cluster, seconds:

```bash
cd publication_runs/<dataset>
python generate_slurm.py --no-tag --outdir /tmp/<dataset>_dry
```

Open a handful of files in `configs/` and check by eye:
- `data.meta`/`data.counts` point where you expect
- `model.exclude_guides` looks right for a couple of different genes (Morris:
  should differ per gene — that gene's own SNPs kept, everyone else's excluded)
- the `sum_factor:` block matches what that dataset actually needs (Domingo:
  `refit_sumfactor` enabled, default `sum_factor_col_old`; Morris:
  `refit_sumfactor` disabled, `compute_scran` enabled)
- `exclude_trans_genes` block present on every trans/permutation/recapitulation config

This catches config-plumbing mistakes but **not** whether the fitting code
actually runs — a rendered YAML can be structurally perfect and still fail
the moment `bayesDREAM` tries to use it. That's what the rest of this
checklist is for.

## 3. Risks specific to this pipeline — check these explicitly

Each of these was flagged during development as unverified or a known
upstream issue. Don't assume any of them are fine just because the config
renders correctly.

- **`rpy2` + R (`scran`/`Matrix`/`SingleCellExperiment`/`S4Vectors`)** must be
  installed in whichever conda env runs Morris's `cis`/`trans`/
  `permutation`/`recapitulation` stages (`bayesdream_cpu`/`bayesdream_rocm`
  on Dardel). Confirmed **not** present in the local `bayesdream` dev env —
  don't assume it's on Dardel either. Check directly:
  ```bash
  <env>/bin/python -c "import rpy2; from rpy2.robjects.packages import importr; importr('scran')"
  ```
- **`fit_cis()` high-MOI dtype bug** (`RuntimeError: expected scalar type
  Float but found Double`, `bayesDREAM/fitting/cis.py`'s `_model_x`) — a
  known, pre-existing upstream issue as of the last check, unrelated to this
  pipeline's config. Run Morris's `02_cis_<gene>.sh` for ONE gene first and
  confirm it doesn't hit this before submitting the full batch — if it does,
  that's a bayesDREAM fix needed, not something to debug in
  `generate_slurm.py`. See `morris/README.md`.
- **Domingo modality directories** — `config_modalities.yaml`'s
  `acceptor_choice`, `donor_efficiency`, `acceptor_efficiency` entries were
  added by inference (symmetry with confirmed working types), not verified
  against real `loader_inputs/<stype>/` directories. Confirm each exists
  with the expected `cell_meta.tsv.gz`/`feature_meta.tsv.gz`/`counts.npz`
  (+`denominator.npz` for binomial) layout before running that modality's
  job, or remove the entry.
- **High-MOI shared-ntc sanity check** — now real library code
  (`add_cis_gene()`), not a workaround, but still worth one cheap check the
  first time: fit one primary gene, sanity-check its `alpha_x_prefit`/cis
  results aren't obviously wrong (e.g. wildly different scale from what
  `check_systematic_shift()` or a rough manual estimate would suggest).

## 4. Size the jobs

Before setting `--cpus-per-task` for real (Dardel's `shared` partition is a
fixed 888MB/core — see main `README.md`'s "Memory" section):

```bash
python common/profile_memory.py --config <a_real_gene_config.yaml> --stage cis --niters 10
```

Do this once per dataset at real (or close-to-real) data scale, not once per
gene — peak memory is set by tensor shapes (cells × features), which don't
vary much gene-to-gene within a dataset.

## 5. Run for real

1. Get the code + data onto Dardel (`git pull` if the repo is already
   cloned there).
2. Morris only: `python morris/preprocess.py --indir <raw> --outdir
   <raw>/preprocessed` once. It has hard alignment assertions and fails
   loudly on mismatch rather than silently producing misaligned data — a
   failure here is the assertion doing its job, not a bug to route around.
3. Regenerate for real (drop `--no-tag` this time, so the batch gets a git
   snapshot tag): `python generate_slurm.py`.
4. Skim `slurm/configs/` once more now that it's rendered against real
   paths (same checks as step 2).
5. `bash slurm/submit_all.sh` — writes `submitted_jobs.tsv`.
6. Monitor:
   ```bash
   python common/slurm/list_job_status.py slurm/submitted_jobs.tsv
   ```
   Only `trans`/`permutation`/`recapitulation` auto-resubmit on timeout.
   Everything else (`ntc`/`cis`/`compensation`/`cis_sweep`) that fails or
   times out needs your manual look and decision — see main `README.md`'s
   "Restart policy".

## 6. After the first genes land — check outputs, not just exit codes

A job exiting 0 doesn't mean the science is right. Spot-check:

- `ntc_feature_summary_<modality>.csv`, `cis_guide_summary.csv`,
  `trans_feature_summary_<modality>.csv` exist with plausible row counts
  (not all-NaN, not suspiciously few features surviving
  `exclude_trans_genes`).
- `compensation_<modality>.csv` — `p_adj` isn't uniformly ~0 or ~1 across
  every row (usually means the sum-factor or `exclude_cells` wiring is off,
  not a real biological signal).
- One permutation replicate's `trans_feature_summary` should look like
  *noise* (few/no significant hits). If it looks like the real fit, the
  permutation or `exclude_trans_genes` step isn't actually being applied —
  check the rendered config for that replicate.
