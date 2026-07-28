# Running the Single-Hill Recovery Study on Berzelius

Operational runbook for executing the simulation study (see
`docs/SIMULATION_STUDY_PLAN.md` for the design/statistics) on the Berzelius cluster.
Seven steps: environment variables → build the design matrix → critical validation
run → generate SLURM scripts → dry run → submit full → monitor.

## Prerequisites

- A conda/mamba environment built from `environment_cuda.yml` (or `_cpu`/`_rocm`),
  which includes `bioconductor-scran` and `r-data.table` alongside torch/pyro —
  `simulate_scenario.py` shells out to `Rscript` for sum-factor recomputation
  (see plan §4.3), so this is a hard requirement, not optional.
- This repository cloned/synced onto Berzelius (`git clone`, or `scp -r`).
- Your Berzelius SLURM account string. If unsure, run `projinfo` — the `/proj/...`
  directory name is not always the same as the account string `sbatch` expects.

Verify both before continuing:

```bash
$PYTHON_ENV --version
$PYTHON_ENV -c "import torch, pyro; print(torch.__version__, pyro.__version__)"
/path/to/your/env/bin/Rscript -e 'library(scran); packageVersion("scran")'
```

## 1. Environment variables

Set these in every shell session before running anything below — a lost variable
(e.g. after opening a new terminal/pane) is the most common source of confusing
failures here (see [Troubleshooting](#troubleshooting)).

```bash
PROJ=/proj/<your-project>/users/<you>
CODE=$PROJ/bayesDREAM_forClaude            # the repo (contains examples/simulation_study)
OUT=$PROJ/<wherever-you-want-outputs>      # e.g. $PROJ/bayesDREAM/Simulations
DATA=$OUT/data
EXAMPLES=$CODE/examples/simulation_study
PYTHON_ENV=/path/to/your/conda/env/bin/python
ACCOUNT=<your-account-from-projinfo>

# sanity check — do this every time you open a new shell:
echo "PROJ=$PROJ  CODE=$CODE  OUT=$OUT  PYTHON_ENV=$PYTHON_ENV  ACCOUNT=$ACCOUNT"
$PYTHON_ENV --version
```

`CODE` and `OUT` are deliberately allowed to live under different parent
directories — the generator only needs to know where the code is
(`--bayesdream_path`/`--examples_path`) and where you want data written
(`--data_path`), independently.

## 2. Build the design matrix

Cheap, runs on the login node. **Commit and push any pending code changes first** —
this step creates a git tag pinning the exact commit used, and refuses to tag a dirty
working tree (a tag on a dirty tree would only pin the last commit, not what's about
to actually run):

```bash
mkdir -p $OUT
$PYTHON_ENV $EXAMPLES/build_design_matrix.py --outdir $OUT
```

Expected output: `Wrote 720 rows (144 scenarios x 5 replicates) to .../design_matrix.csv`,
followed by `Stable snapshot tag: sim-study-<timestamp> (pushed: True)`. If you instead
see a `[WARNING] Working tree has uncommitted changes` line, commit/push first and
re-run — otherwise `design_matrix.csv` will carry `bayesdream_tag` empty, meaning the
recorded commit hash is only reliably reproducible for as long as the `LUR` branch
itself isn't deleted or rewritten (see [Reproducibility](#reproducibility) below).

This `design_matrix.csv` is the single source of truth every later step reads by
`row_index` — nothing downstream re-derives the parameter grid.

## 3. Critical validation run

**Do this before generating or submitting anything at scale.** It answers two
questions in one job: does a full recovery fit actually complete without error at
the largest scenario size, and how long does it really take (which calibrates
`--min_fit_hours` in step 4, replacing a guess with a measurement).

Find a `cells_per_gene=1000` row (the worst case — every scenario shares the same
1736-feature trans panel, so only cell count varies):

```bash
ROW=$(awk -F, 'NR>1 && $5==1000 {print $1; exit}' $OUT/design_matrix.csv)
echo $ROW
```

Simulate it (cheap, CPU, includes the scran sum-factor step — fine on the login node):

```bash
$PYTHON_ENV $EXAMPLES/simulate_scenario.py --design_matrix $OUT/design_matrix.csv --row_index $ROW --outdir $DATA
# prints: [row <ROW>] wrote /.../data/scenario_<sid>/rep_<r>
```

Copy the printed path into `SCEN_DIR`, then submit the fit as a single job.
**Use `sbatch --wrap`, one physical line — do not paste a multi-line heredoc here.**
Multi-line pastes into an interactive terminal can silently corrupt a heredoc's
`EOF` terminator (invisible trailing whitespace, line-ending mangling), leaving the
shell stuck at a `>` continuation prompt indefinitely. `sbatch --wrap` submits from
a single command-line string, so there's nothing for paste corruption to break:

```bash
SCEN_DIR=$DATA/scenario_<sid>/rep_<r>   # replace with what was actually printed
mkdir -p $OUT/logs
sbatch --job-name=sim_study_validate --account=$ACCOUNT --output=$OUT/logs/validate_%j.out --error=$OUT/logs/validate_%j.err --time=06:00:00 --partition=berzelius -C thin --gpus=1 --cpus-per-task=1 --mem=16G --wrap="$PYTHON_ENV $EXAMPLES/run_recovery_fit.py --scenario_dir $SCEN_DIR --device cuda"
```

`run_recovery_fit.py` uses bayesDREAM's library-default `niters`/`nsamples`
throughout (deliberately not overridable — the point of this study is to validate
the defaults, not some study-specific iteration count), and `fit_ntc` defaults to
the `AutoNormal` guide, which is required at this scale (see
[Troubleshooting](#troubleshooting) if you see NaN errors — that specific failure
mode should already be fixed on the version of `bayesDREAM/fitting/ntc.py` in this
repo, but verify you have that fix before assuming a new crash is something else).

Once it finishes:

```bash
sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State
tail -100 $OUT/logs/validate_<jobid>.out
```

Record the `Elapsed` time — you'll pass a safety-margined version of it as
`--min_fit_hours` next. **Calibrated (2026-07-28)**: a real run on the largest
scenario was predicted to take just over 2h, so `--min_fit_hours 3` (also
`generate_slurm.py`'s default now) — override if your own timing differs.

## 4. Generate the SLURM scripts

```bash
$PYTHON_ENV $EXAMPLES/generate_slurm.py \
  --design_matrix $OUT/design_matrix.csv \
  --outdir $OUT/slurm \
  --data_path $DATA \
  --python_env $PYTHON_ENV \
  --bayesdream_path $CODE \
  --examples_path $EXAMPLES \
  --account $ACCOUNT \
  --min_fit_hours 3
```

This writes `$OUT/slurm/{01_simulate.sh, 02_fit.sh, submit_all.sh, logs/}`. It
internally simulates the largest scenario once more to size `fit_trans`'s
memory/partition choice via `SlurmJobGenerator.estimate_memory_requirements()` —
read the printed sizing summary before proceeding, and note the printed caveat
that the auto time-estimator underestimates at this study's scale (hence step 3).

`01_simulate.sh` and `02_fit.sh` are array scripts, sized to the number of rows in
`design_matrix.csv` (`--array=0-719%50` for the default 720-row grid). `submit_all.sh`
chains them with `--dependency=aftercorr`, meaning fit-array-task *i* starts as soon
as simulate-array-task *i* finishes — not after the whole simulate array completes.

## 5. Dry run

Confirm the generated scripts work under real SLURM before committing to all 720
tasks — module loads, resource grants, and path resolution are all things that can
differ from your interactive session:

```bash
cd $OUT/slurm
sbatch --array=0-2 01_simulate.sh
# wait for those 3 to finish, then:
sbatch --array=0-2 02_fit.sh
```

Check `logs/*.out`/`.err` for errors, and `sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State`
for real memory/time — adjust `--mem`/`--time` in the generated scripts directly if
either looks off before the full submission.

## 6. Submit the full study

```bash
cd $OUT/slurm
bash submit_all.sh
```

## 7. Monitor

```bash
squeue -u $USER
squeue --name=sim_study_sizing_fit                          # just the fit array
sacct -j <fit_job_id> --format=JobID,State,Elapsed,MaxRSS | grep FAILED
tail -f logs/sim_study_sizing_fit_<jobid>_<taskid>.out       # live log for one task
```

Resubmit an individual failed array task (SLURM re-substitutes `$SLURM_ARRAY_TASK_ID`
for just that index):

```bash
sbatch --array=<i> 02_fit.sh
```

## Reproducibility

A commit hash alone is sufficient to reproduce the exact code *content* (git is
content-addressed — the same hash always means the same tree, regardless of which
branch, if any, points to it), but it's only guaranteed to still *exist* in the repo
if something keeps it reachable. If the `LUR` branch is later deleted, rebased, or
force-pushed past the commit used here, that commit can become unreachable and
eventually get garbage-collected by `git gc` — even though its hash was recorded
faithfully at the time.

To guard against this, step 2 (`build_design_matrix.py`) creates and pushes an
**annotated git tag** at the current commit — a tag is a ref, so it keeps the commit
reachable independent of any branch's fate. Every row of `design_matrix.csv` carries
`bayesdream_tag`, `bayesdream_commit`, `bayesdream_branch`, and
`bayesdream_git_dirty` from that moment. Each scenario's own `config.json` then
*independently* re-checks commit/branch/dirty at simulate time (recorded as
`bayesdream_commit`/etc. there, alongside `build_commit`/`build_tag` copied from the
design matrix) — if you built the design matrix, pulled new commits, and only then
ran the SLURM array, `bayesdream_commit` in a scenario's `config.json` will disagree
with its `build_commit`, and that drift is visible by comparing the two.

To check out the exact code later, regardless of what's happened to the `LUR` branch
since:

```bash
git fetch --tags
git checkout <bayesdream_tag from design_matrix.csv or config.json>
```

If `bayesdream_tag` is empty for a run, only the commit hash/branch were recorded —
still reproducible today, but not protected against the branch being deleted or
rewritten later. `--no_tag`/`--no_push` on `build_design_matrix.py` skip this
protection deliberately (e.g. for local testing); omit them for a real study run.

## Output layout

```
$OUT/
├── design_matrix.csv
├── slurm/{01_simulate.sh, 02_fit.sh, submit_all.sh, logs/}
└── data/
    └── scenario_<0..143>/rep_<0..4>/
        ├── config.json, meta.csv, counts.csv
        ├── cis_ground_truth.csv, guide_ground_truth.csv, trans_ground_truth.csv
        ├── sum_factor_scran/            # R script + intermediates (kept for debugging)
        └── fit/recovery/
            ├── posterior_samples_{ntc,cis,trans}.pt
            └── trans_feature_summary_gene.csv
```

## Troubleshooting

**Lost environment variables.** If a command fails with a path that looks like it's
missing a prefix (e.g. bash tries to *execute* a `.py` file directly, or a script
looks for `--outdir` in a weird place), a variable from step 1 is almost certainly
unset in the current shell — re-run the `echo`/`--version` sanity checks from step 1
before anything else.

**Heredoc paste gets stuck at a `>` prompt.** See step 3 — use `sbatch --wrap` with
a single physical line instead of pasting a multi-line `cat > file << EOF` block.

**`fit_ntc` diverges to `NaN` within the first few SVI iterations**, with a
traceback through `pyro/infer/autoguide/guides.py`'s `AutoIAFNormal` and a `Delta`
distribution `ValueError`. This was a real, reproducible bug (root-caused: sparse
NTC features combined with `AutoIAFNormal`'s single jointly-coupled network — see
`docs/methods_section.tex`'s "Variational guide selection" paragraph for the full
mechanism), fixed by making `AutoNormal` the default guide. If you hit this, check
you're running the current `bayesDREAM/fitting/ntc.py` (`fit_ntc` should default to
`AutoNormal` unconditionally; `AutoIAFNormal` is opt-in via `force_iaf=True` and not
recommended for this study's data).

**`Rscript`/`scran` not found even though it's installed.** `simulate_scenario.py`
resolves `Rscript` relative to the *running Python interpreter's* own environment
directory, not bare `PATH` — invoking a specific `python` binary doesn't add its
conda env's `bin/` to subprocess `PATH`. Make sure `bioconductor-scran` is installed
in the *same* environment as `$PYTHON_ENV`, not a separate system R.
