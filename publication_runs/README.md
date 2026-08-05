# publication_runs/

Per-dataset run infrastructure for the publication analyses on Dardel. One
subdirectory per dataset (`domingo/`, `morris/`, ...), each with its own
`config.yaml` and `generate_slurm.py`, plus a `common/` library shared by all
of them.

This directory is orchestration code, not part of the installable `bayesDREAM`
package — it drives `bayesDREAM` (via `python -m bayesDREAM <stage> --config
...`, the existing Typer CLI in `bayesDREAM/cli.py`) and a few extra stages
(compensation check, permutation null, recapitulation simulation) that don't
have CLI subcommands yet.

## Layout

```
publication_runs/
├── common/                       # shared across all datasets
│   ├── config_utils.py           # YAML load/deep-merge/render helpers, reuses bayesDREAM.cli internals
│   ├── git_provenance.py         # commit/branch/dirty capture + optional stable tag
│   ├── run_compensation.py       # standalone: model.check_systematic_shift()
│   ├── run_permutation_null.py   # standalone: permute_from_ntc + permute_x_true, then fit_trans
│   ├── run_recapitulation_sim.py # standalone: simulate_from_trans_summary + refit, compare
│   ├── apply_shared_ntc_high_moi.py  # high-MOI workaround for the shared-fit_ntc pattern (see morris/README.md)
│   └── slurm/
│       ├── sbatch_blocks.py      # SBATCH header builders (cpu step / cpu array / gpu whole-node)
│       ├── run_node_queue.sh     # packs several tasks onto one node allocation
│       └── submit_chain.sh       # submits a list of scripts with --dependency=afterok chaining
│
├── domingo/                      # dataset: 4 cis genes, primary + splicing/velocity modalities
├── morris/                       # dataset: high-MOI, 5 primary + ~hundreds of cis-only genes
├── template_dataset/             # copy this to start a new dataset
└── README.md                     # this file
```

## Conventions

**Stages.** Every dataset run is composed of independently-runnable stages:
`ntc` -> `cis` -> `compensation` -> `trans` -> (`permutation`, `recapitulation`).
Each stage is one `sbatch` job (or one task in an array), invoked either via
the `bayesDREAM` CLI directly (`ntc`/`cis`/`trans`/`report`) or via one of the
`common/run_*.py` scripts (`compensation`/`permutation`/`recapitulation`).
Some datasets skip stages or share one stage's output across many runs of
another (e.g. Domingo shares one `ntc` fit across 4 cis genes; Morris shares
one `ntc` fit across ~5+hundreds of cis genes).

**Config layering.** Two YAML layers, deliberately kept separate:

1. **`<dataset>/config.yaml`** — the *orchestration* config: input/output
   paths, which cis genes to run, per-step SLURM resources, permutation/
   simulation replicate counts, cluster account. Consumed by
   `<dataset>/generate_slurm.py`. Not understood by `bayesDREAM` itself.
2. **Per-gene/per-stage `bayesdream_config` YAML** — the exact schema
   `bayesDREAM/cli.py` expects (`data:`/`model:`/`ntc:`/`cis:`/`trans:`/
   `report:` keys, as in `examples/gfi1b_cli_config.yaml`). These are
   *generated* by `generate_slurm.py` from `config.yaml` (one per gene per
   stage, written to `<output_dir>/<label>/configs/`) rather than hand-written,
   so a dataset's 4-or-500 genes don't need 4-or-500 hand-maintained files.

**Git provenance.** Every generated `sbatch` script sources
`common/git_provenance.py` at the top of its job (prints commit/branch/dirty
state into the log) and every `run_*.py` script in `common/` writes a
`provenance.json` next to its output. `generate_slurm.py` itself also calls
`create_stable_snapshot_tag()` once per dataset at generation time (not once
per job — see the docstring in `git_provenance.py`) so the whole batch of jobs
is pinned to one tag.

**GPU node packing.** When a step only needs a fraction of a Dardel GPU node,
don't request one node per task. Instead write a plain-text task list (one
CLI invocation per line) and submit a single `sbatch` that requests one node
and runs `common/slurm/run_node_queue.sh <tasklist> <concurrency>`, which
runs up to `<concurrency>` tasks at a time via `xargs -P`. See
`common/slurm/run_node_queue.sh`.

**Dardel's MaxSubmitJobs gotcha.** `sbatch` array tasks count against your
account's `MaxSubmitJobs` quota the instant they're submitted, regardless of
`--dependency` (a dependency only delays *running*, not *counting*). This bit
the simulation study (`examples/simulation_study/generate_slurm_dardel.py`,
see its module docstring) with two chained 720-task arrays exceeding a
1024-job quota. `common/slurm/submit_chain.sh` and each dataset's
`generate_slurm.py` keep this in mind: prefer one array with an internal
per-task step sequence over multiple dependent arrays, and check
`sacctmgr show assoc user=$USER format=MaxSubmitJobs` before submitting a
large batch (Morris's ~500-gene cis-only sweep is the one to watch here).

## Adding a new dataset

Copy `template_dataset/` to `<new_name>/`, fill in `config.yaml`, and adapt
`generate_slurm.py` (the per-dataset quirks — which stages run, how genes
share an `ntc` fit, any custom modality loading — live here; everything
generic lives in `common/`).
