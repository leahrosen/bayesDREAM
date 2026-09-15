# template_dataset

Copy this directory to `publication_runs/<new_name>/`, rename its files'
internal references (`dataset: template_dataset` in `config.yaml`, etc.),
and fill in `config.yaml`. `generate_slurm.py` as shipped here is a working
low-MOI pipeline (shared `fit_ntc` -> per-gene cis/compensation/trans/
permutation/recapitulation) -- run it as-is against a filled-in config to
get a first working batch, then adapt for whatever's dataset-specific (see
the module docstring at the top of `generate_slurm.py` for the three most
common adaptations: high-MOI, no shared fit_ntc, extra modalities).

See `publication_runs/README.md` for the shared conventions, and
`domingo/` / `morris/` for two worked, more elaborate examples.
