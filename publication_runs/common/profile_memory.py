"""
Measure real peak memory for a bayesdream-CLI-schema config's model
construction, and (optionally) a cheap fit_ntc()/fit_cis()/fit_trans() call
-- to pick --cpus-per-task on Dardel's `shared` partition, where RAM is a
FIXED 888MB per core (confirmed via `scontrol show partition shared` -- see
common/slurm/sbatch_blocks.py's docstring). cores_needed = ceil(peak_gb *
1024 / 888).

Constructor-only memory is a LOWER BOUND, not a good proxy on its own --
fit_ntc/fit_cis/fit_trans allocate on top of it (Adam optimizer state,
posterior sample draws sized by `nsamples`, gradient buffers). Pass
--stage ntc/cis/trans to measure a real fit call too; use a tiny --niters
(default 10) since peak memory is set by tensor SHAPES, not convergence --
you don't need to wait for a real fit to complete to see its real peak.

Reuses the _peak_rss_mb / _timed_step pattern from
examples/simulation_study/run_recovery_fit.py (that script wraps fit calls
only; this one also wraps the bare constructor, which it doesn't).

Stage chain: each stage does everything the previous one does, plus one
more call -- init -> ntc(primary [+ --modality-name's own fit_ntc]) ->
cis -> trans(primary or --modality-name). Reported peak RSS is CUMULATIVE
("peak so far in this process"), not an isolated per-step delta, matching
every step already printing "peak RSS so far".

--modality-name profiles a NON-primary modality's OWN fit_ntc()/fit_trans()
call (e.g. Domingo's binomial splicing modalities, which get a genuinely
separate fit_ntc() -- see domingo/README.md) IN ADDITION TO the primary
modality's. Only meaningful with --stage ntc or --stage trans. For a custom
modality that doesn't already exist on the model at construction time (i.e.
anything added via add_custom_modality(), not the primary/cis modalities),
the config needs a top-level `attach_modality:` block -- same schema as
run_permutation_null.py's (see its docstring), resolved once right after
construction, before any fit call.

Usage
-----
    python profile_memory.py --config <gene_config.yaml> --stage init
    python profile_memory.py --config <gene_config.yaml> --stage ntc --niters 10
    python profile_memory.py --config <gene_config.yaml> --stage cis --niters 10
    python profile_memory.py --config <gene_config.yaml> --stage trans --niters 10
    python profile_memory.py --config <gene_modality_config.yaml> --stage ntc \\
        --modality-name splicing_sj --niters 10
    python profile_memory.py --config <gene_modality_config.yaml> --stage trans \\
        --modality-name splicing_sj --niters 10

Config must have `model.cis_gene` set directly (not the deferred/add_cis_gene
pattern, e.g. Domingo's `<label>_cis.yaml`) for --stage cis/trans to work --
point it at a config that already has cis_gene at construction time instead,
e.g. that same gene's `<label>_compensation.yaml` or `<label>_trans.yaml`
(memory characteristics are essentially the same either way; only cell/
feature subsetting mechanics differ between add_cis_gene() and cis_gene-at-
init, not what fit_cis()/fit_trans() themselves allocate). For a modality's
own profiling run, point --config at that gene's `<label>_modality_<name>.yaml`
(has the `attach_modality:` block already) -- see domingo/generate_slurm.py.

Note --stage cis/trans do NOT require a real completed ntc_shared run:
fit_cis() only requires alpha_x_prefit if you pass technical_covariates
(the pipeline's own configs never do), so it runs fine off the tiny
in-process fit_ntc() this script does anyway -- no manual placeholder
needed.
"""

import argparse
import importlib
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config_utils import (  # noqa: E402
    build_model_from_config,
    load_bayesdream_yaml,
    normalize_stage_args,
    ensure_dataset_dir_on_syspath,
)


def _peak_rss_mb() -> float:
    """Peak RSS so far in this process, in MB. ru_maxrss is KB on Linux,
    bytes on macOS/BSD -- Dardel is Linux, so KB is assumed here."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@contextmanager
def _timed_step(name: str):
    t0 = time.time()
    rss_before = _peak_rss_mb()
    yield
    elapsed = time.time() - t0
    rss_after = _peak_rss_mb()
    cores_needed = rss_after / 888.0  # both already MB (888MB/core, NOT KB/core -- see sbatch_blocks.py)
    print(f"[profile_memory] {name}: {elapsed:.1f}s, peak RSS so far {rss_after:.0f} MB "
          f"(was {rss_before:.0f} MB before this step) "
          f"-> ~{cores_needed:.1f} cores needed on Dardel's `shared` partition (888MB/core)")


def _attach_modality_if_configured(model, cfg: dict):
    """Same contract as run_permutation_null.py's helper of the same name --
    no-op if the config has no `attach_modality:` block (i.e. this run
    targets only modalities that already exist on `model` after
    build_model_from_config, namely the primary and 'cis' modalities)."""
    spec = cfg.get("attach_modality")
    if not spec:
        return None
    ensure_dataset_dir_on_syspath(cfg)
    mod = importlib.import_module(spec["module"])
    fn = getattr(mod, spec["function"])
    return fn(model, **spec.get("kwargs", {}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--stage", choices=["init", "ntc", "cis", "trans"], default="init")
    parser.add_argument("--niters", type=int, default=10, help="Tiny -- peak memory is shape-determined, not convergence-determined.")
    parser.add_argument("--modality-name", default=None,
                         help="Profile this modality's OWN fit_ntc()/fit_trans() call too (only used "
                              "with --stage ntc or --stage trans). For a custom modality, --config's "
                              "attach_modality: block must be able to attach it first.")
    args = parser.parse_args()

    cfg = load_bayesdream_yaml(Path(args.config))
    cfg["_dataset_dir"] = cfg.get("_dataset_dir") or str(Path(args.config).resolve().parents[2])

    with _timed_step("model construction (bayesDREAM.__init__)"):
        model = build_model_from_config(cfg)

    _attach_modality_if_configured(model, cfg)

    if args.stage == "init":
        return

    ntc_cfg = cfg.get("ntc") or cfg.get("technical") or {}
    if ntc_cfg.get("set_technical_groups"):
        model.set_technical_groups(ntc_cfg["set_technical_groups"])

    with _timed_step(f"fit_ntc(primary, niters={args.niters})"):
        fit_args = dict(normalize_stage_args(ntc_cfg.get("fit")))
        fit_args["niters"] = args.niters
        model.fit_ntc(**fit_args)

    modality_name = args.modality_name
    if modality_name and modality_name != model.primary_modality:
        with _timed_step(f"fit_ntc({modality_name}, niters={args.niters})"):
            fit_args = dict(normalize_stage_args(ntc_cfg.get("fit")))
            fit_args["niters"] = args.niters
            fit_args["modality_name"] = modality_name
            model.fit_ntc(**fit_args)

    if args.stage == "ntc":
        return

    cis_cfg = cfg.get("cis") or {}
    with _timed_step(f"fit_cis(niters={args.niters})"):
        fit_args = dict(normalize_stage_args(cis_cfg.get("fit")))
        fit_args["niters"] = args.niters
        model.fit_cis(**fit_args)

    if args.stage == "cis":
        return

    trans_cfg = cfg.get("trans") or {}
    with _timed_step(f"fit_trans({modality_name or 'primary'}, niters={args.niters})"):
        fit_args = dict(normalize_stage_args(trans_cfg.get("fit")))
        fit_args["niters"] = args.niters
        if modality_name:
            fit_args.setdefault("modality_name", modality_name)
        model.fit_trans(**fit_args)


if __name__ == "__main__":
    main()
