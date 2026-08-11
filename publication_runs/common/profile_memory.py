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
cis -> {compensation, trans(primary or --modality-name)}. compensation and
trans both branch off the same post-cis state; passing --stage compensation
does NOT also run trans (matches the real pipeline, where compensation and
trans are independent jobs that both just depend on cis). Reported peak RSS
is CUMULATIVE ("peak so far in this process"), not an isolated per-step
delta, matching every step already printing "peak RSS so far".

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
        --modality-name splicing_sj --modality-spec domingo/config_modalities.yaml --niters 10
    python profile_memory.py --config <label>_cis.yaml --stage cis --niters 10   # deferred
    python profile_memory.py --config <label>_compensation.yaml --stage compensation --niters 10

Two shapes of config, auto-detected from whether `cis_gene` sits under
`model:` (eager, cis_gene known at construction) or as a top-level key with
no `model.cis_gene` (deferred/add_cis_gene() pattern, e.g. Domingo/Morris's
`<label>_cis.yaml` -- see run_cis_deferred.py):

- Eager (`<label>_compensation.yaml`, `<label>_trans.yaml`, plain
  `<label>_modality_<name>.yaml`): --stage ntc/cis/compensation/trans do
  their OWN tiny in-process fit_ntc()/fit_cis() (niters=10) -- no real
  ntc_shared run needed first, since fit_cis() only requires alpha_x_prefit
  if you pass technical_covariates (the pipeline's own configs never do).
  --stage compensation calls check_systematic_shift() straight off that
  same tiny fit_cis() state -- see "compensation" note below.
- Deferred (`<label>_cis.yaml`): mirrors run_cis_deferred.py exactly --
  model.load_ntc_fit(ntc_shared_dir, mask_features=True) then
  model.add_cis_gene(cis_gene), so this DOES require a real completed
  ntc_shared run on disk first (that's the whole point of profiling THIS
  data path -- it measures the real memory load_ntc_fit's full-panel
  posteriors add on top of the tiny cis_only data, which the eager path
  above never touches). Only --stage init/cis are meaningful here -- ntc is
  shared/already-fit elsewhere and compensation/trans always use a separate
  eager config, so --stage ntc/compensation/trans raise on a deferred config.

For a modality's own profiling run, point --config at that gene's
`<label>_modality_<name>.yaml` (has the `attach_modality:` block --
generate_slurm.py adds it to every modality's plain fit config now, not
just its permutation/simulation configs) -- see domingo/generate_slurm.py.

`--stage compensation` profiles `check_systematic_shift()` directly (reads
the same `full` data subset as `trans`, but allocates less -- no Adam state,
no posterior draws, no checkpointing). It reuses `run_compensation.py`'s own
`exclude_cells` resolution, including dataset-specific dynamic-import hooks
(e.g. Morris' padj-based rule, `morris/compensation_exclude_cells.py`) --
point `--config` at the same `<label>_compensation.yaml` the real pipeline
uses.

`--modality-spec` (required for `--stage trans --modality-name <non-primary>`):
a non-primary modality's `function_type`/`min_denominator` are NOT read from
the config's `trans:` block -- plain `<label>_modality_<name>.yaml` configs
never carry one (domingo/load_modalities.py's real fit_trans() call gets
these straight from `config_modalities.yaml`'s per-stype `spec`, not from a
bayesdream-config `trans:` block -- see its module docstring). Point
`--modality-spec` at that same file (e.g. domingo/config_modalities.yaml) so
this script can look up the same two values by matching
`f"{spec.get('name_prefix', 'splicing')}_{spec['stype']}"` against
`--modality-name`. Omitting this for a non-primary modality raises a clear
error here instead of the confusing "min_denominator is required for
distribution='binomial'" ValueError several stack frames deep inside
fit_trans().
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
    apply_sum_factor_adjustments,
)
from run_compensation import _resolve_exclude_cells  # noqa: E402


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


def _find_modality_spec(modality_spec_path: str, modality_name: str) -> dict:
    """Look up modality_name's (e.g. 'splicing_sj') entry in a
    config_modalities.yaml-shaped file, matching on
    f"{spec.get('name_prefix', 'splicing')}_{spec['stype']}" -- same name
    construction as load_modalities.attach_modality()/attach_modality_precomputed()."""
    import yaml
    with open(modality_spec_path) as f:
        specs = yaml.safe_load(f)["modalities"]
    for spec in specs:
        if f"{spec.get('name_prefix', 'splicing')}_{spec['stype']}" == modality_name:
            return spec
    raise ValueError(f"_find_modality_spec: no entry for modality_name={modality_name!r} in {modality_spec_path}")


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
    parser.add_argument("--stage", choices=["init", "ntc", "cis", "compensation", "trans"], default="init")
    parser.add_argument("--niters", type=int, default=10, help="Tiny -- peak memory is shape-determined, not convergence-determined.")
    parser.add_argument("--modality-name", default=None,
                         help="Profile this modality's OWN fit_ntc()/fit_trans() call too (only used "
                              "with --stage ntc or --stage trans). For a custom modality, --config's "
                              "attach_modality: block must be able to attach it first.")
    parser.add_argument("--modality-spec", default=None,
                         help="Path to a config_modalities.yaml-shaped file (list of {stype, "
                              "distribution, function_type, min_denominator, ...} dicts). Required for "
                              "--stage trans on a non-primary --modality-name -- see module docstring.")
    parser.add_argument("--ntc-shared-dir", default=None,
                         help="Deferred configs only: override the config's own ntc_shared_dir, e.g. to "
                              "point at a profile_bootstrap_ntc.py scratch fit instead of editing the "
                              "real rendered config (which the actual pipeline run will also use).")
    args = parser.parse_args()

    cfg = load_bayesdream_yaml(Path(args.config))
    cfg["_dataset_dir"] = cfg.get("_dataset_dir") or str(Path(args.config).resolve().parents[2])
    if args.ntc_shared_dir:
        cfg["ntc_shared_dir"] = args.ntc_shared_dir
    is_deferred = bool(cfg.get("cis_gene")) and not (cfg.get("model") or {}).get("cis_gene")

    with _timed_step("model construction (bayesDREAM.__init__)"):
        model = build_model_from_config(cfg)

    _attach_modality_if_configured(model, cfg)

    if args.stage == "init":
        return

    ntc_cfg = cfg.get("ntc") or cfg.get("technical") or {}
    if ntc_cfg.get("set_technical_groups"):
        model.set_technical_groups(ntc_cfg["set_technical_groups"])

    if is_deferred:
        # Mirrors run_cis_deferred.py exactly -- see module docstring's
        # "Two shapes of config" section for why ntc/compensation/trans
        # don't apply here.
        if args.stage != "cis":
            raise ValueError(
                f"profile_memory: --stage {args.stage} doesn't apply to a deferred cis_gene config "
                f"({args.config}). ntc is already fit in a separate ntc_shared job -- profile that "
                "config directly with --stage ntc. compensation/trans always run off a separate eager "
                "(cis_gene-at-construction) config, e.g. <label>_compensation.yaml/<label>_trans.yaml "
                "-- profile those instead."
            )
        ntc_shared_dir = cfg.get("ntc_shared_dir")
        if not ntc_shared_dir:
            raise ValueError(
                "profile_memory: deferred config needs a top-level 'ntc_shared_dir' key (same as "
                "run_cis_deferred.py), pointing at a REAL, already-completed ntc_shared output "
                "directory -- that's the whole point of profiling this path: it measures the real "
                "memory load_ntc_fit()'s full-panel posteriors add on top of the tiny cis_only data."
            )
        with _timed_step("load_ntc_fit + add_cis_gene"):
            model.load_ntc_fit(input_dir=ntc_shared_dir, mask_features=True)
            model.add_cis_gene(cfg["cis_gene"])

        apply_sum_factor_adjustments(
            model, cfg.get("sum_factor") or {}, steps=("compute_scran", "adjust_ntc_sum_factor"))

        cis_cfg = cfg.get("cis") or {}
        with _timed_step(f"fit_cis(niters={args.niters})"):
            fit_args = dict(normalize_stage_args(cis_cfg.get("fit")))
            fit_args["niters"] = args.niters
            model.fit_cis(**fit_args)
        return

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

    if args.stage == "compensation":
        # check_systematic_shift() has no Adam state/posterior draws/
        # checkpointing (see run_compensation.py) -- the tiny in-process
        # fit_cis() above already gives it correctly-shaped x_true/alpha_x,
        # same shape-not-convergence principle as every other stage here.
        # No real completed ntc/cis fit on disk needed, unlike the deferred
        # cis path -- see module docstring.
        comp_cfg = cfg.get("compensation") or {}
        comp_args = dict(normalize_stage_args(comp_cfg.get("args")))
        comp_args["exclude_cells"] = _resolve_exclude_cells(comp_args.get("exclude_cells"), model, cfg)
        with _timed_step("check_systematic_shift"):
            model.check_systematic_shift(**comp_args)
        return

    trans_cfg = cfg.get("trans") or {}
    with _timed_step(f"fit_trans({modality_name or 'primary'}, niters={args.niters})"):
        fit_args = dict(normalize_stage_args(trans_cfg.get("fit")))
        fit_args["niters"] = args.niters
        if modality_name:
            fit_args.setdefault("modality_name", modality_name)
            if modality_name != model.primary_modality:
                # Plain <label>_modality_<name>.yaml configs never carry a
                # trans: block (the real job reads function_type/
                # min_denominator from config_modalities.yaml instead -- see
                # module docstring's "--modality-spec" section) -- without
                # this, fit_trans() below fails several frames deep with
                # "min_denominator is required for distribution='binomial'".
                if not args.modality_spec:
                    raise ValueError(
                        f"profile_memory: --stage trans --modality-name {modality_name!r} is a "
                        "non-primary modality but no --modality-spec was given -- pass "
                        "--modality-spec <dataset>/config_modalities.yaml so function_type/"
                        "min_denominator can be looked up (see module docstring)."
                    )
                spec = _find_modality_spec(args.modality_spec, modality_name)
                fit_args.setdefault("function_type", spec["function_type"])
                fit_args.setdefault("min_denominator", spec.get("min_denominator", 0))
        model.fit_trans(**fit_args)


if __name__ == "__main__":
    main()
