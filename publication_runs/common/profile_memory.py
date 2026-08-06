"""
Measure real peak memory for a bayesdream-CLI-schema config's model
construction, and (optionally) a cheap fit_ntc()/fit_cis() call -- to pick
--cpus-per-task on Dardel's `shared` partition, where RAM is a FIXED 888MB
per core (confirmed via `scontrol show partition shared` -- see
common/slurm/sbatch_blocks.py's docstring). cores_needed = ceil(peak_gb *
1024 / 888).

Constructor-only memory is a LOWER BOUND, not a good proxy on its own --
fit_ntc/fit_cis/fit_trans allocate on top of it (Adam optimizer state,
posterior sample draws sized by `nsamples`, gradient buffers). Pass
--stage ntc or --stage cis to measure a real fit call too; use a tiny
--niters (default 10) since peak memory is set by tensor SHAPES, not
convergence -- you don't need to wait for a real fit to complete to see its
real peak.

Reuses the _peak_rss_mb / _timed_step pattern from
examples/simulation_study/run_recovery_fit.py (that script wraps fit calls
only; this one also wraps the bare constructor, which it doesn't).

Usage
-----
    python profile_memory.py --config <gene_config.yaml> --stage init
    python profile_memory.py --config <gene_config.yaml> --stage ntc --niters 10
    python profile_memory.py --config <gene_config.yaml> --stage cis --niters 10

Config must have `model.cis_gene` set directly (not the deferred/add_cis_gene
pattern, e.g. Domingo's `<label>_cis.yaml`) for --stage cis to work --
point it at a config that already has cis_gene at construction time instead,
e.g. that same gene's `<label>_compensation.yaml` or `<label>_trans.yaml`
(memory characteristics are essentially the same either way; only cell/
feature subsetting mechanics differ between add_cis_gene() and cis_gene-at-
init, not what fit_cis() itself allocates).
"""

import argparse
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config_utils import build_model_from_config, load_bayesdream_yaml, normalize_stage_args  # noqa: E402


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
    cores_needed = (rss_after * 1024) / 888.0  # MB -> KB / (KB/core)
    print(f"[profile_memory] {name}: {elapsed:.1f}s, peak RSS so far {rss_after:.0f} MB "
          f"(was {rss_before:.0f} MB before this step) "
          f"-> ~{cores_needed:.1f} cores needed on Dardel's `shared` partition (888MB/core)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--stage", choices=["init", "ntc", "cis"], default="init")
    parser.add_argument("--niters", type=int, default=10, help="Tiny -- peak memory is shape-determined, not convergence-determined.")
    args = parser.parse_args()

    cfg = load_bayesdream_yaml(Path(args.config))

    with _timed_step("model construction (bayesDREAM.__init__)"):
        model = build_model_from_config(cfg)

    if args.stage == "init":
        return

    ntc_cfg = cfg.get("ntc") or cfg.get("technical") or {}
    if ntc_cfg.get("set_technical_groups"):
        model.set_technical_groups(ntc_cfg["set_technical_groups"])

    with _timed_step(f"fit_ntc(niters={args.niters})"):
        fit_args = dict(normalize_stage_args(ntc_cfg.get("fit")))
        fit_args["niters"] = args.niters
        model.fit_ntc(**fit_args)

    if args.stage == "ntc":
        return

    cis_cfg = cfg.get("cis") or {}
    with _timed_step(f"fit_cis(niters={args.niters})"):
        fit_args = dict(normalize_stage_args(cis_cfg.get("fit")))
        fit_args["niters"] = args.niters
        model.fit_cis(**fit_args)


if __name__ == "__main__":
    main()
