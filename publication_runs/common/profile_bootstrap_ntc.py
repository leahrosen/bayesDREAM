"""
Cheap, REAL (disk-saved, not just in-memory) fit_ntc bootstrap for profiling
the deferred cis stage's load_ntc_fit() call.

Why this exists: profile_memory.py's own --stage ntc never calls
save_ntc_fit() (it's a pure in-memory profiling tool) -- so there is nothing
on disk yet for a deferred `<label>_cis.yaml` config's load_ntc_fit() to
read, until either the real ntc_shared/per-gene-ntc job has actually run, or
this script has. See profile_memory.py's module docstring ("Two shapes of
config") for why the deferred cis stage needs a real completed ntc fit at
all, unlike the eager compensation/trans/modality configs.

Writes into `{model.output_dir}/_profile_scratch/...` -- NEVER the real
ntc_shared_dir a dataset's generate_slurm.py wires into its `<label>_cis.yaml`
configs -- so this is safe to run before the real pipeline submission and
won't be mistaken for, or silently reused as, a converged fit. Peak memory is
shape-determined, not convergence-determined (same principle profile_memory.py
itself relies on), so a tiny --niters gives the same tensor shapes as a real
run, cheaply.

Usage
-----
    python profile_bootstrap_ntc.py --config <real_ntc_shared_or_gene_ntc.yaml> --niters 10

Prints the scratch directory to pass as profile_memory.py's
--ntc-shared-dir override when profiling that dataset's `<label>_cis.yaml`
with --stage cis.

Config must be a plain (non-deferred-commit, non-cis_gene) ntc config -- i.e.
Domingo's `<label>_ntc_shared.yaml` or Morris's per-gene `<label>_ntc.yaml`,
never a `<label>_cis.yaml` itself.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config_utils import build_model_from_config, load_bayesdream_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--niters", type=int, default=10)
    args = parser.parse_args()

    cfg = load_bayesdream_yaml(Path(args.config))
    model_cfg = cfg.setdefault("model", {})
    if model_cfg.get("cis_gene") or cfg.get("cis_gene"):
        raise ValueError(
            "profile_bootstrap_ntc: config must be a plain ntc config with no cis_gene anywhere "
            "(neither model.cis_gene nor a top-level cis_gene key) -- point it at the dataset's "
            "ntc_shared/per-gene-ntc config, not a <label>_cis.yaml."
        )
    if "output_dir" not in model_cfg:
        raise ValueError("profile_bootstrap_ntc: config missing model.output_dir")
    model_cfg["output_dir"] = f"{model_cfg['output_dir']}/_profile_scratch"

    model = build_model_from_config(cfg)
    ntc_cfg = cfg.get("ntc") or cfg.get("technical") or {}
    if ntc_cfg.get("set_technical_groups"):
        model.set_technical_groups(ntc_cfg["set_technical_groups"])
    model.fit_ntc(niters=args.niters)
    model.save_ntc_fit()

    scratch_dir = f"{model_cfg['output_dir']}/{model_cfg['label']}"
    print(f"[profile_bootstrap_ntc] scratch ntc dir (use as --ntc-shared-dir): {scratch_dir}")


if __name__ == "__main__":
    main()
