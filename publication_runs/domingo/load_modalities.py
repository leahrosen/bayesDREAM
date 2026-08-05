"""
Domingo-specific hook: attach the extra (non-primary) modalities listed in
config_modalities.yaml to an already-cis-fit model, using your existing
custom splicing-loading functions.

TODO: replace `_load_one_modality` below with your actual loader(s). Per
your description, each takes (data_dir: str, modality_name: str) and is the
same function for every modality -- swap in that import/call here. If
different modality types (sj/ir/es/mxe/velocity vs donor/acceptor
usage-vs-efficiency) actually need different loader functions, dispatch on
`spec['distribution']` or `spec['name']` inside `_load_one_modality` instead.

Contract this module needs to satisfy: `attach_modality(model, spec,
data_dir)` must leave `model.get_modality(spec['name'])` populated and ready
for `fit_trans(modality_name=spec['name'], ...)` -- i.e. it must end by
calling one of `model.add_custom_modality(...)` /
`model.add_splicing_modality(...)` (see CLAUDE.md's Multi-Modal Architecture
section for both signatures).
"""

from typing import Dict


def _load_one_modality(data_dir: str, modality_name: str):
    """TODO: call your real loader here.

    Expected to return whatever your loader currently returns -- adjust the
    unpacking in `attach_modality` below to match. Sketched here as
    returning (counts, feature_meta, denominator) since binomial dominates
    this dataset's modality list; multinomial loaders presumably return a
    3-D counts array instead of (counts, denominator).

    Example (replace with your actual import):

        from my_splicing_lib import load_modality_data
        return load_modality_data(data_dir, modality_name)
    """
    raise NotImplementedError(
        f"Plug in your real splicing/velocity loader for modality={modality_name!r}, "
        f"data_dir={data_dir!r}."
    )


def attach_modality(model, spec: Dict, data_dir: str) -> None:
    """Load and attach one modality (per config_modalities.yaml `spec`) to `model`.

    Called once per (cis_gene, modality) by generate_slurm.py's per-task
    invocation (see domingo/generate_slurm.py's `modality_task_command`).
    """
    name = spec["name"]
    distribution = spec["distribution"]

    loaded = _load_one_modality(data_dir, name)

    if distribution == "multinomial":
        counts, feature_meta, cell_names = loaded
        model.add_custom_modality(
            name=name, counts=counts, feature_meta=feature_meta,
            distribution="multinomial", cell_names=cell_names,
        )
    elif distribution == "binomial":
        counts, feature_meta, denominator = loaded
        model.add_custom_modality(
            name=name, counts=counts, feature_meta=feature_meta,
            distribution="binomial", denominator=denominator,
        )
    else:
        raise ValueError(f"load_modalities.attach_modality: unsupported distribution {distribution!r} for {name!r}")


def main() -> None:
    """CLI entry point for one (gene, modality) attach-and-fit-trans task.

    Usage:
        python load_modalities.py --config <gene_cis_stage_config.yaml> \\
            --modality-name sj --modality-spec config_modalities.yaml \\
            --data-dir <modalities.data_dir>/<gene>
    """
    import argparse
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from config_utils import build_model_from_config, load_bayesdream_yaml, normalize_stage_args, is_enabled  # noqa: E402
    from git_provenance import save_provenance_json  # noqa: E402
    import yaml  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True, help="Gene's cis-stage bayesdream config (needs cis already fit and saved).")
    parser.add_argument("--modality-name", required=True)
    parser.add_argument("--modality-spec", required=True, help="Path to config_modalities.yaml")
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    cfg = load_bayesdream_yaml(Path(args.config))
    with open(args.modality_spec) as f:
        specs = {m["name"]: m for m in yaml.safe_load(f)["modalities"]}
    spec = specs[args.modality_name]

    model = build_model_from_config(cfg)
    cis_cfg = cfg.get("cis") or {}
    ntc_cfg = cfg.get("ntc") or cfg.get("technical") or {}
    if is_enabled(cis_cfg.get("load_ntc", cis_cfg.get("load_technical")), default=True):
        model.load_ntc_fit(**normalize_stage_args(cis_cfg.get("load_ntc") or cis_cfg.get("load_technical")))
    model.load_cis_fit(**normalize_stage_args(cis_cfg.get("load_cis")))
    if "technical_group_code" not in model.meta.columns:
        covariates = ntc_cfg.get("set_technical_groups")
        if covariates:
            model.set_technical_groups(covariates)

    attach_modality(model, spec, args.data_dir)

    model.fit_trans(modality_name=args.modality_name, function_type=spec["function_type"])

    model_cfg = cfg.get("model") or {}
    output_dir = os.path.join(model_cfg.get("output_dir", "output"), model_cfg.get("label"))
    model.save_trans_fit(output_dir=output_dir, modalities=[args.modality_name])
    model.save_trans_summary(output_dir=output_dir, modality_name=args.modality_name)

    save_provenance_json(
        os.path.join(output_dir, f"provenance_trans_{args.modality_name}.json"),
        extra={"stage": "trans_modality", "modality": args.modality_name, "label": model_cfg.get("label")},
    )


if __name__ == "__main__":
    main()
