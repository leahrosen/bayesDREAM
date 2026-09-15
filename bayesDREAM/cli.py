"""Typer-based command line interface for bayesDREAM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import typer
import yaml

from . import bayesDREAM

app = typer.Typer(help="CLI wrapper for bayesDREAM analysis workflows.")


def _load_yaml(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise typer.BadParameter(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise typer.BadParameter("Config root must be a YAML mapping/object")
    return cfg


def _read_table(path: str, kwargs: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    read_kwargs = dict(kwargs or {})
    return pd.read_csv(path, **read_kwargs)


def _load_guide_assignment(path: str) -> np.ndarray:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        return np.load(p)
    if suffix == ".npz":
        loaded = np.load(p)
        if len(loaded.files) != 1:
            raise ValueError(
                f"guide_assignment .npz should contain exactly one array, found {loaded.files}"
            )
        return loaded[loaded.files[0]]
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix in {".tsv", ".txt"} else ","
        return pd.read_csv(p, header=None, sep=sep).to_numpy()
    raise ValueError(
        f"Unsupported guide_assignment format: {path}. Use .npy, .npz, .csv, .tsv, or .txt"
    )


def _normalize_stage_args(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, bool):
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"Stage config must be a dict/bool, got {type(obj).__name__}")
    if "args" in obj:
        args = obj.get("args")
        if args is None:
            return {}
        if not isinstance(args, dict):
            raise ValueError("Stage 'args' must be a dict")
        return dict(args)
    return {k: v for k, v in obj.items() if k != "enabled"}


def _is_enabled(obj: Any, default: bool = True) -> bool:
    if obj is None:
        return default
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, dict):
        return bool(obj.get("enabled", default))
    return default


def _build_model(cfg: Dict[str, Any]) -> bayesDREAM:
    data_cfg = cfg.get("data") or {}
    model_cfg = cfg.get("model") or {}

    if "meta" not in data_cfg:
        raise typer.BadParameter("Config missing required key: data.meta")
    if "counts" not in data_cfg:
        raise typer.BadParameter("Config missing required key: data.counts")

    meta = _read_table(data_cfg["meta"], data_cfg.get("meta_read_csv_kwargs"))
    counts = _read_table(data_cfg["counts"], data_cfg.get("counts_read_csv_kwargs") or {"index_col": 0})

    feature_meta = None
    if data_cfg.get("feature_meta"):
        feature_meta = _read_table(
            data_cfg["feature_meta"],
            data_cfg.get("feature_meta_read_csv_kwargs") or {"index_col": 0},
        )

    guide_assignment = None
    if data_cfg.get("guide_assignment"):
        guide_assignment = _load_guide_assignment(data_cfg["guide_assignment"])

    guide_meta = None
    if data_cfg.get("guide_meta"):
        guide_meta = _read_table(data_cfg["guide_meta"], data_cfg.get("guide_meta_read_csv_kwargs"))

    guide_target = None
    if data_cfg.get("guide_target"):
        guide_target = _read_table(data_cfg["guide_target"], data_cfg.get("guide_target_read_csv_kwargs"))

    allowed_model_keys = {
        "modality_name",
        "cis_gene",
        "cis_feature",
        "guide_covariates",
        "guide_covariates_ntc",
        "sum_factor_col",
        "output_dir",
        "label",
        "device",
        "random_seed",
        "cores",
        "exclude_targets",
        "require_ntc",
    }
    model_kwargs = {k: v for k, v in model_cfg.items() if k in allowed_model_keys}

    if feature_meta is not None:
        model_kwargs["feature_meta"] = feature_meta

    return bayesDREAM(
        meta=meta,
        counts=counts,
        guide_assignment=guide_assignment,
        guide_meta=guide_meta,
        guide_target=guide_target,
        **model_kwargs,
    )


def _run_fit_ntc(model: bayesDREAM, cfg: Dict[str, Any]) -> None:
    tech_cfg = cfg.get("ntc") or cfg.get("technical") or {}

    if tech_cfg.get("set_technical_groups"):
        model.set_technical_groups(tech_cfg["set_technical_groups"])

    fit_args = _normalize_stage_args(tech_cfg.get("fit"))
    model.fit_ntc(**fit_args)

    if _is_enabled(tech_cfg.get("save"), default=True):
        save_args = _normalize_stage_args(tech_cfg.get("save"))
        model.save_ntc_fit(**save_args)


def _run_fit_cis(model: bayesDREAM, cfg: Dict[str, Any]) -> None:
    cis_cfg = cfg.get("cis") or {}
    tech_cfg = cfg.get("ntc") or cfg.get("technical") or {}

    if _is_enabled(cis_cfg.get("load_ntc", cis_cfg.get("load_technical")), default=True):
        load_args = _normalize_stage_args(cis_cfg.get("load_ntc") or cis_cfg.get("load_technical"))
        model.load_ntc_fit(**load_args)

    # Stepwise mode may start from a fresh model object where technical_group_code
    # was not persisted in metadata. Recreate it before fit_cis when possible.
    fit_args = _normalize_stage_args(cis_cfg.get("fit"))
    if (
        model.alpha_x_prefit is not None
        and "technical_group_code" not in model.meta.columns
    ):
        covariates = fit_args.get("technical_covariates") or tech_cfg.get("set_technical_groups")
        if covariates:
            model.set_technical_groups(covariates)

    if _is_enabled(cis_cfg.get("adjust_ntc_sum_factor"), default=False):
        adjust_args = _normalize_stage_args(cis_cfg.get("adjust_ntc_sum_factor"))
        model.adjust_ntc_sum_factor(**adjust_args)

    model.fit_cis(**fit_args)

    if _is_enabled(cis_cfg.get("save"), default=True):
        save_args = _normalize_stage_args(cis_cfg.get("save"))
        model.save_cis_fit(**save_args)


def _run_fit_trans(model: bayesDREAM, cfg: Dict[str, Any]) -> None:
    trans_cfg = cfg.get("trans") or {}

    if _is_enabled(trans_cfg.get("load_ntc", trans_cfg.get("load_technical")), default=True):
        load_tech_args = _normalize_stage_args(trans_cfg.get("load_ntc") or trans_cfg.get("load_technical"))
        model.load_ntc_fit(**load_tech_args)

    if _is_enabled(trans_cfg.get("load_cis"), default=True):
        load_cis_args = _normalize_stage_args(trans_cfg.get("load_cis"))
        model.load_cis_fit(**load_cis_args)

    fit_args = _normalize_stage_args(trans_cfg.get("fit"))
    model.fit_trans(**fit_args)

    if _is_enabled(trans_cfg.get("save"), default=True):
        save_args = _normalize_stage_args(trans_cfg.get("save"))
        model.save_trans_fit(**save_args)


def _run_report(model: bayesDREAM, cfg: Dict[str, Any]) -> None:
    report_cfg = cfg.get("report") or {}

    if _is_enabled(report_cfg.get("load_ntc", report_cfg.get("load_technical")), default=False):
        load_args = _normalize_stage_args(report_cfg.get("load_ntc") or report_cfg.get("load_technical"))
        model.load_ntc_fit(**load_args)

    if _is_enabled(report_cfg.get("load_cis"), default=False):
        load_args = _normalize_stage_args(report_cfg.get("load_cis"))
        model.load_cis_fit(**load_args)

    if _is_enabled(report_cfg.get("load_trans"), default=False):
        load_args = _normalize_stage_args(report_cfg.get("load_trans"))
        model.load_trans_fit(**load_args)

    if _is_enabled(report_cfg.get("ntc", report_cfg.get("technical")), default=True):
        args = _normalize_stage_args(report_cfg.get("ntc") or report_cfg.get("technical"))
        model.save_ntc_summary(**args)

    if _is_enabled(report_cfg.get("cis"), default=True):
        args = _normalize_stage_args(report_cfg.get("cis"))
        model.save_cis_summary(**args)

    if _is_enabled(report_cfg.get("trans"), default=True):
        args = _normalize_stage_args(report_cfg.get("trans"))
        model.save_trans_summary(**args)


@app.command("run")
def run_pipeline(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file."),
    steps: Optional[str] = typer.Option(
        None,
        help="Comma-separated override for run steps (technical,cis,trans,report).",
    ),
) -> None:
    """Run pipeline steps in sequence using config defaults."""
    cfg = _load_yaml(config)
    model = _build_model(cfg)

    run_cfg = cfg.get("run") or {}
    pipeline_steps = run_cfg.get("steps", ["ntc", "cis", "trans", "report"])
    if steps:
        pipeline_steps = [s.strip() for s in steps.split(",") if s.strip()]

    valid_steps = {"ntc", "technical", "cis", "trans", "report"}
    unknown = [s for s in pipeline_steps if s not in valid_steps]
    if unknown:
        raise typer.BadParameter(f"Unknown run steps: {unknown}. Valid: {sorted(valid_steps)}")

    for step in pipeline_steps:
        typer.echo(f"[bayesdream] running step: {step}")
        if step in ("ntc", "technical"):
            _run_fit_ntc(model, cfg)
        elif step == "cis":
            _run_fit_cis(model, cfg)
        elif step == "trans":
            _run_fit_trans(model, cfg)
        elif step == "report":
            _run_report(model, cfg)


@app.command("fit-ntc")
def fit_ntc(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file."),
) -> None:
    """Run NTC fit stage (overdispersion estimation and batch effect correction)."""
    cfg = _load_yaml(config)
    model = _build_model(cfg)
    _run_fit_ntc(model, cfg)


@app.command("fit-cis")
def fit_cis(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file."),
) -> None:
    """Run cis fit stage (optionally loading technical fit first)."""
    cfg = _load_yaml(config)
    model = _build_model(cfg)
    _run_fit_cis(model, cfg)


@app.command("fit-trans")
def fit_trans(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file."),
) -> None:
    """Run trans fit stage (optionally loading technical/cis fits first)."""
    cfg = _load_yaml(config)
    model = _build_model(cfg)
    _run_fit_trans(model, cfg)


@app.command("report")
def report(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file."),
) -> None:
    """Export summary CSV reports."""
    cfg = _load_yaml(config)
    model = _build_model(cfg)
    _run_report(model, cfg)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
