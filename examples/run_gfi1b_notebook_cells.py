#!/usr/bin/env python3
"""Reproduce the notebook workflow from the provided GFI1B cells.

This script:
1) Loads toydata meta/counts
2) Builds bayesDREAM model with cis_gene=GFI1B
3) Loads existing technical/cis fits if present, otherwise fits and saves
4) Applies NTC-based sum-factor adjustment before cis fit
5) Compares model x_true to an external posterior_samples_cis.pt + meta_cis_GFI1B.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bayesDREAM import bayesDREAM


def resolve_input_path(repo: Path, user_path: str | None, candidates: list[str], label: str) -> Path:
    if user_path:
        p = Path(user_path)
        if not p.is_absolute():
            p = (repo / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"{label} file not found: {p}")
        return p

    for rel in candidates:
        p = (repo / rel).resolve()
        if p.exists():
            return p

    cand = ", ".join(str((repo / c).resolve()) for c in candidates)
    raise FileNotFoundError(
        f"Could not find {label} automatically. Looked for: {cand}. "
        f"Pass --{label}-path explicitly."
    )


def pick_device(device_index: int) -> torch.device:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if 0 <= device_index < n:
            return torch.device(f"cuda:{device_index}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the notebook-equivalent GFI1B workflow")
    parser.add_argument("--repo-root", default=".", help="Path to bayesDREAM repo root")
    parser.add_argument("--label", default="testing_20260224", help="Run label")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--cis-gene", default="GFI1B", help="Cis gene")
    parser.add_argument("--device-index", type=int, default=1, help="Preferred CUDA device index")
    parser.add_argument("--subset-genes", action="store_true", help="Use only GFI1B/MYB/GAPDH genes")
    parser.add_argument("--meta-path", default="", help="Path to metadata CSV (absolute or repo-relative)")
    parser.add_argument("--counts-path", default="", help="Path to counts CSV (absolute or repo-relative)")
    parser.add_argument("--tech-niters", type=int, default=50000, help="Technical fit iterations")
    parser.add_argument("--cis-niters", type=int, default=100000, help="Cis fit iterations")
    parser.add_argument(
        "--predictive-on-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run cis posterior Predictive sampling on CPU (default: True). Use --no-predictive-on-cpu to keep it on GPU.",
    )
    parser.add_argument(
        "--external-run-dir",
        default="GFI1B_run",
        help="Directory with posterior_samples_cis.pt and meta_cis_GFI1B.csv",
    )
    parser.add_argument("--save-plot", default="", help="Optional path to save scatter plot PNG")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    device = pick_device(args.device_index)
    print(f"[INFO] Using device: {device}")

    meta_path = resolve_input_path(
        repo,
        args.meta_path or None,
        ["toydata/cell_meta.csv", "toydata/Josie_cell_meta.csv"],
        "meta",
    )
    counts_path = resolve_input_path(
        repo,
        args.counts_path or None,
        ["toydata/gene_counts.csv"],
        "counts",
    )
    print(f"[INFO] Using meta: {meta_path}")
    print(f"[INFO] Using counts: {counts_path}")

    meta = pd.read_csv(meta_path)
    gene_counts = pd.read_csv(counts_path, index_col=0)
    if args.subset_genes:
        gene_counts = gene_counts.loc[["GFI1B", "MYB", "GAPDH"], :]

    model = bayesDREAM(
        meta=meta,
        counts=gene_counts,
        cis_gene=args.cis_gene,
        output_dir=args.output_dir,
        label=args.label,
        guide_covariates=["cell_line"],
        device=device,
    )
    model.set_technical_groups(["cell_line"])

    run_dir = Path(model.output_dir) / model.label
    tech_fit_path = run_dir / "posterior_samples_technical_gene.pt"
    if tech_fit_path.exists():
        print("[INFO] Loading existing technical fit...")
        model.load_ntc_fit()
    else:
        print("[INFO] Running technical fit (this may take a while)...")
        model.fit_ntc(tolerance=0, niters=args.tech_niters)
        model.save_ntc_fit()

    if "adjustment_factor" in model.meta.columns:
        model.meta["adjustment_factor_old"] = model.meta["adjustment_factor"].copy()
        del model.meta["adjustment_factor"]

    model.adjust_ntc_sum_factor(covariates=["lane", "cell_line"])

    cis_fit_path = run_dir / "x_true.pt"
    if cis_fit_path.exists():
        print("[INFO] Loading existing cis fit...")
        model.load_cis_fit()
    else:
        print("[INFO] Running cis fit (this may take a while)...")
        model.fit_cis(
            sum_factor_col="sum_factor_adj",
            tolerance=0,
            niters=args.cis_niters,
            predictive_on_cpu=args.predictive_on_cpu,
        )
        model.save_cis_fit()

    external = (repo / args.external_run_dir).resolve()
    posterior_path = external / "posterior_samples_cis.pt"
    external_meta_path = external / f"meta_cis_{model.cis_gene}.csv"

    if not posterior_path.exists() or not external_meta_path.exists():
        print("[WARN] External posterior comparison files not found; skipping comparison.")
        print(f"[WARN] Expected: {posterior_path}")
        print(f"[WARN] Expected: {external_meta_path}")
        return

    cis_posterior = torch.load(posterior_path, map_location="cpu")
    external_meta = pd.read_csv(external_meta_path)

    x_post_reordered = (
        pd.Series(cis_posterior["x_true"].mean(dim=0).numpy(), index=external_meta["cell"])
        .loc[model.meta["cell"]]
        .to_numpy()
    )

    x = model.x_true.mean(dim=0).log2().cpu().numpy()
    y = np.log2(x_post_reordered)
    corr = float(np.corrcoef(x, y)[0, 1])

    print("[RESULT] Comparison complete")
    print(f"[RESULT] n_cells={x.shape[0]}")
    print(f"[RESULT] Pearson corr(log2 model.x_true, log2 external x_true)={corr:.6f}")

    if args.save_plot:
        import matplotlib.pyplot as plt

        cell_line = model.meta["cell_line"].to_numpy()
        cell_lines = np.unique(cell_line)
        cmap = plt.get_cmap("tab10")
        colors = {cl: cmap(i) for i, cl in enumerate(cell_lines)}

        plt.figure(figsize=(4, 4))
        for cl in cell_lines:
            mask = cell_line == cl
            plt.scatter(x[mask], y[mask], s=10, alpha=0.4, label=cl, color=colors[cl])

        plt.xlabel("log2(model.x_true)")
        plt.ylabel("log2(posterior x_true)")
        plt.legend(title="cell_line", frameon=False)
        plt.tight_layout()
        out = Path(args.save_plot).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        print(f"[RESULT] Saved plot: {out}")


if __name__ == "__main__":
    main()
