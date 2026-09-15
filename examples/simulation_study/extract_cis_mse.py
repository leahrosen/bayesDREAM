"""
Extract per-scenario/replicate fit_cis accuracy (MSE of log2(x_true), fitted vs true)
across the full simulation study, for evaluation plotting (docs/SIMULATION_STUDY_PLAN.md
§8). Writes one row per scenario/replicate to cis_eval.csv.

Must run where torch is available (the bayesdream conda env) -- log2_x_true.pt and
posterior_samples_cis.pt are plain torch.save() files, not readable from R directly.
See examples/simulation_study/extract_trans_eval.R for the (pure-R, no torch needed)
trans-side extraction, and plot_evaluation.R for the plots that consume both outputs.

Why cell_names come from posterior_samples_cis.pt, not positional alignment:
log2_x_true.pt is a bare tensor with no embedded cell labels. It's *very likely* in the
same cell order as cis_ground_truth.csv (both ultimately derive from the same per-scenario
`cells` list), but that isn't independently guaranteed for log2_x_true.pt alone -- only
posterior_samples_cis.pt carries an explicit 'cell_names' list (bayesDREAM/io/save.py),
saved from the same fit_cis() call as log2_x_true.pt in the same cell order. Joining by
that explicit name list, rather than trusting positional alignment across two independently
loaded files, avoids a silent misalignment bug that would corrupt every MSE value without
raising an error.

Usage:
    python extract_cis_mse.py --design_matrix $OUT/design_matrix.csv --data_root $DATA \
        --outfile $OUT/cis_eval.csv
"""

import argparse
import os
import sys

import pandas as pd
import torch


def _scenario_dir(data_root: str, scenario_id: int, replicate_id: int) -> str:
    return os.path.join(data_root, f"scenario_{scenario_id}", f"rep_{replicate_id}")


def compute_one(scenario_dir: str):
    """Returns (mse, n_cells_compared) or (None, reason_str) if this scenario/replicate
    isn't ready yet (fit incomplete) or its files don't match expectations."""
    recovery_dir = os.path.join(scenario_dir, 'fit', 'recovery')
    log2_x_true_path = os.path.join(recovery_dir, 'log2_x_true.pt')
    posterior_cis_path = os.path.join(recovery_dir, 'posterior_samples_cis.pt')
    ground_truth_path = os.path.join(scenario_dir, 'cis_ground_truth.csv')

    for p in (log2_x_true_path, posterior_cis_path, ground_truth_path):
        if not os.path.exists(p):
            return None, f"missing {os.path.basename(p)}"

    # weights_only=False (explicit, not just the pre-2.6 default) to match how the rest
    # of bayesDREAM loads these self-generated checkpoints (e.g. bayesDREAM/fitting/
    # trans.py, bayesDREAM/io/load.py) and to silence torch's FutureWarning -- these are
    # files this same pipeline wrote, not untrusted third-party data.
    log2_x_true_fitted = torch.load(log2_x_true_path, map_location='cpu', weights_only=False)
    if hasattr(log2_x_true_fitted, 'numpy'):
        log2_x_true_fitted = log2_x_true_fitted.numpy()

    posterior_cis = torch.load(posterior_cis_path, map_location='cpu', weights_only=False)
    if 'cell_names' not in posterior_cis:
        return None, "posterior_samples_cis.pt has no 'cell_names' key"
    cell_names = list(posterior_cis['cell_names'])

    if len(cell_names) != len(log2_x_true_fitted):
        return None, (f"length mismatch: {len(cell_names)} cell_names vs "
                       f"{len(log2_x_true_fitted)} log2_x_true values")

    fitted_df = pd.DataFrame({'cell': cell_names, 'log2_x_true_fitted': log2_x_true_fitted})

    truth_df = pd.read_csv(ground_truth_path)
    if 'cell' not in truth_df.columns or 'log2_x_true' not in truth_df.columns:
        return None, "cis_ground_truth.csv missing 'cell' or 'log2_x_true' column"
    truth_df = truth_df[['cell', 'log2_x_true']].rename(
        columns={'log2_x_true': 'log2_x_true_true'})

    merged = fitted_df.merge(truth_df, on='cell', how='inner')
    if len(merged) == 0:
        return None, "no matching cell names between fit and ground truth"
    if len(merged) < len(fitted_df):
        # Not fatal -- report what we found, but flag the mismatch count for visibility.
        print(f"[WARN] {scenario_dir}: only {len(merged)}/{len(fitted_df)} cells matched "
              f"by name between log2_x_true.pt and cis_ground_truth.csv", file=sys.stderr)

    mse = float(((merged['log2_x_true_fitted'] - merged['log2_x_true_true']) ** 2).mean())
    return mse, len(merged)


def main(design_matrix_path: str, data_root: str, outfile: str):
    design_matrix = pd.read_csv(design_matrix_path)
    design_cols = ['scenario_id', 'replicate_id', 'cells_per_gene', 'n_guides',
                   'guide_shape', 'sigma_eff', 'log2_X_NTC', 'log2_o_x', 'seed']
    design_cols = [c for c in design_cols if c in design_matrix.columns]

    rows = []
    n_ok, n_skipped = 0, 0
    for _, row in design_matrix.iterrows():
        scenario_id = int(row['scenario_id'])
        replicate_id = int(row['replicate_id'])
        scenario_dir = _scenario_dir(data_root, scenario_id, replicate_id)
        mse, extra = compute_one(scenario_dir)
        out_row = {c: row[c] for c in design_cols}
        if mse is None:
            n_skipped += 1
            out_row['mse_log2_x_true'] = None
            out_row['n_cells_compared'] = None
            out_row['skip_reason'] = extra
        else:
            n_ok += 1
            out_row['mse_log2_x_true'] = mse
            out_row['n_cells_compared'] = extra
            out_row['skip_reason'] = None
        rows.append(out_row)

    result = pd.DataFrame(rows)
    result.to_csv(outfile, index=False)
    print(f"Wrote {len(result)} rows to {outfile} ({n_ok} with a computed MSE, "
          f"{n_skipped} skipped -- see skip_reason column for why).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design_matrix', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--outfile', required=True)
    args = parser.parse_args()
    main(args.design_matrix, args.data_root, args.outfile)
