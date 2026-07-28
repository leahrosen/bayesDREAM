"""
Fit the recovery model for one simulated cell-design scenario/replicate, following
bayesDREAM's documented full workflow (docs/FIT_TRANS_GUIDE.md "Complete Workflow
Example"): fit_ntc -> adjust_ntc_sum_factor -> fit_cis -> refit_sumfactor ->
fit_trans(single_hill). See docs/SIMULATION_STUDY_PLAN.md §7.

niters/nsamples are intentionally NOT exposed here -- every fit_ntc/fit_cis/fit_trans
call uses bayesDREAM's own library defaults, deliberately: the point of this study is
to test whether the default settings recover known parameters, not some other,
study-specific number of iterations.

Reuses the scenario's own seed (recorded in config.json by simulate_scenario.py) for
numpy/torch/pyro, so a rerun of a given scenario+replicate is deterministic end to end
(plan §6).

Usage (single task):
    python run_recovery_fit.py --scenario_dir ./sim_study_out/data/scenario_0/rep_0

Usage (SLURM array over design_matrix rows; resolves scenario_dir from
--data_root/--design_matrix/$SLURM_ARRAY_TASK_ID):
    python run_recovery_fit.py --data_root ./sim_study_out/data \
        --design_matrix ./sim_study_out/design_matrix.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import pyro
import torch

from bayesDREAM import bayesDREAM


def resolve_scenario_dir(data_root: str, design_matrix_path: str, row_index: int) -> str:
    design_matrix = pd.read_csv(design_matrix_path)
    row = design_matrix.loc[design_matrix['row_index'] == row_index].iloc[0]
    return os.path.join(
        data_root, f"scenario_{int(row['scenario_id'])}", f"rep_{int(row['replicate_id'])}",
    )


def run_recovery_fit(
    scenario_dir: str,
    device: str = None,
) -> bayesDREAM:
    with open(os.path.join(scenario_dir, 'config.json')) as f:
        config = json.load(f)
    seed = config['seed']

    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    meta = pd.read_csv(os.path.join(scenario_dir, 'meta.csv'))
    counts = pd.read_csv(os.path.join(scenario_dir, 'counts.csv'), index_col=0)

    fit_dir = os.path.join(scenario_dir, 'fit')
    os.makedirs(fit_dir, exist_ok=True)

    model = bayesDREAM(
        meta=meta,
        counts=counts,
        cis_gene=config['cis_gene_name'],
        output_dir=fit_dir,
        label='recovery',
        device=device,
        random_seed=seed,
    )

    # single technical group (plan §2 / §4.1): C=1 is an explicit no-group-effect
    # code path in fit_ntc/fit_cis/fit_trans, not a degenerate/unsupported case.
    model.set_technical_groups(['cell_line'])
    model.fit_ntc(sum_factor_col='sum_factor')

    # meta's 'sum_factor' is scran-recomputed from realized counts (see
    # simulate_scenario.py), so guide identity is genuinely correlated with it (strong
    # cis perturbations shift library composition) -- follow the documented full
    # workflow (docs/FIT_TRANS_GUIDE.md) rather than feeding raw sum_factor straight
    # into fit_cis/fit_trans.
    model.adjust_ntc_sum_factor(
        sum_factor_col_old='sum_factor',
        sum_factor_col_adj='sum_factor_adj',
        covariates=['cell_line'],
    )
    model.fit_cis(sum_factor_col='sum_factor_adj')

    model.refit_sumfactor(
        sum_factor_col_old='sum_factor_adj',
        sum_factor_col_refit='sum_factor_refit',
        covariates=['cell_line'],
    )
    model.fit_trans(
        sum_factor_col='sum_factor_refit',
        function_type='single_hill',
    )

    model.save_ntc_fit()
    model.save_cis_fit()
    model.save_trans_fit()
    model.save_trans_summary(fdr_threshold=0.05)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario_dir', default=None,
                         help="Directory written by simulate_scenario.py for one "
                              "scenario/replicate. Mutually exclusive with "
                              "--data_root/--design_matrix/--row_index.")
    parser.add_argument('--data_root', default=None)
    parser.add_argument('--design_matrix', default=None)
    parser.add_argument('--row_index', type=int, default=None,
                         help="Defaults to $SLURM_ARRAY_TASK_ID when using "
                              "--data_root/--design_matrix.")
    parser.add_argument('--device', default=None,
                         help="'cpu' or 'cuda'. Defaults to bayesDREAM's own "
                              "auto-detection (cuda if available, else cpu) — pass "
                              "explicitly to force one or the other.")
    args = parser.parse_args()

    if args.scenario_dir is not None:
        scenario_dir = args.scenario_dir
    else:
        row_index = args.row_index
        if row_index is None:
            row_index = int(os.environ['SLURM_ARRAY_TASK_ID'])
        scenario_dir = resolve_scenario_dir(args.data_root, args.design_matrix, row_index)

    run_recovery_fit(scenario_dir=scenario_dir, device=args.device)
    print(f"Done: {scenario_dir}")
