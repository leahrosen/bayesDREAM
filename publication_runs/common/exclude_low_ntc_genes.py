"""
Exclude genes with log2(mu_ntc) < threshold from trans fitting.

There is no `exclude`/`feature_subset` kwarg on fit_trans() (checked against
bayesDREAM/fitting/trans.py's full signature). The only supported way to
drop features before fit_trans() is to build a subsetted Modality via
Modality.get_feature_subset() and manually re-attach posterior_samples_ntc/
alpha_y_prefit_* -- which is exactly what bayesDREAM's own internal
add_cis_gene() -> _refilter_zero_count_features() does
(bayesDREAM/model.py:651-697) for a different reason (dropping zero-count
features after cell subsetting). This module reuses those SAME internal
methods (_trim_feature_axis_in_posteriors, get_feature_subset) rather than
reimplementing the tensor-trimming logic -- same reuse principle as
apply_shared_ntc_high_moi.py, same caveat: `_trim_feature_axis_in_posteriors`
is a private (underscore) method, called here as a stable, already-exercised
internal utility rather than something re-derived from scratch.

Call AFTER load_ntc_fit() (needs posterior_samples_ntc['mu_ntc']) and BEFORE
fit_trans(). Applies to the primary 'gene' modality only, per instruction
("genes with log2(mu_ntc) < -4") -- not extended to non-gene modalities
(splicing junctions, donor/acceptor sites, ...) since those aren't "genes"
and mu_ntc's interpretation there may differ; ask before extending scope.
"""

import numpy as np
import torch


def exclude_low_ntc_genes(model, modality_name: str = None, threshold: float = -4.0, verbose: bool = True) -> None:
    modality_name = modality_name or model.primary_modality
    mod = model.get_modality(modality_name)

    ps = mod.posterior_samples_ntc
    if ps is None or "mu_ntc" not in ps:
        raise ValueError(
            f"exclude_low_ntc_genes: modality '{modality_name}' has no "
            f"posterior_samples_ntc['mu_ntc'] -- call load_ntc_fit() first."
        )

    mu_ntc = ps["mu_ntc"]
    mu_ntc_np = mu_ntc.detach().cpu().numpy() if isinstance(mu_ntc, torch.Tensor) else np.asarray(mu_ntc)
    if mu_ntc_np.ndim > 1:
        mu_ntc_np = np.median(mu_ntc_np, axis=0)  # collapse sample dim if present, e.g. [S, T] -> [T]

    T = mod.dims["n_features"]
    if mu_ntc_np.shape[-1] != T:
        raise ValueError(
            f"exclude_low_ntc_genes: mu_ntc has {mu_ntc_np.shape[-1]} entries but "
            f"modality '{modality_name}' has {T} features -- posterior/modality out of sync."
        )

    log2_mu = np.log2(mu_ntc_np)
    keep_idx = np.where(log2_mu >= threshold)[0]
    n_dropped = T - len(keep_idx)

    if n_dropped == 0:
        if verbose:
            print(f"[exclude_low_ntc_genes] {modality_name}: no genes below log2(mu_ntc) < {threshold}")
        return

    # Reuses bayesDREAM's own internal trim/subset machinery -- see module docstring.
    model._trim_feature_axis_in_posteriors(mod, keep_idx, T)
    new_mod = mod.get_feature_subset(keep_idx.tolist())
    new_mod.posterior_samples_ntc = mod.posterior_samples_ntc  # already trimmed, in place
    new_mod.alpha_y_prefit_mult = mod.alpha_y_prefit_mult
    new_mod.alpha_y_prefit_add = mod.alpha_y_prefit_add
    if hasattr(mod, "loss_ntc"):
        new_mod.loss_ntc = mod.loss_ntc

    model.modalities[modality_name] = new_mod

    if verbose:
        print(f"[exclude_low_ntc_genes] {modality_name}: dropped {n_dropped}/{T} genes "
              f"with log2(mu_ntc) < {threshold} ({len(keep_idx)} remain)")
