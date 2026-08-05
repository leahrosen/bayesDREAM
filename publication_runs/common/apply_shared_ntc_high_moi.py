"""
Workaround: reuse one shared high-MOI fit_ntc() run across many cis genes,
even though ``add_cis_gene()`` (the mechanism that normally makes this
possible, see bayesDREAM_forClaude/CLAUDE.md "Deferred Cis-Gene Workflow")
explicitly raises ``ValueError`` in high-MOI mode:

    bayesDREAM/core.py:181-186 -- "cis_gene must be provided at
    initialization time in high-MOI mode. Deferred cis-gene specification
    via add_cis_gene() is not supported for high-MOI models."

This matters for Morris (high-MOI): fit_ntc on ~hundreds of genes
individually is exactly what the shared-fit_ntc pattern exists to avoid.

THIS IS A USERLAND WORKAROUND, NOT A CORE BAYESDREAM CHANGE. It reproduces,
using only the public save/load API plus one direct read of the saved
posterior tensor, what ``add_cis_gene()``'s internal
``_extract_cis_alpha_from_ntc_posteriors()`` does for the low-MOI case
(bayesDREAM/model.py:571-601: ``alpha_x_prefit = median of the cis gene's
own alpha_y_mult samples``). No bayesDREAM source was modified to build
this -- per project convention (see CLAUDE.md), core code changes need
explicit sign-off, and this doesn't need one since it's entirely outside the
package.

VALIDATE BEFORE TRUSTING AT SCALE (once, not per gene): pick one of the 5
"primary" Morris genes that gets fully fit_ntc'd normally as part of its own
dedicated run, and *also* run it through this shared-NTC path. Compare the
two genes' alpha_x_prefit / trans-partner alpha_y_prefit -- they should
match closely (not identically -- the shared run's NTC cell composition
differs slightly by whatever guides were subsetted per-gene). If they
diverge a lot, this workaround's assumptions don't hold and add_cis_gene()
for high-MOI needs an actual upstream fix instead. See morris/README.md.

How it works
------------
1. A separate, one-off "shared NTC" model is built with the FULL gene panel
   as its primary modality and cis_gene set to some placeholder gene that is
   NOT among the genes you actually need fit -- see morris/config.yaml's
   ``ntc_shared.placeholder_cis_gene``. fit_ntc() is run once on it and
   saved via save_ntc_fit() to ``ntc_shared_dir``.
2. For each real target gene, a fresh high-MOI model is built the normal
   way with ``cis_gene=<target gene>`` (required at init). This function is
   called on it BEFORE fit_cis():
   - ``model.load_ntc_fit(input_dir=ntc_shared_dir, modalities=[primary],
     mask_features=True)`` populates the primary modality's alpha_y_prefit
     for every OTHER gene via the existing, well-tested name-based feature
     alignment (bayesDREAM/io/load.py's ``_align_posterior_features``) --
     this part is NOT a workaround, it's the documented public API used as
     intended.
   - The target gene's OWN alpha (needed as alpha_x for the 'cis' modality,
     which load_ntc_fit can't populate since the target gene isn't part of
     the current model's primary modality) is read directly from the shared
     run's raw ``posterior_samples_ntc_<modality>.pt`` file and set via the
     public ``model.set_alpha_x()``.

Usage (as a library call from a dataset's generate_slurm.py-generated task,
or directly)::

    from apply_shared_ntc_high_moi import apply_shared_ntc

    model = bayesDREAM(meta=meta, counts=counts, cis_gene='GFI1B',
                        guide_assignment=..., guide_meta=..., ...)
    model.set_technical_groups(['cell_line'])
    apply_shared_ntc(model, ntc_shared_dir='.../ntc_shared', modality_name='gene')
    model.fit_cis(sum_factor_col='sum_factor')
"""

import os

import torch


def apply_shared_ntc(model, ntc_shared_dir: str, modality_name: str = None, verbose: bool = True) -> None:
    """Populate a freshly-built high-MOI model's alpha_x_prefit and primary
    modality alpha_y_prefit from a shared fit_ntc() run, without calling
    add_cis_gene() (which is unavailable in high-MOI mode).

    Must be called after ``model.set_technical_groups(...)`` (so
    ``technical_group_code`` exists -- ``set_alpha_x`` needs it) and before
    ``model.fit_cis()``.

    Parameters
    ----------
    model : bayesDREAM
        A freshly-built high-MOI model with ``cis_gene`` already set at init.
    ntc_shared_dir : str
        Directory the shared model's ``save_ntc_fit()`` wrote to.
    modality_name : str, optional
        Defaults to ``model.primary_modality`` (usually 'gene').
    """
    modality_name = modality_name or model.primary_modality

    if "technical_group_code" not in model.meta.columns:
        raise ValueError(
            "apply_shared_ntc: call model.set_technical_groups(...) first -- "
            "set_alpha_x() requires technical_group_code to already exist."
        )
    if model.cis_gene is None:
        raise ValueError("apply_shared_ntc: model.cis_gene must be set (high-MOI requires this at init).")

    # ---- 1. trans-partner alphas for every gene EXCEPT the focal one: -------
    # this is the documented public API, not part of the workaround.
    model.load_ntc_fit(input_dir=ntc_shared_dir, modalities=[modality_name], mask_features=True)

    # ---- 2. the focal gene's own alpha (-> alpha_x), read directly: ---------
    posterior_path = os.path.join(ntc_shared_dir, f"posterior_samples_ntc_{modality_name}.pt")
    if not os.path.exists(posterior_path):
        raise FileNotFoundError(
            f"apply_shared_ntc: {posterior_path} not found. Did the shared NTC "
            f"run's save_ntc_fit() write to ntc_shared_dir={ntc_shared_dir!r}?"
        )
    saved = torch.load(posterior_path, map_location=model.device)
    saved_feature_names = saved.get("feature_names")
    if saved_feature_names is None:
        raise ValueError(f"apply_shared_ntc: {posterior_path} has no 'feature_names' -- old save format?")
    if model.cis_gene not in saved_feature_names:
        raise ValueError(
            f"apply_shared_ntc: cis_gene '{model.cis_gene}' not found in the shared "
            f"NTC run's feature_names. It must not be the shared run's own "
            f"placeholder_cis_gene, and must be present in the shared run's gene panel."
        )
    idx = saved_feature_names.index(model.cis_gene)

    ps = saved["posterior_samples"]
    alpha_y_mult = ps.get("alpha_y_mult", ps.get("alpha_y"))
    if alpha_y_mult is None:
        raise ValueError(
            f"apply_shared_ntc: {posterior_path} has neither 'alpha_y_mult' nor "
            f"'alpha_y' in posterior_samples -- expected for a negbinom primary modality."
        )
    alpha_x_focal = alpha_y_mult[..., idx].median(dim=0).values  # shape [C], mirrors
    # bayesDREAM/model.py:_extract_cis_alpha_from_ntc_posteriors's own computation.

    model.set_alpha_x(alpha_x_focal)

    if verbose:
        print(f"[apply_shared_ntc] {model.cis_gene}: alpha_x_prefit set from shared NTC "
              f"run ({posterior_path}, feature index {idx}); {modality_name} modality "
              f"alpha_y_prefit loaded for other genes via load_ntc_fit(mask_features=True).")
