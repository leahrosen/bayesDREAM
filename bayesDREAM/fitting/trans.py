"""
Trans effects fitting for bayesDREAM.

This module contains the trans model and fitting logic.
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
# Pre-import torch._dynamo to avoid circular import issues with OneCycleLR scheduler
# This is a workaround for a known PyTorch issue in some versions/builds (especially ROCm)
try:
    import torch._dynamo
except (ImportError, AttributeError):
    pass  # Older PyTorch versions may not have _dynamo
import pyro
import pyro.distributions as dist
from pyro.distributions.transforms import iterated, affine_autoregressive
import pyro.optim as optim
import pyro.infer as infer
import pyro.poutine as poutine
import pyro.distributions as dist

from ..utils import find_beta, Hill_based_positive, Hill_based_positive_logK, Polynomial_function, check_tensor


def _soft_clamp(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """
    Soft (differentiable) clamp to the interval (lo, hi) using tanh.

    Replaces torch.clamp, which has zero gradient outside [lo, hi] (dead-gradient
    problem: once n_raw overshoots the boundary, the optimizer has no signal to pull
    it back).

    Mapping:  x_clamped = center + half * tanh(x / half)
      where center = (hi + lo) / 2  and  half = (hi - lo) / 2

    Properties:
    - Linear (identity) for |x - center| << half  →  prior unchanged in typical range
    - Gradient = sech²(x/half) ≥ sech²(1) ≈ 0.42 at the "boundary" (|x| = half)
    - Gradient never reaches zero for finite x  →  no dead gradient
    - Asymptotes to ±half as |x| → ∞
    """
    half = 0.5 * (hi - lo)
    center = 0.5 * (hi + lo)
    return center + half * torch.tanh(x / half)


class TransFitter:
    """Handles trans effects fitting."""

    def _save_checkpoint_atomic(self, checkpoint_path: str, data: dict) -> None:
        """Write checkpoint atomically (tmp → rename) and keep previous as .bak.

        Three-step protocol:
          1. torch.save  →  <path>.tmp        (new data, not yet visible)
          2. os.replace  old <path>  →  .bak  (old data safely preserved)
          3. os.replace  .tmp  →  <path>      (atomic on POSIX; new data visible)

        Failure modes:
          Die in step 1 (during write): .tmp corrupt, <path> unchanged → resume loads <path>.
          Die between 1 and 2: .tmp complete, <path> unchanged → resume loads <path>.
          Die between 2 and 3: .tmp complete, <path> gone (→ .bak) → resume tries .tmp
                                first (complete), falls back to .bak (one interval behind).
          Die during step 3:  os.replace is atomic on POSIX → either completes or not,
                              leaving either <path> or .tmp visible; .bak always intact.
        """
        backup_path = checkpoint_path + '.bak'
        tmp_path = checkpoint_path + '.tmp'
        torch.save(data, tmp_path)
        if os.path.exists(checkpoint_path):
            os.replace(checkpoint_path, backup_path)
        os.replace(tmp_path, checkpoint_path)

    def __init__(self, model):
        """
        Initialize trans fitter.

        Parameters
        ----------
        model : _BayesDREAMCore
            The parent model instance
        """
        self.model = model

    def _t(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self.model.device)

    def _to_cpu(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        return x

    # ----------------------------
    # Debug helpers (Option 1)
    # ----------------------------
    def _tensor_summary(self, x: torch.Tensor):
        x_det = x.detach()
        finite = torch.isfinite(x_det)
        n = x_det.numel()
        n_bad = (~finite).sum().item()
        n_nan = torch.isnan(x_det).sum().item()
        n_inf = torch.isinf(x_det).sum().item()
        if finite.any():
            x_f = x_det[finite]
            mn = x_f.min().item()
            mx = x_f.max().item()
            mean = x_f.mean().item()
            std = x_f.std().item()
        else:
            mn = mx = mean = std = float("nan")
        return dict(n=n, n_bad=n_bad, n_nan=n_nan, n_inf=n_inf, min=mn, max=mx, mean=mean, std=std)

    def _print_tensor(self, name: str, x: torch.Tensor, prefix=""):
        s = self._tensor_summary(x)
        print(
            f"{prefix}{name}: shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
            f"bad={s['n_bad']}/{s['n']} (nan={s['n_nan']}, inf={s['n_inf']}) "
            f"min={s['min']:.3g} max={s['max']:.3g} mean={s['mean']:.3g} std={s['std']:.3g}"
        )

    def _check_param_store_all(self, step: int, svi=None, where="", prev_finite=None, verbose_on_change=True):
        store = pyro.get_param_store()
        if prev_finite is None:
            prev_finite = {}

        changed = []
        bad_vals = []
        bad_grads = []

        for name, p in store.items():
            if not isinstance(p, torch.Tensor):
                continue

            now_finite = torch.isfinite(p).all().item()
            was_finite = prev_finite.get(name, True)
            prev_finite[name] = now_finite

            if was_finite and (not now_finite):
                changed.append(name)

            if not now_finite:
                bad_vals.append(name)

            if p.grad is not None and (not torch.isfinite(p.grad).all().item()):
                bad_grads.append(name)

        if changed and verbose_on_change:
            print(f"\n[ERROR] Param(s) flipped finite→nonfinite at step {step} {where}: {changed}")
            for n in changed:
                self._print_tensor(n, store[n], prefix="  ")
                if store[n].grad is not None:
                    self._print_tensor(n + ".grad", store[n].grad, prefix="  ")

        if bad_vals and verbose_on_change and not changed:
            print(f"\n[ERROR] Non-finite param value(s) present at step {step} {where}: "
                  f"{bad_vals[:10]}{'...' if len(bad_vals)>10 else ''}")
            for n in bad_vals[:10]:
                self._print_tensor(n, store[n], prefix="  ")

        if bad_grads and verbose_on_change:
            print(f"\n[ERROR] Non-finite gradient(s) present at step {step} {where}: "
                  f"{bad_grads[:10]}{'...' if len(bad_grads)>10 else ''}")
            for n in bad_grads[:10]:
                if store[n].grad is not None:
                    self._print_tensor(n + ".grad", store[n].grad, prefix="  ")

        # Best-effort optimizer-state check (optional)
        if svi is not None and hasattr(svi, "optim"):
            opt = svi.optim
            optim_objs = getattr(opt, "optim_objs", None)
            if isinstance(optim_objs, dict):
                bad_state = []
                for pname, torch_opt in optim_objs.items():
                    base_opt = getattr(torch_opt, "optimizer", torch_opt)
                    if not hasattr(base_opt, "state"):
                        continue
                    for group in base_opt.param_groups:
                        for param in group.get("params", []):
                            st = base_opt.state.get(param, {})
                            for key, val in st.items():
                                if torch.is_tensor(val) and (not torch.isfinite(val).all().item()):
                                    bad_state.append((pname, key))
                if bad_state and verbose_on_change:
                    print(f"\n[ERROR] Non-finite optimizer state at step {step} {where}: "
                          f"{bad_state[:10]}{'...' if len(bad_state)>10 else ''}")

        any_bad = bool(bad_vals or bad_grads)
        return prev_finite, any_bad

    def _collect_dist_tensors(self, fn):
        tensors = []
        if isinstance(fn, dist.TransformedDistribution):
            tensors += self._collect_dist_tensors(fn.base_dist)
            return tensors

        for attr in [
            "loc", "scale", "total_count", "logits", "probs", "rate",
            "concentration", "concentration0", "concentration1", "df"
        ]:
            if hasattr(fn, attr):
                v = getattr(fn, attr)
                if torch.is_tensor(v):
                    tensors.append((attr, v))
        return tensors

    def diagnose_nonfinite_sites(self, model, guide, *args, **kwargs):
        print("\n[DIAG] Running traced guide+model to locate non-finite site...")

        guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
        replayed_model = poutine.replay(model, trace=guide_trace)
        model_trace = poutine.trace(replayed_model).get_trace(*args, **kwargs)

        def scan_trace(tr, label):
            print(f"[DIAG] Scanning {label} trace...")
            for name, node in tr.nodes.items():
                if node.get("type") != "sample":
                    continue

                fn = node["fn"]
                val = node["value"]

                if torch.is_tensor(val) and (not torch.isfinite(val).all().item()):
                    print(f"  [BAD VALUE] site={name}")
                    self._print_tensor(f"{label}:{name}.value", val, prefix="    ")
                    return name

                for attr, ten in self._collect_dist_tensors(fn):
                    if not torch.isfinite(ten).all().item():
                        print(f"  [BAD DIST PARAM] site={name} param={attr}")
                        self._print_tensor(f"{label}:{name}.fn.{attr}", ten, prefix="    ")
                        return name

                if "log_prob" in node:
                    lp = node["log_prob"]
                    if torch.is_tensor(lp) and (not torch.isfinite(lp).all().item()):
                        print(f"  [BAD LOG_PROB] site={name}")
                        self._print_tensor(f"{label}:{name}.log_prob", lp, prefix="    ")
                        return name

            print(f"[DIAG] No non-finite sample sites found in {label} trace.")
            return None

        bad_guide = scan_trace(guide_trace, "guide")
        bad_model = scan_trace(model_trace, "model")
        print(f"[DIAG] First bad site guide={bad_guide} model={bad_model}\n")

    def _debug_svi_step(self, svi, step, prev_finite, *args, **kwargs):
        """
        A 'manual' SVI step that lets us check:
          - pre-step params
          - post-grads, pre-update params/grads
          - post-update params
        and run trace diagnosis on first failure.
        """
        # Pre-step check
        prev_finite, any_bad = self._check_param_store_all(
            step, svi=svi, where="(pre-step)", prev_finite=prev_finite
        )
        if any_bad:
            store = pyro.get_param_store()
            if "locs.log_K_a" in store and store["locs.log_K_a"].grad is not None:
                g = store["locs.log_K_a"].grad
                bad_idx = torch.nonzero(torch.isnan(g), as_tuple=False).squeeze(-1)
                if bad_idx.numel() > 0:
                    self._diagnose_log_Ka_nan_grad(svi, bad_idx, *args, **kwargs)
        
            self.diagnose_nonfinite_sites(svi.model, svi.guide, *args, **kwargs)
            raise FloatingPointError("Non-finite grads/params after loss_and_grads (before update).")


        # Compute loss + grads (no update yet)
        with poutine.trace(param_only=True) as param_capture:
            loss = svi.loss_and_grads(svi.model, svi.guide, *args, **kwargs)

        # Post-grads check (this often catches the real cause)
        prev_finite, any_bad = self._check_param_store_all(
            step, svi=svi, where="(post-grads, pre-update)", prev_finite=prev_finite
        )
        if any_bad:
            store = pyro.get_param_store()
            if "locs.log_K_a" in store and store["locs.log_K_a"].grad is not None:
                g = store["locs.log_K_a"].grad
                bad_idx = torch.nonzero(torch.isnan(g), as_tuple=False).squeeze(-1)
                if bad_idx.numel() > 0:
                    self._diagnose_log_Ka_nan_grad(svi, bad_idx, *args, **kwargs)
            self.diagnose_nonfinite_sites(svi.model, svi.guide, *args, **kwargs)
            # Fitting will stop: FloatingPointError is caught in the main loop → [STOP] + break
            raise FloatingPointError("Non-finite grads/params after loss_and_grads (before update).")

        # Apply optimizer update
        params = set(site["value"].unconstrained() for site in param_capture.trace.nodes.values())
        svi.optim(params)

        # Post-update check
        prev_finite, any_bad = self._check_param_store_all(
            step, svi=svi, where="(post-update)", prev_finite=prev_finite
        )
        if any_bad:
            store = pyro.get_param_store()
            if "locs.log_K_a" in store and store["locs.log_K_a"].grad is not None:
                g = store["locs.log_K_a"].grad
                bad_idx = torch.nonzero(torch.isnan(g), as_tuple=False).squeeze(-1)
                if bad_idx.numel() > 0:
                    self._diagnose_log_Ka_nan_grad(svi, bad_idx, *args, **kwargs)
            self.diagnose_nonfinite_sites(svi.model, svi.guide, *args, **kwargs)
            # Fitting will stop: FloatingPointError is caught in the main loop → [STOP] + break
            raise FloatingPointError("Optimizer update produced non-finite params.")

        return loss, prev_finite

    def _diagnose_log_Ka_nan_grad(self, svi, bad_idx, *args, **kwargs):
        """
        bad_idx: 1D LongTensor of indices where locs.log_K_a.grad is NaN
        Prints model/guide site values for those indices and derived quantities
        (phi, Hill logit range, NB logits range).
        """
        import pyro.poutine as poutine
        import torch
    
        print(f"\n[DIAG+] NaN grad indices for log_K_a: {bad_idx.tolist()}")
    
        guide_trace = poutine.trace(svi.guide).get_trace(*args, **kwargs)
        replayed_model = poutine.replay(svi.model, trace=guide_trace)
        model_trace = poutine.trace(replayed_model).get_trace(*args, **kwargs)
    
        def get_site(tr, name):
            return tr.nodes[name]["value"] if name in tr.nodes else None
    
        # --- pull site values (all should be finite, but may be extreme) ---
        def _first_nonnone(*vals):
            return next((v for v in vals if v is not None), None)
        log_K_a = _first_nonnone(get_site(model_trace, "log_K_a"), get_site(guide_trace, "log_K_a"))
        log_Vmax_a = _first_nonnone(get_site(model_trace, "log_Vmax_a"), get_site(guide_trace, "log_Vmax_a"))
        Vmax_a = get_site(model_trace, "Vmax_a")
        K_a = get_site(model_trace, "K_a")
        n_a = get_site(model_trace, "n_a")
        A = get_site(model_trace, "A")
        o_y = get_site(model_trace, "o_y")  # sampled per feature
        alpha = get_site(model_trace, "alpha")  # relaxed bernoulli
    
        # Some of these might not exist depending on branch / naming
        for name, tensor in [
            ("log_K_a", log_K_a),
            ("K_a", K_a),
            ("log_Vmax_a", log_Vmax_a),
            ("Vmax_a", Vmax_a),
            ("n_a", n_a),
            ("A", A),
            ("o_y", o_y),
            ("alpha", alpha),
        ]:
            if tensor is None:
                continue
            sel = tensor[bad_idx]
            self._print_tensor(f"{name}[bad_idx]", sel, prefix="  ")
    
        # --- derived phi and ranges (very informative) ---
        if o_y is not None:
            phi = 1.0 / (o_y ** 2)
            self._print_tensor("phi_y[bad_idx]", phi[bad_idx], prefix="  ")
    
        # --- derived Hill logit range (this catches x**n instabilities) ---
        # kwargs must include x_true_sample, and we prefer to use log_K_a directly
        x_true = kwargs.get("x_true_sample", None)
        if x_true is not None and log_K_a is not None and n_a is not None:
            tiny = torch.finfo(x_true.dtype).tiny
            log_x = torch.log(x_true.clamp_min(tiny)).unsqueeze(-1)  # [N,1]
            # logit for each bad feature: [N, |bad_idx|]
            logit = (n_a[bad_idx].unsqueeze(0) * (log_x - log_K_a[bad_idx].unsqueeze(0)))
            # print range across cells (per bad feature)
            logit_min = logit.min(dim=0).values
            logit_max = logit.max(dim=0).values
            self._print_tensor("hill_logit_min[bad_idx]", logit_min, prefix="  ")
            self._print_tensor("hill_logit_max[bad_idx]", logit_max, prefix="  ")
    
            # approximate Hill output range
            if Vmax_a is not None:
                H = Vmax_a[bad_idx].unsqueeze(0) * torch.sigmoid(logit)
                self._print_tensor("Hilla_min[bad_idx]", H.min(dim=0).values, prefix="  ")
                self._print_tensor("Hilla_max[bad_idx]", H.max(dim=0).values, prefix="  ")
    
        # --- derived NB logits range for bad features (if you pass mu_y pieces) ---
        # We can’t see mu_final unless you expose it; but we can at least compute the part driven by Hill:
        # If you want full mu_final, expose it via pyro.deterministic (recommended).
        print("[DIAG+] If hill_logit range is extreme (e.g. > ~80 in magnitude), Hill math/backward is a prime suspect.\n")

    
    #########################################
    ## Step 3: Fit trans effects (model_y) ##
    #########################################
    def _model_y(
        self,
        N,
        T,
        y_obs_tensor,
        sum_factor_tensor,
        beta_o_alpha_tensor,
        beta_o_beta_tensor,
        alpha_alpha_mu_tensor,
        K_max_tensor,
        K_alpha_tensor,
        Vmax_mean_tensor,
        Vmax_alpha_tensor,
        n_mu_tensor,
        Amean_tensor,
        p_n_logits_tensor,
        epsilon_tensor,
        x_true_sample,
        log2_x_true_sample,
        nmin,
        nmax,
        alpha_y_sample=None,
        C=None,
        groups_tensor=None,
        temperature=1.0,
        use_straight_through=False,
        function_type='single_hill',
        polynomial_degree=6,
        use_alpha=True,
        distribution='negbinom',
        denominator_tensor=None,
        K=None,
        D=None,
        mean_within_guide_var=None,
        x_true_CV=None,
        x_ntc_mean=None,
        use_data_driven_priors=True,
        use_epsilon=True,
        vmax_log_sigma_floor_tensor=None,
        k_log_sigma_min_tensor=None,
        k_center_tensor=None,
        y_ntc_tensor=None,
        mean_y_corrected_tensor=None,
        o_y_ntc_tensor=None,
        alpha_n_coupling: float = 10.0,
        latents_only=False,
    ):

        if use_alpha:
            alpha_dist = dist.RelaxedBernoulliStraightThrough if use_straight_through else dist.RelaxedBernoulli

        trans_plate = pyro.plate("trans_plate", T, dim=-1)
        
        ##########
        ## x_true (Now Fully Observed) ##
        ##########
        #x_true = pyro.deterministic("x_true", x_true_sample)
        x_true = x_true_sample
    
        ####################
        ## alpha_y ##
        ####################
        alpha_y = None
        if alpha_y_sample is not None:
            alpha_y = alpha_y_sample
        elif groups_tensor is not None:
            with pyro.plate("technical_groups_plate", C-1, dim=-2):  # **Now correctly uses C-1**
                with trans_plate:
                    alpha_alpha = pyro.sample("alpha_alpha", dist.Exponential(1 / alpha_alpha_mu_tensor))  # shape = [C-1, T]
                    alpha_mu = pyro.sample("alpha_mu", dist.Gamma(1, 1))  # shape = [C-1, T]
                    alpha_y = pyro.sample("alpha_y", dist.Gamma(alpha_alpha, alpha_alpha / alpha_mu))  # shape = [C-1, T]
    
        ####################
        ## Overdispersion ##
        ####################
        beta_o = pyro.sample("beta_o", dist.Gamma(beta_o_alpha_tensor, beta_o_beta_tensor))
        with trans_plate:
            o_y = pyro.sample("o_y", dist.Exponential(beta_o))
            phi_y = 1 / (o_y**2)
        phi_y_used = phi_y.unsqueeze(-2)

        # Per-gene overdispersion weight for the adaptive A prior.
        # weight → 1 for noisier genes (o_y >> prior mean), → 0 for quieter genes.
        # If o_y_ntc_tensor is provided (from fit_ntc), use it as a fixed pre-fit weight
        # rather than the sampled o_y — this avoids the unidentifiability of o_y during
        # fit_trans (posterior stays near prior mean for all genes, making the weight
        # uninformative). The NTC-estimated o_y reflects true gene-level noisiness.
        _prior_mean_o_y = (beta_o_beta_tensor / beta_o_alpha_tensor).clamp_min(epsilon_tensor)
        if o_y_ntc_tensor is not None:
            _o_y_weight = o_y_ntc_tensor / (o_y_ntc_tensor + _prior_mean_o_y)  # [T], fixed
        else:
            _o_y_weight = o_y / (o_y + _prior_mean_o_y)  # [T], sampled (fallback)

        # Degrees of freedom for Student-t distribution (nu_y)
        # Only needed if using studentt distribution
        nu_y = None
        if distribution == 'studentt':
            # Two options (must match fit_ntc choice):
            # Option 1: Fixed value (simpler, faster) - COMMENTED OUT FOR NOW
            # nu_y = self._t(3.0)
            # Option 2: Sample per-feature (more flexible, slower) - ACTIVE
            with trans_plate:
                nu_y = pyro.sample("nu_y", dist.Gamma(self._t(10.0), self._t(2.0)))  # mean~5, ensures df>2
    
        #################
        ## Hill-based: ##
        #################
        if function_type in ['polynomial']:
            #sigma_coeff = pyro.sample("sigma_coeff", dist.Exponential(100)) #   -> controls how variable n_a can be across genes
            sigma_coeff = pyro.sample("sigma_coeff", dist.HalfCauchy(scale=self._t(1.0)))
        
        # Now enter the trans_plate (T dimension)
        with trans_plate:

            # For multinomial, Amean_tensor and Vmax_mean_tensor are already [T, K] shaped
            # For binomial, they are [T] shaped
            # We keep them in their native shape for per-category priors in multinomial

            # Baseline parameter A depends on distribution:
            # - normal/studentt: can be negative (natural value space)
            # - negbinom: positive count
            # - binomial/multinomial: probability in [0,1], using NEW reparameterization
            if distribution in ['normal', 'studentt']:
                # A prior: NTC-anchored Normal (analogous to negbinom LogNormal, but in linear
                # space since normal/studentt values can be negative).
                #
                # When y_ntc available (fit_ntc has been run):
                #   w → 0 (quiet feature): prior centred at Amean; y_ntc is 1σ above
                #   w → 1 (noisy feature): prior centred at y_ntc
                #   sigma = |y_ntc − Amean| floored at 10% of response amplitude
                #
                # Fallback (no NTC posterior): fixed-shift approach using response amplitude.
                delta_A = (Vmax_mean_tensor - Amean_tensor).clamp_min(epsilon_tensor)
                if y_ntc_tensor is not None:
                    sigma_A = (y_ntc_tensor - Amean_tensor).abs().clamp_min(delta_A * 0.1)
                    mean_A  = (1.0 - _o_y_weight) * Amean_tensor + _o_y_weight * y_ntc_tensor
                else:
                    sigma_A = delta_A
                    _shift_A = (1.0 - _o_y_weight) * 0.55
                    mean_A  = Amean_tensor - _shift_A * delta_A
                A = pyro.sample("A", dist.Normal(mean_A, sigma_A))

            elif distribution in ['binomial', 'multinomial']:
                # Beta/Dirichlet priors for bounded [0, 1] likelihoods.
                # For binomial:    A ~ Beta(1, β) with mean = 0.5×Q05
                # For multinomial: A ~ Dirichlet with mean = 0.5×Q05 (row-normalized)
                #
                # POTENTIAL IMPROVEMENT: noise-adaptive shift (not yet implemented).
                # Analogous to the o_y weighting for negbinom/normal/studentt, the A prior
                # shift could be attenuated for low-coverage features where PSI estimates
                # are unreliable. The natural noise proxy is mean denominator read depth:
                #   _binom_weight = n_ref / (n_ref + mean_denom)   ∈ (0, 1)
                #   shift = 0.5 * (1 - _binom_weight)              (0.5 → 0 as reads increase)
                # where mean_denom = obs_denom.mean(cells) for binomial, or
                # obs_counts.sum(categories).mean(cells) for multinomial.
                # A data-adaptive n_ref = median(mean_denom) across features would be
                # more principled than a fixed count threshold.

                # Amean_tensor: [T] for binomial, [T, K] for multinomial
                # Vmax_mean_tensor: [T] for binomial, [T, K] for multinomial

                # Sample A
                if distribution == 'multinomial' and Amean_tensor.ndim > 1:
                    # For multinomial: LogisticNormal prior on A (Normal in logit space + softmax).
                    #
                    # We previously used dist.Dirichlet, but Pyro's autoguide represents the
                    # K-simplex via a stick-breaking transform from R^(K-1).  With K=65 and
                    # many phantom (all-zero) categories, float32 rounding in the stick-breaking
                    # product chain makes the sample sum drift away from 1.0 beyond PyTorch's
                    # 1e-6 Simplex tolerance, raising a ValueError in both compute_log_prob and
                    # compute_score_parts — even with validate_args=False on the model, the
                    # guide's TransformedDistribution re-validates the same sample.
                    #
                    # LogisticNormal sidesteps this entirely: the guide uses an unconstrained
                    # Normal site (no simplex constraint, no stick-breaking), and softmax
                    # guarantees a valid probability vector in the model.
                    K_dim = Amean_tensor.shape[-1]

                    # Build phantom mask (categories with zero observations across all cells)
                    phantom_conc_mask = None
                    if y_obs_tensor.dim() == 3:  # multinomial: [N, T, K]
                        obs_total = y_obs_tensor.sum(dim=0)  # [T, K]
                        phantom_conc_mask = (obs_total == 0)  # [T, K]

                    # Logit-space prior mean (base: from Q05 or uniform)
                    if use_data_driven_priors:
                        A_mean_clamped = (0.5 * Amean_tensor).clamp(min=epsilon_tensor, max=1.0 - epsilon_tensor)
                        A_mean_normalized = A_mean_clamped / A_mean_clamped.sum(dim=-1, keepdim=True)  # [T, K]
                        A_logit_mean_base = torch.log(A_mean_normalized.clamp(min=1e-10))              # [T, K]
                    else:
                        A_logit_mean_base = torch.zeros(T, K_dim, device=self.model.device)

                    # A prior: w-interpolated LogisticNormal anchored between data-driven
                    # lower anchor and y_ntc. Mode at (1-w)*A_logit_mean_base + w*logit(y_ntc),
                    # where w = o_y_ntc / (o_y_ntc + prior_mean_o_y).
                    if y_ntc_tensor is not None:
                        if y_ntc_tensor.ndim != 2 or y_ntc_tensor.shape[0] != T or y_ntc_tensor.shape[1] != K_dim:
                            raise ValueError(
                                f"y_ntc_tensor shape {tuple(y_ntc_tensor.shape)} does not match "
                                f"expected multinomial shape [{T}, {K_dim}]."
                            )
                        logit_y_ntc       = torch.log(y_ntc_tensor.clamp(min=1e-10))      # [T, K], log-simplex
                        A_logit_mean_base = torch.maximum(A_logit_mean_base, logit_y_ntc - 4.0)
                        w_expanded        = _o_y_weight.unsqueeze(-1)                      # [T, 1]
                        A_logit_mean      = (1.0 - w_expanded) * A_logit_mean_base + w_expanded * logit_y_ntc  # [T, K]
                        sigma_A_logit     = (logit_y_ntc - A_logit_mean_base).abs().clamp_min(1.0)             # [T, K]
                    else:
                        A_logit_mean  = A_logit_mean_base
                        sigma_A_logit = self._t(1.0)

                    # Push phantom categories to -inf in prior mean so the guide learns
                    # to assign them zero probability.
                    if phantom_conc_mask is not None:
                        A_logit_mean = A_logit_mean.masked_fill(phantom_conc_mask, -1e4)

                    # Sample in unconstrained logit space — no simplex constraint, no stick-breaking.
                    A_logit = pyro.sample("A", dist.Normal(A_logit_mean, sigma_A_logit).to_event(1))  # [T, K]

                    # Softmax with phantom masking → valid probability vector [T, K]
                    if phantom_conc_mask is not None:
                        A_logit = A_logit.masked_fill(phantom_conc_mask, -1e9)
                    A = torch.softmax(A_logit, dim=-1)  # [T, K]

                else:
                    # For binomial: Logit-Normal prior when y_ntc available (analogous to negbinom
                    # LogNormal, adapted to the [0,1] simplex via logit transform).
                    #
                    # When y_ntc available:
                    #   Lower anchor: logit(Amean/2); upper anchor: logit(y_ntc)
                    #   w → 0 (quiet feature): centred at logit(Amean/2); y_ntc is 1σ above
                    #   w → 1 (noisy feature): centred at logit(y_ntc)
                    #   sigma = (logit(y_ntc) − logit(Amean/2)).clamp_min(1 logit unit)
                    #
                    # Fallback (no NTC posterior): Beta(1, β) with mean = 0.5×Q05.
                    if y_ntc_tensor is not None and use_data_driven_priors:
                        # Lift upper anchor to mean(y_corrected) when y_ntc < mean.
                        if mean_y_corrected_tensor is not None:
                            effective_y_ntc = torch.maximum(y_ntc_tensor, mean_y_corrected_tensor)
                        else:
                            effective_y_ntc = y_ntc_tensor
                        logit_Amean_half = torch.logit((0.5 * Amean_tensor).clamp(1e-6, 1.0 - 1e-6))      # [T]
                        logit_y_ntc      = torch.logit(effective_y_ntc.clamp(1e-6, 1.0 - 1e-6))            # [T]
                        logit_Amean_half = torch.maximum(logit_Amean_half, logit_y_ntc - 4.0)              # cap: sigma ≤ 4
                        sigma_logit = (logit_y_ntc - logit_Amean_half).clamp_min(1.0)                      # [T]
                        mu_logit    = (1.0 - _o_y_weight) * logit_Amean_half + _o_y_weight * logit_y_ntc  # [T]
                        logit_A = pyro.sample("logit_A", dist.Normal(mu_logit, sigma_logit))            # [T]
                        A = pyro.deterministic("A", torch.sigmoid(logit_A))                             # [T]
                    elif use_data_driven_priors:
                        A_mean_shifted = (0.5 * Amean_tensor).clamp_min(epsilon_tensor)
                        beta_A = (1.0 - A_mean_shifted) / A_mean_shifted                               # [T]
                        A = pyro.sample("A", dist.Beta(self._t(1.0), beta_A).expand([T]))              # [T]
                    else:
                        A = pyro.sample("A", dist.Beta(self._t(1.0), self._t(1.0)).expand([T]))        # [T]

            else:
                # For negbinom: LogNormal A prior (Normal in log2 space) with noise-adaptive mean.
                # log2(A) ~ Normal(mu_log2_A, sigma_log2_A), so A = 2^log2(A).
                #
                # mu_log2_A interpolates in log2 space between two gene-specific anchors:
                #   w → 0 (quiet gene, informative likelihood): log2(Amean/2)
                #   w → 1 (noisy gene, weak likelihood):        log2(y_ntc)
                #
                # sigma_log2_A = log2(y_ntc) - lower_anchor, floored at 1 octave, capped at 4.
                if y_ntc_tensor is not None:
                    log2_y_ntc = torch.log2(y_ntc_tensor)
                    # Lift upper anchor to mean(y_corrected) when y_ntc < mean — prevents
                    # both anchors collapsing to the floor for genes absent in NTC but
                    # expressed in perturbed conditions.
                    if mean_y_corrected_tensor is not None:
                        log2_upper_anchor = torch.maximum(log2_y_ntc, torch.log2(mean_y_corrected_tensor))
                    else:
                        log2_upper_anchor = log2_y_ntc
                    log2_Amean_half = torch.maximum(
                        torch.log2(Amean_tensor / 2.0),
                        log2_upper_anchor - 4.0,
                    )
                    sigma_log2_A = (log2_upper_anchor - log2_Amean_half).clamp_min(1.0)  # ≤ 4 octaves
                    mu_log2_A    = (1.0 - _o_y_weight) * log2_Amean_half + _o_y_weight * log2_upper_anchor
                else:
                    log2_Amean_half = torch.log2(Amean_tensor / 2.0)
                    sigma_log2_A    = self._t(4.0)
                    mu_log2_A       = log2_Amean_half
                log2_A = pyro.sample("log2_A", dist.Normal(mu_log2_A, sigma_log2_A))
                A = pyro.deterministic("A", torch.pow(self._t(2.0), log2_A))

            if use_alpha:
                # Relaxed Bernoulli: alpha ~ (0,1), becomes more discrete as temperature -> 0
                # For multinomial: per-category gate — each of the K-1 categories can
                # independently be "on" or "off". This allows, e.g., only one acceptor to
                # respond while others remain flat. The Kth (residual) category is affected
                # iff at least one K-1 category is on (conservation of probability).
                if distribution == 'multinomial' and K is not None:
                    alpha = pyro.sample("alpha", alpha_dist(temperature=temperature, logits=p_n_logits_tensor).expand([K - 1]).to_event(1))  # [T, K-1]
                else:
                    alpha = pyro.sample("alpha", alpha_dist(temperature=temperature, logits=p_n_logits_tensor))  # [T]
            else:
                if distribution == 'multinomial' and K is not None:
                    alpha = torch.ones((T, K - 1), device=self.model.device)
                else:
                    alpha = torch.ones((T,), device=self.model.device)
            
            if function_type in ['single_hill', 'additive_hill', 'nested_hill']:

                #####################################
                ## function priors (depend on o_y) ##
                #####################################
                # Gamma and delta depend on T dimension
                # Reduce over group dimension if necessary
                K_sigma = (K_max_tensor / (self._t(2) * torch.sqrt(K_alpha_tensor))) + epsilon_tensor

                # For multinomial, reduce [T, K] → [T] for the Hill function amplitude prior
                # For other distributions, use Vmax_mean_tensor directly
                if distribution == 'multinomial' and Vmax_mean_tensor.ndim > 1:
                    Vmax_prior_mean = Vmax_mean_tensor.mean(dim=-1)  # [T]
                else:
                    Vmax_prior_mean = Vmax_mean_tensor  # [T]

                Vmax_sigma = (Vmax_prior_mean / torch.sqrt(Vmax_alpha_tensor)) + epsilon_tensor

                # Compute the unconstrained prior mean for n_raw that maps to n_mu under
                # soft_clamp.  Without this correction, Normal(0, sigma) on n_raw has its
                # constrained mode at center_n = (nmin + nmax)/2, which can be far from 0
                # when nmin/nmax are asymmetric (e.g. all x > 1 → nmin = −100, center ≈ −31).
                _center_n = 0.5 * (nmax + nmin)
                _half_n   = 0.5 * (nmax - nmin).clamp_min(epsilon_tensor)
                _ratio_n  = ((n_mu_tensor - _center_n) / _half_n).clamp(-1 + 1e-6, 1 - 1e-6)
                n_mu_raw_tensor = _half_n * torch.atanh(_ratio_n)

                # Per-gene n scale: sampled inside trans_plate so each gene is
                # independently regularised (prevents one noisy gene from inflating
                # the global sigma and loosening the prior for all other genes).
                sigma_n_a = pyro.sample("sigma_n_a", dist.Exponential(self._t(1.0)))

                # For multinomial, we need per-category parameters (K-1 categories, Kth is residual)
                if distribution == 'multinomial' and K is not None:
                    K_minus_1 = K - 1
                    # Sample parameters for K-1 categories
                    # Each category gets its own Hill function parameters
                    # Use .to_event(1) instead of a nested plate to avoid dim=-1 collision with trans_plate
                    n_a_raw = pyro.sample("n_a_raw", dist.Normal(n_mu_raw_tensor, sigma_n_a.unsqueeze(-1).expand(T, K_minus_1)).to_event(1))  # [T, K-1]
                    n_a = pyro.deterministic(
                        "n_a",
                        _soft_clamp(n_a_raw, nmin, nmax)
                    )  # [T, K-1]
                else:
                    # For non-multinomial: single set of parameters per feature
                    n_a_raw = pyro.sample("n_a_raw", dist.Normal(n_mu_raw_tensor, sigma_n_a))
                    n_a = pyro.deterministic(
                        "n_a",
                        _soft_clamp(n_a_raw, nmin, nmax)
                    )

                # Identifiability coupling: when n_a≈0 the Hill is flat and alpha·Vmax is
                # just a constant offset, indistinguishable from A.  Penalise alpha by
                # exp(-|n_a|): coupling=1 at n=0, 0.37 at n=1, 0.14 at n=2.
                # This keeps alpha near 0 until the data pushes n_a away from 0.
                if use_alpha and alpha_n_coupling > 0.0:
                    _flatness_a = torch.exp(-torch.abs(n_a))  # [T] or [T, K-1]
                    if distribution == 'multinomial' and K is not None:
                        pyro.factor("alpha_n_coupling_a", -(alpha * _flatness_a * alpha_n_coupling).sum(dim=-1))
                    else:
                        pyro.factor("alpha_n_coupling_a", -(alpha * _flatness_a * alpha_n_coupling))

                # Scale for Vmax, K is multiplied by alpha
                #eff_Vmax_sigma = alpha * Vmaxa_sigma + epsilon_tensor
                #eff_Ka_sigma   = alpha * Ka_sigma    + epsilon_tensor

                # UNIFIED Vmax and K priors (Log-Normal for all distributions)
                # K uses CV-based std (scale-invariant, works without guides)
                # Vmax uses raw variance (data-driven) for negbinom/normal/studentt

                # K parameterization (UNIFIED for all distributions)
                # When NTC mean is known, center the prior there with ±5 log2FC coverage.
                # This ensures the P.O.I. starts within the observed x-range regardless of
                # whether the subset is KO-only, CA-only, or full.
                #
                # Minimum prior width: K_log_sigma >= 5*ln(2)/2 ≈ 1.73
                # This guarantees the lower end of the 95% CI extends at least 2^{-5} (1/32x)
                # below the prior centre, matching the NTC-centred branch below.
                _K_log_sigma_min = (k_log_sigma_min_tensor
                                    if k_log_sigma_min_tensor is not None
                                    else self._t(5.0 * 0.6931 / 2.0))

                if x_ntc_mean is not None:
                    # K prior: log-normal with MEDIAN at x_ntc_mean (= log2FC 0).
                    # In log2FC space: log2(K_a) ~ Normal(0, sigma_log2²) where
                    # sigma_log2 = K_log_sigma / ln(2) = 2.5, giving a 95% CI of ±5 log2FC.
                    # Setting K_log_mu = log(x_ntc_mean) centres the median (not the mean)
                    # at NTC, which is the natural reference in log2FC space.
                    K_log_sigma = _K_log_sigma_min  # = 5*ln(2)/2 → sigma_log2 = 2.5
                    K_log_mu = torch.log(x_ntc_mean.clamp_min(epsilon_tensor))
                else:
                    # Fallback: centre median at user-specified quantile of x_true (k_prior_center).
                    # Default: middle of observed range (K_max/2). Width from CV.
                    # Floor sigma so the prior always spans at least ±5 log2FC.
                    _k_fallback = (k_center_tensor if k_center_tensor is not None
                                   else K_max_tensor / 2.0)
                    K_mean_prior = _k_fallback.clamp_min(epsilon_tensor)
                    if x_true_CV is not None:
                        K_std_prior = K_mean_prior * x_true_CV
                    else:
                        K_std_prior = K_max_tensor / (self._t(2.0) * torch.sqrt(K_alpha_tensor))
                    ratio_K = (K_std_prior / K_mean_prior).clamp_min(self._t(1e-6))
                    K_log_sigma = torch.sqrt(torch.log1p(ratio_K ** 2)).clamp_min(_K_log_sigma_min)
                    K_log_mu = torch.log(K_mean_prior)

                if distribution in ['binomial', 'multinomial']:
                    # For binomial/multinomial: Sample Vmax_a INDEPENDENTLY (like BCD1C4F)
                    # Even for multinomial, the Kth category is residual, so we can have independent Vmax
                    # This gives Vmax_a direct gradient signal, not mediated through alpha
                    # Avoids chicken-and-egg: alpha needs signal → signal needs Vmax → Vmax needs alpha
                    Vmax_mean_clamped = Vmax_mean_tensor.clamp(min=0.01, max=0.99)
                    # concentration=2 gives CV ≈ 0.6-1.0 depending on the mean —
                    # much more diffuse than the previous 10, allowing the posterior
                    # to explore Vmax values well above the data-driven estimate when
                    # only part of the dose-response range is observed.
                    concentration_vmax = self._t(2.0)
                    alpha_vmax = Vmax_mean_clamped * concentration_vmax
                    beta_vmax = (1 - Vmax_mean_clamped) * concentration_vmax

                    if distribution == 'multinomial' and K is not None:
                        # For multinomial: Sample Vmax_a for K-1 categories (Kth is residual)
                        # Use .to_event(1) instead of nested plates to avoid dim=-1 collision with trans_plate
                        K_minus_1 = K - 1
                        Vmax_a = pyro.sample("Vmax_a", dist.Beta(alpha_vmax[:, :K_minus_1], beta_vmax[:, :K_minus_1]).to_event(1))  # [T, K-1]

                        # K_a: Log-Normal for K-1 categories
                        log_K_a = pyro.sample("log_K_a", dist.Normal(K_log_mu, K_log_sigma).expand([K_minus_1]).to_event(1))  # [T, K-1]
                        K_a = pyro.deterministic("K_a", torch.exp(log_K_a))  # [T, K-1]
                    else:
                        # For binomial: per-feature Vmax_a and K_a
                        Vmax_a = pyro.sample("Vmax_a", dist.Beta(alpha_vmax, beta_vmax))  # [T]
                        log_K_a = pyro.sample("log_K_a", dist.Normal(K_log_mu, K_log_sigma))  # [T]
                        K_a = pyro.deterministic("K_a", torch.exp(log_K_a))  # [T]

                else:
                    # Vmax prior: LogNormal for negbinom, normal, and studentt alike.
                    # Direction of effect is carried by n (negative n = repressor Hill),
                    # so Vmax must be strictly positive for all three distributions.
                    # Wide log_sigma (floor ≥ 1.5, ≈ 20× range) keeps the prior diffuse
                    # enough for one-sided subsets (single technical group).
                    _Vmax_log_sigma_floor = (vmax_log_sigma_floor_tensor
                                             if vmax_log_sigma_floor_tensor is not None
                                             else self._t(1.5))
                    Vmax_sigma = (Vmax_prior_mean / torch.sqrt(Vmax_alpha_tensor)) + epsilon_tensor
                    Vmax_log_sigma = torch.sqrt(
                        torch.log1p((Vmax_sigma / Vmax_prior_mean) ** 2)
                    ).clamp_min(_Vmax_log_sigma_floor)
                    Vmax_log_mu = (torch.log(Vmax_prior_mean.clamp_min(epsilon_tensor))
                                   - 0.5 * Vmax_log_sigma ** 2)
                    log_Vmax_a = pyro.sample("log_Vmax_a", dist.Normal(Vmax_log_mu, Vmax_log_sigma))
                    Vmax_a = pyro.deterministic("Vmax_a", torch.exp(log_Vmax_a))

                    log_K_a = pyro.sample("log_K_a", dist.Normal(K_log_mu, K_log_sigma))
                    K_a = pyro.deterministic("K_a", torch.exp(log_K_a))

                # Sample all required parameters (additive_hill and nested_hill need second set)
                if function_type in ['additive_hill', 'nested_hill']:
                    sigma_n_b = pyro.sample("sigma_n_b", dist.Exponential(self._t(1.0)))
                    if distribution == 'multinomial' and K is not None:
                        # Use .to_event(1) instead of nested plate to avoid dim=-1 collision with trans_plate
                        beta = pyro.sample("beta", alpha_dist(temperature=temperature, logits=p_n_logits_tensor).expand([K - 1]).to_event(1))  # [T, K-1]
                    else:
                        beta = pyro.sample("beta", alpha_dist(temperature=temperature, logits=p_n_logits_tensor))  # [T]

                    # n_b: per-category for multinomial, single for others
                    if distribution == 'multinomial' and K is not None:
                        # Use .to_event(1) instead of nested plate to avoid dim=-1 collision with trans_plate
                        K_minus_1 = K - 1
                        n_b_raw = pyro.sample("n_b_raw", dist.Normal(n_mu_raw_tensor, sigma_n_b.unsqueeze(-1).expand(T, K_minus_1)).to_event(1))  # [T, K-1]
                        n_b = pyro.deterministic(
                            "n_b",
                            _soft_clamp(n_b_raw, nmin, nmax)
                        )  # [T, K-1]
                    else:
                        n_b_raw = pyro.sample("n_b_raw", dist.Normal(n_mu_raw_tensor, sigma_n_b))
                        n_b = pyro.deterministic(
                            "n_b",
                            _soft_clamp(n_b_raw, nmin, nmax)
                        )

                    # Identifiability coupling for beta/n_b (same logic as alpha/n_a above).
                    if use_alpha and alpha_n_coupling > 0.0:
                        _flatness_b = torch.exp(-torch.abs(n_b))  # [T] or [T, K-1]
                        if distribution == 'multinomial' and K is not None:
                            pyro.factor("alpha_n_coupling_b", -(beta * _flatness_b * alpha_n_coupling).sum(dim=-1))
                        else:
                            pyro.factor("alpha_n_coupling_b", -(beta * _flatness_b * alpha_n_coupling))

                    # Vmax_b and K_b: same structure as Vmax_a and K_a
                    if distribution in ['binomial', 'multinomial']:
                        if distribution == 'multinomial' and K is not None:
                            # For multinomial: Sample Vmax_b for K-1 categories (Kth is residual)
                            # Use .to_event(1) instead of nested plates to avoid dim=-1 collision with trans_plate
                            K_minus_1 = K - 1
                            Vmax_b = pyro.sample("Vmax_b", dist.Beta(alpha_vmax[:, :K_minus_1], beta_vmax[:, :K_minus_1]).to_event(1))  # [T, K-1]

                            # K_b: Log-Normal for K-1 categories
                            log_K_b = pyro.sample("log_K_b", dist.Normal(K_log_mu, K_log_sigma).expand([K_minus_1]).to_event(1))  # [T, K-1]
                            K_b = pyro.deterministic("K_b", torch.exp(log_K_b))  # [T, K-1]
                        else:
                            # For binomial: per-feature Vmax_b and K_b
                            Vmax_b = pyro.sample("Vmax_b", dist.Beta(alpha_vmax, beta_vmax))  # [T]
                            log_K_b = pyro.sample("log_K_b", dist.Normal(K_log_mu, K_log_sigma))  # [T]
                            K_b = pyro.deterministic("K_b", torch.exp(log_K_b))  # [T]

                    else:
                        # Vmax_b: same LogNormal prior as Vmax_a (parameters computed above).
                        log_Vmax_b = pyro.sample("log_Vmax_b", dist.Normal(Vmax_log_mu, Vmax_log_sigma))
                        Vmax_b = pyro.deterministic("Vmax_b", torch.exp(log_Vmax_b))

                        log_K_b = pyro.sample("log_K_b", dist.Normal(K_log_mu, K_log_sigma))
                        K_b = pyro.deterministic("K_b", torch.exp(log_K_b))
                
                # Early exit for memory-efficient posterior sampling.
                # All gene-level latent variables (A, alpha, beta, Vmax_a/b, K_a/b, n_a/b,
                # o_y, sigma_n_a/b) have been sampled above as [T]-shaped tensors.
                # Skipping the cell-level Hill computation avoids creating [N, T] tensors
                # (each ~2.68 GB for N=29318, T=22834), which is the source of OOM during
                # pyro.infer.Predictive with num_samples > 1.
                if latents_only:
                    return

                # Compute Hill function(s)
                # Hill_based_positive returns values in [0, Vmax]
                # We compute: y = A + alpha * Hill + beta * Hill (for additive)

                if distribution == 'multinomial' and K is not None:
                    # NEW FORMULATION for multinomial (matching binomial structure):
                    # Sample A from Dirichlet (sum to 1)
                    # Sample Vmax_a and Vmax_b INDEPENDENTLY for K-1 categories (Kth is residual)
                    # For each category k in K-1:
                    #   y_k = A_k + (alpha * Hill_a_k(Vmax=Vmax_a_k)) + (beta * Hill_b_k(Vmax=Vmax_b_k))
                    # Then: y_K = 1 - sum(y_1, ..., y_{K-1})
                    # This ensures probabilities sum to 1, and Kth category gets whatever is left

                    K_dim = K
                    K_minus_1 = K - 1
                    x_expanded = x_true.unsqueeze(-1)  # [N, 1]

                    # A is [T, K] from Dirichlet
                    # Extract K-1 for fitting Hills (Kth doesn't get a Hill function)
                    A_kminus1 = A[..., :K_minus_1]  # [T, K-1]

                    # Expand for broadcasting
                    A_kminus1_expanded = A_kminus1.unsqueeze(0)  # [1, T, K-1]

                    if function_type == 'single_hill':
                        # Compute Hills for K-1 categories using log_K_a (numerically stable)
                        Hilla_list = []
                        for k in range(K_minus_1):
                            hill_k = Hill_based_positive_logK(x_expanded, Vmax=Vmax_a[:, k], A=0,
                                                              logK=log_K_a[:, k], n=n_a[:, k])  # [N, T]
                            Hilla_list.append(hill_k.unsqueeze(-1))  # [N, T, 1]
                        Hilla_kminus1 = torch.cat(Hilla_list, dim=-1)  # [N, T, K-1]

                        # Combine: y = A + alpha_k * Hill_a_k(Vmax=Vmax_a_k)
                        # alpha is [T, K-1] (per-category); unsqueeze(0) -> [1, T, K-1]
                        combined_hill = alpha.unsqueeze(0) * Hilla_kminus1  # [N, T, K-1]
                        y_kminus1 = A_kminus1_expanded + combined_hill  # [N, T, K-1]

                    elif function_type == 'additive_hill':
                        # Compute Hills for K-1 categories using log_K_a/log_K_b (numerically stable)
                        Hilla_list = []
                        Hillb_list = []
                        for k in range(K_minus_1):
                            hill_a_k = Hill_based_positive_logK(x_expanded, Vmax=Vmax_a[:, k], A=0,
                                                                logK=log_K_a[:, k], n=n_a[:, k])  # [N, T]
                            hill_b_k = Hill_based_positive_logK(x_expanded, Vmax=Vmax_b[:, k], A=0,
                                                                logK=log_K_b[:, k], n=n_b[:, k])  # [N, T]
                            Hilla_list.append(hill_a_k.unsqueeze(-1))  # [N, T, 1]
                            Hillb_list.append(hill_b_k.unsqueeze(-1))  # [N, T, 1]
                        Hilla_kminus1 = torch.cat(Hilla_list, dim=-1)  # [N, T, K-1]
                        Hillb_kminus1 = torch.cat(Hillb_list, dim=-1)  # [N, T, K-1]

                        # Combine: y = A + (alpha_k * Hill_a_k) + (beta_k * Hill_b_k)
                        # alpha, beta are [T, K-1] (per-category); unsqueeze(0) -> [1, T, K-1]
                        combined_hill = (alpha.unsqueeze(0) * Hilla_kminus1 +
                                       beta.unsqueeze(0) * Hillb_kminus1)  # [N, T, K-1]
                        y_kminus1 = A_kminus1_expanded + combined_hill  # [N, T, K-1]

                    elif function_type == 'nested_hill':
                        # Compute nested Hills for K-1 categories
                        # First Hill uses log_K_a (stable); second Hill feeds Hilla output as x
                        Hillb_list = []
                        for k in range(K_minus_1):
                            hill_a_k = Hill_based_positive_logK(x_expanded, Vmax=Vmax_a[:, k], A=0,
                                                                logK=log_K_a[:, k], n=n_a[:, k])  # [N, T]
                            hill_b_k = Hill_based_positive(hill_a_k.unsqueeze(-1), Vmax=Vmax_b[:, k], A=0,
                                                          K=K_b[:, k], n=n_b[:, k],
                                                          epsilon=epsilon_tensor)  # [N, T]
                            Hillb_list.append(hill_b_k.unsqueeze(-1))  # [N, T, 1]
                        Hillb_kminus1 = torch.cat(Hillb_list, dim=-1)  # [N, T, K-1]

                        # alpha is [T, K-1] (per-category); unsqueeze(0) -> [1, T, K-1]
                        combined_hill = alpha.unsqueeze(0) * Hillb_kminus1  # [N, T, K-1]
                        y_kminus1 = A_kminus1_expanded + combined_hill  # [N, T, K-1]

                    # Clamp K-1 probabilities to [epsilon, 1-epsilon] to ensure valid residual
                    y_kminus1 = torch.clamp(y_kminus1, min=epsilon_tensor, max=1.0 - epsilon_tensor)

                    # Ensure sum of K-1 doesn't exceed 1
                    sum_kminus1 = y_kminus1.sum(dim=-1, keepdim=True)  # [N, T, 1]
                    # If sum > (1-epsilon), rescale proportionally
                    y_kminus1 = torch.where(
                        sum_kminus1 > (1.0 - epsilon_tensor),
                        y_kminus1 * (1.0 - epsilon_tensor) / sum_kminus1,
                        y_kminus1
                    )

                    # Compute Kth category as residual: y_K = 1 - sum(y_1, ..., y_{K-1})
                    y_K = 1.0 - y_kminus1.sum(dim=-1, keepdim=True)  # [N, T, 1]
                    y_K = torch.clamp(y_K, min=epsilon_tensor, max=1.0 - epsilon_tensor)

                    # Concatenate to get all K probabilities
                    y_dose_response = torch.cat([y_kminus1, y_K], dim=-1)  # [N, T, K]

                else:
                    # For non-multinomial: standard Hill computation
                    if distribution == 'binomial':
                        # For binomial: use log_K_a/log_K_b (numerically stable sigmoid formulation)
                        # y = A + (alpha * Hill_a(Vmax=Vmax_a)) + (beta * Hill_b(Vmax=Vmax_b))
                        # Note: Vmax_a + Vmax_b CAN exceed 1 - we'll try without clamp first

                        if function_type == 'single_hill':
                            Hilla = Hill_based_positive_logK(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, logK=log_K_a, n=n_a)
                            y_dose_response = A + (alpha * Hilla)

                        elif function_type == 'additive_hill':
                            Hilla = Hill_based_positive_logK(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, logK=log_K_a, n=n_a)
                            Hillb = Hill_based_positive_logK(x_true.unsqueeze(-1), Vmax=Vmax_b, A=0, logK=log_K_b, n=n_b)
                            y_dose_response = A + (alpha * Hilla) + (beta * Hillb)

                        elif function_type == 'nested_hill':
                            Hilla = Hill_based_positive_logK(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, logK=log_K_a, n=n_a)
                            Hillb = Hill_based_positive(Hilla, Vmax=Vmax_b, A=0, K=K_b, n=n_b, epsilon=epsilon_tensor)
                            y_dose_response = A + (alpha * Hillb)

                        # OPTIONAL CLAMP (commented out for now - try without first):
                        # If Vmax_a + Vmax_b > 1 causes issues, uncomment:
                        # y_dose_response = torch.clamp(y_dose_response, min=epsilon_tensor, max=1.0 - epsilon_tensor)

                    else:
                        # For negbinom/normal/studentt: log_K_a/b are always available
                        # (log-normal prior is now the only path), so always use logK variant.
                        if function_type == 'single_hill':
                            Hilla = Hill_based_positive_logK(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, logK=log_K_a, n=n_a)
                            y_dose_response = A + (alpha * Hilla)
                        elif function_type == 'additive_hill':
                            Hilla = Hill_based_positive_logK(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, logK=log_K_a, n=n_a)
                            Hillb = Hill_based_positive_logK(x_true.unsqueeze(-1), Vmax=Vmax_b, A=0, logK=log_K_b, n=n_b)
                            y_dose_response = A + (alpha * Hilla) + (beta * Hillb)
                        elif function_type == 'nested_hill':
                            Hilla = Hill_based_positive_logK(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, logK=log_K_a, n=n_a)
                            Hillb = Hill_based_positive_logK(Hilla, Vmax=Vmax_b, A=0, logK=log_K_b, n=n_b)
                            y_dose_response = A + (alpha * Hillb)

            elif function_type == 'polynomial':
                assert polynomial_degree is not None and polynomial_degree >= 1, \
                    "polynomial_degree must be ≥ 1 (no intercept, A is handled separately)"

                if distribution == 'multinomial' and K is not None:
                    # For multinomial: fit K independent polynomials (one per category) in logit space
                    # Then apply softmax to get K probabilities that sum to 1
                    # This is more standard than K-1 with residual for unbounded logit space

                    # Sample per-category polynomial coefficients
                    coeffs_per_category = []
                    for d in range(1, polynomial_degree + 1):
                        with pyro.plate(f"poly_category_plate_deg{d}", K, dim=-2):
                            coeff = pyro.sample(f"poly_coeff_{d}", dist.Normal(0., sigma_coeff))  # [K, T]
                            coeffs_per_category.append(coeff)
                    coeffs = torch.stack(coeffs_per_category, dim=-3)  # [degree, K, T]

                    if latents_only:
                        return

                    # Compute polynomial for each category
                    # coeffs: [degree, K, T]
                    # Need to permute to [degree, T, K] for Polynomial_function to work correctly
                    coeffs_permuted = coeffs.permute(0, 2, 1)  # [degree, T, K]
                    poly_val = Polynomial_function(x_true_sample, coeffs_permuted)  # [N, T, K]

                    # A is baseline logit for each category (need K logits)
                    # Sample K baseline logits (unbounded)
                    with pyro.plate("category_plate_A", K, dim=-2):
                        A_clamped = torch.clamp(A.unsqueeze(-2), min=epsilon_tensor, max=1.0 - epsilon_tensor)  # [1, T]
                        logit_A_transpose = torch.log(A_clamped) - torch.log(1 - A_clamped)  # [K, T] logits
                    logit_A = logit_A_transpose.transpose(-1, -2)  # [T, K]

                    # Compute logits for each category
                    logits_K = logit_A.unsqueeze(0) + alpha.unsqueeze(0).unsqueeze(-1) * poly_val  # [N, T, K]

                    # Apply softmax to get probabilities that sum to 1
                    y_dose_response = torch.softmax(logits_K, dim=-1)  # [N, T, K]

                else:
                    # For non-multinomial: single set of coefficients per feature
                    # Stack polynomial coefficients
                    coeffs = []
                    for d in range(1, polynomial_degree + 1):  # start at degree 1, no intercept
                        coeff = pyro.sample(f"poly_coeff_{d}", dist.Normal(0., sigma_coeff))
                        coeffs.append(coeff)
                    coeffs = torch.stack(coeffs, dim=-2)  # [degree, T]
                    if (coeffs.shape[1] == 1) & (coeffs.ndim == 4):
                        coeffs = coeffs.squeeze(1)        # [S, D, T]

                    if latents_only:
                        return

                    # Distribution-specific polynomial computation:
                    if distribution in ['normal', 'studentt']:
                        # For continuous distributions: work in natural space (no logs!)
                        # Polynomial is applied to x_true directly: y = A + alpha * poly(x)
                        poly_val = Polynomial_function(x_true_sample, coeffs)  # [N, T]
                        y_dose_response = A + alpha * poly_val  # [N, T] - can be negative!

                    elif distribution in ['negbinom']:
                        # For negbinom: work in log space
                        # Polynomial is applied to log2(x): log2(y) = log2(A) + alpha * poly(log2(x))
                        log2_x_true = torch.log2(x_true_sample)  # [N]
                        poly_val = Polynomial_function(log2_x_true, coeffs)  # [N, T]
                        log2_y_dose_response = torch.log2(A) + alpha * poly_val  # [N, T]
                        y_dose_response = 2 ** log2_y_dose_response  # Convert back to count space

                    elif distribution == 'binomial':
                        # For binomial: work in LOGIT space (unbounded: -inf to +inf)
                        # A is in [0,1] from Beta prior, convert to logit
                        # Polynomial is applied to x_true: logit(p) = logit(A) + alpha * poly(x)
                        A_clamped = torch.clamp(A, min=epsilon_tensor, max=1.0 - epsilon_tensor)
                        logit_A = torch.log(A_clamped) - torch.log(1 - A_clamped)  # logit(A)
                        poly_val = Polynomial_function(x_true_sample, coeffs)  # [N, T]
                        logit_p = logit_A + alpha * poly_val  # [N, T] - logit space (unbounded)
                        # Convert back to probability space for sampler
                        y_dose_response = torch.sigmoid(logit_p)  # [N, T] in [0, 1]

                    else:
                        raise ValueError(f"Unknown distribution for polynomial: {distribution}")
            else:
                raise ValueError(f"Unknown function_type: {function_type}")
            

        ##########################
        ## Cell-level variables ##
        ##########################
        # At this point, y_dose_response contains the dose-response function output.
        # We need to transform it to the format expected by samplers.
        #
        # IMPORTANT: Samplers in distributions.py handle technical group effects themselves!
        # Do NOT apply alpha_y here - pass it to the sampler via alpha_y_full.

        # Prepare alpha_y_full (full C technical groups, including reference)
        # This will be passed to samplers, which apply technical groups themselves
        # Multinomial now supported (additive on logit scale, like binomial)
        if alpha_y is not None and groups_tensor is not None:
            # CRITICAL: Check if alpha_y already includes reference group
            # If alpha_y.shape[0 or 1] == C, it already has reference, don't add it!

            if alpha_y.dim() == 4:  # Predictive multinomial: Could be (S, C-1, T, K) or (S, C, T, K)
                if alpha_y.shape[1] == C:
                    # Already includes reference
                    alpha_y_full = alpha_y
                else:
                    # Need to add reference (zeros for additive on logit scale)
                    baseline_shape = (alpha_y.shape[0], 1, alpha_y.shape[2], alpha_y.shape[3])
                    baseline = torch.zeros(baseline_shape, device=self.model.device)
                    alpha_y_full = torch.cat([baseline, alpha_y], dim=1)

            elif alpha_y.dim() == 3:
                # Could be Predictive 2D (S, C-1, T) or Training multinomial (C-1, T, K) or (C, T, K)
                # Check if last dimension matches K (multinomial) or T (2D predictive)
                if distribution == 'multinomial' and K is not None and alpha_y.shape[-1] == K:
                    # Training multinomial: (C-1, T, K) or (C, T, K)
                    if alpha_y.shape[0] == C:
                        # Already includes reference
                        alpha_y_full = alpha_y
                    else:
                        # Need to add reference (zeros for additive on logit scale)
                        baseline_shape = (1, alpha_y.shape[1], alpha_y.shape[2])
                        baseline = torch.zeros(baseline_shape, device=self.model.device)
                        alpha_y_full = torch.cat([baseline, alpha_y], dim=0)
                else:
                    # Predictive 2D: (S, C-1, T) or (S, C, T)
                    if alpha_y.shape[1] == C:
                        # Already includes reference
                        alpha_y_full = alpha_y
                    else:
                        # Need to add reference (zeros for additive, ones for multiplicative)
                        ones_shape = (alpha_y.shape[0], 1, T)
                        if distribution == 'negbinom':
                            baseline = torch.ones(ones_shape, device=self.model.device)
                        else:
                            baseline = torch.zeros(ones_shape, device=self.model.device)
                        alpha_y_full = torch.cat([baseline, alpha_y], dim=1)

            elif alpha_y.dim() == 2:  # Training 2D: Could be (C-1, T) or (C, T)
                if alpha_y.shape[0] == C:
                    # Already includes reference
                    alpha_y_full = alpha_y
                else:
                    # Need to add reference (zeros for additive, ones for multiplicative)
                    ones_shape = (1, T)
                    if distribution == 'negbinom':
                        baseline = torch.ones(ones_shape, device=self.model.device)
                    else:
                        baseline = torch.zeros(ones_shape, device=self.model.device)
                    alpha_y_full = torch.cat([baseline, alpha_y], dim=0)
            else:
                raise ValueError(f"Unexpected alpha_y shape: {alpha_y.shape}")
        else:
            alpha_y_full = None

        # Transform dose-response to sampler-expected format
        # (samplers will handle technical groups and normalization)
        if distribution == 'negbinom':
            # Sampler expects: mu_y = dose-response in count space (NO technical groups, NO sum factors)
            # Sampler will apply: mu_final = mu_y * alpha_y * sum_factor
            mu_y = y_dose_response  # [N, T] - just the dose-response

        elif distribution == 'binomial':
            # Sampler expects: mu_y = probability in [0, 1]
            # y_dose_response should already be a probability from Hill/polynomial
            # Sampler will apply technical groups on logit scale
            mu_y = y_dose_response  # [N, T] - already probability

        elif distribution == 'multinomial':
            # Sampler expects: mu_y = baseline probabilities [N, T, K]
            # y_dose_response is already [N, T, K] probabilities that sum to 1
            # Sampler will apply technical groups on logit scale per category
            # Zero out phantom categories so _masked_softmax in sample_multinomial_trans
            # correctly excludes them (it detects masked positions via mu_y == 0).
            obs_total = y_obs_tensor.sum(dim=0, keepdim=True)  # [1, T, K]
            mu_y = y_dose_response.masked_fill(obs_total == 0, 0.0)  # [N, T, K]

        elif distribution in ['normal', 'studentt']:
            # Sampler expects: mu_y = natural value space
            # Sampler will apply technical groups additively
            mu_y = y_dose_response  # [N, T] - can be negative!

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Debug checks (keep for troubleshooting)
        if torch.isnan(mu_y).any() or torch.isinf(mu_y).any():
            check_tensor("mu_y", mu_y)
            check_tensor("y_dose_response", y_dose_response)
            check_tensor("sum_factor_tensor", sum_factor_tensor)
            check_tensor("phi_y_used", phi_y_used)
            check_tensor("A", A)
            if function_type in ['single_hill', 'additive_hill', 'nested_hill']:
                check_tensor("n_a", n_a)
                check_tensor("Vmax_a", Vmax_a)
                check_tensor("K_a", K_a)
            if function_type in ['additive_hill', 'nested_hill']:
                check_tensor("n_b", n_b)
                check_tensor("Vmax_b", Vmax_b)
                check_tensor("K_b", K_b)
            if function_type == 'polynomial':
                check_tensor("coeffs", coeffs)

        # Call distribution-specific observation sampler
        from .distributions import get_observation_sampler
        observation_sampler = get_observation_sampler(distribution, 'trans')

        # Call the appropriate sampler based on distribution
        if distribution == 'negbinom':
            observation_sampler(
                y_obs_tensor=y_obs_tensor,
                mu_y=mu_y,
                phi_y_used=phi_y_used,
                alpha_y_full=alpha_y_full,
                groups_tensor=groups_tensor,
                sum_factor_tensor=sum_factor_tensor,
                N=N,
                T=T,
                C=C,
                use_epsilon=use_epsilon
            )
        elif distribution == 'multinomial':
            # For multinomial, mu_y should be probabilities [N, T, K]
            observation_sampler(
                y_obs_tensor=y_obs_tensor,
                mu_y=mu_y,  # Should be [N, T, K] probabilities
                alpha_y_full=alpha_y_full,  # [C, T, K] or [S, C, T, K] - additive on logit scale
                groups_tensor=groups_tensor,
                N=N,
                T=T,
                K=K,
                C=C
            )
        elif distribution == 'binomial':
            observation_sampler(
                y_obs_tensor=y_obs_tensor,
                denominator_tensor=denominator_tensor,
                mu_y=mu_y,  # Should be probabilities [N, T]
                alpha_y_full=alpha_y_full,
                groups_tensor=groups_tensor,
                N=N,
                T=T,
                C=C
            )
        elif distribution == 'normal':
            # For normal, we need sigma_y (standard deviation)
            sigma_y = 1.0 / torch.sqrt(phi_y)  # Convert from precision to std dev
            observation_sampler(
                y_obs_tensor=y_obs_tensor,
                mu_y=mu_y,
                sigma_y=sigma_y,
                alpha_y_full=alpha_y_full,
                groups_tensor=groups_tensor,
                N=N,
                T=T,
                C=C
            )
        elif distribution == 'studentt':
            # For studentt, we need sigma_y (standard deviation) and nu_y (degrees of freedom)
            sigma_y = 1.0 / torch.sqrt(phi_y)  # Convert from precision to std dev
            observation_sampler(
                y_obs_tensor=y_obs_tensor,
                mu_y=mu_y,
                sigma_y=sigma_y,
                nu_y=nu_y,
                alpha_y_full=alpha_y_full,
                groups_tensor=groups_tensor,
                N=N,
                T=T,
                C=C
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    ########################################################
    # Step 3: Fit trans effects (model_y)
    ########################################################
    def fit_trans(
        self,
        sum_factor_col: str = None,
        function_type: str = 'single_hill',  # or 'additive', 'nested'
        polynomial_degree: int = 6,
        lr: float = None,
        niters: int = None,
        nsamples: int = 1000,
        alpha_ewma: float = 0.05,
        tolerance: float = 1e-4, # recommended to keep based on cell2location
        beta_o_beta: float = 3, # recommended to keep based on cell2location
        beta_o_alpha: float = 9, # recommended to keep based on cell2location
        alpha_alpha_mu: float = 5.8,
        K_alpha: float = 2,
        Vmax_alpha: float = 2,
        n_mu: float = 0,
        p_n: float = 1e-6,
        epsilon: float = 1e-6,
        init_temp: float = 1.0,
        #final_temp: float = 1e-8,
        final_temp: float = 0.1,
        minibatch_size: int = None,
        distribution: str = None,
        denominator: np.ndarray = None,
        modality_name: str = None,
        min_denominator: int = None,
        use_data_driven_priors: bool = True,
        correct_priors_for_technical: bool = True,
        use_archive_prior_computation: bool = False,
        use_epsilon: bool = False,
        warmup: bool = True,
        warmup_T_min: float = 0.5,
        vmax_log_sigma_floor: float = 1.5,
        k_log_sigma_min: float = None,
        k_prior_center: str = 'middle',
        checkpoint_interval: int = 10_000,
        checkpoint_dir: str = None,
        predictive_checkpoint: str = None,
        restart_from_checkpoint: bool = True,
        alpha_n_coupling: float = 10.0,
        **kwargs
    ):
        """
        Fit trans effects using distribution-specific likelihood.

        Parameters
        ----------
        modality_name : str, optional
            Name of modality to fit. If None, uses primary modality.
        distribution : str, optional
            Distribution type: 'negbinom', 'multinomial', 'binomial', 'normal'
            If None, auto-detected from modality.
        sum_factor_col : str, optional
            Sum factor column name. Required for negbinom, ignored for others.
        denominator : np.ndarray, optional
            Denominator array for binomial distribution (e.g., total counts for PSI)
            If None, auto-detected from modality.
        min_denominator : int, optional
            Minimum denominator value for binomial observations. Observations where
            denominator < min_denominator are masked (excluded from fitting).
            Useful for filtering low-coverage splicing junctions. Default: None (no filtering).
        use_data_driven_priors : bool, optional
            If True (default), use Beta priors for A and upper_limit based on data percentiles.
            If False, use uniform priors (Beta(1, 1)). Useful for testing if data-driven
            priors are too strong or causing issues. Default: True.
        correct_priors_for_technical : bool, optional
            If True (default), correct data for technical effects before computing priors (Amean, Vmax_mean).
            If False (archive behavior), compute priors from raw sum_factor-normalized data.
            Technical effects are still corrected during model fitting via alpha_y_sample.
        use_archive_prior_computation : bool, optional
            If True, compute Amean and Vmax_mean using archive method:
            - Amean = min(guide_means) per feature
            - Vmax_mean = max(guide_means) per feature
            If False (default), use percentile-based method:
            - Amean = 5th percentile of guide means
            - Vmax_mean = 95th percentile - 5th percentile (range)
        use_epsilon : bool, optional
            If True, add 1e-8 epsilon for numerical stability in NegativeBinomial logits.
            If False (default), use log(mu) - log(phi) directly.
        warmup : bool, optional
            For additive_hill/nested_hill: run a single_hill warmup phase before the main
            fit. Default True (warmup is on by default for additive/nested hill).
            Ignored for single_hill and polynomial.
        warmup_T_min : float, optional
            Temperature at which Phase 1 (single_hill warmup) ends. Phase 1 cools from
            init_temp down to warmup_T_min at the same rate as Phase 2, so the number of
            warmup steps is computed automatically:
                warmup_steps = round(niters * (init_temp - warmup_T_min) /
                                               (init_temp - final_temp))
            niters always refers to Phase 2 (additive_hill) steps; total steps =
            warmup_steps + niters. Default 0.5 (roughly half the annealing range,
            saves ~44% of a full single_hill run while still reaching the separation
            point where active and null genes are clearly distinguished).
        vmax_log_sigma_floor : float, optional
            Minimum log-sigma for the Vmax LogNormal prior (negbinom only).
            The data-driven log-sigma (≈ sqrt(log(1 + 1/Vmax_alpha))) is typically
            ≈0.65 and is always below this floor, so this parameter directly controls
            prior width. Default 1.5 (95% CI spans ≈exp(±3) ≈ 20× range).
            Previous default was 1.0 (7× range). Increase to allow Vmax to range
            more freely; set to 1.0 to restore the old behavior.
        k_log_sigma_min : float, optional
            Minimum log-sigma for the K LogNormal prior. Default None, which uses
            5*ln(2)/2 ≈ 1.733 (±5 log2FC = 32× range). Increase sparingly — K is
            already very wide, and wider K worsens the EC50 drift problem for genes
            where the Hill hasn't saturated within the observed x-range.
        k_prior_center : str, optional
            Where to centre the K (EC50) LogNormal prior when NTC cells are absent.
            Has no effect when NTC cells are present (prior is always centred at the
            NTC mean). Options:
            - 'lower'  : 5th percentile of x_true guide means. Use when the EC50 is
                         expected near baseline (e.g. strong activators where
                         half-maximum is reached at low cis expression).
            - 'middle' : half the maximum observed x_true (default, current behaviour).
            - 'upper'  : 95th percentile of x_true guide means. Use when the EC50 is
                         expected near the top of the observed range (e.g. repressors
                         where the effect only kicks in at high cis expression).
        predictive_checkpoint : str, optional
            Path to a checkpoint file to use for Predictive sampling only.  When set,
            all prior-computation setup runs as normal (same arguments as the original
            fit_trans call), but the training loop is skipped and the param_store is
            loaded from the checkpoint instead.  Use this to draw posterior samples
            from any intermediate checkpoint without re-running training.

            **function_type is auto-detected**: every checkpoint stores
            ``effective_function_type`` — the function that was *actually being
            fitted* when that checkpoint was written (``'single_hill'`` during
            warmup, the target type afterwards).  ``fit_trans`` reads this value
            before initialising the guide, so the guide architecture always matches
            the checkpoint.  ``warmup`` is also forced to ``False`` automatically.
            You therefore only need to pass ``sum_factor_col`` (and any other
            data-access arguments from the original call)::

                model.fit_trans(
                    sum_factor_col='sum_factor_adj',
                    predictive_checkpoint='outdir/label/trans_checkpoint_gene_warmup.pt',
                )

            The same minimal call works for any checkpoint — mid-warmup, post-warmup,
            or complete.

            To inspect a checkpoint manually::

                import torch
                ckpt = torch.load('trans_checkpoint_gene_step0030000.pt',
                                  map_location='cpu', weights_only=False)
                # Was this step in warmup?
                print(ckpt['effective_function_type'])  # 'single_hill' → yes, 'additive_hill' → no
                print(ckpt['phase2_announced'])         # False → in warmup, True → past it

            Structural mismatches (N, T, K, C) raise an error; ``distribution``
            mismatches produce a warning.
        function_type : str
            Dose-response function: 'single_hill', 'additive_hill', 'polynomial'
        **kwargs
            Additional parameters for specific distributions

        Notes
        -----
        Each modality stores its own fitting results.
        Primary modality results are also stored at model level for backward compatibility.
        Trans fitting requires that technical fit has been performed for the modality.

        If technical_group_code is set (via set_technical_groups()), it will be used for
        correction. Otherwise, no group correction is applied.

        Examples
        --------
        >>> model.set_technical_groups(['cell_line'])  # Optional, for correction
        >>> model.fit_trans(sum_factor_col='sum_factor', function_type='additive_hill')

        >>> # Test without data-driven priors
        >>> model.fit_trans(sum_factor_col='sum_factor', function_type='additive_hill',
        ...                 use_data_driven_priors=False)
        """

        if self.model.x_true is None:
            warnings.warn(
                "x_true not set. You should run fit_cis() before fit_trans(). "
                "Proceeding without cis effects."
            )

        # Determine which modality to use
        if modality_name is None:
            modality_name = self.model.primary_modality
        modality = self.model.get_modality(modality_name)

        # Auto-detect distribution from modality
        if distribution is None:
            distribution = modality.distribution

        # Auto-detect denominator from modality (for binomial)
        if denominator is None and modality.denominator is not None:
            denominator = modality.denominator

        # ---------------------------
        # Set conditional default for niters
        # ---------------------------
        if niters is None:
            # Default: 100,000 unless multinomial OR polynomial function, then 200,000
            # niters always means Phase 2 (additive/nested hill) steps.
            if distribution == 'multinomial':
                niters = 200_000
                if predictive_checkpoint is None:
                    print(f"[INFO] Using default niters=200,000 for multivariate distribution '{distribution}'")
            elif function_type == 'polynomial':
                niters = 200_000
                if predictive_checkpoint is None:
                    print(f"[INFO] Using default niters=200,000 for polynomial function")
            else:
                niters = 100_000
                if predictive_checkpoint is None:
                    print(f"[INFO] Using default niters=100,000 for distribution '{distribution}' and function_type '{function_type}'")

        # ---------------------------
        # When predictive_checkpoint is given, read function_type directly from
        # the checkpoint's effective_function_type so the caller doesn't have to
        # specify it.  This must happen before _do_warmup and guide initialisation,
        # both of which depend on function_type.
        # ---------------------------
        if predictive_checkpoint is not None:
            try:
                _peek = torch.load(predictive_checkpoint, map_location='cpu', weights_only=False)
                _peek_ft = _peek.get('effective_function_type', _peek.get('function_type'))
                if _peek_ft is not None:
                    if _peek_ft != function_type:
                        print(f"[INFO] predictive_checkpoint: overriding "
                              f"function_type={function_type!r} → {_peek_ft!r} "
                              f"(read from checkpoint)")
                    function_type = _peek_ft
                # Suppress warmup — there's nothing to warm up when loading from a checkpoint
                warmup = False
            except Exception as e:
                warnings.warn(
                    f"Could not peek at predictive_checkpoint for function_type: {e}. "
                    f"Using caller-provided function_type={function_type!r}."
                )

        # ---------------------------
        # Curriculum warmup: compute warmup_steps so Phase 1 cools at the same
        # rate as Phase 2 but stops at warmup_T_min instead of final_temp.
        # niters always refers to Phase 2 steps; total = warmup_steps + niters.
        # ---------------------------
        _do_warmup = (warmup and function_type in ['additive_hill', 'nested_hill'])
        if _do_warmup:
            if warmup_T_min >= init_temp:
                raise ValueError(
                    f"warmup_T_min ({warmup_T_min}) must be less than init_temp ({init_temp})"
                )
            if warmup_T_min < final_temp:
                import warnings as _w
                _w.warn(
                    f"warmup_T_min ({warmup_T_min}) is below final_temp ({final_temp}); "
                    f"Phase 1 will cool past the Phase 2 endpoint."
                )
            # Same cooling rate per step as Phase 2
            warmup_steps = round(niters * (init_temp - warmup_T_min) /
                                 (init_temp - final_temp))
            total_steps = warmup_steps + niters
            print(
                f"[INFO] Curriculum warmup: {warmup_steps} steps as single_hill "
                f"(T: {init_temp} → {warmup_T_min}), then {niters} steps as "
                f"{function_type} (T: {init_temp} → {final_temp}). "
                f"Total: {total_steps} steps."
            )
        else:
            warmup_steps = 0
            total_steps = niters
        
        #if lr is None:
        #    # Default: 100,000 unless multinomial OR polynomial function, then 200,000
        #    if distribution in ['binomial', 'multinomial']:
        #        lr = 1e-4
        #        print(f"[INFO] Using default niters=200,000 for distribution '{distribution}'")
        #    else:
        #        lr = 1e-3
        #        print(f"[INFO] Using default lr=1e-3 for distribution '{distribution}'")

        if (modality.alpha_y_prefit is None
                and 'technical_group_code' in self.model.meta.columns):
            raise ValueError(
                f"Modality '{modality_name}' has not been fit with fit_ntc(). "
                f"Please run fit_ntc(modality_name='{modality_name}') first."
            )

        # Get counts from modality (densify if sparse)
        counts_to_fit = modality.counts
        if hasattr(counts_to_fit, 'toarray'):
            counts_to_fit = counts_to_fit.toarray()

        # Get cell names from modality
        if modality.cell_names is not None:
            modality_cells = modality.cell_names
        else:
            # Modality doesn't have cell names - assume same order as model.meta['cell']
            modality_cells = self.model.meta['cell'].values[:counts_to_fit.shape[modality.cells_axis]]

        # Get technical fit results from modality (NOT self.model.alpha_y_prefit!)
        # DEFENSIVE: Use distribution-specific attributes first (like plot_xy_data does)
        # This is more robust than the generic alpha_y_prefit
        alpha_y_prefit = None

        if distribution == 'negbinom':
            if hasattr(modality, 'alpha_y_prefit_mult') and modality.alpha_y_prefit_mult is not None:
                alpha_y_prefit = modality.alpha_y_prefit_mult
                print(f"[INFO] Using distribution-specific alpha_y_prefit_mult for {distribution}")
        else:
            if hasattr(modality, 'alpha_y_prefit_add') and modality.alpha_y_prefit_add is not None:
                alpha_y_prefit = modality.alpha_y_prefit_add
                print(f"[INFO] Using distribution-specific alpha_y_prefit_add for {distribution}")

        # Fallback to generic attribute if distribution-specific not available
        if alpha_y_prefit is None and hasattr(modality, 'alpha_y_prefit') and modality.alpha_y_prefit is not None:
            alpha_y_prefit = modality.alpha_y_prefit
            print(f"[WARNING] Distribution-specific alpha_y not found, falling back to generic alpha_y_prefit. "
                  f"This may be incorrect if using old technical fit results. "
                  f"Consider re-running fit_ntc with current code.")

        print(f"[INFO] Fitting trans model for modality '{modality_name}' (distribution: {distribution})")

        # Validate min_denominator is specified for binomial/multinomial
        if distribution in ['binomial', 'multinomial'] and min_denominator is None:
            raise ValueError(
                f"min_denominator is required for distribution='{distribution}'. "
                f"Please specify min_denominator (e.g., min_denominator=0 for no filtering, "
                f"or min_denominator=3 for standard quality filtering)."
            )

        # Validate distribution-specific requirements
        from .distributions import requires_sum_factor, requires_denominator

        if requires_sum_factor(distribution) and sum_factor_col is None:
            raise ValueError(f"Distribution '{distribution}' requires sum_factor_col parameter")

        if requires_sum_factor(distribution) and modality.sum_factors is None:
            raise ValueError(
                f"Modality '{modality_name}' has distribution '{distribution}' but no sum_factors "
                f"DataFrame has been set. Assign a cell-indexed DataFrame to "
                f"modality.sum_factors before calling fit_trans()."
            )

        if requires_denominator(distribution) and denominator is None:
            raise ValueError(f"Distribution '{distribution}' requires denominator parameter")

        # convert to gpu for fitting if applicable
        if self.model.x_true is not None and self.model.x_true.device != self.model.device:
            self.model.x_true = self.model.x_true.to(self.model.device)
        if alpha_y_prefit is not None and alpha_y_prefit.device != self.model.device:
            alpha_y_prefit = alpha_y_prefit.to(self.model.device)

        if not hasattr(self.model, "log2_x_true") or self.model.log2_x_true is None:
            if self.model.x_true is not None:
                self.model.log2_x_true = torch.log2(self.model.x_true)

        # Handle cell subsetting
        # CRITICAL: Preserve exact modality cell order when subsetting meta
        # Use .set_index().loc[] to ensure meta_subset has same order as modality_cells
        meta_with_index = self.model.meta.set_index('cell')
        meta_subset = meta_with_index.loc[modality_cells].reset_index()

        # Verify we got all cells (debugging)
        if len(meta_subset) != len(modality_cells):
            missing = set(modality_cells) - set(meta_subset['cell'].values)
            raise ValueError(f"Some modality cells not found in meta: {missing}")

        # Check if technical_group_code exists (for correction)
        if "technical_group_code" in meta_subset.columns:
            C = meta_subset['technical_group_code'].nunique()
            groups_tensor = torch.tensor(meta_subset['technical_group_code'].values, dtype=torch.long, device=self.model.device)
            print(f"[INFO] Using technical_group_code with {C} groups for correction")
        else:
            C = None
            groups_tensor = None
            if alpha_y_prefit is None:
                warnings.warn("no alpha_y_prefit and no technical_group_code, assuming no confounding effect.")

        # Compute x_true subset for this modality's cells.
        # Modalities may cover a strict subset of model cells; we index x_true
        # by position in self.model.meta so the N-length x_true_subset aligns
        # exactly with the N-length y_obs.
        _all_model_cells = self.model.meta['cell'].tolist()
        _cell_index_map = {c: i for i, c in enumerate(_all_model_cells)}
        _cell_indices = [_cell_index_map[c] for c in modality_cells]
        _cell_idx_tensor = torch.tensor(_cell_indices, dtype=torch.long, device=self.model.device)
        x_true_subset = self.model.x_true[_cell_idx_tensor]
        log2_x_true_subset = torch.log2(x_true_subset)

        N = len(modality_cells)

        # Modality counts are already in correct order matching modality_cells
        # Handle both 2D and 3D arrays
        if counts_to_fit.ndim == 2:
            if modality.cells_axis == 1:
                y_obs = counts_to_fit.T  # [T, N] -> [N, T]
            else:
                y_obs = counts_to_fit  # Already [N, T]
            T = y_obs.shape[1]
        elif counts_to_fit.ndim == 3:
            # 3D data: (features, cells, categories/dimensions)
            # Transpose to (cells, features, categories/dimensions)
            y_obs = counts_to_fit.transpose(1, 0, 2)  # [T, N, K] -> [N, T, K]
            T = counts_to_fit.shape[0]
        else:
            raise ValueError(f"Unexpected number of dimensions: {counts_to_fit.ndim}")

        # Handle sum factors for modality cells
        if sum_factor_col is not None:
            sum_factor_vals = modality.sum_factors.loc[meta_subset['cell'].values, sum_factor_col].values
            # Guard: zero, NaN, or inf sum factors will produce -inf/NaN logits in the likelihood
            # (log(0) = -inf → NaN gradients for all features of that cell).
            n_zero = int((sum_factor_vals == 0).sum())
            n_nonfinite = int(~np.isfinite(sum_factor_vals).sum())
            if n_nonfinite > 0:
                raise ValueError(
                    f"[fit_trans] Column '{sum_factor_col}' has {n_nonfinite} non-finite (NaN/inf) value(s). "
                    f"Please fix before fitting."
                )
            if n_zero > 0:
                raise ValueError(
                    f"[fit_trans] Column '{sum_factor_col}' has {n_zero} zero value(s). "
                    f"Zero sum factors produce log(0)=-inf logits and NaN gradients for all features. "
                    f"If using sum_factor_refit, re-run refit_sumfactor() — it now floors values above zero."
                )
            sum_factor_tensor = torch.tensor(
                sum_factor_vals,
                dtype=torch.float32, device=self.model.device
            )
        else:
            sum_factor_tensor = torch.ones(N, dtype=torch.float32, device=self.model.device)

        # Handle denominator for modality cells
        denominator_tensor = None
        if denominator is not None:
            if denominator.ndim == 2:
                if modality.cells_axis == 1:
                    denominator_subset = denominator.T  # [T, N] -> [N, T]
                else:
                    denominator_subset = denominator  # Already [N, T]
                denominator_tensor = torch.tensor(denominator_subset, dtype=torch.float32, device=self.model.device)
            elif denominator.ndim == 3:
                # 3D denominator (shouldn't happen for current distributions, but handle it)
                denominator_subset = denominator.transpose(1, 0, 2)
                denominator_tensor = torch.tensor(denominator_subset, dtype=torch.float32, device=self.model.device)

            # Apply min_denominator filter if specified
            if min_denominator is not None and min_denominator > 0:
                # Create mask for observations where denominator < threshold
                low_coverage_mask = denominator_tensor < min_denominator
                n_masked = low_coverage_mask.sum().item()
                n_total = denominator_tensor.numel()
                pct_masked = 100 * n_masked / n_total if n_total > 0 else 0

                print(f"[INFO] Filtering observations with denominator < {min_denominator}")
                print(f"[INFO] Masked {n_masked}/{n_total} observations ({pct_masked:.1f}%)")

                # For binomial distributions, we'll pass the mask to the model
                # The sampler will need to handle it (for now, set those observations to special value)
                # We'll use a very negative value that the sampler can detect
                # Actually, better approach: modify y_obs to have NaN for masked observations
                # But binomial doesn't support NaN observations...
                # Best approach: set denominator=0 for masked observations, and sampler handles it
                denominator_tensor = torch.where(low_coverage_mask,
                                                 torch.zeros_like(denominator_tensor),
                                                 denominator_tensor)

        # Detect data dimensions (for multinomial)
        from .distributions import is_3d_distribution
        K = None
        D = None
        if is_3d_distribution(distribution):
            if y_obs.ndim == 3:
                if distribution == 'multinomial':
                    K = y_obs.shape[2]  # Number of categories
            else:
                raise ValueError(f"Distribution '{distribution}' requires 3D data but got shape {y_obs.shape}")
        x_true_mean = x_true_subset
        beta_o_alpha_tensor = torch.tensor(beta_o_alpha, dtype=torch.float32, device=self.model.device)
        beta_o_beta_tensor = torch.tensor(beta_o_beta, dtype=torch.float32, device=self.model.device)
        alpha_alpha_mu_tensor = torch.tensor(alpha_alpha_mu, dtype=torch.float32, device=self.model.device)
        K_alpha_tensor = torch.tensor(K_alpha, dtype=torch.float32, device=self.model.device)
        Vmax_alpha_tensor = torch.tensor(Vmax_alpha, dtype=torch.float32, device=self.model.device)
        n_mu_tensor = torch.tensor(n_mu, dtype=torch.float32, device=self.model.device)
        _k_log_sigma_min_val = k_log_sigma_min if k_log_sigma_min is not None else 5.0 * 0.6931 / 2.0
        vmax_log_sigma_floor_tensor = torch.tensor(vmax_log_sigma_floor, dtype=torch.float32, device=self.model.device)
        k_log_sigma_min_tensor = torch.tensor(_k_log_sigma_min_val, dtype=torch.float32, device=self.model.device)
        y_obs_tensor = torch.tensor(y_obs, dtype=torch.float32, device=self.model.device)
        epsilon_tensor = torch.tensor(epsilon, dtype=torch.float32, device=self.model.device)
        p_n_tensor = torch.tensor(p_n, dtype=torch.float32, device=self.model.device)
        # Pre-compute logits for RelaxedBernoulli (more numerically stable than probs)
        # logits = log(p / (1-p))
        p_n_logits_tensor = torch.tensor(np.log(p_n / (1 - p_n)), dtype=torch.float32, device=self.model.device)

        # --- robust, finite bounds for n to avoid overflow in x**n ---
        # use the same x_true sample type you use elsewhere
        x_for_bounds = x_true_subset
        x_min = torch.clamp(x_for_bounds.min(), min=1e-12)  # strictly > 0 to avoid log(0)
        x_max = x_for_bounds.max()

        log_fmax = torch.log(torch.tensor(torch.finfo(torch.float32).max, device=self.model.device))

        # candidates can be inf if denominator ~ 0; we cap them later
        nmin_cand = (-log_fmax / torch.abs(torch.log(x_min))) if (x_min < 1) else torch.tensor(float('-inf'), device=self.model.device)
        nmax_cand = ( log_fmax / torch.abs(torch.log(x_max))) if (x_max > 1) else torch.tensor(float('inf'),  device=self.model.device)

        # Use physically-derived overflow bounds directly; fall back to ±100 if infinite
        # (nmin_cand is -inf when x_min >= 1; nmax_cand is +inf when x_max <= 1)
        nmin = torch.where(torch.isfinite(nmin_cand), nmin_cand, torch.tensor(-100.0, device=self.model.device))
        nmax = torch.where(torch.isfinite(nmax_cand), nmax_cand, torch.tensor( 100.0, device=self.model.device))
        # ensure proper ordering just in case
        nmin = torch.minimum(nmin, nmax)

        # CRITICAL: Use meta_subset (correctly ordered) instead of full self.model.meta
        guides_tensor = torch.tensor(meta_subset['guide_code'].values, dtype=torch.long, device=self.model.device)

        # Distribution-specific normalization for data-driven priors
        # For building priors later:
        y_obs_for_prior = None

        # Note: min_denominator is now required for binomial/multinomial (validated earlier)
        # For negbinom/normal/studentt, it's not used so None is fine

        if distribution == 'binomial' and denominator_tensor is not None:
            # Full probabilities for the likelihood
            y_obs_factored = y_obs_tensor / denominator_tensor.clamp_min(epsilon_tensor)

            # Store valid_mask for later (will apply AFTER technical correction)
            valid_mask_binomial = (denominator_tensor >= min_denominator)
            print(f"[INFO] Binomial: will use {valid_mask_binomial.float().mean().item()*100:.1f}% of entries with denominator >= {min_denominator} for priors (after technical correction)")

            # Don't create y_obs_for_prior yet - technical correction happens first
            y_obs_for_prior = y_obs_factored

        elif distribution == 'multinomial' and y_obs_tensor.ndim == 3:
            total_counts = y_obs_tensor.sum(dim=-1, keepdim=True).clamp_min(epsilon_tensor)  # [N, T, 1]
            y_obs_factored = y_obs_tensor / total_counts  # [N, T, K]

            # Store valid_mask for later (will apply AFTER technical correction)
            valid_mask_multinomial = (total_counts >= min_denominator)  # [N, T, 1]
            print(f"[INFO] Multinomial: will use {valid_mask_multinomial.float().mean().item()*100:.1f}% of entries with total counts >= {min_denominator} for priors (after technical correction)")

            # Don't create y_obs_for_prior yet - technical correction happens first
            y_obs_for_prior = y_obs_factored

        elif sum_factor_col is not None:
            y_obs_factored = y_obs_tensor / sum_factor_tensor.view(-1, 1)
            y_obs_for_prior = y_obs_factored
        else:
            y_obs_factored = y_obs_tensor
            y_obs_for_prior = y_obs_factored

        # ---- Multinomial zero-category mask (analogous to fit_ntc zero_cat_mask) ----
        # Identifies categories structurally absent across ALL trans cells.
        # These phantom positions are already masked inline inside _model_y; this block
        # performs an early diagnostic check and warns when features have ≤1 active
        # category (which should have been filtered at Modality initialisation).
        if distribution == 'multinomial' and y_obs_tensor.ndim == 3:
            _obs_total_per_cat = y_obs_tensor.sum(dim=0)  # [T, K]
            zero_cat_mask_trans = (_obs_total_per_cat == 0)  # [T, K] bool
            _active_k = (~zero_cat_mask_trans).sum(dim=-1)  # [T]
            if (_active_k <= 1).any():
                import warnings as _warnings
                _bad_t = torch.nonzero(_active_k <= 1, as_tuple=False).squeeze(-1).tolist()
                _warnings.warn(
                    f"[fit_trans] {len(_bad_t)} multinomial feature(s) have ≤1 active "
                    f"category in the trans data (feature indices: "
                    f"{_bad_t[:10]}{'...' if len(_bad_t) > 10 else ''}). "
                    f"These features should be filtered before fitting (at Modality "
                    f"initialisation). Fitting will be unreliable for these features.",
                    UserWarning,
                    stacklevel=3,
                )
        else:
            zero_cat_mask_trans = None

        # ===================================================================
        # CORRECT FOR TECHNICAL EFFECTS BEFORE COMPUTING PRIORS
        # ===================================================================
        # The observed data includes technical batch effects. To compute unbiased
        # priors for A and Vmax (baseline parameters), we can optionally remove these effects
        # using the inverse transformation.
        # NOTE: Archive code does NOT correct priors - it uses raw sum_factor-normalized data.
        # Technical effects are only applied during model fitting via alpha_y_sample.
        # Setting correct_priors_for_technical=False (default) matches archive behavior.
        if correct_priors_for_technical and alpha_y_prefit is not None and groups_tensor is not None:
            print(f"[INFO] Correcting for technical effects before computing priors (distribution: {distribution})")

            # alpha_y_prefit is always [C, T] (point estimate/mean), index groups dimension
            alpha_y_expanded = alpha_y_prefit[groups_tensor, ...]  # [N, T] or [N, T, K]

            # alpha_y_expanded is now [N, T] (or [N, T, K] for multinomial), matching y_obs_for_prior
            # NO TRANSPOSE NEEDED - both are [N, T]

            # Diagnostic: Check alpha_y_expanded for issues
            if not torch.isfinite(alpha_y_expanded).all():
                n_invalid_alpha = (~torch.isfinite(alpha_y_expanded)).sum().item()
                print(f"[WARNING] alpha_y_expanded contains {n_invalid_alpha} non-finite values before correction")
            else:
                # Check if values look like multiplicative (around 1.0) vs additive (logit scale)
                alpha_mean = alpha_y_expanded.mean().item()
                alpha_std = alpha_y_expanded.std().item()
                print(f"[DEBUG] alpha_y_expanded stats: mean={alpha_mean:.4f}, std={alpha_std:.4f}")
                if distribution == 'binomial' and abs(alpha_mean - 1.0) < 0.5:
                    print(f"[WARNING] alpha_y_expanded looks like multiplicative correction (mean~1.0). "
                          f"For binomial, expected additive correction on logit scale (mean~0, range~[-5,5]). "
                          f"You may be using technical fit results from old (buggy) code. "
                          f"Re-run fit_ntc with the fixed code.")

            if distribution == 'negbinom':
                # Technical effect: multiplicative (mu_corrected = mu * alpha_y_mult)
                # Inverse: divide by alpha_y_mult to get baseline
                # alpha_y_prefit for negbinom is multiplicative (from fit_ntc)
                y_obs_for_prior = y_obs_for_prior / alpha_y_expanded.clamp_min(epsilon_tensor)
                print(f"[INFO] negbinom: Applied inverse multiplicative correction (divide by alpha_y_mult)")

            elif distribution in ['normal', 'studentt']:
                # Technical effect: additive (mu_corrected = mu + alpha_y_add)
                # Inverse: subtract alpha_y_add to get baseline
                # alpha_y_prefit for normal/studentt is additive (from fit_ntc)
                y_obs_for_prior = y_obs_for_prior - alpha_y_expanded
                print(f"[INFO] {distribution}: Applied inverse additive correction (subtract alpha_y_add)")

            elif distribution == 'binomial':
                # Technical effect: logit scale (logit(p_corrected) = logit(p) + alpha_y_add)
                # Inverse: logit(p_baseline) = logit(p_observed) - alpha_y_add
                # Then: p_baseline = sigmoid(logit(p_baseline))

                # Convert observed proportions to logit scale
                p_obs_clamped = torch.clamp(y_obs_for_prior, min=epsilon_tensor, max=1.0 - epsilon_tensor)
                logit_obs = torch.log(p_obs_clamped) - torch.log(1.0 - p_obs_clamped)

                # Diagnostic: Check logit_obs range
                print(f"[DEBUG] logit_obs range: [{logit_obs.min().item():.2f}, {logit_obs.max().item():.2f}]")
                print(f"[DEBUG] alpha_y_expanded range: [{alpha_y_expanded.min().item():.4f}, {alpha_y_expanded.max().item():.4f}]")

                # Apply inverse correction on logit scale
                logit_baseline = logit_obs - alpha_y_expanded

                # Diagnostic: Check logit_baseline range and finite values
                print(f"[DEBUG] logit_baseline range: [{logit_baseline.min().item():.2f}, {logit_baseline.max().item():.2f}]")
                if not torch.isfinite(logit_baseline).all():
                    n_invalid = (~torch.isfinite(logit_baseline)).sum().item()
                    print(f"[WARNING] logit_baseline contains {n_invalid} non-finite values after correction")

                # Convert back to probability scale
                y_obs_for_prior = torch.sigmoid(logit_baseline)
                print(f"[INFO] binomial: Applied inverse logit correction (subtract alpha_y_add on logit scale)")

                # DIAGNOSTIC: Check correction for specific feature
                sj_check_idx = 1298  # User's specific SJ position
                if sj_check_idx < T:
                    # Get uncorrected values (observed PSI)
                    y_uncorrected = y_obs_factored[:, sj_check_idx]

                    # Get corrected values (baseline PSI)
                    y_corrected = y_obs_for_prior[:, sj_check_idx]

                    # Check by group
                    print(f"[DEBUG] Checking inverse correction for feature {sj_check_idx} (chr6:34236964:34237203):")
                    for grp in range(C):
                        grp_mask = groups_tensor == grp
                        if grp_mask.sum() > 0:
                            uncorr_mean = y_uncorrected[grp_mask].mean().item()
                            corr_mean = y_corrected[grp_mask].mean().item()
                            alpha_val = alpha_y_prefit[grp, sj_check_idx].item() if alpha_y_prefit.ndim == 2 else alpha_y_prefit[0, grp, sj_check_idx].mean().item()
                            print(f"  Group {grp} (n={grp_mask.sum()}): α={alpha_val:.4f}, Observed PSI={uncorr_mean:.4f}, Baseline PSI={corr_mean:.4f}")

            elif distribution == 'multinomial':
                # Technical effect: log scale (log(probs_corrected) = log(probs) + alpha_y_add)
                # Inverse: log(probs_baseline) = log(probs_observed) - alpha_y_add
                # Then: probs_baseline = exp(log_probs_baseline) / sum(exp(...))
                # For multinomial, alpha_y_expanded is already [N, T, K] from above (matches y_obs_for_prior)

                # Convert observed proportions to log scale
                p_obs_clamped = torch.clamp(y_obs_for_prior, min=epsilon_tensor)
                log_probs_obs = torch.log(p_obs_clamped)

                # Apply inverse correction on log scale
                log_probs_baseline = log_probs_obs - alpha_y_expanded

                # Normalize (softmax) to get valid probabilities
                y_obs_for_prior = torch.softmax(log_probs_baseline, dim=-1)
                print(f"[INFO] multinomial: Applied inverse log correction (subtract alpha_y_add on log scale)")

            # Handle any NaN or invalid values that may result from correction
            # (e.g., if correction pushes values outside valid range)
            if not torch.isfinite(y_obs_for_prior).all():
                n_invalid = (~torch.isfinite(y_obs_for_prior)).sum().item()
                n_total = y_obs_for_prior.numel()
                print(f"[WARNING] Technical correction produced {n_invalid}/{n_total} "
                      f"({100*n_invalid/n_total:.2f}%) non-finite values. "
                      f"These will be excluded from prior computation.")
                # Mark invalid values as NaN so they're excluded by nanmean/nanvar
                y_obs_for_prior = torch.where(
                    torch.isfinite(y_obs_for_prior),
                    y_obs_for_prior,
                    torch.full_like(y_obs_for_prior, float('nan'))
                )
        else:
            if alpha_y_prefit is not None:
                print(f"[INFO] alpha_y_prefit provided but groups_tensor is None - skipping technical correction")

        # NOW apply min_denominator masking (AFTER technical correction)
        # This prevents computing priors from low-coverage observations
        if distribution == 'binomial' and 'valid_mask_binomial' in locals():
            # Mask low-coverage entries with NaN
            y_obs_for_prior = torch.where(
                valid_mask_binomial,
                y_obs_for_prior,
                torch.full_like(y_obs_for_prior, float('nan'))
            )
            n_masked = (~valid_mask_binomial).sum().item()
            n_total = valid_mask_binomial.numel()
            print(f"[INFO] Masked {n_masked}/{n_total} ({100*n_masked/n_total:.2f}%) low-coverage observations (denominator < {min_denominator})")

        elif distribution == 'multinomial' and 'valid_mask_multinomial' in locals():
            # Mask low-coverage entries with NaN
            y_obs_for_prior = torch.where(
                valid_mask_multinomial,
                y_obs_for_prior,
                torch.full_like(y_obs_for_prior, float('nan'))
            )
            n_masked = (~valid_mask_multinomial).sum().item()
            n_total = valid_mask_multinomial.numel()
            print(f"[INFO] Masked {n_masked}/{n_total} ({100*n_masked/n_total:.2f}%) low-coverage observations (total counts < {min_denominator})")

        unique_guides = torch.unique(guides_tensor)

        # nanmean and nanvar helpers (in case torch.nanmean/nanvar isn't available)
        if hasattr(torch, "nanmean"):
            def nanmean(x, dim):
                return torch.nanmean(x, dim=dim)
        else:
            def nanmean(x, dim):
                mask = ~torch.isnan(x)
                num = torch.where(mask, x, torch.zeros_like(x)).sum(dim=dim)
                den = mask.sum(dim=dim).clamp_min(1)
                return num / den

        if hasattr(torch, "nanvar"):
            def nanvar(x, dim):
                return torch.var(x, dim=dim)  # Note: torch.var ignores NaN in older versions
        else:
            def nanvar(x, dim):
                mask = ~torch.isnan(x)
                n = mask.sum(dim=dim).clamp_min(2)  # Need at least 2 values for variance
                x_mean = nanmean(x, dim=dim)
                # Expand mean to match x shape for broadcasting
                if dim == 0:
                    x_mean_expanded = x_mean.unsqueeze(0)
                else:
                    x_mean_expanded = x_mean
                sq_diff = torch.where(mask, (x - x_mean_expanded) ** 2, torch.zeros_like(x))
                return sq_diff.sum(dim=dim) / (n - 1)  # Unbiased variance

        # nanquantile helper
        def nanquantile(x, q, dim):
            """Compute quantile ignoring NaN values."""
            if x.ndim == 2:  # [G, T]
                result = []
                for t in range(x.shape[1]):
                    vals = x[:, t]
                    valid = vals[~torch.isnan(vals)]
                    if valid.numel() > 0:
                        result.append(torch.quantile(valid, q))
                    else:
                        result.append(torch.tensor(float('nan'), device=x.device))
                return torch.stack(result)  # [T]
            elif x.ndim == 3:  # [G, T, K]
                result = []
                for t in range(x.shape[1]):
                    result_k = []
                    for k in range(x.shape[2]):
                        vals = x[:, t, k]
                        valid = vals[~torch.isnan(vals)]
                        if valid.numel() > 0:
                            result_k.append(torch.quantile(valid, q))
                        else:
                            result_k.append(torch.tensor(float('nan'), device=x.device))
                    result.append(torch.stack(result_k))
                return torch.stack(result)  # [T, K]
            else:
                raise ValueError(f"Unsupported ndim for nanquantile: {x.ndim}")

        # Pre-extract q01 of NTC posterior means for the Q05=0 floor below.
        # Only needed for negbinom/binomial; full NTC anchor extraction happens later.
        _q01_ntc_for_floor = None
        if distribution in ('negbinom', 'binomial'):
            _post_tech_pre = getattr(modality, 'posterior_samples_ntc', None)
            if _post_tech_pre is not None and 'mu_ntc' in _post_tech_pre:
                try:
                    _mu_ntc_pre = _post_tech_pre['mu_ntc']
                    if not torch.is_tensor(_mu_ntc_pre):
                        _mu_ntc_pre = torch.tensor(_mu_ntc_pre, dtype=torch.float32, device=self.model.device)
                    else:
                        _mu_ntc_pre = _mu_ntc_pre.float().to(self.model.device)
                    _mu_ntc_pre_flat = _mu_ntc_pre
                    while _mu_ntc_pre_flat.ndim > 1:
                        _mu_ntc_pre_flat = _mu_ntc_pre_flat.median(dim=0).values
                    _finite_pre = _mu_ntc_pre_flat[(_mu_ntc_pre_flat > 0) & ~torch.isnan(_mu_ntc_pre_flat)]
                    if _finite_pre.numel() > 0:
                        _q01_ntc_for_floor = torch.quantile(_finite_pre, 0.01)
                except Exception:
                    pass

        # Compute cell-level Q05/Q95 for Amean/Vmax priors (always, regardless of guide count)
        _q05_was_zero = None
        mean_y_corrected_tensor = None  # [T]; set below for 2D distributions (negbinom, binomial)
        if y_obs_for_prior.ndim == 2:  # [N, T]
            Amean_list, Vmax_list, cell_var_list = [], [], []
            for t in range(T):
                vals_t = y_obs_for_prior[:, t]
                valid_t = vals_t[~torch.isnan(vals_t)]
                if valid_t.numel() > 0:
                    Amean_list.append(torch.quantile(valid_t, 0.05))
                    Vmax_list.append(torch.quantile(valid_t, 0.95))
                    cell_var_list.append(torch.var(valid_t))
                else:
                    Amean_list.append(torch.tensor(float('nan'), device=self.model.device))
                    Vmax_list.append(torch.tensor(float('nan'), device=self.model.device))
                    cell_var_list.append(torch.tensor(float('nan'), device=self.model.device))

            Amean_tensor = torch.stack(Amean_list)    # [T]; cell Q05
            Vmax_raw     = torch.stack(Vmax_list)     # [T]; cell Q95
            cell_var     = torch.stack(cell_var_list) # [T]

            # Q05=0 floor: for count/proportion distributions (negbinom, binomial), Q05=0
            # means the gene is sparse and needs a small positive lower bound because the
            # negbinom A prior uses log2(Amean/2), which requires Amean > 0.
            # Use a fixed small constant (2^-10 ≈ 1e-3 normalized counts) rather than the
            # data-driven min non-zero Q05, which can be inflated (~0.5–1 count) and would
            # wrongly pull the prior lower bound above the NTC expression level for sparse genes.
            # The y_ntc anchor provides the upper reference; Amean just needs a valid lower bound.
            #
            # For normal/studentt, Q05 can legitimately be ≤ 0 (e.g. negative SpliZ scores),
            # and the linear-space prior works fine with negative Amean — so only floor NaNs.
            # NaN Amean values (no valid observations) are handled by the fallback below.
            if distribution in ('negbinom', 'binomial'):
                _q05_was_zero = (Amean_tensor <= 0) | torch.isnan(Amean_tensor)
            else:  # normal, studentt: Q05 can be ≤ 0; only catch NaN
                _q05_was_zero = torch.isnan(Amean_tensor)
            if _q05_was_zero.any():
                n_q05_zero = _q05_was_zero.sum().item()
                _amean_floor = self._t(2.0 ** -10)
                if _q01_ntc_for_floor is not None:
                    _amean_floor = torch.minimum(_q01_ntc_for_floor, _amean_floor)
                Amean_tensor = torch.where(_q05_was_zero, _amean_floor.expand_as(Amean_tensor), Amean_tensor)
                print(f"[INFO] Q05 floor: {n_q05_zero} features raised to {_amean_floor.item():.3e} (min(q01_ntc, 2^-10))")
            Amean_tensor     = Amean_tensor.clamp_min(self._t(1e-12))
            Vmax_mean_tensor = (Vmax_raw - Amean_tensor).clamp_min(self._t(1e-3))

            # Mean across all cells — used as a data-driven floor for the A prior upper
            # anchor. When y_ntc ≈ 0 (gene absent in NTC but expressed in perturbations),
            # this prevents both anchors collapsing to the same floor value.
            mean_y_corrected_tensor = torch.nanmean(y_obs_for_prior, dim=0).clamp_min(self._t(1e-12))  # [T]

        elif y_obs_for_prior.ndim == 3:  # [N, T, K] — multinomial
            Amean_list, Vmax_list, cell_var_list = [], [], []
            for t in range(T):
                Amean_k, Vmax_k, var_k = [], [], []
                for k in range(y_obs_for_prior.shape[2]):
                    vals_tk = y_obs_for_prior[:, t, k]
                    valid_tk = vals_tk[~torch.isnan(vals_tk)]
                    if valid_tk.numel() > 0:
                        Amean_k.append(torch.quantile(valid_tk, 0.05))
                        Vmax_k.append(torch.quantile(valid_tk, 0.95))
                        var_k.append(torch.var(valid_tk))
                    else:
                        Amean_k.append(torch.tensor(float('nan'), device=self.model.device))
                        Vmax_k.append(torch.tensor(float('nan'), device=self.model.device))
                        var_k.append(torch.tensor(float('nan'), device=self.model.device))
                Amean_list.append(torch.stack(Amean_k))
                Vmax_list.append(torch.stack(Vmax_k))
                cell_var_list.append(torch.stack(var_k))

            Amean_tensor = torch.stack(Amean_list)  # [T, K]
            # Q05=0 floor: use fixed small constant (proportions clamped at 1e-3 below anyway).
            _q05_was_zero = (Amean_tensor <= 0) | torch.isnan(Amean_tensor)
            if _q05_was_zero.any():
                n_q05_zero = _q05_was_zero.sum().item()
                _amean_floor_mn = self._t(2.0 ** -10)
                Amean_tensor = torch.where(_q05_was_zero, _amean_floor_mn.expand_as(Amean_tensor), Amean_tensor)
                print(f"[INFO] Q05=0 floor (multinomial): {n_q05_zero} (feature, category) cells raised to 2^-10={2.0**-10:.3e}")
            Amean_tensor     = Amean_tensor.clamp_min(self._t(1e-12))   # [T, K]
            Vmax_raw         = torch.stack(Vmax_list)                    # [T, K]
            cell_var         = torch.stack(cell_var_list)                # [T, K]
            Vmax_mean_tensor = (Vmax_raw - Amean_tensor).clamp_min(self._t(1e-3))

        # mean_within_guide_var: guide-level if guides available, else cell-level
        if 'guide_code' in self.model.meta.columns and len(unique_guides) > 1:
            _guide_vars = []
            for g in unique_guides:
                vals_g = y_obs_for_prior[guides_tensor == g, ...]
                _guide_vars.append(nanvar(vals_g, dim=0))
            mean_within_guide_var = nanmean(torch.stack(_guide_vars, dim=0), dim=0)
            print(f"[INFO] Cell-level Q05/Q95 priors ({T} features, {len(unique_guides)} guides for within-guide variance)")
        else:
            mean_within_guide_var = cell_var
            print(f"[INFO] Cell-level Q05/Q95 priors ({T} features, no guide stratification)")

        # For binomial/multinomial: clamp Vmax_mean to valid Beta range
        if distribution in ['binomial', 'multinomial']:
            Vmax_mean_tensor = Vmax_mean_tensor.clamp(min=self._t(1e-3), max=self._t(1.0 - 1e-6))
            Amean_tensor = Amean_tensor.clamp(min=self._t(1e-3), max=self._t(1.0 - 1e-6))

        # Handle NaN values (features where ALL observations were filtered out)
        nan_mask = torch.isnan(Amean_tensor) | torch.isnan(Vmax_mean_tensor)
        if nan_mask.any():
            n_nan = nan_mask.sum().item() if nan_mask.ndim == 1 else nan_mask.any(dim=-1).sum().item()
            print(f"[WARNING] {n_nan} features have all observations filtered (denominator < {min_denominator}). Using fallback values.")

            # Fallback: median of valid Amean/cell_var across features
            if distribution in ['binomial', 'multinomial']:
                valid_A   = Amean_tensor[~torch.isnan(Amean_tensor)]
                valid_var = cell_var[~torch.isnan(cell_var)]
                if valid_A.numel() > 0:
                    valid_mean    = torch.median(valid_A)
                    valid_var_val = torch.median(valid_var) if valid_var.numel() > 0 else self._t(0.1)
                else:
                    print("[WARNING] No valid observations found! Using generic defaults: A=0.3, var=0.1")
                    valid_mean    = self._t(0.3)
                    valid_var_val = self._t(0.1)
                print(f"[INFO] Using fallback: mean={valid_mean.item():.3f}, var={valid_var_val.item():.3f}")
                fallback_A    = torch.clamp(valid_mean * 0.5, min=0.01, max=0.99)
                fallback_Vmax = torch.clamp(valid_mean,       min=0.01, max=0.99)
            else:  # negbinom, normal, studentt
                valid_A   = Amean_tensor[~torch.isnan(Amean_tensor)]
                valid_var = cell_var[~torch.isnan(cell_var)]
                if valid_A.numel() > 0:
                    valid_mean    = torch.median(valid_A)
                    valid_var_val = torch.median(valid_var) if valid_var.numel() > 0 else valid_mean * 0.5
                else:
                    print("[WARNING] No valid observations found! Using generic defaults: A=5.0, var=5.0")
                    valid_mean    = self._t(5.0)
                    valid_var_val = self._t(5.0)
                print(f"[INFO] Using fallback: mean={valid_mean.item():.3f}, var={valid_var_val.item():.3f}")
                fallback_A    = torch.clamp(valid_mean * 0.5, min=1e-3)
                fallback_Vmax = torch.clamp(valid_mean,       min=1e-3)

            Amean_tensor = torch.where(
                torch.isnan(Amean_tensor), torch.full_like(Amean_tensor, fallback_A), Amean_tensor)
            Vmax_mean_tensor = torch.where(
                torch.isnan(Vmax_mean_tensor), torch.full_like(Vmax_mean_tensor, fallback_Vmax), Vmax_mean_tensor)
            mean_within_guide_var = torch.where(
                torch.isnan(mean_within_guide_var), torch.full_like(mean_within_guide_var, valid_var_val), mean_within_guide_var)

        # --- A prior: NTC anchor (all distributions) ---
        # Extract y_ntc_tensor (posterior mean of NTC expression per feature) to anchor the
        # noisy-feature end of the A prior in _model_y, preventing A-collapse.
        # For negbinom: also floors Amean at q01(mu_ntc) to prevent near-zero Q05 estimates.
        # Keys per distribution:
        #   negbinom/normal/studentt/binomial: posterior_samples_ntc['mu_ntc']  → [S, T]
        #   multinomial:                       posterior_samples_ntc['probs_baseline'] → [S, T, K]
        y_ntc_tensor = None
        _post_tech = getattr(modality, 'posterior_samples_ntc', None)
        if _post_tech is not None:
            try:
                if distribution in ('negbinom', 'normal', 'studentt', 'binomial') and 'mu_ntc' in _post_tech:
                    _mu_ntc = _post_tech['mu_ntc']
                    if not torch.is_tensor(_mu_ntc):
                        _mu_ntc = torch.tensor(_mu_ntc, dtype=torch.float32, device=self.model.device)
                    else:
                        _mu_ntc = _mu_ntc.float().to(self.model.device)
                    # Collapse any leading dimensions (samples, groups, etc.) so that
                    # _mu_ntc ends up as [T] regardless of whether it is stored as
                    # [S, T], [S, C, T], or already [T].
                    if _mu_ntc.ndim > 1 and _mu_ntc.shape[-1] == T:
                        # Flatten all leading dims by successive mean over dim=0
                        _mu_ntc_flat = _mu_ntc
                        while _mu_ntc_flat.ndim > 1:
                            _mu_ntc_flat = _mu_ntc_flat.median(dim=0).values
                        if distribution == 'negbinom':
                            n_nonpos = (_mu_ntc_flat <= 0).sum().item()
                            if n_nonpos > 0:
                                print(f"[DEBUG] {n_nonpos}/{T} genes have mean mu_ntc ≤ 0 "
                                      f"(min={_mu_ntc_flat.min().item():.3e}); possible float32 underflow in NTC posterior")
                        y_ntc_tensor = _mu_ntc_flat  # [T]
                        # Compute q01_ntc_global BEFORE NaN fill: 1st percentile of per-gene NTC means
                        # across genes where fit_ntc was run (ignores NaN genes).
                        _finite_ntc = _mu_ntc_flat[~torch.isnan(_mu_ntc_flat)]
                        q01_ntc_global = torch.quantile(_finite_ntc, 0.01) if _finite_ntc.numel() > 0 else None
                        # NaN fill: genes where fit_ntc was not run get q01_ntc_global as anchor
                        _ntc_nan_mask = torch.isnan(y_ntc_tensor)
                        if _ntc_nan_mask.any():
                            n_ntc_nan = _ntc_nan_mask.sum().item()
                            if q01_ntc_global is not None:
                                print(f"[WARNING] {n_ntc_nan}/{T} genes have NaN mu_ntc (fit_ntc not run); "
                                      f"using q01_ntc_global={q01_ntc_global.item():.3e} as y_ntc anchor")
                                y_ntc_tensor = torch.where(_ntc_nan_mask, q01_ntc_global.expand_as(y_ntc_tensor), y_ntc_tensor)
                            else:
                                print(f"[WARNING] {n_ntc_nan}/{T} genes have NaN mu_ntc and no finite NTC posterior; "
                                      f"using Amean as y_ntc fallback")
                                y_ntc_tensor = torch.where(_ntc_nan_mask, Amean_tensor, y_ntc_tensor)
                        if distribution == 'negbinom':
                            if (y_ntc_tensor <= 0).any():
                                raise ValueError(
                                    f"y_ntc_tensor contains non-positive values "
                                    f"(min={y_ntc_tensor.min().item():.3e}). "
                                    f"mu_ntc posterior has strictly positive support — this should not occur."
                                )
                    elif _mu_ntc.ndim == 1 and _mu_ntc.shape[0] == T:
                        y_ntc_tensor = _mu_ntc
                        _finite_ntc_1d = _mu_ntc[~torch.isnan(_mu_ntc)]
                        q01_ntc_global = torch.quantile(_finite_ntc_1d, 0.01) if _finite_ntc_1d.numel() > 0 else None
                        _ntc_nan_mask = torch.isnan(y_ntc_tensor)
                        if _ntc_nan_mask.any():
                            n_ntc_nan = _ntc_nan_mask.sum().item()
                            if q01_ntc_global is not None:
                                print(f"[WARNING] {n_ntc_nan}/{T} genes have NaN mu_ntc (fit_ntc not run); "
                                      f"using q01_ntc_global={q01_ntc_global.item():.3e} as y_ntc anchor")
                                y_ntc_tensor = torch.where(_ntc_nan_mask, q01_ntc_global.expand_as(y_ntc_tensor), y_ntc_tensor)
                            else:
                                print(f"[WARNING] {n_ntc_nan}/{T} genes have NaN mu_ntc and no finite NTC posterior; "
                                      f"using Amean as y_ntc fallback")
                                y_ntc_tensor = torch.where(_ntc_nan_mask, Amean_tensor, y_ntc_tensor)
                        if distribution == 'negbinom':
                            if (y_ntc_tensor <= 0).any():
                                raise ValueError(
                                    f"y_ntc_tensor contains non-positive values "
                                    f"(min={y_ntc_tensor.min().item():.3e}). "
                                    f"mu_ntc posterior has strictly positive support — this should not occur."
                                )
                    else:
                        print(f"[WARNING] mu_ntc shape {tuple(_mu_ntc.shape)} doesn't match T={T}; skipping A prior NTC anchor")
                elif distribution == 'multinomial' and 'probs_baseline' in _post_tech:
                    _probs = _post_tech['probs_baseline']
                    if not torch.is_tensor(_probs):
                        _probs = torch.tensor(_probs, dtype=torch.float32, device=self.model.device)
                    else:
                        _probs = _probs.float().to(self.model.device)
                    _probs_flat = None
                    if _probs.ndim == 3 and _probs.shape[1] == T:
                        _probs_flat = _probs.median(dim=0).values  # [T, K]
                    elif _probs.ndim == 2 and _probs.shape[0] == T:
                        _probs_flat = _probs              # [T, K]
                    else:
                        print(f"[WARNING] probs_baseline shape {tuple(_probs.shape)} doesn't match T={T}; skipping multinomial NTC anchor")
                    if _probs_flat is not None:
                        # q01_ntc_global for multinomial: q01 across non-NaN features per category → [K], normalized
                        _valid_rows = ~torch.isnan(_probs_flat).any(dim=-1)  # [T]
                        if _valid_rows.any():
                            _probs_valid = _probs_flat[_valid_rows]  # [T_valid, K]
                            q01_ntc_global_probs = torch.quantile(_probs_valid, 0.01, dim=0)  # [K]
                            q01_ntc_global_probs = q01_ntc_global_probs / q01_ntc_global_probs.sum().clamp_min(1e-10)
                        else:
                            _K_dim = _probs_flat.shape[1]
                            q01_ntc_global_probs = torch.ones(_K_dim, device=self.model.device) / _K_dim
                        # NaN fill: features where fit_ntc not run get q01_ntc_global_probs as y_ntc anchor
                        _ntc_nan_mask = torch.isnan(_probs_flat).all(dim=-1)  # [T]
                        if _ntc_nan_mask.any():
                            n_ntc_nan = _ntc_nan_mask.sum().item()
                            print(f"[WARNING] {n_ntc_nan}/{T} features have NaN probs_baseline (fit_ntc not run); "
                                  f"using q01_ntc_global probability vector as y_ntc anchor")
                            fill = q01_ntc_global_probs.unsqueeze(0).expand_as(_probs_flat)
                            mask = _ntc_nan_mask.unsqueeze(-1).expand_as(_probs_flat)
                            _probs_flat = torch.where(mask, fill, _probs_flat)
                        y_ntc_tensor = _probs_flat  # [T, K]
                else:
                    print(f"[INFO] No NTC posterior available for distribution='{distribution}'; A prior uses Q05 anchor only")
            except Exception as _e:
                print(f"[WARNING] Could not compute y_ntc anchor from technical posterior: {_e}")

        # --- o_y NTC anchor: pre-fit overdispersion for adaptive A prior weight ---
        # Uses o_y from fit_ntc rather than the sampled o_y from fit_trans.
        # The sampled o_y is unidentified (posterior ≈ prior mean for all genes), so
        # _o_y_weight would be ~0.6 for every gene regardless of true noisiness.
        # The NTC-estimated o_y is gene-specific and reflects actual count dispersion.
        # Only used for negbinom (where o_y is meaningfully estimated in fit_ntc).
        o_y_ntc_tensor = None
        if distribution == 'negbinom' and _post_tech is not None and 'o_y' in _post_tech:
            try:
                _o_y_ntc = _post_tech['o_y']
                if not torch.is_tensor(_o_y_ntc):
                    _o_y_ntc = torch.tensor(_o_y_ntc, dtype=torch.float32, device=self.model.device)
                else:
                    _o_y_ntc = _o_y_ntc.float().to(self.model.device)
                # Collapse leading sample/group dims → [T]
                if _o_y_ntc.ndim > 1 and _o_y_ntc.shape[-1] == T:
                    while _o_y_ntc.ndim > 1:
                        _o_y_ntc = _o_y_ntc.median(dim=0).values
                elif _o_y_ntc.ndim == 1 and _o_y_ntc.shape[0] == T:
                    pass  # already [T]
                else:
                    print(f"[WARNING] o_y from NTC posterior has unexpected shape "
                          f"{tuple(_post_tech['o_y'].shape if hasattr(_post_tech['o_y'], 'shape') else [])}; "
                          f"falling back to sampled o_y for A prior weight")
                    _o_y_ntc = None

                if _o_y_ntc is not None:
                    # NaN fill: genes where fit_ntc wasn't run get prior_mean_o_y (w=0.5)
                    _o_y_ntc_prior_mean = (beta_o_beta_tensor / beta_o_alpha_tensor).clamp_min(self._t(1e-12))
                    _o_y_nan_mask = torch.isnan(_o_y_ntc) | (_o_y_ntc <= 0)
                    if _o_y_nan_mask.any():
                        n_nan = _o_y_nan_mask.sum().item()
                        print(f"[INFO] o_y NTC anchor: {n_nan}/{T} genes have NaN/zero o_y; "
                              f"using prior mean ({_o_y_ntc_prior_mean.item():.3f}) → w=0.5 for those genes")
                        _o_y_ntc = torch.where(_o_y_nan_mask, _o_y_ntc_prior_mean.expand_as(_o_y_ntc), _o_y_ntc)
                    _o_y_ntc = _o_y_ntc.clamp_min(self._t(1e-6))
                    o_y_ntc_tensor = _o_y_ntc  # [T], fixed weight for A prior
                    print(f"[INFO] o_y NTC anchor: using pre-fit o_y for A prior weight "
                          f"(min={o_y_ntc_tensor.min().item():.3f}, "
                          f"median={o_y_ntc_tensor.median().item():.3f}, "
                          f"max={o_y_ntc_tensor.max().item():.3f})")
            except Exception as _e:
                print(f"[WARNING] Could not extract o_y from NTC posterior: {_e}; "
                      f"falling back to sampled o_y for A prior weight")

        # For K: use CV (coefficient of variation) of x_true (works with or without guides)
        # CV = std(x_true) / mean(x_true) - scale-invariant measure of variability
        x_true_mean_global = x_true_mean.mean()
        x_true_std_global = x_true_mean.std()
        x_true_CV = x_true_std_global / x_true_mean_global.clamp_min(epsilon_tensor)

        # K_max: max of x_true (or max of guide means if guides exist)
        if 'guide_code' in self.model.meta.columns and len(unique_guides) > 1:
            guide_x_means = torch.stack([x_true_mean[guides_tensor == g].mean() for g in torch.unique(guides_tensor)])
            K_max_tensor = guide_x_means.max()
        else:
            K_max_tensor = x_true_mean.max()

        # NTC mean of x_true: center K prior here so the P.O.I. starts within the observed range.
        # Any fold change within ±5 log2 of NTC covers the full CRISPR perturbation range.
        x_ntc_mean = None
        if 'target' in meta_subset.columns:
            ntc_mask = meta_subset['target'].str.lower() == 'ntc'
            if ntc_mask.any():
                ntc_idx = torch.tensor(ntc_mask.values, dtype=torch.bool, device=self.model.device)
                x_ntc_mean = x_true_mean[ntc_idx].mean()

        # K prior center tensor for no-NTC fallback (ignored when x_ntc_mean is present)
        _valid_k_prior_centers = ('lower', 'middle', 'upper')
        if k_prior_center not in _valid_k_prior_centers:
            raise ValueError(
                f"k_prior_center must be one of {_valid_k_prior_centers}, got {k_prior_center!r}"
            )
        _x_for_kcenter = x_true_mean  # per-guide (or per-cell) means of x_true
        if k_prior_center == 'lower':
            k_center_tensor = torch.quantile(_x_for_kcenter, 0.05).clamp_min(epsilon_tensor)
        elif k_prior_center == 'upper':
            k_center_tensor = torch.quantile(_x_for_kcenter, 0.95).clamp_min(epsilon_tensor)
        else:  # 'middle'
            k_center_tensor = (K_max_tensor / 2.0).clamp_min(epsilon_tensor)

        print("[DEBUG] Amean:", Amean_tensor.min().item(), Amean_tensor.max().item())
        print("[DEBUG] Vmax_mean:", Vmax_mean_tensor.min().item(), Vmax_mean_tensor.max().item())
        print("[DEBUG] Mean within-guide variance:", mean_within_guide_var.min().item(), mean_within_guide_var.max().item())
        print("[DEBUG] x_true CV:", x_true_CV.item(), "K_max:", K_max_tensor.item(),
              "x_ntc_mean:", x_ntc_mean.item() if x_ntc_mean is not None else "N/A",
              "k_center:", k_center_tensor.item(), f"(k_prior_center={k_prior_center!r})")

        # Diagnostic: Verify alpha_y_prefit is correctly structured
        if alpha_y_prefit is not None and groups_tensor is not None:
            print(f"[INFO] Technical correction setup: alpha_y_prefit.shape={alpha_y_prefit.shape}, C={C}")
            if alpha_y_prefit.shape[0] == C:
                print(f"[INFO] alpha_y_prefit already includes reference group (correct!)")
            else:
                print(f"[WARNING] alpha_y_prefit shape mismatch - may need to add reference group")

        # Ensure x_true is on the correct device (may have been loaded from CPU)
        # Always move to device to avoid device comparison issues (cuda vs cuda:0)
        self.model.x_true = self.model.x_true.to(self.model.device)
        if alpha_y_prefit is not None:
            alpha_y_prefit = alpha_y_prefit.to(self.model.device)

        def init_loc_fn(site):
            name = site["name"]
            if "poly_coeff" in name:
                return torch.zeros(T)
            return pyro.infer.autoguide.initialization.init_to_median(site)
        
        from torch.optim.lr_scheduler import OneCycleLR
        if function_type == "polynomial":
            guide_y = pyro.infer.autoguide.AutoMultivariateNormal(self._model_y, init_loc_fn=init_loc_fn)

            guide_y(
                N,
                T,
                y_obs_tensor,
                sum_factor_tensor,
                beta_o_alpha_tensor,
                beta_o_beta_tensor,
                alpha_alpha_mu_tensor,
                K_max_tensor,
                K_alpha_tensor,
                Vmax_mean_tensor,
                Vmax_alpha_tensor,
                n_mu_tensor,
                Amean_tensor,
                p_n_logits_tensor,
                epsilon_tensor,
                x_true_sample = x_true_subset,
                log2_x_true_sample = log2_x_true_subset,
                nmin = nmin,
                nmax = nmax,
                alpha_y_sample = alpha_y_prefit,
                C = C,
                groups_tensor=groups_tensor,
                temperature=torch.tensor(init_temp, dtype=torch.float32, device=self.model.device),
                use_straight_through=False,
                function_type=function_type,
                polynomial_degree=polynomial_degree,
                use_alpha=True,
                distribution=distribution,
                denominator_tensor=denominator_tensor,
                K=K,
                D=D,
                mean_within_guide_var=mean_within_guide_var,
                x_true_CV=x_true_CV,
                x_ntc_mean=x_ntc_mean,
                use_data_driven_priors=use_data_driven_priors,
                use_epsilon=use_epsilon,
                vmax_log_sigma_floor_tensor=vmax_log_sigma_floor_tensor,
                k_log_sigma_min_tensor=k_log_sigma_min_tensor,
                k_center_tensor=k_center_tensor,
                y_ntc_tensor=y_ntc_tensor,
                mean_y_corrected_tensor=mean_y_corrected_tensor,
                o_y_ntc_tensor=o_y_ntc_tensor,
                alpha_n_coupling=alpha_n_coupling,
            )
            # OneCycleLR for polynomial only
            base_lr = 1e-3 if lr is None else lr

            optimizer = pyro.optim.PyroLRScheduler(
                scheduler_constructor=OneCycleLR,
                optim_args={
                    # underlying torch optimizer
                    "optimizer": torch.optim.Adam,
                    "optim_args": {
                        "lr": base_lr,      # initial lr (OneCycle will move it)
                        "betas": (0.9, 0.999),
                    },
                    # OneCycleLR hyperparameters
                    "max_lr":          base_lr * 10,  # was 1e-2 when base_lr=1e-3
                    "total_steps":     niters,
                    "pct_start":       0.1,
                    "div_factor":      25.0,
                    "final_div_factor": 1e4,
                },
                clip_args={"clip_norm": 5},
            )

            svi = pyro.infer.SVI(
                self._model_y,
                guide_y,
                optimizer,
                loss=pyro.infer.Trace_ELBO()
            )
        else:
            # Simple Adam for Hill-based function types (single_hill, additive_hill, nested_hill)
            guide_y = pyro.infer.autoguide.AutoNormalMessenger(self._model_y)
            hill_lr = 1e-3 if lr is None else lr
            optimizer = pyro.optim.ClippedAdam({"lr": hill_lr, "clip_norm": 10.0})
            svi = pyro.infer.SVI(
                self._model_y,
                guide_y,
                optimizer,
                loss=pyro.infer.Trace_ELBO()
            )

        
        for name, value in pyro.get_param_store().items():
            if "poly_coeff" in name and "loc" in name:
                print(name, value.shape, value.min().item(), value.max().item())

        self.losses_trans = []
        smoothed_loss = None
        _phase2_announced = False
        start_step = 0

        # ── Checkpoint setup ──────────────────────────────────────────────────
        # Metadata stored in every checkpoint; validated on reload to catch
        # mismatches (e.g. different gene set, different number of cells).
        _ckpt_metadata = dict(
            N=N, T=T, K=K, C=C,
            modality_name=modality_name,
            function_type=function_type,
            distribution=distribution,
        )

        def _load_and_validate_ckpt(path, context="loading checkpoint"):
            """Load checkpoint and validate structural metadata against current call."""
            try:
                ckpt = torch.load(path, map_location=self.model.device, weights_only=False)
            except Exception as e:
                print(f"[WARNING] Could not load checkpoint ({e}): {path}")
                return None
            # Structural mismatches → error (param shapes won't match)
            _bad = []
            for key, expected, label in [
                ('N', N, 'number of cells'),
                ('T', T, 'number of features'),
                ('K', K, 'number of categories (K)'),
                ('C', C, 'number of technical groups (C)'),
            ]:
                stored = ckpt.get(key)
                if stored is not None and stored != expected:
                    _bad.append(f"  {label}: checkpoint={stored}, current call={expected}")
            if _bad:
                raise ValueError(
                    f"[{context}] Checkpoint metadata mismatch — cannot resume "
                    f"(parameter shapes differ):\n" + "\n".join(_bad)
                )
            # Non-structural mismatches → warn only
            # Use effective_function_type when available (warmup checkpoints store
            # 'single_hill' even though target function_type may be 'additive_hill').
            _stored_ft = ckpt.get('effective_function_type', ckpt.get('function_type'))
            if _stored_ft is not None and _stored_ft != function_type:
                warnings.warn(
                    f"[{context}] function_type mismatch: checkpoint effective={_stored_ft!r}, "
                    f"current call={function_type!r}. Ensure model and guide are compatible."
                )
            _stored_dist = ckpt.get('distribution')
            if _stored_dist is not None and _stored_dist != distribution:
                warnings.warn(
                    f"[{context}] distribution mismatch: checkpoint={_stored_dist!r}, "
                    f"current call={distribution!r}."
                )
            return ckpt

        _predictive_only_mode = predictive_checkpoint is not None
        checkpoint_path = None

        if _predictive_only_mode:
            # ── Predictive-only: load specified checkpoint, skip training ──────
            print(f"[INFO] Predictive-only mode: loading checkpoint {predictive_checkpoint}")
            ckpt = _load_and_validate_ckpt(predictive_checkpoint, context="predictive_checkpoint")
            if ckpt is None:
                raise FileNotFoundError(
                    f"Could not load predictive checkpoint: {predictive_checkpoint}"
                )
            self.losses_trans = ckpt.get('losses', [])
            smoothed_loss = ckpt.get('smoothed_loss')
            _phase2_announced = ckpt.get('phase2_announced', False)
            pyro.get_param_store().set_state(ckpt['param_store'])
            start_step = total_steps  # skip training loop entirely
            print(f"[INFO] Loaded checkpoint: step={ckpt['step']}, "
                  f"complete={ckpt.get('complete', False)}, "
                  f"phase2_announced={_phase2_announced}")

        elif checkpoint_interval is not None:
            # ── Normal checkpoint resume ──────────────────────────────────────
            _ckpt_dir = (checkpoint_dir if checkpoint_dir is not None
                         else os.path.join(self.model.output_dir, self.model.label))
            os.makedirs(_ckpt_dir, exist_ok=True)
            checkpoint_path = os.path.join(_ckpt_dir, f'trans_checkpoint_{modality_name}_latest.pt')
            _backup_path = checkpoint_path + '.bak'
            _tmp_path = checkpoint_path + '.tmp'
            _ckpt = None
            if not restart_from_checkpoint:
                print(f"[INFO] restart_from_checkpoint=False — skipping checkpoint resume, starting from step 0.")
            else:
                # Try candidates in priority order:
                #   1. _latest.pt  — normal case
                #   2. _latest.pt.tmp  — process died after torch.save but before renames;
                #                        .tmp may be a complete, valid checkpoint
                #   3. _latest.pt.bak  — previous interval; always valid but one step behind
                for _cand, _label in [
                    (checkpoint_path, 'primary (_latest.pt)'),
                    (_tmp_path,       '.tmp (rename interrupted)'),
                    (_backup_path,    '.bak (previous interval)'),
                ]:
                    if os.path.exists(_cand):
                        print(f"[INFO] Attempting to load {_label}: {os.path.basename(_cand)}")
                        _ckpt = _load_and_validate_ckpt(_cand, context="resume")
                        if _ckpt is not None:
                            break
                        print(f"[WARNING] {_label} failed to load; trying next fallback…")
            if _ckpt is not None:
                if _ckpt.get('complete', False):
                    print(f"[INFO] Checkpoint is marked complete — training already finished. "
                          f"Skipping to Predictive sampling.")
                start_step = _ckpt['step'] + 1
                self.losses_trans = _ckpt['losses']
                smoothed_loss = _ckpt['smoothed_loss']
                _phase2_announced = _ckpt.get('phase2_announced', False)
                pyro.get_param_store().set_state(_ckpt['param_store'])
                try:
                    optimizer.set_state(_ckpt['optimizer_state'])
                except Exception as e:
                    print(f"[WARNING] Could not restore optimizer state: {e}. "
                          f"Continuing with fresh optimizer.")
                print(f"[INFO] Resumed from step {start_step} / {total_steps}")

        prev_finite = None  # initialize NaN tracker (before loop so checkpoint-resume works)
        for step in range(start_step, total_steps):
            # ── Curriculum: choose effective function type and temperature ──
            if _do_warmup and step < warmup_steps:
                effective_function_type = 'single_hill'
                # Phase 1: cool from init_temp to warmup_T_min at same rate as Phase 2
                phase_fraction = step / float(warmup_steps)
                current_temp = init_temp + (warmup_T_min - init_temp) * phase_fraction
            else:
                effective_function_type = function_type
                if _do_warmup and not _phase2_announced:
                    print(f"[INFO] Warmup complete at step {step}. Switching to {function_type}.")
                    _phase2_announced = True
                    if checkpoint_path is not None:
                        _warmup_data = {
                            'step': step,
                            'losses': self.losses_trans,
                            'smoothed_loss': smoothed_loss,
                            'phase2_announced': True,
                            'param_store': pyro.get_param_store().get_state(),
                            'optimizer_state': optimizer.get_state(),
                            'complete': False,
                            # effective_function_type='single_hill' lets the user pass
                            # function_type='single_hill' to predictive_checkpoint= without a warning
                            'effective_function_type': 'single_hill',
                            **_ckpt_metadata,
                        }
                        _warmup_path = checkpoint_path.replace('_latest.pt', '_warmup.pt')
                        self._save_checkpoint_atomic(_warmup_path, _warmup_data)
                        print(f"[CKPT] Warmup checkpoint saved at step {step} → {os.path.basename(_warmup_path)}")
                # Phase 2: restart from init_temp, cool to final_temp over niters steps
                phase_step = step - warmup_steps if _do_warmup else step
                phase_fraction = phase_step / float(niters) if niters > 0 else 1.0
                if effective_function_type in ['single_hill', 'additive_hill', 'nested_hill']:
                    current_temp = init_temp + (final_temp - init_temp) * phase_fraction
                elif effective_function_type == 'polynomial':
                    current_temp = init_temp + (final_temp - init_temp) * (2*phase_fraction-1)
                else:
                    raise ValueError(f"Unknown function_type: {effective_function_type}")
                
            #if step < 0.7 * niters:
            #    # First 70% of training: linearly decrease from 1.0 to 0.1
            #    current_temp = 1.0 - (0.9 * (step / (0.7 * niters)))
            #else:
            #    # Last 30% of training: exponentially cool down to 0.0005
            #    current_temp = 0.1 * (final_temp/0.1) ** ((step - 0.7 * niters) / (0.3 * niters))


            # Anneal coupling from 0 → target over first half of total training.
            # Lets alpha and n co-explore freely early on; identifiability constraint
            # ramps in gradually rather than blocking from step 0.
            _coupling_frac = min(1.0, 2.0 * step / total_steps) if total_steps > 0 else 1.0
            current_coupling = alpha_n_coupling * _coupling_frac

            # x_true, log2_x_true, alpha_y_prefit are always point estimates (means)
            x_true_sample = x_true_subset
            log2_x_true_sample = log2_x_true_subset
            alpha_y_sample = alpha_y_prefit  # [C, T] point estimate or None

            #use_straight_through = step >= int(0.7 * niters)
            use_straight_through = False

            '''
            loss = svi.step(
                N,
                T,
                y_obs_tensor,
                sum_factor_tensor,
                beta_o_alpha_tensor,
                beta_o_beta_tensor,
                alpha_alpha_mu_tensor,
                K_max_tensor,
                K_alpha_tensor,
                Vmax_mean_tensor,
                Vmax_alpha_tensor,
                n_mu_tensor,
                Amean_tensor,
                p_n_logits_tensor,
                epsilon_tensor,
                x_true_sample = x_true_sample,
                log2_x_true_sample = log2_x_true_sample,
                nmin = nmin,
                nmax = nmax,
                alpha_y_sample = alpha_y_sample,
                C = C,
                groups_tensor=groups_tensor,
                temperature=torch.tensor(current_temp, dtype=torch.float32, device=self.model.device),
                use_straight_through=use_straight_through,
                function_type=effective_function_type,
                polynomial_degree=polynomial_degree,
                use_alpha=True if effective_function_type != 'polynomial' else True if phase_fraction>=0.5 else False,
                distribution=distribution,
                denominator_tensor=denominator_tensor,
                K=K,
                D=D,
                mean_within_guide_var=mean_within_guide_var,
                x_true_CV=x_true_CV,
                x_ntc_mean=x_ntc_mean,
                use_data_driven_priors=use_data_driven_priors,
                use_epsilon=use_epsilon,
            )
            '''

            try:
                loss, prev_finite = self._debug_svi_step(
                    svi, step, prev_finite,
                    N, T, y_obs_tensor, sum_factor_tensor, beta_o_alpha_tensor, beta_o_beta_tensor,
                    alpha_alpha_mu_tensor, K_max_tensor, K_alpha_tensor, Vmax_mean_tensor, Vmax_alpha_tensor,
                    n_mu_tensor, Amean_tensor, p_n_logits_tensor, epsilon_tensor,
                    x_true_sample=x_true_sample,
                    log2_x_true_sample=log2_x_true_sample,
                    nmin=nmin,
                    nmax=nmax,
                    alpha_y_sample=alpha_y_sample,
                    C=C,
                    groups_tensor=groups_tensor,
                    temperature=torch.tensor(current_temp, dtype=torch.float32, device=self.model.device),
                    use_straight_through=use_straight_through,
                    function_type=effective_function_type,
                    polynomial_degree=polynomial_degree,
                    use_alpha=True if effective_function_type != 'polynomial' else True if phase_fraction>=0.5 else False,
                    distribution=distribution,
                    denominator_tensor=denominator_tensor,
                    K=K,
                    D=D,
                    mean_within_guide_var=mean_within_guide_var,
                    x_true_CV=x_true_CV,
                    use_data_driven_priors=use_data_driven_priors,
                    use_epsilon=use_epsilon,
                    vmax_log_sigma_floor_tensor=vmax_log_sigma_floor_tensor,
                    k_log_sigma_min_tensor=k_log_sigma_min_tensor,
                    k_center_tensor=k_center_tensor,
                    y_ntc_tensor=y_ntc_tensor,
                    mean_y_corrected_tensor=mean_y_corrected_tensor,
                    o_y_ntc_tensor=o_y_ntc_tensor,
                    alpha_n_coupling=current_coupling,
                )
            except FloatingPointError as e:
                print(f"[STOP] {e} at step {step}")
                break


            # NaN detection and early stopping
            if np.isnan(loss) or np.isinf(loss):
                print(f"[WARNING] NaN/Inf detected in loss at step {step}! Stopping optimization.")
                print(f"  - temperature: {current_temp}")
                print(f"  - p_n_tensor: {p_n_tensor.item()}")
                # Check parameters for NaN
                for name, value in pyro.get_param_store().items():
                    if torch.isnan(value).any():
                        nan_count = torch.isnan(value).sum().item()
                        print(f"  - Parameter '{name}' has {nan_count} NaN values")
                break

            self.losses_trans.append(loss)
            if step % 1000 == 0:
                print(f"Step {step} : loss = {loss:.5e}, device: {Vmax_mean_tensor.device}")
            if checkpoint_path is not None and step > 0 and step % checkpoint_interval == 0:
                _eff_ft = ('single_hill' if (_do_warmup and step < warmup_steps)
                           else function_type)
                _interval_data = {
                    'step': step,
                    'losses': self.losses_trans,
                    'smoothed_loss': smoothed_loss,
                    'phase2_announced': _phase2_announced,
                    'param_store': pyro.get_param_store().get_state(),
                    'optimizer_state': optimizer.get_state(),
                    'complete': False,
                    'effective_function_type': _eff_ft,
                    **_ckpt_metadata,
                }
                # Rolling checkpoint (overwrites previous; used for crash-resume)
                self._save_checkpoint_atomic(checkpoint_path, _interval_data)
                # Numbered checkpoint (persistent; named by step for analysis)
                _numbered_path = checkpoint_path.replace('_latest.pt', f'_step{step:07d}.pt')
                self._save_checkpoint_atomic(_numbered_path, _interval_data)
                print(f"[CKPT] Checkpoint saved at step {step} → {os.path.basename(_numbered_path)}")
            if smoothed_loss is None:
                smoothed_loss = loss
            else:
                if abs(alpha_ewma * (loss - smoothed_loss)) < tolerance:
                    print(f"Converged at step {step}! Loss = {loss:.5e}")
                    break
                smoothed_loss = alpha_ewma * loss + (1 - alpha_ewma) * smoothed_loss

        # Save complete=True checkpoint so Predictive can be retried if it OOMs.
        # Also save a named _complete.pt that persists after the rolling checkpoint
        # is cleaned up, so the fitted state is always recoverable.
        if checkpoint_path is not None and not _predictive_only_mode:
            _complete_data = {
                'step': total_steps - 1,
                'losses': self.losses_trans,
                'smoothed_loss': smoothed_loss,
                'phase2_announced': _phase2_announced,
                'param_store': pyro.get_param_store().get_state(),
                'optimizer_state': optimizer.get_state(),
                'complete': True,
                'effective_function_type': function_type,
                **_ckpt_metadata,
            }
            self._save_checkpoint_atomic(checkpoint_path, _complete_data)
            _complete_path = checkpoint_path.replace('_latest.pt', '_complete.pt')
            self._save_checkpoint_atomic(_complete_path, _complete_data)
            print(f"[CKPT] Final checkpoint saved (complete=True) → {os.path.basename(_complete_path)}")

        # Move to CPU if using too much GPU memory for Predictive
        _original_device = self.model.device  # Save exact device (e.g. cuda:6) before any switch
        run_on_cpu = self.model.device.type != "cpu"
        if run_on_cpu:
            print("[INFO] Running Predictive on CPU to reduce GPU memory pressure...")
            guide_y.to("cpu")
            self.model.device = torch.device("cpu")
        
            model_inputs = {
                "N": N,
                "T": T,
                "y_obs_tensor": self._to_cpu(y_obs_tensor),
                "sum_factor_tensor": self._to_cpu(sum_factor_tensor),
                "beta_o_alpha_tensor": self._to_cpu(beta_o_alpha_tensor),
                "beta_o_beta_tensor": self._to_cpu(beta_o_beta_tensor),
                "alpha_alpha_mu_tensor": self._to_cpu(alpha_alpha_mu_tensor),
                "K_max_tensor": self._to_cpu(K_max_tensor),
                "K_alpha_tensor": self._to_cpu(K_alpha_tensor),
                "Vmax_mean_tensor": self._to_cpu(Vmax_mean_tensor),
                "Vmax_alpha_tensor": self._to_cpu(Vmax_alpha_tensor),
                "n_mu_tensor": self._to_cpu(n_mu_tensor),
                "Amean_tensor": self._to_cpu(Amean_tensor),
                "p_n_logits_tensor": self._to_cpu(p_n_logits_tensor),
                "epsilon_tensor": self._to_cpu(epsilon_tensor),
                "x_true_sample": self._to_cpu(x_true_subset),
                "log2_x_true_sample": self._to_cpu(log2_x_true_subset),
                "nmin": self._to_cpu(nmin),
                "nmax": self._to_cpu(nmax),
                "alpha_y_sample": self._to_cpu(alpha_y_prefit) if alpha_y_prefit is not None else None,
                "C": C,
                "groups_tensor": self._to_cpu(groups_tensor) if groups_tensor is not None else None,
                # create on CPU explicitly since we just set self.model.device="cpu"
                "temperature": torch.tensor(final_temp, dtype=torch.float32, device=torch.device("cpu")),
                "use_straight_through": True,
                "function_type": function_type,
                "polynomial_degree": polynomial_degree,
                "use_alpha": True,
                "distribution": distribution,
                "denominator_tensor": self._to_cpu(denominator_tensor) if denominator_tensor is not None else None,
                "K": K,
                "D": D,
                "mean_within_guide_var": self._to_cpu(mean_within_guide_var) if mean_within_guide_var is not None else None,
                "x_true_CV": self._to_cpu(x_true_CV) if x_true_CV is not None else None,
                "use_data_driven_priors": use_data_driven_priors,
                "use_epsilon": use_epsilon,
                "vmax_log_sigma_floor_tensor": self._to_cpu(vmax_log_sigma_floor_tensor),
                "k_log_sigma_min_tensor": self._to_cpu(k_log_sigma_min_tensor),
                "k_center_tensor": self._to_cpu(k_center_tensor),
                "y_ntc_tensor": self._to_cpu(y_ntc_tensor) if y_ntc_tensor is not None else None,
                "x_ntc_mean": self._to_cpu(x_ntc_mean) if x_ntc_mean is not None else None,
                "o_y_ntc_tensor": self._to_cpu(o_y_ntc_tensor) if o_y_ntc_tensor is not None else None,
                "alpha_n_coupling": alpha_n_coupling,
            }
        else:
            model_inputs = {
                "N": N,
                "T": T,
                "y_obs_tensor": y_obs_tensor,
                "sum_factor_tensor": sum_factor_tensor,
                "beta_o_alpha_tensor": beta_o_alpha_tensor,
                "beta_o_beta_tensor": beta_o_beta_tensor,
                "alpha_alpha_mu_tensor": alpha_alpha_mu_tensor,
                "K_max_tensor": K_max_tensor,
                "K_alpha_tensor": K_alpha_tensor,
                "Vmax_mean_tensor": Vmax_mean_tensor,
                "Vmax_alpha_tensor": Vmax_alpha_tensor,
                "n_mu_tensor": n_mu_tensor,
                "Amean_tensor": Amean_tensor,
                "p_n_logits_tensor": p_n_logits_tensor,
                "epsilon_tensor": epsilon_tensor,
                "x_true_sample": x_true_subset,
                "log2_x_true_sample": log2_x_true_subset,
                "nmin": nmin,
                "nmax": nmax,
                "alpha_y_sample": alpha_y_prefit,
                "C": C,
                "groups_tensor": groups_tensor if groups_tensor is not None else None,
                "temperature": torch.tensor(final_temp, dtype=torch.float32, device=self.model.device),
                "use_straight_through": True,
                "function_type": function_type,
                "polynomial_degree": polynomial_degree,
                "use_alpha": True,
                "distribution": distribution,
                "denominator_tensor": denominator_tensor if denominator_tensor is not None else None,
                "K": K,
                "D": D,
                "mean_within_guide_var": mean_within_guide_var,
                "x_true_CV": x_true_CV,
                "use_data_driven_priors": use_data_driven_priors,
                "use_epsilon": use_epsilon,
                "vmax_log_sigma_floor_tensor": vmax_log_sigma_floor_tensor,
                "k_log_sigma_min_tensor": k_log_sigma_min_tensor,
                "k_center_tensor": k_center_tensor,
                "y_ntc_tensor": y_ntc_tensor,
                "o_y_ntc_tensor": o_y_ntc_tensor,
                "alpha_n_coupling": alpha_n_coupling,
            }

        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        max_samples = nsamples
        keep_sites = kwargs.get("keep_sites", lambda name, site: name != "y_obs")

        # Add latents_only=True so _model_y exits before [N, T] Hill/polynomial computation.
        # All downstream-used sites (A, o_y, alpha, beta, Vmax_a/b, K_a/b, n_a/b, etc.) are
        # [T]-shaped and sampled BEFORE the cell-level computation, so nothing is lost.
        # This avoids allocating multiple [N=cells, T=genes] tensors (~2.68 GB each) per
        # Predictive sample, which is the primary source of OOM on large datasets.
        predictive_model_inputs = {**model_inputs, "latents_only": True}

        if minibatch_size is not None:
            from collections import defaultdict

            print(f"[INFO] Running Predictive in minibatches of {minibatch_size}...")
            predictive_y = pyro.infer.Predictive(
                self._model_y,
                guide=guide_y,
                num_samples=minibatch_size,
                parallel=True
            )
            all_samples = defaultdict(list)
            with torch.no_grad():
                for i in range(0, max_samples, minibatch_size):
                    samples = predictive_y(**predictive_model_inputs)
                    for k, v in samples.items():
                        if keep_sites(k, {"value": v}):
                            all_samples[k].append(self._to_cpu(v))
                    if self.model.device.type == "cuda":
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()

            posterior_samples_y = {k: torch.cat(v, dim=0) for k, v in all_samples.items()}

        else:
            predictive_y = pyro.infer.Predictive(
                self._model_y,
                guide=guide_y,
                num_samples=nsamples
            )
            with torch.no_grad():
                _raw_samples = predictive_y(**predictive_model_inputs)
                # keep_sites filters out y_obs (never used downstream, and not computed
                # when latents_only=True anyway — this is a safety net for other callers).
                posterior_samples_y = {
                    k: v for k, v in _raw_samples.items()
                    if keep_sites(k, {"value": v})
                }
                del _raw_samples
                if self.model.device.type == "cuda":
                    torch.cuda.empty_cache()
                import gc
                gc.collect()

        if run_on_cpu:
            self.model.device = _original_device  # Restore exact original device (not generic "cuda")
            print("[INFO] Reset self.model.device to:", self.model.device)

        for k, v in posterior_samples_y.items():
            posterior_samples_y[k] = self._to_cpu(v)

        # ----------------------------------------
        # Store nmin/nmax and check boundaries
        # ----------------------------------------
        posterior_samples_y['nmin'] = self._to_cpu(nmin)
        posterior_samples_y['nmax'] = self._to_cpu(nmax)

        # Warn if fitted n_a parameters are close to boundaries
        if 'n_a' in posterior_samples_y and function_type in ['single_hill', 'additive_hill', 'nested_hill']:
            n_a_samples = posterior_samples_y['n_a']  # Shape: [S, T]
            nmin_val = nmin.item()
            nmax_val = nmax.item()

            # Define "close to boundary" threshold (e.g., within 10% of range)
            boundary_threshold = 0.1 * (nmax_val - nmin_val)

            # Check how many samples are close to boundaries
            close_to_min = (n_a_samples < (nmin_val + boundary_threshold)).float().mean().item()
            close_to_max = (n_a_samples > (nmax_val - boundary_threshold)).float().mean().item()

            # Warn if >10% of samples are at boundaries
            if close_to_min > 0.1:
                warnings.warn(
                    f"[WARNING] {close_to_min*100:.1f}% of n_a samples are close to lower boundary (nmin={nmin_val:.2f}). "
                    f"Consider: (1) checking if x_true range is appropriate, or (2) relaxing nmin constraint.",
                    UserWarning
                )
            if close_to_max > 0.1:
                warnings.warn(
                    f"[WARNING] {close_to_max*100:.1f}% of n_a samples are close to upper boundary (nmax={nmax_val:.2f}). "
                    f"Consider: (1) checking if x_true range is appropriate, or (2) relaxing nmax constraint.",
                    UserWarning
                )

            # Summary statistics
            print(f"[INFO] n_a boundary check: nmin={nmin_val:.2f}, nmax={nmax_val:.2f}")
            print(f"[INFO]   {close_to_min*100:.1f}% of samples near lower bound, {close_to_max*100:.1f}% near upper bound")

        # ── Compute and save prior parameters for diagnostic plotting ─────────
        # These let plot_parameter_ci_panel overlay prior distributions as violins
        trans_prior_params = None
        if function_type in ['single_hill', 'additive_hill']:
            _K_log_sigma_min_val = float(k_log_sigma_min_tensor.item())  # parameterized, default 5*ln(2)/2 ≈ 1.733

            # n prior: unconstrained mean (maps to n_mu=0 in constrained space)
            _half_n_p   = 0.5 * (nmax - nmin).clamp_min(epsilon_tensor)
            _center_n_p = 0.5 * (nmax + nmin)
            _ratio_n_p  = ((n_mu_tensor - _center_n_p) / _half_n_p).clamp(-1 + 1e-6, 1 - 1e-6)
            n_mu_raw_prior = float((_half_n_p * torch.atanh(_ratio_n_p)).item())

            # Vmax prior (log-normal): Vmax_log_sigma is the same for all genes
            # because sigma/mean = 1/sqrt(Vmax_alpha) regardless of Vmax_mean.
            # Apply the same floor that _model_y uses (parameterized via vmax_log_sigma_floor).
            if distribution not in ['binomial', 'multinomial']:
                Vmax_alpha_val   = float(Vmax_alpha_tensor.item())
                _Vmax_log_sigma_floor_p = float(vmax_log_sigma_floor_tensor.item())
                Vmax_log_sigma_p = max(
                    float(np.sqrt(np.log1p(1.0 / Vmax_alpha_val))),
                    _Vmax_log_sigma_floor_p,
                )
                Vmax_log_mu_p    = (np.log(np.maximum(Vmax_mean_tensor.cpu().numpy(), 1e-12))
                                    - 0.5 * Vmax_log_sigma_p ** 2)  # [T]
            else:
                Vmax_log_sigma_p = None
                Vmax_log_mu_p    = None

            # K prior (log-normal): median centred at x_ntc_mean or k_center fallback.
            # K_log_mu = log(centre) with no -0.5*sigma² correction, so the MEDIAN of
            # K_a equals the centre (log2FC = 0 for the NTC-centred case).
            if x_ntc_mean is not None:
                K_log_sigma_p = _K_log_sigma_min_val
                K_log_mu_p    = float(torch.log(x_ntc_mean.clamp_min(epsilon_tensor)).item())
            else:
                _K_mean_p  = k_center_tensor.clamp_min(epsilon_tensor)
                _K_std_p   = (_K_mean_p * x_true_CV if x_true_CV is not None
                              else K_max_tensor / (2.0 * torch.sqrt(K_alpha_tensor)))
                _ratio_K_p = (_K_std_p / _K_mean_p).clamp_min(self._t(1e-6))
                K_log_sigma_p = float(torch.sqrt(torch.log1p(_ratio_K_p ** 2))
                                      .clamp_min(self._t(_K_log_sigma_min_val)).item())
                K_log_mu_p    = float(torch.log(_K_mean_p).item())

            trans_prior_params = {
                'function_type':    function_type,
                'distribution':     distribution,
                # n_a / n_b
                'nmin':             float(nmin.item()),
                'nmax':             float(nmax.item()),
                'n_mu_raw':         n_mu_raw_prior,
                'sigma_n_prior_rate': 1.0,           # Exp(1) prior per gene → mean sigma = 1
                # Vmax_a / Vmax_b (log-normal)
                'Vmax_log_mu':      Vmax_log_mu_p,   # [T] numpy array or None
                'Vmax_log_sigma':   Vmax_log_sigma_p, # scalar or None
                # K_a / K_b (log-normal)
                'K_log_mu':         K_log_mu_p,      # scalar
                'K_log_sigma':      K_log_sigma_p,   # scalar
                # A prior (negbinom: log2-Normal; others: None)
                # A_log2_mu / A_log2_sigma: per-gene anchors in log2 space, matching _model_y.
                # mu = lower anchor (log2(Amean/2), capped so sigma ≤ 4 octaves).
                # sigma = log2(y_ntc) - lower_anchor, clamped to [1, 4] octaves.
                # Without y_ntc: sigma = 4 (flat fallback), mu = log2(Amean/2).
                'Amean':            (Amean_tensor.cpu().numpy()
                                     if distribution not in ['binomial', 'multinomial']
                                     else None),      # [T] or None (kept for backward compat)
                'A_log2_mu':        None,             # [T] or None — set below for negbinom
                'A_log2_sigma':     None,             # [T] or None — set below for negbinom
                # alpha / beta (RelaxedBernoulli)
                'p_n_logits':       float(p_n_logits_tensor.item()),
                'temperature_prior': 1.0,
            }

            # Fill A_log2_mu / A_log2_sigma for negbinom (matches _model_y lines 658-671).
            # Replicate the exact weight used inside _model_y so the violin is accurate.
            if distribution == 'negbinom':
                _Amean_np = Amean_tensor.cpu().numpy()
                if y_ntc_tensor is not None:
                    _y_ntc_np = y_ntc_tensor.cpu().numpy()
                    # Mirror _model_y: lift upper anchor to mean(y_corrected) when y_ntc < mean.
                    if mean_y_corrected_tensor is not None:
                        _mean_ycorr_np = mean_y_corrected_tensor.cpu().numpy()
                        _upper_np = np.maximum(_y_ntc_np, _mean_ycorr_np)
                    else:
                        _upper_np = _y_ntc_np
                    _log2_upper = np.log2(np.maximum(_upper_np, 1e-12))
                    _log2_lower = np.maximum(
                        np.log2(np.maximum(_Amean_np / 2.0, 1e-12)),
                        _log2_upper - 4.0,
                    )
                    _sigma_log2_A = np.clip(_log2_upper - _log2_lower, 1.0, 4.0)
                    if o_y_ntc_tensor is not None:
                        _prior_mean_oy = float((beta_o_beta_tensor / beta_o_alpha_tensor)
                                               .clamp_min(epsilon_tensor).item())
                        _w = (o_y_ntc_tensor / (o_y_ntc_tensor + _prior_mean_oy)).cpu().numpy()
                    else:
                        _w = np.zeros_like(_Amean_np)
                    _mu_log2_A = (1.0 - _w) * _log2_lower + _w * _log2_upper
                else:
                    _mu_log2_A    = np.log2(np.maximum(_Amean_np / 2.0, 1e-12))
                    _sigma_log2_A = np.full_like(_mu_log2_A, 4.0)
                trans_prior_params['A_log2_mu']    = _mu_log2_A    # [T]
                trans_prior_params['A_log2_sigma'] = _sigma_log2_A  # [T]

        # Store results
        # Store in modality
        modality.posterior_samples_trans = posterior_samples_y
        modality.trans_prior_params = trans_prior_params
        modality.losses_trans = self.losses_trans  # Store loss history

        # Update alpha_y_prefit in modality if it was None and alpha_y was sampled
        if modality.alpha_y_prefit is None and groups_tensor is not None and "alpha_y" in posterior_samples_y:
            modality.alpha_y_prefit = posterior_samples_y["alpha_y"].median(dim=0).values

        # If primary modality, also store at model level (backward compatibility)
        if modality_name == self.model.primary_modality:
            self.model.posterior_samples_trans = posterior_samples_y
            self.model.losses_trans = self.losses_trans
            self.model.trans_prior_params = trans_prior_params
            # NOTE: alpha_y_prefit is stored per-modality (already done above), not at model level
            print(f"[INFO] Stored results in modality '{modality_name}' and at model level (primary modality)")
        else:
            print(f"[INFO] Stored results in modality '{modality_name}'")

        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        pyro.clear_param_store()

        # Remove the rolling checkpoint (_latest.pt) and its temporaries now that
        # results are stored on the model.  Numbered (_step*.pt), warmup, and
        # complete checkpoints are retained for post-hoc analysis.
        if checkpoint_path is not None and not _predictive_only_mode:
            for _p in [checkpoint_path, checkpoint_path + '.bak', checkpoint_path + '.tmp']:
                if os.path.exists(_p):
                    os.remove(_p)
            print(f"[INFO] Rolling checkpoint removed; "
                  f"numbered/warmup/complete checkpoints retained in "
                  f"{os.path.dirname(checkpoint_path)}")

        print("Finished fit_trans.")


