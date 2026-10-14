"""Technical-group and externally supplied overdispersion state."""

from ..utils import assign_or_warn_technical_group_code, sample_or_use_point


class TechnicalStateMixin:
    def set_alpha_x(
        self,
        alpha_x,
        covariates: list[str] = None,
    ):
        """
        Set cis-gene overdispersion scaling factors from a pre-fitted or external estimate.

        Stores ``alpha_x`` as a point-estimate tensor of shape ``[C]``, where index 0
        is the reference group (value fixed to 1.0).  Must be called before
        ``fit_cis()`` when overdispersion is supplied externally rather than estimated
        by ``fit_ntc()``.

        Parameters
        ----------
        alpha_x : array-like or torch.Tensor
            Overdispersion scale factors, shape ``[C]`` (including reference group at
            index 0).  Accepts numpy arrays, lists, or tensors.
        covariates : list of str, optional
            Column names in ``meta`` used to define technical groups (e.g.,
            ``['cell_line']``).  If provided, creates/overwrites
            ``meta['technical_group_code']``.  If ``None``, assumes
            ``technical_group_code`` was already set (raises ``ValueError`` if not).
        """
        self.meta["technical_group_code"] = assign_or_warn_technical_group_code(
            self.meta, covariates, reuse_warning="technical_group previously set. Assuming alpha_x corresponds."
        )
        self.alpha_x_prefit = sample_or_use_point("alpha_x_posterior", alpha_x, self.device).flatten()

    def set_alpha_y(
        self,
        alpha_y,
        covariates: list[str] = None,
    ):
        """
        Set trans-gene overdispersion scaling factors from a pre-fitted or external estimate.

        Stores ``alpha_y`` as a point-estimate tensor of shape ``[C, T]`` on the
        primary modality, where ``C`` is the number of technical groups (index 0 =
        reference group) and ``T`` is the number of trans features.  Must be called
        before ``fit_trans()`` when overdispersion is supplied externally rather than
        estimated by ``fit_ntc()``.

        Parameters
        ----------
        alpha_y : array-like or torch.Tensor
            Overdispersion scale factors, shape ``[C, T]`` (reference group at index 0).
            Accepts numpy arrays, lists, or tensors.
        covariates : list of str, optional
            Column names in ``meta`` used to define technical groups (e.g.,
            ``['cell_line']``).  If provided, creates/overwrites
            ``meta['technical_group_code']``.  If ``None``, assumes
            ``technical_group_code`` was already set (raises ``ValueError`` if not).
        """
        self.meta["technical_group_code"] = assign_or_warn_technical_group_code(
            self.meta, covariates, reuse_warning="technical_group previously set. Assuming alpha_xy corresponds."
        )
        alpha_y = sample_or_use_point("alpha_y_posterior", alpha_y, self.device)
        self.get_modality(self.primary_modality).alpha_y_prefit = alpha_y