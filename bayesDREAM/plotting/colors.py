"""
Color scheme management for bayesDREAM plots.

Provides consistent coloring across all visualization functions for targets, guides,
and technical groups.
"""

import numpy as np
import matplotlib.colors as mcolors
from matplotlib import cm


class ColorScheme:
    """
    Manages consistent color schemes for bayesDREAM visualizations.

    Attributes
    ----------
    palette : dict
        Target-specific color palettes (colormaps with linspace gradients)
    guide_colors : dict
        Guide-specific colors built from palette
    target_colors : dict
        Target-level representative colors
    target_cmaps : dict
        Colormap function for each target (for dynamic color generation)
    """

    # Default colormap rotation for targets
    DEFAULT_CMAPS = [cm.Greens, cm.Blues, cm.Reds, cm.Purples,
                    cm.Oranges, cm.YlOrBr, cm.PuBu, cm.BuGn,
                    cm.YlGn, cm.OrRd, cm.RdPu, cm.BuPu]

    def __init__(self, palette=None, target_cmaps=None):
        """
        Initialize ColorScheme.

        Parameters
        ----------
        palette : dict, optional
            Target -> list of colors mapping. If None, uses default palette.
            Example: {'GFI1B': [cm.Greens(i) for i in np.linspace(0.4, 0.9, 3)]}
        target_cmaps : dict, optional
            Target -> colormap function mapping for dynamic color generation.
            If None, uses default colormap rotation.
        """
        if palette is None:
            # Default palette based on 10x.ipynb
            self.palette = {
                'GFI1B': [cm.Greens(i) for i in np.linspace(0.4, 0.9, 3)],
                'NTC':   [cm.Greys(i)  for i in np.linspace(0.4, 0.8, 5)],
                'ntc':   [cm.Greys(i)  for i in np.linspace(0.4, 0.8, 5)],
                'GEMIN5': [cm.Blues(i)  for i in np.linspace(0.4, 0.8, 2)],
                'DDX6':  [cm.Reds(i)   for i in np.linspace(0.4, 0.8, 3)],
            }
        else:
            self.palette = palette

        # Store colormap functions for dynamic generation
        if target_cmaps is None:
            self.target_cmaps = {
                'GFI1B': cm.Greens,
                'NTC': cm.Greys,
                'ntc': cm.Greys,
                'GEMIN5': cm.Blues,
                'DDX6': cm.Reds,
            }
        else:
            self.target_cmaps = target_cmaps

        # Build guide and target colors
        self.guide_colors = build_guide_colors(self.palette)
        self.target_colors = self._build_target_colors()

        # Track which target index we're on for unknown targets
        self._unknown_target_idx = 0

    def _build_target_colors(self):
        """Build target-level representative colors from palette."""
        target_colors = {}
        for gene, colors in self.palette.items():
            # Use middle color as representative
            target_colors[gene] = colors[len(colors) // 2]
            # Also handle lowercase 'ntc'
            if gene.upper() == 'NTC':
                target_colors['ntc'] = colors[len(colors) // 2]
        return target_colors

    def _get_target_cmap(self, target):
        """Get colormap for a target, assigning new one if unknown."""
        target_str = str(target)

        # Check if we already have a cmap for this target
        if target_str in self.target_cmaps:
            return self.target_cmaps[target_str]

        # Handle NTC variants
        if target_str.upper() in ('NTC', 'NON-TARGETING', 'NON_TARGETING'):
            self.target_cmaps[target_str] = cm.Greys
            return cm.Greys

        # Assign a new colormap from the rotation
        cmap = self.DEFAULT_CMAPS[self._unknown_target_idx % len(self.DEFAULT_CMAPS)]
        self.target_cmaps[target_str] = cmap
        self._unknown_target_idx += 1
        return cmap

    def get_guide_color(self, guide, default='black'):
        """
        Get color for a specific guide.

        Dynamically generates colors for guides not in the predefined palette.
        Parses guide names like 'GFI1B_1', 'GFI1B_25', 'NTC_3', etc.
        """
        guide_str = str(guide)

        # First check if we have an exact match
        if guide_str in self.guide_colors:
            return self.guide_colors[guide_str]

        # Try to parse the guide name (e.g., "GFI1B_1" -> target="GFI1B", idx=1)
        if '_' in guide_str:
            parts = guide_str.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                target = parts[0]
                idx = int(parts[1])

                # Get colormap for this target
                cmap = self._get_target_cmap(target)

                # Generate color based on index
                # Use a range that gives good visual distinction
                # Map index to [0.3, 0.9] range with cycling for many guides
                if idx <= 30:
                    # For up to 30 guides, spread evenly
                    t = 0.3 + (idx / 35) * 0.6
                else:
                    # For more guides, cycle with slight offset
                    t = 0.3 + ((idx % 30) / 35) * 0.6

                color = cmap(t)

                # Cache for future use
                self.guide_colors[guide_str] = color
                return color

        return default

    def get_target_color(self, target, default='gray'):
        """Get color for a specific target."""
        target_str = str(target)

        if target_str in self.target_colors:
            return self.target_colors[target_str]

        # Generate a color for unknown target
        cmap = self._get_target_cmap(target_str)
        color = cmap(0.6)  # Middle of range
        self.target_colors[target_str] = color
        return color

    @classmethod
    def from_model(cls, model):
        """
        Create ColorScheme from a bayesDREAM model.

        Automatically detects all guides and targets and assigns appropriate
        colors.  Works for both low-MOI (reads ``model.meta``) and high-MOI
        (reads ``model.guide_meta`` / ``model.guide_targets_dict``).

        Parameters
        ----------
        model : bayesDREAM
            Fitted model.

        Returns
        -------
        ColorScheme
        """
        if not hasattr(model, 'meta'):
            return cls()

        is_high_moi = getattr(model, 'is_high_moi', False)

        if is_high_moi and hasattr(model, 'guide_meta'):
            # ---- High-MOI: build from guide_meta ----
            gm = model.guide_meta
            guides  = gm['guide'].astype(str).tolist()
            targets = []

            # Get targets from guide_targets_dict if available
            gtd = getattr(model, 'guide_targets_dict', {})
            if gtd:
                all_targets = set()
                for tlist in gtd.values():
                    all_targets.update(str(t) for t in tlist)
                targets = sorted(all_targets)
            elif 'target' in gm.columns:
                targets = gm['target'].astype(str).unique().tolist()

            # Count guides per target
            n_guides_per_target = {}
            if gtd:
                for guide, tlist in gtd.items():
                    for t in tlist:
                        t_str = str(t)
                        n_guides_per_target[t_str] = (
                            n_guides_per_target.get(t_str, 0) + 1
                        )
            elif 'target' in gm.columns:
                for t in targets:
                    n_guides_per_target[t] = int((gm['target'].astype(str) == t).sum())

        else:
            # ---- Low-MOI: build from meta ----
            targets = []
            if 'target' in model.meta.columns:
                targets = model.meta['target'].astype(str).unique().tolist()

            guides = []
            if 'guide' in model.meta.columns:
                guides = model.meta['guide'].astype(str).unique().tolist()

            # Count guides per target
            n_guides_per_target = {}
            for guide in guides:
                parts = str(guide).rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    target = parts[0]
                    idx = int(parts[1])
                    n_guides_per_target[target] = max(
                        n_guides_per_target.get(target, 0), idx
                    )
            if 'target' in model.meta.columns and 'guide' in model.meta.columns:
                for target in targets:
                    if str(target).lower() not in ('ntc', 'non-targeting'):
                        count = model.meta.loc[
                            model.meta['target'].astype(str) == target, 'guide'
                        ].nunique()
                        n_guides_per_target[target] = max(
                            n_guides_per_target.get(target, 0), count
                        )

        return cls.from_targets(targets, n_guides_per_target=n_guides_per_target)

    @classmethod
    def from_targets(cls, targets, colormaps=None, n_guides_per_target=None):
        """
        Create ColorScheme from target list.

        Parameters
        ----------
        targets : list of str
            Target names
        colormaps : list of str or colormaps, optional
            Colormap for each target. If None, uses default rotation.
        n_guides_per_target : dict or int, optional
            Number of guides per target. If int, uses same for all.
            If None, defaults to 10.

        Returns
        -------
        ColorScheme
        """
        if colormaps is None:
            colormaps = [cls.DEFAULT_CMAPS[i % len(cls.DEFAULT_CMAPS)]
                        for i in range(len(targets))]

        if n_guides_per_target is None:
            n_guides_per_target = 10  # Default to more guides

        if isinstance(n_guides_per_target, int):
            n_guides = {t: n_guides_per_target for t in targets}
        else:
            n_guides = n_guides_per_target

        palette = {}
        target_cmaps = {}
        for i, target in enumerate(targets):
            cmap = colormaps[i] if i < len(colormaps) else cls.DEFAULT_CMAPS[i % len(cls.DEFAULT_CMAPS)]
            n = n_guides.get(target, 10)

            # Special handling for NTC (use greys)
            if str(target).upper() in ('NTC', 'NON-TARGETING'):
                palette[target] = [cm.Greys(i) for i in np.linspace(0.4, 0.8, n)]
                target_cmaps[target] = cm.Greys
            else:
                palette[target] = [cmap(i) for i in np.linspace(0.3, 0.9, n)]
                target_cmaps[target] = cmap

        return cls(palette=palette, target_cmaps=target_cmaps)


def build_guide_colors(palette_dict):
    """
    Build guide-level colors from target palette.

    Parameters
    ----------
    palette_dict : dict
        Target -> list of colors mapping

    Returns
    -------
    dict
        Guide -> color mapping (e.g., 'GFI1B_1' -> color)
    """
    guide_colors = {}
    for gene, colors in palette_dict.items():
        for i, color in enumerate(colors, start=1):
            guide_colors[f"{gene}_{i}"] = color
    return guide_colors


def lighten(color, amount=0.3):
    """
    Lighten an RGBA/RGB color by mixing with white.

    Parameters
    ----------
    color : color spec
        Any matplotlib color specification
    amount : float
        Amount to lighten (0=no change, 1=white)

    Returns
    -------
    tuple
        RGBA color tuple
    """
    c = np.array(mcolors.to_rgba(color))
    white = np.array([1, 1, 1, 1])
    return tuple((1 - amount) * c + amount * white)


def darken(color, amount=0.3):
    """
    Darken an RGBA/RGB color by mixing with black.

    Parameters
    ----------
    color : color spec
        Any matplotlib color specification
    amount : float
        Amount to darken (0=no change, 1=black)

    Returns
    -------
    tuple
        RGBA color tuple
    """
    c = np.array(mcolors.to_rgba(color))
    black = np.array([0, 0, 0, 1])
    return tuple((1 - amount) * c + amount * black)
