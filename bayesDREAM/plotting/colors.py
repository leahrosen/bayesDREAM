"""
Color scheme management for bayesDREAM plots.

Provides consistent coloring across all visualization functions for targets, guides,
and technical groups.

Defaults
--------
- NTC target:         grey60  (#999999)
- Targeting target:   forestgreen (#228B22), subsequent genes use colormap rotation
- NTC guides:         shades of grey  (cm.Greys, 0.35–0.70)
- Targeting guides:   shades of green (cm.Greens, 0.40–0.85) for first gene, etc.
- CRISPRi:            steelblue
- CRISPRa:            tomato

Usage
-----
The recommended way to get correct guide colors is via the model's own scheme::

    # Built automatically at model init:
    model.color_scheme.get_guide_color(guide_name)

When passing an explicit scheme to a plotting function, the function calls
``color_scheme.connect(model)`` internally so that actual guide names are
always resolved correctly — regardless of naming convention.
"""

import copy
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import cm
from collections import defaultdict


# Recognised NTC target name variants
_NTC_VARIANTS_SET = frozenset({
    'ntc', 'non-targeting', 'non-targeting-control',
    'non_targeting', 'nontargeting',
})

# Colormap rotation for non-NTC targets (Greens first)
_TARGET_CMAPS = [cm.Greens, cm.Blues, cm.Reds, cm.Purples,
                 cm.Oranges, cm.YlOrBr, cm.PuBu, cm.BuGn]


def _is_ntc(target_str):
    return str(target_str).lower() in _NTC_VARIANTS_SET


def _guide_sort_key(g):
    """Sort guides by root name then trailing number."""
    parts = str(g).rsplit('_', 1)
    root = parts[0]
    idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return (root, idx)


def _detect_crispri_crispra(model):
    """Scan model metadata for CRISPRi/CRISPRa values; return color dict."""
    result = {
        'CRISPRi': 'steelblue', 'crispri': 'steelblue',
        'CRISPRa': 'tomato',    'crispra': 'tomato',
    }
    if not hasattr(model, 'meta'):
        return result
    for col in model.meta.columns:
        try:
            for v in model.meta[col].astype(str).unique():
                vl = v.lower()
                if 'crispri' in vl:
                    result[v] = 'steelblue'
                elif 'crispra' in vl:
                    result[v] = 'tomato'
        except Exception:
            pass
    return result


def _build_guide_to_target(model):
    """
    Build ``{guide_name: normalised_target}`` from model metadata.

    Works for both low-MOI (reads ``model.meta``) and high-MOI (reads
    ``model.guide_meta`` + ``model.guide_targets_dict``).  NTC variants
    are all normalised to ``'NTC'``.

    Returns
    -------
    dict[str, str]
    """
    guide_to_target = {}
    is_high_moi = getattr(model, 'is_high_moi', False)

    if is_high_moi and hasattr(model, 'guide_meta'):
        guide_names = model.guide_meta['guide'].astype(str).tolist()
        gtd = getattr(model, 'guide_targets_dict', {})
        for gn in guide_names:
            targets = [str(t) for t in gtd.get(gn, [])]
            targeting = [t for t in targets if not _is_ntc(t)]
            guide_to_target[gn] = targeting[0] if targeting else 'NTC'
    elif hasattr(model, 'meta'):
        if 'guide' in model.meta.columns and 'target' in model.meta.columns:
            for _, row in model.meta.drop_duplicates('guide').iterrows():
                t = str(row['target'])
                guide_to_target[str(row['guide'])] = 'NTC' if _is_ntc(t) else t

    guide_to_target['multiple_NTC'] = 'NTC'
    # Normalise all NTC variants → 'NTC'
    return {g: ('NTC' if _is_ntc(t) else t) for g, t in guide_to_target.items()}


def build_guide_colors(palette_dict):
    """
    Build guide-level colors from target palette.

    Generates ``TARGET_N`` style keys.  For actual model guide names use
    ``ColorScheme.from_model()`` or ``ColorScheme.connect(model)`` instead.

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
    """Lighten an RGBA/RGB color by mixing with white."""
    c = np.array(mcolors.to_rgba(color))
    white = np.array([1, 1, 1, 1])
    return tuple((1 - amount) * c + amount * white)


def darken(color, amount=0.3):
    """Darken an RGBA/RGB color by mixing with black."""
    c = np.array(mcolors.to_rgba(color))
    black = np.array([0, 0, 0, 1])
    return tuple((1 - amount) * c + amount * black)


class ColorScheme:
    """
    Manages consistent color schemes for bayesDREAM visualizations.

    Three independent color lookups:

    * **guide colors**   – one color per guide, shades within target family
    * **target colors**  – one representative color per target
    * **technical group colors** – CRISPRi → steelblue, CRISPRa → tomato, etc.

    Construction
    ------------
    * ``ColorScheme.from_model(model)`` — recommended; auto-detects everything
    * ``ColorScheme(palette=..., technical_group_colors=...)`` — manual palette

    Key method
    ----------
    ``connect(model)``
        Augments any scheme with actual guide names from the model, using this
        scheme's palette / colormaps for color generation.  Always call this
        inside plotting functions before looking up guide colors::

            color_scheme = color_scheme.connect(model)

    Attributes
    ----------
    guide_colors : dict
        guide name → RGBA color
    target_colors : dict
        target name → RGBA color
    technical_group_colors : dict
        group name/value → color
    target_cmaps : dict
        target → colormap (for dynamic generation of unseen guides)
    _guide_to_target : dict
        guide name → target name (populated by ``from_model`` and ``connect``)
    """

    # Default colormap rotation for targets
    DEFAULT_CMAPS = _TARGET_CMAPS

    def __init__(self, palette=None, target_cmaps=None,
                 technical_group_colors=None,
                 guide_colors=None, target_colors=None):
        """
        Parameters
        ----------
        palette : dict, optional
            Target -> list-of-colors mapping.  Colors are indexed as
            ``TARGET_1``, ``TARGET_2``, … For actual model guide names that
            don't follow this convention, call ``connect(model)`` after
            construction (plotting functions do this automatically).
        target_cmaps : dict, optional
            Target -> colormap.  If ``None``, inferred from palette keys:
            NTC-variant keys → ``cm.Greys``; others rotate through
            ``DEFAULT_CMAPS``.
        technical_group_colors : dict, optional
            Explicit map from technical group values to colors.
        guide_colors : dict, optional
            Explicit guide-name → color overrides / additions.
        target_colors : dict, optional
            Explicit target-name → color overrides / additions.
        """
        # ---- palette-based colours ----
        if palette is None:
            self.palette = {
                'GFI1B': [cm.Greens(i) for i in np.linspace(0.40, 0.85, 5)],
                'NTC':   [cm.Greys(i)  for i in np.linspace(0.35, 0.70, 10)],
                'ntc':   [cm.Greys(i)  for i in np.linspace(0.35, 0.70, 10)],
            }
        else:
            self.palette = palette

        # Infer target_cmaps from palette when not supplied explicitly
        if target_cmaps is None:
            self.target_cmaps = {}
            non_ntc_idx = 0
            for t in self.palette:
                if _is_ntc(t):
                    self.target_cmaps[t] = cm.Greys
                else:
                    self.target_cmaps[t] = self.DEFAULT_CMAPS[
                        non_ntc_idx % len(self.DEFAULT_CMAPS)]
                    non_ntc_idx += 1
        else:
            self.target_cmaps = dict(target_cmaps)

        # Build from palette (TARGET_N style keys), then apply explicit overrides
        self.guide_colors = build_guide_colors(self.palette)
        if guide_colors:
            self.guide_colors.update(guide_colors)

        self.target_colors = self._build_target_colors()
        if target_colors:
            self.target_colors.update(target_colors)

        # Technical group colours
        if technical_group_colors is None:
            self.technical_group_colors = {
                'CRISPRi': 'steelblue', 'crispri': 'steelblue',
                'CRISPRa': 'tomato',    'crispra': 'tomato',
            }
        else:
            self.technical_group_colors = dict(technical_group_colors)

        self._unknown_target_idx = 0

        # Guide → target map; populated by from_model / connect
        self._guide_to_target = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_target_colors(self):
        """Build target-level representative colors from palette."""
        target_colors = {}
        for gene, colors in self.palette.items():
            if _is_ntc(gene):
                target_colors[gene] = mcolors.to_rgba('#999999')  # grey60
            else:
                target_colors[gene] = colors[len(colors) // 2]
        return target_colors

    def _get_target_cmap(self, target):
        """Get colormap for a target, assigning new one if unknown."""
        target_str = str(target)
        if target_str in self.target_cmaps:
            return self.target_cmaps[target_str]
        if _is_ntc(target_str):
            self.target_cmaps[target_str] = cm.Greys
            return cm.Greys
        cmap = self.DEFAULT_CMAPS[self._unknown_target_idx % len(self.DEFAULT_CMAPS)]
        self.target_cmaps[target_str] = cmap
        self._unknown_target_idx += 1
        return cmap

    # ------------------------------------------------------------------
    # Public colour lookup
    # ------------------------------------------------------------------

    def get_guide_color(self, guide, default='black'):
        """
        Get color for a guide.

        Checks explicit ``guide_colors`` first (populated by ``from_model``
        or ``connect``), then falls back to ``TARGET_NUMBER`` dynamic
        generation, then to ``_guide_to_target`` lookup, then to ``default``.

        For reliable coloring of non-``TARGET_N`` guide names, call
        ``connect(model)`` on the scheme before plotting (plotting functions
        do this automatically).
        """
        guide_str = str(guide)

        if guide_str in self.guide_colors:
            return self.guide_colors[guide_str]

        # Special pseudo-guide from high-MOI resolve_guide_labels
        if guide_str == 'multiple_NTC':
            return mcolors.to_rgba('#999999')

        # Try TARGET_NUMBER dynamic generation
        if '_' in guide_str:
            parts = guide_str.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                target = parts[0]
                idx = int(parts[1])
                cmap = self._get_target_cmap(target)
                t = 0.35 + (min(idx, 30) / 35) * 0.5
                color = cmap(t)
                self.guide_colors[guide_str] = color
                return color

        # _guide_to_target fallback (available after connect / from_model)
        if guide_str in self._guide_to_target:
            target = self._guide_to_target[guide_str]
            cmap = self._get_target_cmap(target)
            shade = 0.35 if _is_ntc(target) else 0.55
            color = cmap(shade)
            self.guide_colors[guide_str] = color
            return color

        return default

    def get_target_color(self, target, default='gray'):
        """Get representative color for a target."""
        target_str = str(target)
        if target_str in self.target_colors:
            return self.target_colors[target_str]
        if _is_ntc(target_str):
            c = mcolors.to_rgba('#999999')
            self.target_colors[target_str] = c
            return c
        cmap = self._get_target_cmap(target_str)
        color = cmap(0.65)
        self.target_colors[target_str] = color
        return color

    def get_technical_group_color(self, group, default='gray'):
        """Get color for a technical group value (e.g. 'CRISPRi', 'CRISPRa')."""
        g = str(group)
        if g in self.technical_group_colors:
            return self.technical_group_colors[g]
        g_lower = g.lower()
        if 'crispri' in g_lower:
            return 'steelblue'
        if 'crispra' in g_lower:
            return 'tomato'
        return default

    def guide_target(self, guide):
        """
        Return the target for a guide name, or ``None`` if unknown.

        Consults ``_guide_to_target`` (populated by ``from_model`` /
        ``connect``), then falls back to the ``TARGET_NUMBER`` naming
        convention.
        """
        g = str(guide)
        if g in self._guide_to_target:
            return self._guide_to_target[g]
        # TARGET_NUMBER fallback
        parts = g.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return None

    # ------------------------------------------------------------------
    # Model connection
    # ------------------------------------------------------------------

    def connect(self, model):
        """
        Return a copy of this scheme with guide_colors populated for every
        guide present in *model*.

        This is the key method that makes palette-based schemes work with any
        guide naming convention — not just ``TARGET_NUMBER`` format.

        * If all guides are already in ``guide_colors`` **and**
          ``_guide_to_target`` is already populated, returns ``self``
          unchanged (no copy made).
        * Colors for new guides are generated from this scheme's palette (if
          the target is present) or from the target's colormap, distributing
          shades evenly across guides within each target.
        * The returned scheme's ``_guide_to_target`` is always fully
          populated.

        Parameters
        ----------
        model : bayesDREAM

        Returns
        -------
        ColorScheme
        """
        g2t = _build_guide_to_target(model)
        missing = {g: t for g, t in g2t.items() if g not in self.guide_colors}

        # Fast path: nothing to do
        if not missing and self._guide_to_target:
            return self

        new_cs = copy.copy(self)
        new_cs._guide_to_target = {**self._guide_to_target, **g2t}
        new_cs.guide_colors = dict(self.guide_colors)
        new_cs.target_colors = dict(self.target_colors)
        new_cs.target_cmaps = dict(self.target_cmaps)
        new_cs._unknown_target_idx = self._unknown_target_idx

        if not missing:
            return new_cs

        # Group missing guides by target
        by_target = defaultdict(list)
        for g, t in missing.items():
            by_target[t].append(g)

        for target, guides in by_target.items():
            guides_sorted = sorted(guides, key=_guide_sort_key)
            n = len(guides_sorted)

            # If user supplied palette colors for this target, distribute them
            palette_key = next(
                (k for k in (target, target.lower(), target.upper())
                 if k in self.palette), None)

            if palette_key is not None:
                palette_colors = self.palette[palette_key]
                n_p = len(palette_colors)
                for i, g in enumerate(guides_sorted):
                    idx = int(round(i / max(n - 1, 1) * (n_p - 1))) if n > 1 else 0
                    new_cs.guide_colors[g] = palette_colors[idx]
            else:
                cmap = new_cs._get_target_cmap(target)
                if _is_ntc(target):
                    shade_vals = np.linspace(0.35, 0.70, n)
                else:
                    shade_vals = np.linspace(0.40, 0.85, n)
                for g, v in zip(guides_sorted, shade_vals):
                    new_cs.guide_colors[g] = cmap(v)

        return new_cs

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_model(cls, model):
        """
        Build a ColorScheme from a bayesDREAM model.

        Uses actual guide names from the model (works with any naming
        convention).  Automatically detects CRISPRi / CRISPRa in covariate
        columns.

        Parameters
        ----------
        model : bayesDREAM

        Returns
        -------
        ColorScheme
        """
        if not hasattr(model, 'meta'):
            return cls()

        is_high_moi = getattr(model, 'is_high_moi', False)

        # ---- 1. guide → primary target ----
        guide_to_target = _build_guide_to_target(model)

        # ---- 2. Ordered target list: NTC, then cis_gene, then others ----
        cis_gene = getattr(model, 'cis_gene', None)
        ordered_targets = []
        seen = set()

        for preferred in (['NTC', cis_gene] if cis_gene else ['NTC']):
            if preferred and preferred not in seen:
                if preferred == 'NTC' and any(t == 'NTC' for t in guide_to_target.values()):
                    seen.add('NTC')
                    ordered_targets.append('NTC')
                elif preferred != 'NTC' and preferred in guide_to_target.values():
                    seen.add(preferred)
                    ordered_targets.append(preferred)

        for t in guide_to_target.values():
            if t not in seen:
                seen.add(t)
                ordered_targets.append(t)

        # ---- 3. Colormap per target ----
        target_cmap = {}
        non_ntc_idx = 0
        for t in ordered_targets:
            if t == 'NTC':
                target_cmap['NTC'] = cm.Greys
            else:
                target_cmap[t] = _TARGET_CMAPS[non_ntc_idx % len(_TARGET_CMAPS)]
                non_ntc_idx += 1

        # ---- 4. Guide shades + target representative colours ----
        guides_per_target = defaultdict(list)
        for g, t in guide_to_target.items():
            guides_per_target[t].append(g)

        guide_colors_out  = {}
        target_colors_out = {}
        first_non_ntc = next((t for t in ordered_targets if t != 'NTC'), None)

        for t in ordered_targets:
            guides = sorted(guides_per_target.get(t, []), key=_guide_sort_key)
            cmap = target_cmap.get(t, cm.Greens)
            n = max(len(guides), 1)

            if t == 'NTC':
                shade_vals = np.linspace(0.35, 0.70, n)
                target_colors_out['NTC'] = mcolors.to_rgba('#999999')
            else:
                shade_vals = np.linspace(0.40, 0.85, n)
                if t == first_non_ntc:
                    target_colors_out[t] = mcolors.to_rgba('forestgreen')
                else:
                    target_colors_out[t] = cmap(0.65)

            for g, v in zip(guides, shade_vals):
                guide_colors_out[g] = cmap(v)

        # ---- 5. Technical group colours ----
        tech_colors = _detect_crispri_crispra(model)

        instance = cls(
            technical_group_colors=tech_colors,
            guide_colors=guide_colors_out,
            target_colors=target_colors_out,
            target_cmaps=target_cmap,
        )
        instance._guide_to_target = guide_to_target
        return instance

    @classmethod
    def from_targets(cls, targets, colormaps=None, n_guides_per_target=None):
        """
        Create ColorScheme from a list of target names.

        Parameters
        ----------
        targets : list of str
        colormaps : list of colormaps, optional
        n_guides_per_target : dict or int, optional

        Returns
        -------
        ColorScheme
        """
        if colormaps is None:
            colormaps = [cls.DEFAULT_CMAPS[i % len(cls.DEFAULT_CMAPS)]
                        for i in range(len(targets))]

        if n_guides_per_target is None:
            n_guides = {t: 10 for t in targets}
        elif isinstance(n_guides_per_target, int):
            n_guides = {t: n_guides_per_target for t in targets}
        else:
            n_guides = {t: n_guides_per_target.get(t, 10) for t in targets}

        palette = {}
        target_cmaps = {}
        non_ntc_cmap_idx = 0

        for target in targets:
            n = max(n_guides.get(target, 10), 1)
            if _is_ntc(str(target)):
                palette[target]       = [cm.Greys(v) for v in np.linspace(0.35, 0.70, n)]
                target_cmaps[target]  = cm.Greys
            else:
                cmap = colormaps[non_ntc_cmap_idx] if non_ntc_cmap_idx < len(colormaps) \
                       else cls.DEFAULT_CMAPS[non_ntc_cmap_idx % len(cls.DEFAULT_CMAPS)]
                palette[target]      = [cmap(v) for v in np.linspace(0.40, 0.85, n)]
                target_cmaps[target] = cmap
                non_ntc_cmap_idx += 1

        return cls(palette=palette, target_cmaps=target_cmaps)
