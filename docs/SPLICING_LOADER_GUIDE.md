# Splicing Modality Loader Guide

Pre-calculated splicing inputs come in a standard directory layout.  Each type
lives in its own subdirectory.  Loading a type into a `bayesDREAM` model means:

1. Reading the files
2. Aligning cells to the model's cell order
3. Reconstructing 3-D arrays for multinomial types (donor/acceptor choice)
4. Calling `model.add_custom_modality()`

---

## Directory structure

```
loader_inputs/
├── manifest.json
└── {type}/
    ├── cell_meta.tsv.gz       # cell identifiers
    ├── feature_meta.tsv.gz    # feature annotations
    ├── counts.npz             # numerator counts  (features × cells, sparse)
    └── denominator.npz        # denominator counts (binomial only; optional if use_counts=True)
```

**Distribution detection:**

| Type         | Distribution  | Denominator source                                      |
|--------------|---------------|---------------------------------------------------------|
| binomial     | `binomial`    | `model.counts` (default) or `denominator.npz`          |
| multinomial  | `multinomial` | none                                                    |

Binomial denominators can be sourced two ways (controlled by `use_counts` in the loader):

- **`use_counts=True` (default)**: denominator taken from `model.counts` via
  `feature_meta['gene_for_denominator']`.  Ensures consistency with the gene
  expression data already in the model.  No `denominator.npz` needed.
- **`use_counts=False`**: denominator loaded from `denominator.npz` in the type
  directory (original behaviour).

Types in a typical run:

| Type                 | Distribution  | Notes                              |
|----------------------|---------------|-------------------------------------|
| `sj`                 | binomial      | Raw SJ counts / gene expression    |
| `exon_skip`          | binomial      | Inclusion / total reads            |
| `mxe`                | binomial      | Mutually exclusive exon PSI        |
| `intron_retention`   | binomial      | Intron reads / total reads         |
| `donor_efficiency`   | binomial      | Donor reads / regional reads       |
| `acceptor_efficiency`| binomial      | Acceptor reads / regional reads    |
| `gene_velocity`      | binomial      | Nascent / total                    |
| `donor_choice`       | multinomial   | Which acceptor chosen per donor    |
| `acceptor_choice`    | multinomial   | Which donor chosen per acceptor    |

---

## Helper imports

Copy these helper functions into your loading script — they are **not** part of
the `bayesDREAM` package.

```python
import os
import warnings
import numpy as np
import pandas as pd


def load_npz(path):
    """Load a .npz file as a dense 2-D numpy array."""
    try:
        import scipy.sparse
        return scipy.sparse.load_npz(path).toarray()
    except Exception:
        pass
    data = np.load(path, allow_pickle=False)
    keys = list(data.keys())
    for key in ('data', 'arr_0', 'matrix', 'counts'):
        if key in keys:
            return data[key]
    return data[keys[0]]


def align_cells(counts, file_cells, model_cells):
    """
    Reorder/subset the cell axis (axis 1) of counts to match model_cells.
    Cells in model_cells but absent from file_cells are filled with zeros.
    """
    file_idx = {c: i for i, c in enumerate(file_cells)}
    n_missing = sum(1 for c in model_cells if c not in file_idx)
    if n_missing:
        warnings.warn(
            f"{n_missing}/{len(model_cells)} model cells not found in file; "
            "filling missing cells with zeros."
        )
    out_shape = list(counts.shape)
    out_shape[1] = len(model_cells)
    out = np.zeros(out_shape, dtype=counts.dtype)
    for j, cell in enumerate(model_cells):
        if cell in file_idx:
            if counts.ndim == 2:
                out[:, j] = counts[:, file_idx[cell]]
            elif counts.ndim == 3:
                out[:, j, :] = counts[:, file_idx[cell], :]
    return out


def reconstruct_multinomial_3d(counts_2d, feature_meta):
    """
    Reshape a 2-D counts matrix into a 3-D multinomial array using
    row_start / row_end columns in feature_meta.

    The 2-D layout is (total_rows × n_cells); rows row_start:row_end belong to
    feature i.  Returns shape (n_features × n_cells × max_categories).
    """
    n_features = len(feature_meta)
    n_cells    = counts_2d.shape[1]
    max_cats   = int(feature_meta['n_categories'].max())
    counts_3d  = np.zeros((n_features, n_cells, max_cats), dtype=counts_2d.dtype)
    for i, (_, row) in enumerate(feature_meta.iterrows()):
        s, e  = int(row['row_start']), int(row['row_end'])
        n_cat = e - s
        counts_3d[i, :, :n_cat] = counts_2d[s:e, :].T
    return counts_3d
```

---

## Loading binomial types

All binomial types share the same pattern.

```python
def load_binomial_type(model, base_dir, stype,
                       name_prefix='splicing',
                       use_counts=True,
                       gene_subset='counts',
                       overwrite=False):
    """
    Parameters
    ----------
    use_counts : bool, default True
        If True, derive the binomial denominator from model.counts (gene
        expression) via feature_meta['gene_for_denominator'].
        If False, load denominator from denominator.npz in the type directory.
    gene_subset : str or None, default 'counts'
        Filter features before loading:
        - None / 'none' : keep all features.
        - 'counts'      : keep features whose gene_for_denominator is in
                          model.counts.index.  Recommended with use_counts=True.
        - 'primary'     : keep features whose gene_for_denominator is in the
                          primary modality gene set (post zero-variance filtering).
    """
    type_dir    = os.path.join(base_dir, stype)
    model_cells = model.meta['L_cell_barcode'].tolist()

    # Cell order in file
    cell_meta  = pd.read_csv(os.path.join(type_dir, 'cell_meta.tsv.gz'), sep='\t')
    file_cells = cell_meta['cell_barcode'].tolist()

    # Feature metadata — add 'gene' alias for plot_xy_data compatibility
    feature_meta = pd.read_csv(os.path.join(type_dir, 'feature_meta.tsv.gz'), sep='\t')
    if 'gene' not in feature_meta.columns and 'gene_for_denominator' in feature_meta.columns:
        feature_meta = feature_meta.assign(gene=feature_meta['gene_for_denominator'])

    # Build feature row mask for gene_subset option
    row_mask = None
    if gene_subset not in (None, 'none') and 'gene_for_denominator' in feature_meta.columns:
        if gene_subset == 'counts':
            valid_genes = set(model.counts.index)
        elif gene_subset == 'primary':
            primary_mod = model.get_modality(model.primary_modality)
            for col in ('gene', 'gene_name', 'gene_id'):
                if col in primary_mod.feature_meta.columns:
                    valid_genes = set(primary_mod.feature_meta[col])
                    break
            else:
                valid_genes = set(primary_mod.feature_meta.index)
        else:
            raise ValueError(f"gene_subset must be None, 'none', 'counts', or 'primary'; "
                             f"got {gene_subset!r}")
        row_mask = feature_meta['gene_for_denominator'].isin(valid_genes).values
        n_dropped = int((~row_mask).sum())
        if n_dropped:
            print(f"[INFO] gene_subset='{gene_subset}': dropping {n_dropped} features "
                  f"with unrecognised gene ({int(row_mask.sum())} kept)")

    # Load and (optionally row-filter) counts, then align cells
    counts_raw = load_npz(os.path.join(type_dir, 'counts.npz'))
    if row_mask is not None:
        counts_raw   = counts_raw[row_mask]
        feature_meta = feature_meta[row_mask].reset_index(drop=True)
    counts = align_cells(counts_raw, file_cells, model_cells)

    # Denominator
    if use_counts:
        # One row per feature; .loc with a list handles multiple features per gene.
        genes       = feature_meta['gene_for_denominator'].tolist()
        denominator = model.counts.loc[genes, model_cells].values
    else:
        denom_raw = load_npz(os.path.join(type_dir, 'denominator.npz'))
        if row_mask is not None:
            denom_raw = denom_raw[row_mask]
        denominator = align_cells(denom_raw, file_cells, model_cells)

    model.add_custom_modality(
        name=f'{name_prefix}_{stype}',
        counts=counts,
        feature_meta=feature_meta.reset_index(drop=True),
        distribution='binomial',
        denominator=denominator,
        cell_names=model_cells,
        overwrite=overwrite,
    )
    print(f"Added '{name_prefix}_{stype}': {counts.shape[0]} features")
```

### Usage

```python
# Recommended: use model.counts as denominator, subset to genes in model
for stype in ['exon_skip', 'intron_retention', 'donor_efficiency',
              'acceptor_efficiency', 'sj', 'mxe', 'gene_velocity']:
    load_binomial_type(model, 'loader_inputs/', stype,
                       use_counts=True, gene_subset='counts')

# Use pre-computed denominator.npz, no gene filtering
load_binomial_type(model, 'loader_inputs/', 'sj',
                   use_counts=False, gene_subset=None)

# Only keep features for genes that passed zero-variance filtering in the primary modality
load_binomial_type(model, 'loader_inputs/', 'sj',
                   use_counts=True, gene_subset='primary')
```

### `feature_meta` requirements for `plot_xy_data`

`plot_xy_data` can accept a gene name and return all matching features.  For
this to work, `feature_meta` must contain a `gene`, `gene_name`, or `gene_id`
column.  The loader above satisfies this by aliasing `gene_for_denominator` →
`gene`.

---

## Loading multinomial types (donor/acceptor choice)

The `counts.npz` for these types is a **flat 2-D matrix**
(total_rows × n_cells), where each donor/acceptor site occupies a contiguous
block of rows.  `feature_meta` records which rows belong to which site:

| Column           | Meaning                                         |
|------------------|-------------------------------------------------|
| `feature_id`     | Donor/acceptor identifier, e.g. `D:chr1:+:999834` |
| `n_categories`   | Number of acceptors (donors) for this site      |
| `row_start`      | First row index (inclusive) in the 2-D matrix   |
| `row_end`        | Last row index (exclusive) in the 2-D matrix    |
| `category_labels`| JSON list of the other-side site IDs            |

`reconstruct_multinomial_3d` uses `row_start` / `row_end` to build a 3-D
array of shape **(n_features × n_cells × max_categories)**, zero-padded.

### Category-level metadata

Each row of the flat 2-D matrix corresponds to one junction (one
acceptor choice for a given donor, or one donor choice for a given acceptor).
If you want per-category annotations (strand, coordinates, novelty flags, …)
load the observed-junction metadata separately:

```python
# One row per junction, matching the flat row order of counts.npz
sj_meta = pd.read_csv('loader_inputs/sj_observed_meta.tsv.gz', sep='\t')
# sj_meta row i  →  flat counts row i
# feature_meta row_start:row_end  →  sj_meta rows for that site
```

### Loading code

```python
def load_multinomial_type(model, base_dir, stype, cell_col='cell',
                          name_prefix='splicing', overwrite=False):
    type_dir = os.path.join(base_dir, stype)
    model_cells = model.meta[cell_col].tolist()

    cell_meta    = pd.read_csv(os.path.join(type_dir, 'cell_meta.tsv.gz'), sep='\t')
    file_cells   = cell_meta[cell_col].tolist()
    feature_meta = pd.read_csv(os.path.join(type_dir, 'feature_meta.tsv.gz'), sep='\t')

    # 2-D flat matrix: (total_rows × n_file_cells)
    counts_2d = load_npz(os.path.join(type_dir, 'counts.npz'))

    # Align cells first (axis 1 of the flat 2-D matrix)
    counts_2d = align_cells(counts_2d, file_cells, model_cells)

    # Reconstruct 3-D: (n_features × n_model_cells × max_categories)
    counts_3d = reconstruct_multinomial_3d(counts_2d, feature_meta)

    print(f"  shape: {counts_3d.shape}  "
          f"(features × cells × max_categories={counts_3d.shape[2]})")

    model.add_custom_modality(
        name=f'{name_prefix}_{stype}',
        counts=counts_3d,
        feature_meta=feature_meta.reset_index(drop=True),
        distribution='multinomial',
        cell_names=model_cells,
        overwrite=overwrite,
    )
    print(f"Added '{name_prefix}_{stype}': {counts_3d.shape[0]} features")
```

### Usage

```python
for stype in ['donor_choice', 'acceptor_choice']:
    load_multinomial_type(model, 'loader_inputs/', stype)
```

---

## Complete loading script

```python
import os, warnings
import numpy as np
import pandas as pd

# --- helpers (not part of bayesDREAM) ---

def load_npz(path):
    try:
        import scipy.sparse
        return scipy.sparse.load_npz(path).toarray()
    except Exception:
        pass
    data = np.load(path, allow_pickle=False)
    keys = list(data.keys())
    for key in ('data', 'arr_0', 'matrix', 'counts'):
        if key in keys:
            return data[key]
    return data[keys[0]]

def align_cells(counts, file_cells, model_cells):
    file_idx = {c: i for i, c in enumerate(file_cells)}
    n_missing = sum(1 for c in model_cells if c not in file_idx)
    if n_missing:
        warnings.warn(f"{n_missing}/{len(model_cells)} model cells missing; filling with zeros.")
    out_shape = list(counts.shape)
    out_shape[1] = len(model_cells)
    out = np.zeros(out_shape, dtype=counts.dtype)
    for j, cell in enumerate(model_cells):
        if cell in file_idx:
            if counts.ndim == 2:
                out[:, j] = counts[:, file_idx[cell]]
            elif counts.ndim == 3:
                out[:, j, :] = counts[:, file_idx[cell], :]
    return out

def reconstruct_multinomial_3d(counts_2d, feature_meta):
    n_features = len(feature_meta)
    n_cells    = counts_2d.shape[1]
    max_cats   = int(feature_meta['n_categories'].max())
    counts_3d  = np.zeros((n_features, n_cells, max_cats), dtype=counts_2d.dtype)
    for i, (_, row) in enumerate(feature_meta.iterrows()):
        s, e  = int(row['row_start']), int(row['row_end'])
        n_cat = e - s
        counts_3d[i, :, :n_cat] = counts_2d[s:e, :].T
    return counts_3d

# --- loading ---

BINOMIAL_TYPES    = ['sj', 'exon_skip', 'mxe', 'intron_retention',
                     'donor_efficiency', 'acceptor_efficiency', 'gene_velocity']
MULTINOMIAL_TYPES = ['donor_choice', 'acceptor_choice']

BASE_DIR   = 'loader_inputs/'
CELL_COL   = 'cell'
PREFIX     = 'splicing'

model_cells = model.meta[CELL_COL].tolist()


USE_COUNTS  = True      # True: denominator from model.counts; False: from denominator.npz
GENE_SUBSET = 'counts'  # None, 'counts', or 'primary'


def _load_type(stype, distribution):
    d = os.path.join(BASE_DIR, stype)

    file_cells   = pd.read_csv(os.path.join(d, 'cell_meta.tsv.gz'),    sep='\t')[CELL_COL].tolist()
    feature_meta = pd.read_csv(os.path.join(d, 'feature_meta.tsv.gz'), sep='\t')

    if 'gene' not in feature_meta.columns and 'gene_for_denominator' in feature_meta.columns:
        feature_meta = feature_meta.assign(gene=feature_meta['gene_for_denominator'])

    # Build gene_subset row mask
    row_mask = None
    if distribution == 'binomial' and GENE_SUBSET not in (None, 'none') \
            and 'gene_for_denominator' in feature_meta.columns:
        if GENE_SUBSET == 'counts':
            valid_genes = set(model.counts.index)
        elif GENE_SUBSET == 'primary':
            pm = model.get_modality(model.primary_modality)
            for col in ('gene', 'gene_name', 'gene_id'):
                if col in pm.feature_meta.columns:
                    valid_genes = set(pm.feature_meta[col]); break
            else:
                valid_genes = set(pm.feature_meta.index)
        row_mask = feature_meta['gene_for_denominator'].isin(valid_genes).values
        n_dropped = int((~row_mask).sum())
        if n_dropped:
            print(f"[INFO] {stype}: dropping {n_dropped} features with unrecognised gene")
        feature_meta = feature_meta[row_mask].reset_index(drop=True)

    counts_raw = load_npz(os.path.join(d, 'counts.npz'))
    if row_mask is not None:
        counts_raw = counts_raw[row_mask]

    counts = align_cells(counts_raw, file_cells, model_cells)

    if distribution == 'binomial':
        if USE_COUNTS:
            genes       = feature_meta['gene_for_denominator'].tolist()
            denominator = model.counts.loc[genes, model_cells].values
        else:
            denom_raw = load_npz(os.path.join(d, 'denominator.npz'))
            if row_mask is not None:
                denom_raw = denom_raw[row_mask]
            denominator = align_cells(denom_raw, file_cells, model_cells)
        model.add_custom_modality(
            name=f'{PREFIX}_{stype}',
            counts=counts,
            feature_meta=feature_meta.reset_index(drop=True),
            distribution='binomial',
            denominator=denominator,
            cell_names=model_cells,
        )

    elif distribution == 'multinomial':
        counts_3d = reconstruct_multinomial_3d(counts, feature_meta)
        model.add_custom_modality(
            name=f'{PREFIX}_{stype}',
            counts=counts_3d,
            feature_meta=feature_meta.reset_index(drop=True),
            distribution='multinomial',
            cell_names=model_cells,
        )

    print(f"  {PREFIX}_{stype}: {counts.shape[0]} features")


for stype in BINOMIAL_TYPES:
    _load_type(stype, 'binomial')

for stype in MULTINOMIAL_TYPES:
    _load_type(stype, 'multinomial')
```

---

## Notes

- **Cell alignment**: `align_cells` reorders columns to match `model.meta['L_cell_barcode']`
  exactly.  Cells in the model but missing from the file are filled with zeros
  (a warning is printed).  `add_custom_modality` will then drop any cells not
  present in the model (intersection-only; no zero-filling by the model).
- **Denominator (`use_counts`)**: When `use_counts=True` (default), the denominator
  is built from `model.counts` via `feature_meta['gene_for_denominator']`.
  `model.counts.loc[genes, model_cells]` handles duplicates correctly (multiple
  features per gene each get their own row).  When `use_counts=False`, the
  denominator is loaded from `denominator.npz` in the type directory.
- **Gene subsetting (`gene_subset`)**: Controls which features are retained before
  loading.  Use `'counts'` (default) to keep features whose gene is in
  `model.counts.index`; this is required when `use_counts=True` to avoid KeyErrors.
  Use `'primary'` to further restrict to genes that survived zero-variance filtering
  in the primary modality.  Use `None` to disable.
- **Gene column for plotting**: `plot_xy_data` searches `gene`, `gene_name`, or
  `gene_id` columns to look up features by gene.  The loader adds `gene` as an
  alias for `gene_for_denominator` so gene-level lookups work out of the box.
- **Zero-padding**: `reconstruct_multinomial_3d` pads shorter sites to
  `max_categories` with zeros.  The model's multinomial likelihood ignores
  zero-count categories automatically.
- **Filtering**: `add_custom_modality` applies variance filtering (removes
  constant features) after the data is passed in.
- **Category metadata**: The flat row index in `counts.npz` matches the row
  index in `sj_observed_meta.tsv.gz`.  Use `feature_meta['row_start']` and
  `feature_meta['row_end']` to slice it per feature if needed.
