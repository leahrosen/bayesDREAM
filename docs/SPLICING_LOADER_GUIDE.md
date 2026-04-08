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
    └── denominator.npz        # denominator counts (binomial only)
```

**Distribution detection:**

| Files present                      | Distribution |
|------------------------------------|--------------|
| `counts.npz` + `denominator.npz`   | `binomial`   |
| `counts.npz` only                  | `multinomial`|

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
def load_binomial_type(model, base_dir, stype, cell_col='cell',
                       name_prefix='splicing', overwrite=False):
    type_dir = os.path.join(base_dir, stype)
    model_cells = model.meta[cell_col].tolist()

    # Cell order in file
    cell_meta  = pd.read_csv(os.path.join(type_dir, 'cell_meta.tsv.gz'), sep='\t')
    file_cells = cell_meta[cell_col].tolist()

    # Feature metadata
    feature_meta = pd.read_csv(os.path.join(type_dir, 'feature_meta.tsv.gz'), sep='\t')

    # Counts and denominator (features × file_cells)
    counts      = load_npz(os.path.join(type_dir, 'counts.npz'))
    denominator = load_npz(os.path.join(type_dir, 'denominator.npz'))

    # Align to model cell order
    counts      = align_cells(counts,      file_cells, model_cells)
    denominator = align_cells(denominator, file_cells, model_cells)

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
for stype in ['exon_skip', 'intron_retention', 'donor_efficiency',
              'acceptor_efficiency', 'sj', 'mxe', 'gene_velocity']:
    load_binomial_type(model, 'loader_inputs/', stype)
```

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


def _load_type(stype, distribution):
    d = os.path.join(BASE_DIR, stype)

    file_cells   = pd.read_csv(os.path.join(d, 'cell_meta.tsv.gz'),    sep='\t')[CELL_COL].tolist()
    feature_meta = pd.read_csv(os.path.join(d, 'feature_meta.tsv.gz'), sep='\t')
    counts       = align_cells(load_npz(os.path.join(d, 'counts.npz')), file_cells, model_cells)

    if distribution == 'binomial':
        denominator = align_cells(load_npz(os.path.join(d, 'denominator.npz')), file_cells, model_cells)
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

- **Cell alignment**: `align_cells` reorders columns to match `model.meta['cell']`
  exactly.  Cells in the model but missing from the file are filled with zeros
  (a warning is printed).  `add_custom_modality` will also drop any cells in the
  passed `cell_names` that are not in the model's cell list.
- **Zero-padding**: `reconstruct_multinomial_3d` pads shorter sites to
  `max_categories` with zeros.  The model's multinomial likelihood ignores
  zero-count categories automatically.
- **Filtering**: `add_custom_modality` applies variance filtering (removes
  constant features) after the data is passed in.
- **Category metadata**: The flat row index in `counts.npz` matches the row
  index in `sj_observed_meta.tsv.gz`.  Use `feature_meta['row_start']` and
  `feature_meta['row_end']` to slice it per feature if needed.
