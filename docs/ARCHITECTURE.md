# Multi-Modal bayesDREAM Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Interface Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  bayesDREAM (model.py)                                           │
│  - add_transcript_modality()                                    │
│  - add_splicing_modality()                                      │
│  - add_atac_modality()                                          │
│  - add_custom_modality()                                        │
│  - get_modality() / list_modalities()                          │
└────────────┬────────────────────────────────────┬───────────────┘
             │                                    │
             │ inherits                           │ contains
             ↓                                    ↓
┌────────────────────────┐          ┌────────────────────────────┐
│  _BayesDREAMCore       │          │  modalities: Dict[str,     │
│  (core.py)             │          │                 Modality]  │
│                        │          │                            │
│  - set_technical_      │          │  Required: 'cis'           │
│    groups()            │          │  Primary: 'gene' (default) │
│  - fit_ntc()     │          │  Others: 'transcript',     │
│  - fit_cis()           │          │          'splicing_*',     │
│  - fit_trans()         │          │          'atac',           │
│  - save/load methods   │          │          custom, etc.      │
│                        │          │                            │
│  Delegates to:         │          │  All modalities subset to  │
│  - NTCFitter     │          │  cells in 'cis' modality   │
│  - CisFitter           │          │                            │
│  - TransFitter         │          └────────────────────────────┘
│  - ModelSaver          │
│  - ModelLoader         │
└────────────────────────┘
```

## Cis Modality Architecture

bayesDREAM uses a **separate 'cis' modality** for modeling direct perturbation effects:

```
┌─────────────────────────────────────────────────────────────────┐
│              bayesDREAM Initialization (with cis_gene)           │
│                                                                  │
│  Input: gene_counts (92 genes including GFI1B)                  │
│         meta (20,001 cells)                                      │
│         cis_gene='GFI1B'                                        │
└────────────┬────────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────────┐
│           Modality Extraction (during __init__)                  │
│                                                                  │
│  1. Extract 'cis' modality                                      │
│     ├─ Contains: Just GFI1B (1 feature)                        │
│     ├─ Distribution: negbinom                                   │
│     └─ Cells: All 20,001 cells (before filtering)              │
│                                                                  │
│  2. Create 'gene' modality                                      │
│     ├─ Contains: 91 genes (all EXCEPT GFI1B)                   │
│     ├─ Distribution: negbinom                                   │
│     ├─ Filtered: Remove zero-std genes                         │
│     └─ Cells: Subset to match filtered meta                    │
│                                                                  │
│  3. Base class initialization                                   │
│     ├─ self.counts: ORIGINAL 92 genes (for fit_ntc)     │
│     ├─ self.meta: Filtered to ntc + GFI1B cells               │
│     └─ self.cis_gene: 'GFI1B'                                 │
│                                                                  │
│  4. Cell subsetting                                             │
│     └─ All modalities subset to cells present in 'cis'        │
└────────────┬────────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Final Modality Structure                        │
│                                                                  │
│  self.modalities = {                                            │
│    'cis':  Modality(1 feature: GFI1B, 4,281 cells),           │
│    'gene': Modality(91 trans genes, 4,281 cells),              │
│    ...other modalities added by user...                        │
│  }                                                               │
│                                                                  │
│  self.counts:  DataFrame(92 genes × 4,281 cells)               │
│                includes GFI1B for fit_ntc                 │
│                                                                  │
│  self.primary_modality: 'gene'  # Set by modality_name parameter│
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**Why Separate Cis Modality?**
1. **Consistency**: Same interface (`fit_cis()`) works for any modality type (gene, ATAC, etc.)
2. **Clarity**: Explicit separation of cis vs. trans features
3. **Extensibility**: Easy to support new modality types without changing fit_cis logic
4. **Cell Subsetting**: All modalities automatically aligned to cells in 'cis' modality

**Fitting Behavior:**
```
fit_ntc(primary modality):
  ├─ Uses: self.counts (92 genes, includes GFI1B)
  ├─ Fits: All features including cis
  ├─ Extracts: alpha_x_prefit for GFI1B [nsamples, n_groups]
  └─ Stores: alpha_y_prefit for 91 trans genes [nsamples, n_groups, 91]

fit_cis():
  ├─ Uses: 'cis' modality (just GFI1B)
  ├─ Fits: Cis gene expression per guide
  └─ Stores: x_true [nsamples, n_guides]

fit_trans():
  ├─ Uses: 'gene' modality (91 trans genes)
  ├─ Fits: Dose-response f(x_true)
  └─ Stores: posterior_samples_trans
```

**Parameters:**
- `cis_gene`: For gene modality (alias for `cis_feature`)
- `cis_feature`: Generic parameter for any modality type
- Example: `cis_gene='GFI1B'` or `cis_feature='chr9:132283881-132284881'`

**Important Notes:**
- Cis extraction happens **ONLY** during `bayesDREAM()` initialization
- Calling `add_*_modality()` later does **NOT** extract cis (except special ATAC case)
- All modalities are automatically subset to cells in 'cis' modality
- The base class `self.counts` includes the cis feature for technical fitting

## Modality Class Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                         Modality                                 │
├─────────────────────────────────────────────────────────────────┤
│  Attributes:                                                     │
│    - name: str                                                   │
│    - counts: np.ndarray                                         │
│    - feature_meta: pd.DataFrame                                 │
│    - distribution: str (negbinom|multinomial|binomial|...)      │
│    - denominator: np.ndarray (optional, for binomial)           │
│    - dims: Dict (n_features, n_cells, n_categories?, ...)      │
│                                                                  │
│  Methods:                                                        │
│    - to_tensor() → torch.Tensor                                 │
│    - get_feature_subset(indices) → Modality                     │
│    - get_cell_subset(indices) → Modality                        │
└─────────────────────────────────────────────────────────────────┘
                            ↓ created by
    ┌───────────────────────┴───────────────────────────────┐
    │                                                         │
    ↓                                                         ↓
┌─────────────────────┐                    ┌─────────────────────────┐
│  User-created       │                    │  Auto-created via       │
│                     │                    │  helper functions       │
│  Modality(          │                    │                         │
│    name='custom',   │                    │  - add_transcript_      │
│    counts=data,     │                    │    modality()           │
│    feature_meta=... │                    │  - add_splicing_        │
│    distribution=... │                    │    modality()           │
│  )                  │                    │  - create_splicing_     │
└─────────────────────┘                    │    modality()           │
                                            └─────────────────────────┘
```

## Distribution to Data Structure Mapping

```
Distribution Type         Data Shape              Example
─────────────────────────────────────────────────────────────────────
negbinom                  2D: (F, C)             Gene counts
                                                 Transcript counts
                          ┌───┬───┬───┬───┐
                          │ C1│ C2│ C3│...│
                          ├───┼───┼───┼───┤
                        F1│ 5 │ 3 │ 7 │...│
                        F2│ 2 │ 9 │ 1 │...│
                        F3│ 8 │ 0 │ 4 │...│
                          └───┴───┴───┴───┘

multinomial               3D: (F, C, K)          Donor usage
                                                 Isoform usage
                          Cell 1    Cell 2
                          ┌───────┬───────┐
                        F1│[3,2,1]│[4,1,2]│
                        F2│[5,0,3]│[2,3,4]│
                          └───────┴───────┘
                          K=3 categories per feature

binomial                  2D: (F, C) + denom     Exon skipping PSI

                          Inclusion:  Total:
                          ┌───┬───┐  ┌────┬────┐
                        F1│ 5 │ 3 │  │ 10 │  8 │
                        F2│ 2 │ 9 │  │  5 │ 15 │
                          └───┴───┘  └────┴────┘
                          PSI = inclusion / total

normal                    2D: (F, C)             SpliZ scores

                          ┌──────┬──────┬──────┐
                        F1│ 0.5  │-0.2  │ 1.3  │
                        F2│-1.1  │ 0.8  │-0.4  │
                          └──────┴──────┴──────┘
```

## Splicing Data Processing Pipeline

**Note**: Pure Python implementation (no R dependencies as of 2025-10-13)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input: Splice Junction Data                   │
│                                                                  │
│  sj_counts (junctions × cells)   sj_meta (junction metadata)    │
│  ┌───────────────┬─────┐        ┌──────────┬──────┬────────┐  │
│  │  coord.intron │ ... │        │coord.int │chrom │ start  │  │
│  ├───────────────┼─────┤        ├──────────┼──────┼────────┤  │
│  │chr1:100:200:+ │  5  │        │chr1:100..│ chr1 │  100   │  │
│  │chr1:100:300:+ │  3  │        │chr1:100..│ chr1 │  100   │  │
│  └───────────────┴─────┘        └──────────┴──────┴────────┘  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              splicing.py (Pure Python Implementation)            │
│                                                                  │
│  process_sj_counts():                                           │
│    → Raw SJ counts with gene expression denominator            │
│    → Only keeps SJs where start and end are in same gene       │
│    → Returns: 2D binomial modality                             │
│                                                                  │
│  process_donor_usage():                                         │
│    → Groups junctions by donor site (5' splice site)           │
│    → 3D array (donors, cells, acceptors)                       │
│    → metadata: donor coords, list of acceptors                 │
│                                                                  │
│  process_acceptor_usage():                                      │
│    → Groups junctions by acceptor site (3' splice site)        │
│    → 3D array (acceptors, cells, donors)                       │
│    → metadata: acceptor coords, list of donors                 │
│                                                                  │
│  process_exon_skipping():                                       │
│    → Finds cassette exon triplets (inc1, inc2, skip)          │
│    → 2D inclusion + 2D total arrays                            │
│    → metadata: triplet coordinates                             │
│                                                                  │
│  create_splicing_modality():                                    │
│    → High-level function for all splicing types                │
│    → Returns ready-to-use Modality objects                     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Output: Modality Objects                        │
│                                                                  │
│  Modality(name='splicing_sj', distribution='binomial')          │
│  Modality(name='splicing_donor', distribution='multinomial')    │
│  Modality(name='splicing_acceptor', distribution='multinomial') │
│  Modality(name='splicing_exon_skip', distribution='binomial')   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow in bayesDREAM

```
┌─────────────────────────────────────────────────────────────────┐
│                          User Code                               │
│                                                                  │
│  model = bayesDREAM(meta, counts, cis_gene='GFI1B',            │
│                     guide_covariates=['cell_line'])             │
│  model.add_splicing_modality(sj_counts, sj_meta, [...])        │
│  model.add_custom_modality('spliz', scores, meta, 'normal')     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                     bayesDREAM.__init__                          │
│                                                                  │
│  1. Extract 'cis' modality from counts (if cis_gene specified)  │
│  2. Create 'gene' modality from counts (excluding cis gene)     │
│  3. Store both in self.modalities                               │
│  4. Pass ORIGINAL counts (with cis) to super().__init__()       │
│  5. Filter cells based on meta (ntc + cis_gene)                 │
│  6. Subset all modalities to match cells in 'cis' modality      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              self.modalities (runtime state)                     │
│                                                                  │
│  {                                                               │
│    'cis': Modality(negbinom, 1 feature, 500 cells),            │
│    'gene': Modality(negbinom, 999 trans genes, 500 cells),     │
│    'splicing_donor': Modality(multinomial, 50 features, ...),  │
│    'splicing_acceptor': Modality(multinomial, 45 features,...),│
│    'splicing_exon_skip': Modality(binomial, 20 features, ...), │
│    'spliz': Modality(normal, 1000 features, 500 cells),        │
│    'custom_mod': Modality(...),                                 │
│  }                                                               │
│                                                                  │
│  self.primary_modality = 'gene'                                 │
│  self.counts = DataFrame(1000 genes × 500 cells)  # Includes cis│
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Modeling Pipeline                             │
│                                                                  │
│  model.set_technical_groups(['cell_line'])                      │
│    → Required before fit_ntc                              │
│    → Sets technical_group_code for NTC cells                    │
│                                                                  │
│  model.fit_ntc(sum_factor_col='sum_factor',               │
│                      modality_name='gene')                      │
│    → For primary modality: Uses self.counts (includes cis gene) │
│    → Fits _model_ntc (distribution-flexible)              │
│    → Extracts alpha_x_prefit for cis gene [nsamples, n_groups]  │
│    → Stores alpha_y_prefit for trans genes [nsamples, ..., 91]  │
│                                                                  │
│  model.fit_cis(sum_factor_col='sum_factor')                     │
│    → Uses 'cis' modality (just cis gene)                        │
│    → Fits _model_x (cis effects)                                │
│    → Saves self.x_true, self.posterior_samples_cis              │
│                                                                  │
│  model.fit_trans(sum_factor_col='sum_factor_adj',               │
│                  function_type='additive_hill',                 │
│                  modality_name='gene')                          │
│    → Uses 'gene' modality (trans genes only)                    │
│    → Fits _model_y (trans effects as f(x_true))                 │
│    → Distribution-flexible (negbinom, normal, binomial, etc.)   │
│    → Saves self.posterior_samples_trans                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Downstream Analysis                            │
│                                                                  │
│  # Access gene modality (used in modeling)                      │
│  gene_mod = model.get_modality('gene')                          │
│  gene_posteriors = model.posterior_samples_cis                  │
│                                                                  │
│  # Access other modalities (for manual analysis)                │
│  donor_mod = model.get_modality('splicing_donor')               │
│  donor_counts = donor_mod.counts  # 3D array                    │
│  donor_meta = donor_mod.feature_meta                            │
│                                                                  │
│  # Future: fit modality-specific models                         │
│  # model.fit_splicing_trans(modality='splicing_donor', ...)    │
└─────────────────────────────────────────────────────────────────┘
```

## Class Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                    _BayesDREAMCore (base class)                  │
│  Located in: bayesDREAM/core.py                                 │
├─────────────────────────────────────────────────────────────────┤
│  Attributes:                                                     │
│    - meta: pd.DataFrame (cell metadata)                         │
│    - counts: pd.DataFrame (gene counts, features × cells)       │
│    - cis_gene: str                                              │
│    - trans_genes: List[str]                                     │
│    - alpha_y_prefit, alpha_x_prefit: tensors                    │
│    - x_true: posterior cis expression                           │
│    - posterior_samples_cis, posterior_samples_trans: dicts      │
│    - _ntc_fitter, _cis_fitter, _trans_fitter: delegates   │
│    - _saver, _loader: save/load delegates                       │
│                                                                  │
│  Methods:                                                        │
│    - set_technical_groups(covariates)                           │
│    - fit_ntc(sum_factor_col, modality_name, ...)          │
│    - fit_cis(sum_factor_col, ...)                               │
│    - fit_trans(sum_factor_col, function_type, ...)              │
│    - set_alpha_x(), set_alpha_y(), set_x_true()                 │
│    - adjust_ntc_sum_factor(), refit_sumfactor()                 │
│    - permute_genes()                                             │
│    - save_ntc_fit(), load_ntc_fit()                 │
│    - save_cis_fit(), load_cis_fit()                             │
│    - save_trans_fit(), load_trans_fit()                         │
│                                                                  │
│  Internal Models (used by fitters):                             │
│    - _model_ntc(), _model_x(), _model_y()                 │
└────────────────────────────┬────────────────────────────────────┘
                             │ inherits
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              bayesDREAM (main user-facing class)                 │
│  Located in: bayesDREAM/model.py                                │
│  Adds multi-modal support via mixins                            │
├─────────────────────────────────────────────────────────────────┤
│  Additional Attributes:                                          │
│    - modalities: Dict[str, Modality] (includes 'cis', primary)  │
│    - primary_modality: str (set by modality_name parameter)     │
│    - counts_meta: pd.DataFrame (metadata for NTC fit)     │
│                                                                  │
│  Additional Methods (from mixins):                               │
│    - add_modality(name, modality, overwrite)                    │
│    - add_transcript_modality(counts, meta, modality_types)      │
│    - add_splicing_modality(sj_counts, sj_meta, types, ...)     │
│    - add_atac_modality(atac_counts, atac_meta, ...)             │
│    - add_custom_modality(name, counts, feature_meta, dist)      │
│    - get_modality(name) → Modality                              │
│    - list_modalities() → pd.DataFrame                           │
│                                                                  │
│  Inherited Methods (delegated to fitters):                       │
│    - fit_ntc(), fit_cis(), fit_trans() [from Core]        │
│    - All fitting methods accept modality_name parameter         │
│    - fit_cis() always uses 'cis' modality                       │
│    - fit_ntc() on primary uses self.counts (incl. cis)    │
│    - fit_trans() uses primary modality (trans genes only)       │
└─────────────────────────────────────────────────────────────────┘
```

## Distribution-Flexible Fitting (Implemented)

```
Both fit_ntc() and fit_trans() support all distributions:

┌─────────────────────────────────────────────────────────────────┐
│  # Gene counts (negbinom)                                        │
│  model.fit_ntc(covariates=['cell_line'],                  │
│                      sum_factor_col='sum_factor',               │
│                      distribution='negbinom')                    │
│  model.fit_trans(sum_factor_col='sum_factor',                   │
│                  distribution='negbinom',                        │
│                  function_type='additive_hill')                 │
│                                                                  │
│  # Continuous measurements (normal)                             │
│  model.fit_ntc(covariates=['cell_line'],                  │
│                      distribution='normal')                      │
│  model.fit_trans(distribution='normal',                         │
│                  function_type='polynomial')                     │
│                                                                  │
│  # Exon skipping PSI (binomial)                                 │
│  model.fit_trans(distribution='binomial',                       │
│                  denominator=total_counts,                       │
│                  function_type='single_hill')                   │
│                                                                  │
│  # Donor/acceptor usage (multinomial)                           │
│  model.fit_trans(distribution='multinomial',                    │
│                  function_type='additive_hill')                 │
└─────────────────────────────────────────────────────────────────┘

Future:
  - Cross-modality joint models
  - Modality-specific preprocessing and transformations
```

## Module Dependencies

```
model.py (~311 lines)
  ├── pandas, numpy
  ├── core.py (_BayesDREAMCore)
  ├── modality.py (Modality class)
  └── modalities/ (mixin classes)
      ├── transcript.py
      ├── splicing_modality.py
      ├── atac.py
      └── custom.py

core.py (~909 lines)
  ├── pandas, numpy, torch, pyro
  ├── scipy, sklearn
  ├── distributions.py (observation samplers)
  ├── fitting/ (delegated fitters)
  │   ├── technical.py (NTCFitter)
  │   ├── cis.py (CisFitter)
  │   └── trans.py (TransFitter)
  └── io/ (save/load)
      ├── save.py (ModelSaver)
      └── load.py (ModelLoader)

modality.py
  ├── pandas, numpy
  └── torch

distributions.py (observation samplers)
  ├── torch
  └── pyro

splicing.py (Pure Python - no R dependencies)
  ├── pandas, numpy
  └── modality.py

fitting/
  ├── helpers.py (helper functions)
  ├── technical.py (NTCFitter)
  ├── cis.py (CisFitter)
  └── trans.py (TransFitter)

io/
  ├── save.py (ModelSaver)
  └── load.py (ModelLoader)

modalities/
  ├── transcript.py (TranscriptModalityMixin)
  ├── splicing_modality.py (SplicingModalityMixin)
  ├── atac.py (ATACModalityMixin)
  └── custom.py (CustomModalityMixin)

__init__.py
  ├── model.py (bayesDREAM)
  ├── core.py (_BayesDREAMCore)
  ├── modality.py (Modality)
  ├── splicing.py
  ├── distributions.py
  └── modalities/
```
