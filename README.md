# Efficient SMV-SVM Reproduction

This repository contains a notebook-based reproduction of key experiments from the paper:
`Efficient implementation techniques of an SVM-based speech/music classifier in SMV`
(DOI: `10.1007/s11042-014-1859-8`).

The implementation focuses on:
- baseline SVM speech/music classification
- filtering, skipping, and combined compute-reduction mechanisms
- parameter search under an accuracy-degradation constraint
- extended analyses aligned with sections 5.4, 5.5, 5.6, and 5.7

The current notebook code has also been refactored for readability and explainability, including:
- named constants instead of magic numbers
- shared normalization and plotting helpers
- section runner functions
- smaller helper functions for search logic
- consistent variable naming and intent docstrings

## Repository Layout

- `01_smv_svm_paper20m_vs_all_dataset_protocols.ipynb`
  - earlier pipeline notebook used for baseline/cache workflow
- `02_smv_svm_extended_reproduction_sections_5_4_to_5_7.ipynb`
  - main extended notebook with section 5.4 to 5.7 analyses
- `smv_svm_lecture_fa.md`
  - Persian lecture/report material
- `musan/`
  - extracted MUSAN dataset directory (required)
- `musan.tar.gz`
  - MUSAN archive
- `artifacts/full_feature_cache/`
  - cached frame-level features used by notebook execution

## Problem Framing

The target task is frame-level speech/music classification in an SMV-like setting with a hard efficiency constraint.

Using SVM on every frame can provide strong classification performance but can be expensive. The notebook reproduces the paper’s strategy for reducing SVM calls while preserving performance:
- `Filtering`: deterministic rule-based routing of likely music frames
- `Skipping`: reuse recent decision history to avoid repeated SVM calls
- `Combined`: apply both mechanisms to increase compute savings

## Data and Protocol

Because the exact paper dataset/toolchain is not fully available in this environment, the notebook uses reproducible proxies:
- MUSAN for speech/music/noise source material
- cached frame-level features loaded from `artifacts/full_feature_cache`
- synthetic paper-style test streams with alternating speech/music/silence segments
- mode set: `['varied', 2, 5, 10]` with 10 files per mode

Core labels:
- speech: `+1`
- music: `-1`
- silence/noise placeholder: `0`

Frame settings:
- frame duration: `0.02s`
- train corpus target: `20 minutes` per class

## Pipeline Overview

The main notebook (`02_...ipynb`) follows this flow:

1. Configuration and constants
2. Data index creation from cached feature files
3. Train/test split
4. Paper-style corpus and synthetic test-file construction
5. SVM training from scratch (`TorchRBFSVM`)
6. Baseline scoring and threshold calibration
7. Mechanism implementations:
   - filtering
   - skipping by prior labels
   - skipping by prior SVM outputs
   - combined logic with order variants
8. Evaluation metrics and normalization against baseline
9. Parameter search with `<= 2%` degradation rule
10. Section runners for 5.5, 5.6, and 5.7
11. Tables/plots aligned with paper structure

## Key Implemented Components

### Classifiers

- `TorchRBFSVM`
  - custom RBF-kernel SVM optimization in PyTorch
- `DiagGMM`
  - diagonal-covariance GMM implemented from scratch (NumPy)
- `WeightedTorchRBFSVM`
  - weighted feature transform + custom SVM backend

### Shared Helpers

- `normalize_features(raw)`
  - single normalization entry point using train-set `feat_mean` / `feat_std`
- plotting helpers:
  - `plot_norm_bars(...)`
  - `plot_norm_bars_and_tradeoff(...)`
  - `plot_norm_line_compare(...)`
- constantized metric columns:
  - `NORM_METRICS = ['overall_norm', 'speech_norm', 'music_norm']`

### Search and Section Runners

Parameter search is structured into small helpers for readability:
- `_evaluate_filter_candidates(...)`
- `_evaluate_skip_candidates(...)`
- `_evaluate_combined_candidates(...)`
- `_choose_best_candidate(...)`
- `search_params_under_constraint(...)`

Section-level orchestration:
- `run_section55(...)`
- `run_section56(...)`
- `run_section57(...)`

## Outputs by Section

### Section 5.4 Analog

Produces comparisons for mechanism order and parameter effects:
- table-2-like comparison (`F1->S1` vs `S1->F1`)
- table-3-like comparison (`F1->S1` vs `F2->S1`)
- table-4-like comparison (`F2->S1` vs `F2->S2`)

### Section 5.5 Analog

Parameter selection under the `<= 2%` constraint:
- table-5-like summary: selected parameter sets
- table-6-like summary: normalized performance + proxy execution/energy

### Section 5.6 Analog

Generalization to additional classifier families:
- GMM and WSVM baselines/caches
- table-7-like selected parameters by classifier
- table-8-like effectiveness summary by classifier + mechanism

### Section 5.7 Analog

Evaluation on a degraded proxy dataset:
- deterministic feature-space degradation function
- table-9-like selected parameters on degraded data
- table-10-like normalized effectiveness and compute proxies
- line comparison between original and degraded setups

## Environment and Dependencies

Recommended runtime:
- Python 3.10+
- Jupyter Notebook / JupyterLab

Main Python packages:
- `numpy`
- `pandas`
- `torch`
- `matplotlib`
- `IPython`

Example install:

```bash
pip install numpy pandas torch matplotlib ipython jupyter
```

## How to Run

1. Ensure dataset and cache paths exist:
   - `musan/`
   - `artifacts/full_feature_cache/`
2. Open Jupyter in repository root.
3. Run:
   - `01_smv_svm_paper20m_vs_all_dataset_protocols.ipynb` first if cache/baseline artifacts need regeneration.
   - `02_smv_svm_extended_reproduction_sections_5_4_to_5_7.ipynb` for extended section 5.4 to 5.7 results.
4. Execute cells in order from top to bottom.

## Reproducibility Notes

- deterministic seeds are set for `random`, `numpy`, and `torch`
- dataset substitution and feature-space proxies are explicit
- timing/energy values are reported as reproducible proxies, not hardware-exact SMV measurements

## Limitations

The following paper conditions are approximated in this repository:
- exact original speech/music dataset composition
- internal fixed-point SMV feature path
- hardware-level timing/energy simulation environment
- original transfer/degradation data channel

Despite these constraints, the notebook preserves the algorithmic logic and comparison structure of the target sections.

## Quick Troubleshooting

- Error: `musan not found`
  - extract or place MUSAN under `musan/`
- Error: `artifacts/full_feature_cache not found`
  - run upstream cache-generation workflow first (starting from notebook `01_...`)
- Very long runtime
  - reduce candidate-grid sizes in section configs:
    - `SEC55_CFG`
    - `SEC56_CFG`
    - `SEC57_CFG`

## Reference

- `Efficient implementation techniques of an SVM-based speech/music classifier in SMV`
  - DOI: `10.1007/s11042-014-1859-8`
