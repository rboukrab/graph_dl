# graph_dl — Online Graph Learning and Joint Dictionary Learning

This repository implements several online graph-learning algorithms and joint dictionary-learning/graph-learning methods in PyTorch. It is designed for experiments on synthetic Gaussian Markov random field (GMRF) / stationary graph signals (`stat`) and on a real fMRI emotion dataset (`fmri`).

All standalone experiment scripts write results to `./logs/<experiment-name>/` as `.npy` files and print progress via `tqdm`.

---

## Table of Contents

- [Installation](#installation)
- [Core package](#core-package)
- [Graph-learning algorithms](#graph-learning-algorithms)
- [Main runnable scripts](#main-runnable-scripts)
- [Hyper-parameter and scalability sweeps](#hyper-parameter-and-scalability-sweeps)
- [Benchmarking](#benchmarking)
- [Plotting scripts](#plotting-scripts)
- [fMRI preprocessing scripts](#fmri-preprocessing-scripts)
- [Tests](#tests)
- [Known issues / caveats](#known-issues--caveats)
- [Quick start examples](#quick-start-examples)

---

## Installation

Create and activate a virtual environment (the repository ships a `.venv`), then install in editable mode:

```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows
# source .venv/bin/activate # Linux/macOS
pip install -e ".[all]"
```

If you only need the core algorithms:

```bash
pip install -e .
```

For fMRI preprocessing support:

```bash
pip install -e ".[fmri]"
```

---

## Core package

Source code lives under the `core/` package:

| Module | Purpose |
|--------|---------|
| `core/graph_learn.py` | Graph-learning solvers: `BCD`, `GAMA`, `GLAMA`, `HomogeneousSelfDualEmbedding`, `StationaryADMM`, `ProjADMM`, `ProjADMM2`. |
| `core/models.py` | Dictionary-learning/network classes: `LapRegSparseNet`, `StationarySparseNet`, `HeatDiffSparseNet`, `ApproximateKSVD`. |
| `core/datasets.py` | PyTorch datasets: `StatDataset`, `PolyDataset`, `FlankerDataset`, `SmoothDataset`. |
| `core/metrics.py` | Evaluation metrics: MSE, cosine similarity, NAE, F1/precision/recall, Laplacian evaluation, filter error. |
| `core/sparse_coding.py` | Sparse coding algorithms: `_ISTA` and `_AIHT`. |
| `core/utils.py` | Helpers: parameter printer, `bcd_glasso`, duplication matrix, graph generators, matching, small-world/modularity utilities. |
| `core/config.py` | Shared experiment-path builders and subprocess helpers. |
| `core/runners.py` | Algorithm runner classes (`GlamaRunner`, `SmoothRunner`, `PolyRunner`, `PglRunner`). |

---

## Graph-learning algorithms

- **BCD** — block coordinate descent for sparse inverse covariance / GMRF graph learning.
- **GAMA** — ADMM-based graphical lasso variant.
- **GLAMA** — Laplacian-constrained ADMM graph learner (used for GMRF / stationary signals).
- **StationaryADMM** — ADMM solver for stationary (polynomial spectral) graph learning.
- **ProjADMM / ProjADMM2** — projection-ADMM Laplacian learners used by `run_gl.py --algorithm smooth` and `run_gl.py --algorithm poly`.
- **HomogeneousSelfDualEmbedding** — self-dual embedding solver. **Note:** currently implemented only for `dim=3`.

---

## Main runnable scripts

### `run_gl.py` — Unified graph-learning dispatcher (recommended)

All four graph-learning algorithms are available through one dispatcher:

```bash
python run_gl.py --algorithm glama --data_set stat --dim 20 --n_samples 1000000 \
  --gamma 1e-3 --delta 1e-1 --rho 1e-3 --n_iters 10000 --seed 1

python run_gl.py --algorithm smooth --data_set stat --dim 20 --n_samples 1000000 \
  --alpha 1e-2 --beta 0.3 --n_iters 10000 --seed 1

python run_gl.py --algorithm poly --data_set stat --dim 20 --n_samples 1000000 \
  --deg 2 --alpha 4 --n_iters 10000 --seed 2

python run_gl.py --algorithm pgl --data_set stat --dim 20 --n_samples 1000000 \
  --rho 0.1 --eta 1e-1 --n_iters 10 --seed 1
```

Common arguments:

- `--algorithm {glama,smooth,poly,pgl}`
- `--data_set {stat,fmri}`
- `--dim`, `--n_samples`, `--n_iters`, `--sim_time`, `--test_interval`, `--seed`, `--device`

Algorithm-specific arguments are added automatically; run `python run_gl.py --help` to see them.

The actual algorithm logic lives in `core/runners.py` as `GlamaRunner`, `SmoothRunner`, `PolyRunner`, and `PglRunner`.

---

### `joint_dl_gl.py` — Joint dictionary learning + graph learning

Trains a dictionary and a graph jointly.

```bash
python joint_dl_gl.py --data_set stat --method smooth --graph_learn glama \
  --dim 100 --n_atoms 200 --n_iters 10000 --batch_size 2500 \
  --alpha 0.001 --gamma 1e-3 --rho 1e-2 --seed 42
```

Supported methods: `smooth`, `stationary`, `heatdiff`.  
Supported graph learners: `known`, `bcd`, `gama`, `glama`, `admm`.

---

## Hyper-parameter and scalability sweeps

| Script | What it does | How to run |
|--------|--------------|------------|
| `grid_search_gl.py` | Sweeps algorithm hyper-parameters (`gamma`/`delta`, `alpha`/`beta`, `deg`/`alpha`, or `rho`/`eta`). | `python grid_search_gl.py --algorithm glama --data_set stat` |
| `var_n_samples_gl_*.py` | Varies number of samples `N` (log-spaced) and reruns each method with many seeds. | `python var_n_samples_gl_glama.py --data_set stat` |

These scripts now use `subprocess.run([sys.executable, ...])` and work on Windows as well as Linux/macOS.

---

## Benchmarking

### `benchmark_l_update.py`

Benchmarks the inner `L`-update step of polynomial graph learning using `ProjADMM2` with fixed known coefficients.

```bash
python benchmark_l_update.py --dim 100 --deg 2 --alpha 4 \
  --n_samples 1000 --n_iters 100 --mode normal
```

---

## Plotting scripts

| Script | Input | Output |
|--------|-------|--------|
| `plot_grid_search.py` | `logs/gl_<method>/...` | Heatmaps of NAE, F-score, sparsity, etc. |
| `plot_learning_curves.py` | `logs/gl_<method>_curve/...` | Metric-vs-time curves comparing methods. |
| `plot_var_n_samples.py` | `logs/var_<method>/...` | Metric-vs-`N` curves. |
| `other/*.py` | Various | Auxiliary figure-generation scripts. |

Plotting scripts use `matplotlib` with LaTeX text rendering (`text.usetex: True`) and require a working LaTeX installation or will fall back with warnings.

---

## fMRI preprocessing scripts

| Script | Purpose |
|--------|---------|
| `fmri/compute_zmaps.py` | First-level GLM / z-map computation on fMRIprep-preprocessed BOLD data. |
| `fmri/plot_difumo_atlas.py` | Visualization of the DiFuMo atlas. |
| `data/fmri_prep*.sh`, `data/extract_rois.sh` | Shell helpers for fMRIprep and ROI extraction. |

---

## Tests

Run the test suite with pytest:

```bash
pytest
```

The test suite covers:

- Module imports
- Dataset instantiation
- One-iteration smoke tests for `BCD`, `GAMA`, `GLAMA`, `StationaryADMM`, `ProjADMM`
- Metric utilities
- Sparse coding helpers

---

## Known issues / caveats

1. **`synth/` legacy scripts** (`synth/dl/*.py`, `synth/gl/*.py`) reference missing classes such as `SparseNet`, `SynthDataset`, and `cpp.bcd`. They are not currently runnable and are preserved as reference/legacy code.
2. **`HomogeneousSelfDualEmbedding`** is implemented only for `dim=3`; other sizes raise `NotImplementedError`.
3. **`ProjADMM`** only supports `mode='optimized'`; the broken `mode='normal'` branch was removed.
4. Plotting scripts use the TkAgg backend and `plt.show(block=True)`; they are interactive and may hang in headless environments.
5. The `joint_dl_gl.py` `fmri` branch contains hard-coded ROI indices for a specific experiment.

---

## Quick start examples

**Graph learning on synthetic GMRF signals:**

```bash
python run_gl.py --algorithm glama --data_set stat --dim 20 --n_samples 100000 \
  --gamma 1e-3 --delta 1e-1 --n_iters 1000
```

**Smoothness-prior graph learning on fMRI:**

```bash
python run_gl.py --algorithm smooth --data_set fmri --alpha 1e-2 --beta 0.3 --sim_time 30
```

**Polynomial graph learning:**

```bash
python run_gl.py --algorithm poly --data_set stat --dim 20 --deg 2 --alpha 4 --n_iters 1000
```

**Benchmark the L-update:**

```bash
python benchmark_l_update.py --dim 100 --deg 2 --n_iters 100
```

**Run tests:**

```bash
pytest
```
