# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Online graph-learning and joint dictionary-learning/graph-learning algorithms in PyTorch, for experiments on
synthetic GMRF/stationary graph signals (`stat`) and a real fMRI emotion dataset (`fmri`).

## Commands

```bash
pip install -e ".[all]"     # editable install with fmri/plot/dev extras
pip install -e .            # core only
pytest                      # run full test suite
pytest tests/test_graph_learn.py::test_glama_one_iter   # single test
ruff check .                # lint (config in pyproject.toml, line-length 120)
```

Standalone experiment scripts write results to `logs/<experiment-name>/` as `.npy` files and print progress via
`tqdm`. Sweep scripts (`grid_search_gl.py`, `var_n_samples_gl.py`) invoke `run_gl.py` via
`subprocess.run([sys.executable, ...])` (see `core/config.script_cmd`) so they work on Windows too.

Key entry points:
- `run_gl.py --algorithm {glama,smooth,poly,pgl} --data_set {stat,fmri} ...` — unified dispatcher for the four
  graph-learning runners; run `python run_gl.py --help` for algorithm-specific args.
- `joint_dl_gl.py --method {smooth,stationary,heatdiff} --graph_learn {known,bcd,gama,glama,admm}` — joint
  dictionary + graph learning.
- `benchmark_l_update.py` / `benchmark_projection.py` — micro-benchmarks of the `L`-update / Laplacian
  projection solvers (see below).

## Architecture

### `core/graph_learn.py` — graph-learning solvers

All solvers are plain classes (not `nn.Module`s) operating on `torch.float64` tensors, constructed with a
target dimension `dim` and an initial Laplacian `L`:

- `BCD`, `GAMA`, `GLAMA` — sparse-inverse-covariance / Laplacian-constrained ADMM-style learners for GMRF data.
- `StationaryADMM` — ADMM solver for stationary (polynomial spectral) graph learning.
- `HomogeneousSelfDualEmbedding` — self-dual embedding solver, **implemented only for `dim=3`**.
- `ProjADMM`, `ProjIPM`, `ProjActiveSet` — three interchangeable solvers for the same problem (project onto
  the set of valid Laplacians given a target matrix `H_k`), benchmarked against each other in
  `benchmark_projection.py`. Each takes a `mode` kwarg:
  - `mode='optimized'` (default) — sparse/structured linear algebra (duplication matrix, Schur-complement
    updates for `ProjActiveSet`, sparse Newton steps for `ProjIPM`).
  - `mode='naive'` — assembles the dense KKT system directly and solves it with `torch.linalg.solve`. This
    exists because sparse `to_dense()` is unreliable on some PyTorch builds in this environment; dense
    matrices in `mode='naive'` (and in shared helpers like `_dense_equality_matrix`) are built analytically via
    `scatter_` instead of densifying a sparse tensor.
  - `mode='minres'` (`ProjADMM` only) — solves the same KKT system with a matrix-free MINRES Krylov solver
    (`_minres`, adapted from `gpytorch.utils.minres`): O(D^2) per inner iteration via the sparse equality
    matrix and the diagonal `P + rho I`, O(D^2) memory, no dense KKT assembly. Tuning kwargs:
    `minres_tol` (default 1e-6), `minres_max_iter` (default: system size). The benchmark exposes it via
    `--modes minres`; `run_gl.py --admm_mode minres` selects it in the runners.
  - `mode='cudss'` (`ProjADMM` only) — sparse-direct KKT solves via the optional `torch-sla` package: the same
    [[P + rho I, A^T], [A, 0]] system stored as a `SparseTensor` (O(D^2) memory instead of the O(D^4) dense
    KKT), factorized in float64 once and backsolved per ADMM iteration. On CUDA it uses the cuDSS backend
    (`pip install torch-sla[cudss]`; torch-sla 0.3.3 re-factorizes per call — no cached cuDSS factorization
    yet), on CPU the scipy/SuperLU backend via the cached `A.lu()` (whose default COLAMD ordering fills
    heavily on this KKT structure from dim ~100, so the CPU path is for small dims/tests only). The benchmark
    exposes it via `--modes cudss`.
  - `tests/conftest.cpu_lapack_available()` exists because CPU LAPACK (`torch.linalg.cholesky`, used by some
    solver paths) is not guaranteed to be available on every build — tests that need it should skip using this
    helper rather than assuming it works.

### `core/runners.py` — experiment runner framework

`GraphLearningRunner` (ABC) factors out the common experiment loop (iterate, periodically evaluate against
`dataset.graph`, save metrics) shared by all `run_gl.py` algorithms. Each concrete runner
(`GlamaRunner`, `SmoothRunner`, `PolyRunner`, `PglRunner`) implements `add_args`, `setup`, `step`, and
`get_graph_matrix`; `class.name` is used for CLI dispatch in `run_gl.py` and `class.method_name` for the log
directory name via `core/config.gl_experiment_path`.

### Other `core/` modules

| Module | Purpose |
|--------|---------|
| `core/models.py` | Dictionary-learning/network classes: `LapRegSparseNet`, `StationarySparseNet`, `HeatDiffSparseNet`, `ApproximateKSVD`. |
| `core/datasets.py` | PyTorch `Dataset`s: `StatDataset`, `PolyDataset`, `FlankerDataset`, `SmoothDictDataset`, `SmoothDataset`. |
| `core/graphs.py` | Graph generators (`random_geometric_graph`, `rbf_graph`) used to build ground-truth graphs for synthetic datasets. |
| `core/metrics.py` | MSE, cosine similarity, NAE, F1/precision/recall, Laplacian/adjacency evaluation, filter error, `get_object_memory` (used for the `memory_mb` benchmark column). |
| `core/sparse_coding.py` | `_ISTA`, `_AIHT` sparse-coding routines used by the dictionary-learning models. |
| `core/utils.py` | Parameter printing, `bcd_glasso`, `duplication_matrix`, small-world/modularity/matching utilities. |
| `core/config.py` | Log-directory path builders (`gl_experiment_path`, `joint_experiment_path`, `benchmark_experiment_path`) and `script_cmd` for cross-platform subprocess dispatch. |

### Plotting

`plot/*.py` read `.npy`/`.csv` arrays from `logs/` (naming determined by `core/config.py`) and render with
matplotlib using `text.usetex: True` (LaTeX rendering, SVG output) — requires a working LaTeX installation, and
uses the TkAgg backend with `plt.show(block=True)`, so these are interactive and may hang headless.

## Known issues / caveats

- `synth/`, `other/`, and similarly gitignored directories (`data/`, `fmri/`, `img/`, `neuroemo/`) hold
  local/legacy scripts and are not tracked in git; `synth/*.py` in particular references missing classes
  (`SparseNet`, `SynthDataset`, `cpp.bcd`) and is not currently runnable.
- `HomogeneousSelfDualEmbedding` only supports `dim=3`.
- `joint_dl_gl.py`'s `fmri` branch has hard-coded ROI indices for a specific experiment.
