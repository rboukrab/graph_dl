"""Tests for the graph-learning runners."""

from types import SimpleNamespace

import pytest
import torch

from core.datasets import FlankerDataset, StatDataset
from core.runners import RUNNERS, load_dataset
from tests.conftest import cpu_lapack_available


def _make_args(algorithm: str, data_set: str = "stat", dim: int = 6,
               n_samples: int = 50, **overrides):
    base = {
        "algorithm": algorithm,
        "data_set": data_set,
        "device": "cpu",
        "n_samples": n_samples,
        "n_iters": 4,
        "sim_time": 9999999.0,
        "test_interval": 2,
        "dim": dim,
        "seed": 0,
        "metric_mode": "binary",
    }
    if algorithm == "glama":
        base.update({"rho": 1e-3, "gamma": 1e-3, "delta": 1e-1})
    elif algorithm == "smooth":
        base.update({"solver": "projadmm2", "admm_mode": "optimized",
                     "alpha": 1e-2, "beta": 0.3, "tol": 1e-2, "max_iter": 2})
    elif algorithm == "poly":
        base.update({"admm_mode": "optimized", "deg": 1, "alpha": 1.0,
                     "tol": 1e-2, "max_iter": 2})
    elif algorithm == "pgl":
        base.update({"admm_mode": "optimized", "rho": 1e-1, "eta": 1e-1,
                     "delta": 1e-4})
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("name", ["glama", "smooth", "poly"])
def test_runner_can_setup(name):
    args = _make_args(name)
    dataset = load_dataset(args)
    runner = RUNNERS[name](args, dataset, torch.device("cpu"))
    runner.setup()
    assert runner.sim_path != ""


def test_pgl_runner_can_setup():
    args = _make_args("pgl")
    try:
        dataset = load_dataset(args)
    except FileNotFoundError:
        pytest.skip("PolyDataset files not available for test dimensions")
    runner = RUNNERS["pgl"](args, dataset, torch.device("cpu"))
    runner.setup()
    assert runner.sim_path != ""


@pytest.mark.skipif(not cpu_lapack_available(), reason="PyTorch build lacks LAPACK")
@pytest.mark.parametrize("name", ["glama", "smooth", "poly"])
def test_runner_standard_loop_smoke(name):
    args = _make_args(name)
    dataset = load_dataset(args)
    runner = RUNNERS[name](args, dataset, torch.device("cpu"))
    runner.run()
    assert "accum_time" in runner.metrics
    assert "sp" in runner.metrics


@pytest.mark.skipif(not cpu_lapack_available(), reason="PyTorch build lacks LAPACK")
def test_pgl_custom_loop_smoke():
    args = _make_args("pgl", n_iters=2)
    try:
        dataset = load_dataset(args)
    except FileNotFoundError:
        pytest.skip("PolyDataset files not available for test dimensions")
    runner = RUNNERS["pgl"](args, dataset, torch.device("cpu"))
    runner.run()
    assert "accum_time" in runner.metrics
    assert "sp" in runner.metrics
    assert "commerr" in runner.metrics


@pytest.mark.skipif(not cpu_lapack_available(), reason="PyTorch build lacks LAPACK")
@pytest.mark.parametrize("name", ["glama", "smooth", "poly", "pgl"])
def test_runner_get_graph_matrix(name):
    args = _make_args(name)
    try:
        dataset = load_dataset(args)
    except FileNotFoundError:
        pytest.skip("Dataset files not available for test dimensions")
    runner = RUNNERS[name](args, dataset, torch.device("cpu"))
    runner.setup()
    runner.step(0)
    matrix = runner.get_graph_matrix()
    assert matrix.shape == (args.dim, args.dim)


def test_load_dataset_stat():
    args = SimpleNamespace(algorithm="glama", data_set="stat", dim=8, n_samples=30, seed=0)
    dataset = load_dataset(args)
    assert hasattr(dataset, "graph")


def test_load_dataset_fmri():
    args = SimpleNamespace(algorithm="glama", data_set="fmri", dim=8, n_samples=30, seed=0)
    dataset = load_dataset(args)
    assert isinstance(dataset, FlankerDataset)
