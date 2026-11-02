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
        base.update({"solver": "ProjADMM", "admm_mode": "optimized",
                     "alpha": 1e-2, "beta": 0.3, "tol": 1e-2, "max_iter": 2})
    elif algorithm == "poly":
        base.update({"admm_mode": "optimized", "deg": 1, "alpha": 1.0,
                     "penalty": "mcp", "mcp_gamma": 3.0, "eta_scale": 1.0,
                     "tol": 1e-2, "max_iter": 2})
    elif algorithm == "gmrf":
        base.update({"admm_mode": "optimized", "alpha": 1.0, "mcp_gamma": 3.0,
                     "mu": 1.0, "tol": 1e-2, "max_iter": 5})
    elif algorithm == "gs":
        base.update({"admm_mode": "optimized", "alpha": 1.0, "mcp_gamma": 3.0,
                     "mu1": 1.0, "mu2": 1.0, "tol": 1e-2, "max_iter": 5,
                     "pg_iters": 2, "q1_iters": 20, "q1_tol": 1e-8})
    elif algorithm == "pgl":
        base.update({"admm_mode": "optimized", "rho": 1e-1, "eta": 1e-1,
                     "delta": 1e-4})
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("name", ["glama", "smooth", "poly", "gmrf"])
def test_runner_can_setup(name):
    args = _make_args(name)
    dataset = load_dataset(args)
    runner = RUNNERS[name](args, dataset, torch.device("cpu"))
    runner.setup()
    assert runner.sim_path != ""


@pytest.mark.skipif(not cpu_lapack_available(), reason="PyTorch build lacks LAPACK")
def test_gs_runner_can_setup():
    args = _make_args("gs")
    dataset = load_dataset(args)
    runner = RUNNERS["gs"](args, dataset, torch.device("cpu"))
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
@pytest.mark.parametrize("name", ["glama", "smooth", "poly", "gmrf", "gs"])
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
@pytest.mark.parametrize("name", ["glama", "smooth", "poly", "pgl", "gmrf", "gs"])
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


# ---------------------------------------------------------------------------
# GMRF runner-specific checks
# ---------------------------------------------------------------------------

def test_firm_threshold_regimes():
    from core.runners import firm_threshold

    alpha, gamma, tau = 0.3, 3.0, 2.0  # gamma * tau = 6 > 1
    # Dead zone |y| <= alpha/tau maps to 0.
    y_dead = torch.tensor([-alpha / tau, 0.5 * alpha / tau])
    assert torch.all(firm_threshold(y_dead, alpha, gamma, tau) == 0)
    # Middle zone shrinks toward zero.
    y_mid = torch.tensor([0.5 * (alpha / tau + gamma * alpha)])
    out_mid = firm_threshold(y_mid, alpha, gamma, tau)
    expected = (y_mid - alpha / tau) / (1 - 1 / (gamma * tau))
    assert torch.allclose(out_mid, expected)
    assert torch.all(out_mid.abs() < y_mid.abs())
    # Zone |y| > gamma*alpha is not penalized (identity).
    y_big = torch.tensor([gamma * alpha + 0.1, -(gamma * alpha + 0.1)])
    assert torch.allclose(firm_threshold(y_big, alpha, gamma, tau), y_big)
    # gamma -> inf reduces to soft thresholding with threshold alpha/tau.
    y = torch.linspace(-1.0, 1.0, 21)
    out = firm_threshold(y, alpha, float("inf"), tau)
    soft = torch.sign(y) * torch.clamp(y.abs() - alpha / tau, min=0.0)
    assert torch.allclose(out, soft, atol=1e-6)


@pytest.mark.skipif(not cpu_lapack_available(), reason="PyTorch build lacks LAPACK")
def test_logdet_prox_optimality():
    torch.manual_seed(0)
    dim, mu = 6, 1.5
    A = torch.randn(dim, dim)
    X = 0.5 * (A + A.T)
    Q = RUNNERS["gmrf"].logdet_prox(X, mu)
    assert torch.linalg.eigvalsh(Q)[0] > 0  # positive definite
    # First-order optimality of min_Q -log det Q + mu/2 ||Q - X||_F^2:
    # grad = -Q^{-1} + mu (Q - X) = 0.
    grad = -torch.linalg.inv(Q) + mu * (Q - X)
    assert torch.linalg.norm(grad) / torch.linalg.norm(Q) < 1e-4


@pytest.mark.skipif(not cpu_lapack_available(), reason="PyTorch build lacks LAPACK")
def test_gmrf_laplacian_feasible_and_objective_decreases():
    args = _make_args("gmrf", dim=8, n_samples=100)
    dataset = load_dataset(args)
    runner = RUNNERS["gmrf"](args, dataset, torch.device("cpu"))
    runner.setup()

    alpha, gamma = runner.alpha, runner.mcp_gamma

    def objective(L):
        off = L * runner.O_mask
        mcp = torch.where(
            off.abs() <= gamma * alpha,
            alpha * off.abs() - off ** 2 / (2 * gamma),
            gamma * alpha ** 2 / 2,
        )
        return (torch.trace(runner.Cov @ L) - torch.logdet(L + runner.J)
                + mcp.sum())

    eye = torch.eye(args.dim)
    L0 = runner.vol / (args.dim * (args.dim - 1)) * (
        args.dim * eye - torch.ones((args.dim, args.dim))
    )

    for i in range(3):
        runner.step(i)
    L = runner.get_graph_matrix()

    # Feasibility: symmetric, zero row sums, non-positive off-diagonals, trace = vol.
    assert torch.allclose(L, L.T, atol=1e-6)
    assert torch.abs(L.sum(dim=1)).max() < 1e-4
    off_diag = L * runner.O_mask
    assert off_diag.max() <= 1e-6
    assert abs(torch.trace(L).item() - runner.vol) / runner.vol < 1e-4

    # The GMRF objective decreases over the first steps.
    assert objective(L) < objective(L0)


@pytest.mark.skipif(not cpu_lapack_available(), reason="PyTorch build lacks LAPACK")
def test_gs_q1_descent_feasibility_and_objective():
    args = _make_args("gs", dim=8, n_samples=100)
    dataset = load_dataset(args)
    runner = RUNNERS["gs"](args, dataset, torch.device("cpu"))
    runner.setup()

    alpha, gamma = runner.alpha, runner.mcp_gamma

    def q1_objective(Q1, L, Q2):
        comm = Q1 @ L - L @ Q1
        return (torch.trace(runner.Cov @ Q1) + 0.5 * runner.mu1 * (comm ** 2).sum()
                + 0.5 * runner.mu2 * ((Q1 - Q2) ** 2).sum())

    def split_objective(Q1, Q2, L):
        comm = Q1 @ L - L @ Q1
        off = L * runner.O_mask
        mcp = torch.where(off.abs() <= gamma * alpha,
                          alpha * off.abs() - off ** 2 / (2 * gamma),
                          gamma * alpha ** 2 / 2)
        return (torch.trace(runner.Cov @ Q1) - torch.logdet(Q2)
                + 0.5 * runner.mu1 * (comm ** 2).sum()
                + 0.5 * runner.mu2 * ((Q1 - Q2) ** 2).sum() + mcp.sum())

    eye = torch.eye(args.dim)
    L0 = runner.graph_learner.L.detach().clone()

    # One Q1 pass from the identity must not increase the Q1 subproblem
    # objective (holding L and the previous Q2 fixed).
    L_init = runner.graph_learner.L.detach().clone()
    Q2_old = runner.Q2.detach().clone()
    q1_before = q1_objective(runner.Q1, L_init, Q2_old)
    runner.step(0)
    q1_after = q1_objective(runner.Q1, L_init, Q2_old)
    assert q1_after <= q1_before + 1e-9

    for i in range(1, 3):
        runner.step(i)
    L = runner.get_graph_matrix()

    # Feasibility of the Laplacian block and the precision blocks.
    assert torch.allclose(L, L.T, atol=1e-6)
    assert torch.abs(L.sum(dim=1)).max() < 1e-4
    assert (L * runner.O_mask).max() <= 1e-6
    assert abs(torch.trace(L).item() - runner.vol) / runner.vol < 1e-4
    assert torch.linalg.eigvalsh(runner.Q1)[0] >= -1e-6
    assert torch.linalg.eigvalsh(runner.Q2)[0] > 0

    # The split objective decreases over the first steps.
    assert split_objective(runner.Q1, runner.Q2, L) < split_objective(
        torch.eye(args.dim), torch.eye(args.dim), L0)
