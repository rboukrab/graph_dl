"""Tests for graph learning solvers."""

import pytest
import torch

from graph_dl.graph_learn import (
    BCD, GAMA, GLAMA, StationaryADMM, ProjADMM2, LaplacianIPM, LaplacianActiveSet,
)
from tests.conftest import cpu_lapack_available


def _random_cov(dim: int, device: str = "cpu") -> torch.FloatTensor:
    X = torch.randn(50, dim, device=device)
    return X.T @ X / X.shape[0]


@pytest.mark.skipif(not cpu_lapack_available(), reason="PyTorch build lacks LAPACK")
def test_bcd_runs_one_iteration():
    dim = 5
    S = _random_cov(dim)
    learner = BCD(cov_matrix=S, device="cpu")
    learner.forward(S_d=S, tolerance=1e-4, max_iter=1)
    assert learner.Phi.shape == (dim, dim)


@pytest.mark.skipif(not cpu_lapack_available(), reason="PyTorch build lacks LAPACK")
def test_gama_runs_one_iteration():
    dim = 5
    S = _random_cov(dim)
    learner = GAMA(cov_matrix=S, gamma=1e-3, rho=1e-3, device="cpu")
    learner.forward(S_d=S, tolerance=1e-4, max_iter=1)
    assert learner.Phi.shape == (dim, dim)


@pytest.mark.skipif(not cpu_lapack_available(), reason="PyTorch build lacks LAPACK")
def test_glama_runs_one_iteration():
    dim = 5
    S = _random_cov(dim)
    learner = GLAMA(cov_matrix=S, gamma=1e-3, rho=1e-3, eta=1e-3, device="cpu")
    learner.forward(S_d=S, tolerance=1e-4, max_iter=1)
    assert learner.L.shape == (dim, dim)


def test_stationary_admm_runs_one_iteration():
    dim = 5
    learner = StationaryADMM(dim=dim, mode="optimized", device="cpu")
    H = torch.randn(dim, dim, device="cpu")
    learner.forward(b_t=1.0, F_t=H, tolerance=1e-4, max_iter=1)
    assert learner.L.shape == (dim, dim)


def test_proj_admm2_runs_one_iteration():
    dim = 5
    learner = ProjADMM2(dim=dim, mode="optimized", device="cpu")
    F = torch.randn(dim, dim, device="cpu")
    learner.forward(F_t=F, tolerance=1e-4, max_iter=1)
    assert learner.L.shape == (dim, dim)


def test_laplacian_ipm_runs_and_is_laplacian():
    dim = 5
    learner = LaplacianIPM(dim=dim, vol=float(dim), device="cpu")
    F = torch.randn(dim, dim, device="cpu")
    F = (F + F.T) / 2
    learner.forward(F_t=F, tolerance=1e-3, max_iter=20)
    L = learner.L
    assert L.shape == (dim, dim)
    # Symmetry
    assert torch.allclose(L, L.T, atol=1e-5)
    # Row sums approximately zero
    assert torch.abs(L.sum(dim=1)).max().item() < 1e-4
    # Trace equals volume
    assert abs(L.trace().item() - dim) < 1e-4
    # Off-diagonals non-positive
    off_diag = L[torch.eye(dim, device=L.device) == 0]
    assert (off_diag <= 1e-6).all()


def test_laplacian_active_set_runs_and_matches_ipm():
    dim = 5
    learner = LaplacianActiveSet(dim=dim, vol=float(dim), device="cpu")
    F = torch.randn(dim, dim, device="cpu")
    F = (F + F.T) / 2
    learner.forward(F_t=F)
    L = learner.L
    assert L.shape == (dim, dim)
    # Symmetry
    assert torch.allclose(L, L.T, atol=1e-5)
    # Row sums approximately zero
    assert torch.abs(L.sum(dim=1)).max().item() < 1e-4
    # Trace equals volume
    assert abs(L.trace().item() - dim) < 1e-4
    # Off-diagonals non-positive
    off_diag = L[torch.eye(dim, device=L.device) == 0]
    assert (off_diag <= 1e-6).all()

    # Compare with IPM
    learner_ipm = LaplacianIPM(dim=dim, vol=float(dim), device="cpu")
    learner_ipm.forward(F_t=F, tolerance=1e-6, max_iter=20)
    rel_diff = torch.norm(L - learner_ipm.L) / torch.norm(learner_ipm.L)
    assert rel_diff < 1e-3
