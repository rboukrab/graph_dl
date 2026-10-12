"""Tests for graph learning solvers."""

import pytest
import torch

from core.graph_learn import (
    BCD, GAMA, GLAMA, StationaryADMM, ProjADMM, ProjIPM, ProjActiveSet,
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


def test_proj_ipm_runs_one_iteration():
    dim = 5
    learner = ProjIPM(dim=dim, vol=float(dim), device="cpu")
    F = torch.randn(dim, dim, device="cpu")
    learner.forward(F_t=F, tolerance=1e-3, max_iter=2)
    assert learner.L.shape == (dim, dim)


@pytest.mark.skipif(not cpu_lapack_available(), reason="PyTorch build lacks LAPACK")
def test_proj_active_set_naive_matches_optimized():
    dim = 5
    F = torch.randn(dim, dim, device="cpu")
    F = (F + F.T) / 2
    opt = ProjActiveSet(dim=dim, vol=float(dim), device="cpu")
    opt.forward(F_t=F)
    naive = ProjActiveSet(dim=dim, vol=float(dim), mode="naive", device="cpu")
    naive.forward(F_t=F)
    rel_diff = torch.norm(opt.L - naive.L) / torch.norm(opt.L)
    assert rel_diff < 1e-6
    assert naive.W == opt.W


@pytest.mark.skipif(not cpu_lapack_available(), reason="PyTorch build lacks LAPACK")
def test_proj_ipm_naive_matches_optimized():
    dim = 5
    F = torch.randn(dim, dim, device="cpu")
    F = (F + F.T) / 2
    opt = ProjIPM(dim=dim, vol=float(dim), device="cpu")
    opt.forward(F_t=F, tolerance=1e-6, max_iter=50)
    naive = ProjIPM(dim=dim, vol=float(dim), mode="naive", device="cpu")
    naive.forward(F_t=F, tolerance=1e-6, max_iter=50)
    rel_diff = torch.norm(opt.L - naive.L) / torch.norm(opt.L)
    assert rel_diff < 1e-3


def test_proj_admm2_runs_one_iteration():
    dim = 5
    learner = ProjADMM(dim=dim, mode="optimized", device="cpu")
    F = torch.randn(dim, dim, device="cpu")
    learner.forward(F_t=F, tolerance=1e-4, max_iter=1)
    assert learner.L.shape == (dim, dim)


def test_proj_admm_minres_runs_one_iteration():
    dim = 5
    learner = ProjADMM(dim=dim, mode="minres", device="cpu")
    F = torch.randn(dim, dim, device="cpu")
    learner.forward(F_t=F, tolerance=1e-4, max_iter=1)
    assert learner.L.shape == (dim, dim)
    assert learner.total_inner_iter > 0


def test_proj_admm_minres_matches_optimized():
    # MINRES is matrix-free (no LAPACK needed); the optimized anchor applies
    # the closed-form Schur inverse, so neither path needs a dense solver.
    dim = 5
    F = torch.randn(dim, dim, device="cpu")
    F = (F + F.T) / 2
    opt = ProjADMM(dim=dim, vol=float(dim), device="cpu")
    opt.forward(F_t=F, tolerance=1e-6, max_iter=5000)
    mr = ProjADMM(dim=dim, vol=float(dim), mode="minres", device="cpu")
    mr.forward(F_t=F, tolerance=1e-6, max_iter=5000)
    rel_diff = torch.norm(mr.L - opt.L) / torch.norm(opt.L)
    assert rel_diff < 1e-3


torch_sla = pytest.importorskip("torch_sla", reason="mode='cudss' requires torch-sla")


def test_proj_admm_cudss_runs_one_iteration():
    dim = 5
    learner = ProjADMM(dim=dim, mode="cudss", device="cpu")
    F = torch.randn(dim, dim, device="cpu")
    learner.forward(F_t=F, tolerance=1e-4, max_iter=1)
    assert learner.L.shape == (dim, dim)


def test_proj_admm_cudss_matches_optimized():
    # The sparse-direct mode solves the same KKT system as the dense
    # 'naive' mode (scipy SuperLU on CPU here; cuDSS on CUDA), so it must
    # land on the same projection as ALP up to tolerance.
    dim = 5
    F = torch.randn(dim, dim, device="cpu")
    F = (F + F.T) / 2
    opt = ProjADMM(dim=dim, vol=float(dim), device="cpu")
    opt.forward(F_t=F, tolerance=1e-6, max_iter=5000)
    spd = ProjADMM(dim=dim, vol=float(dim), mode="cudss", device="cpu")
    spd.forward(F_t=F, tolerance=1e-6, max_iter=5000)
    rel_diff = torch.norm(spd.L - opt.L) / torch.norm(opt.L)
    assert rel_diff < 1e-3


def test_laplacian_active_set_runs_and_is_laplacian():
    dim = 5
    learner = ProjActiveSet(dim=dim, vol=float(dim), device="cpu")
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


def _random_laplacian(dim: int, vol: float, seed: int = 0) -> torch.FloatTensor:
    gen = torch.Generator().manual_seed(seed)
    W = torch.triu((torch.rand(dim, dim, generator=gen) < 0.4) * torch.rand(dim, dim, generator=gen), 1)
    W = W + W.T
    L = torch.diag(W.sum(1)) - W
    return L * vol / torch.trace(L)


@pytest.mark.parametrize("dim", [2, 5, 12])
def test_proj_admm_closed_form_schur_inverse(dim):
    import numpy as np
    learner = ProjADMM(dim=dim, vol=float(dim))
    m = dim * (dim + 1) // 2
    # Dense A built analytically (row sums of the vech of L, then the trace), since
    # sparse to_dense() is unreliable on some builds; reference solve with NumPy
    # because CPU LAPACK may be unavailable in torch.
    A = np.zeros((dim + 1, m))
    for i in range(dim):
        for j in range(dim):
            pos = i * dim + j - i * (i + 1) // 2 if j >= i else j * dim + i - j * (j + 1) // 2
            A[i, pos] = 1.0
        A[dim, i * (2 * dim - i + 1) // 2] = 1.0
    rinv = np.zeros(m)
    rinv_c = learner.Rinv.coalesce()
    rinv[rinv_c.indices()[1].numpy()] = rinv_c.values().double().numpy()
    S = (A * rinv) @ A.T
    s = np.random.default_rng(dim).standard_normal(dim + 1)
    t = learner._schur_inv_apply(torch.tensor(s, dtype=torch.float32)).double().numpy()
    assert np.allclose(t, np.linalg.solve(S, s), rtol=1e-4, atol=1e-5)


def test_proj_admm_projects_target_matrix():
    """forward(F_t=G) projects G onto the Laplacians: a valid Laplacian is a fixed point."""
    dim, vol = 8, 8.0
    L0 = _random_laplacian(dim, vol)
    learner = ProjADMM(dim=dim, vol=vol)
    learner.forward(F_t=L0, tolerance=1e-10, max_iter=5000)
    assert torch.allclose(learner.L, L0, atol=1e-4)


def test_firm_threshold_is_mcp_prox():
    from core.runners import PolyRunner
    lam, gamma, tau = 0.7, 3.0, 2.0
    y = torch.linspace(-4, 4, 41, dtype=torch.float64)
    xs = torch.linspace(-6, 6, 120001, dtype=torch.float64)
    ax = xs.abs()
    mcp = torch.where(ax <= gamma * lam, lam * ax - ax ** 2 / (2 * gamma),
                      torch.full_like(xs, gamma * lam ** 2 / 2))
    brute = torch.stack([xs[torch.argmin(mcp + tau / 2 * (xs - yi) ** 2)] for yi in y])
    assert torch.allclose(PolyRunner.firm_threshold(y, lam, gamma, tau), brute, atol=2e-4)
