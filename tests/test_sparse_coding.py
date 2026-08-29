"""Tests for sparse coding helpers."""

import pytest
import torch

from graph_dl.sparse_coding import _ISTA, _AIHT
from tests.conftest import cpu_lapack_available


@pytest.mark.skipif(not cpu_lapack_available(), reason="PyTorch build lacks LAPACK")
def test_ista_shape():
    n, m, k = 10, 6, 4
    D = torch.randn(n, k)
    x = torch.randn(m, n)
    y = _ISTA(x, D, alpha=0.01, tolerance=1e-3, max_iter=10, device="cpu")
    assert y.shape == (m, k)


def test_aiht_shape():
    n, m, k = 10, 6, 4
    D = torch.randn(n, k)
    x = torch.randn(m, n)
    y, _ = _AIHT(x, D, m=k, M=2, thresh=1e-3, device="cpu")
    assert y.shape == (m, k)
    assert (y != 0).sum(dim=1).max() <= 2
