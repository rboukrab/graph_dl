"""Shared pytest configuration and helpers."""

import torch


def cpu_lapack_available() -> bool:
    """Return True if this PyTorch build supports LAPACK on CPU."""
    try:
        _ = torch.linalg.cholesky(torch.eye(2))
        return True
    except RuntimeError:
        return False
