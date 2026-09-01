"""Tests for dataset classes."""

import torch

from core.datasets import StatDataset, FlankerDataset, NoisyLaplacianDataset


def test_stat_dataset_shape():
    dim = 10
    n_samples = 50
    dataset = StatDataset(seed=0, dim=dim, n=n_samples)
    assert len(dataset) == n_samples
    sample = dataset[0]
    assert sample.shape == (dim,)
    assert dataset.X.shape == (n_samples, dim)
    assert dataset.graph is not None


def test_stat_dataset_poly_degree():
    dataset = StatDataset(seed=0, dim=8, n=20, deg=2)
    assert dataset.h.shape[0] == 3  # deg + 1


def test_flanker_dataset_loads():
    dataset = FlankerDataset()
    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape[0] == dataset.dim


def test_noisy_laplacian_dataset_shape():
    dim = 10
    dataset = NoisyLaplacianDataset(seed=0, dim=dim, noise_level=0.1)
    assert dataset.L.shape == (dim, dim)
    assert dataset.H.shape == (dim, dim)
    assert dataset.vol > 0
    assert len(dataset) == dim


def test_noisy_laplacian_dataset_noise():
    dim = 20
    dataset_clean = NoisyLaplacianDataset(seed=1, dim=dim, noise_level=0.0)
    dataset_noisy = NoisyLaplacianDataset(seed=1, dim=dim, noise_level=0.5)
    assert torch.allclose(dataset_clean.H, dataset_clean.L)
    assert not torch.allclose(dataset_noisy.H, dataset_noisy.L)
    noise_norm = torch.norm(dataset_noisy.H - dataset_noisy.L)
    lap_norm = torch.norm(dataset_noisy.L)
    assert noise_norm > 0.4 * lap_norm  # noise_level=0.5 -> ~0.5 relative norm
