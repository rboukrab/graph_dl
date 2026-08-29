"""Tests for dataset classes."""

import torch

from graph_dl.datasets import StatDataset, FlankerDataset


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
