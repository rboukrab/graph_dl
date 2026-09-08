"""Tests for metric utilities."""

import numpy as np
import pytest
import torch
from pygsp.graphs import Graph

from core.metrics import (
    laplacian_to_adj,
    top_k_from_adj,
    evaluate_adjacency,
    evaluate_adjacency_weighted,
    evaluate_laplacian,
    evaluate_laplacian_weighted,
    compute_scores,
    get_object_memory,
)


def test_laplacian_to_adj():
    L = torch.tensor([[2.0, -1.0, -1.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    A = laplacian_to_adj(L)
    assert A.shape == (3, 3)
    assert A[0, 1] > 0
    assert A[0, 2] > 0
    assert A[1, 2] == 0
    np.testing.assert_allclose(A, A.T, atol=1e-10)


def test_top_k_from_adj():
    A = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]])
    pred = top_k_from_adj(A, k=2)
    assert pred[0, 2] == 1
    assert pred[0, 1] == 0
    assert pred[1, 2] == 1


def test_evaluate_adjacency():
    W = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    graph = Graph(W.astype(float))
    metrics = evaluate_adjacency(graph, torch.from_numpy(W).float())
    assert "f1_score" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert metrics["f1_score"] == 1.0


def test_evaluate_laplacian():
    W = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    graph = Graph(W.astype(float))
    L = torch.tensor([[2.0, -1.0, -1.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    metrics = evaluate_laplacian(graph, L)
    assert metrics["f1_score"] == 1.0


def test_compute_scores():
    W_true = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    graph = Graph(W_true.astype(float))
    W_pred = torch.tensor([[0.0, 0.6, 0.6], [0.6, 0.0, 0.0], [0.6, 0.0, 0.0]])
    tp, fp, tn, fn = compute_scores(W_pred, graph, thresh=0.5)
    assert tp == 2.0
    assert fp == 0.0
    assert fn == 0.0
    assert tn == 1.0


def test_evaluate_adjacency_weighted_perfect():
    W = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    graph = Graph(W.astype(float))
    metrics = evaluate_adjacency_weighted(graph, torch.from_numpy(W).float())
    assert metrics["f1_score"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_evaluate_adjacency_weighted_partial():
    W_true = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    graph = Graph(W_true.astype(float))
    # Predict half the true-edge weight on true edges, plus a small false edge.
    W_pred = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.1], [0.5, 0.1, 0.0]])
    metrics = evaluate_adjacency_weighted(graph, torch.from_numpy(W_pred).float())
    # TP = 0.5 + 0.5 = 1.0, FP = 0.1, FN = (1-0.5) + (1-0.5) = 1.0
    np.testing.assert_allclose(metrics["precision"], 1.0 / 1.1, rtol=1e-6)
    np.testing.assert_allclose(metrics["recall"], 1.0 / 2.0, rtol=1e-6)
    assert 0.0 < metrics["f1_score"] < 1.0


def test_get_object_memory():
    class Foo:
        def __init__(self):
            self.a = torch.zeros(100, 100)            # 40000 B
            self.b = np.zeros((50, 50))               # 20000 B
            self.c = torch.sparse_coo_tensor(         # values + indices
                indices=torch.tensor([[0, 1], [1, 0]]),
                values=torch.zeros(2),
            )
            self.d = 3.14                             # not a tensor: 0 B

    expected = (100 * 100 * 4 + 50 * 50 * 8 + 2 * 4 + 4 * 8) / (1024.0 ** 2)
    assert get_object_memory(Foo()) == pytest.approx(expected, rel=1e-9)


def test_evaluate_laplacian_weighted():
    W_true = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    graph = Graph(W_true.astype(float))
    L = torch.tensor([[2.0, -1.0, -1.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    metrics = evaluate_laplacian_weighted(graph, L)
    assert metrics["f1_score"] == 1.0
