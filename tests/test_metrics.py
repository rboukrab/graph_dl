"""Tests for metric utilities."""

import numpy as np
import torch
from pygsp.graphs import Graph

from joint_ol_gl.metrics import (
    laplacian_to_adj,
    top_k_from_adj,
    evaluate_adjacency,
    evaluate_laplacian,
    compute_scores,
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
