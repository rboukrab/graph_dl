"""Synthetic ground-truth graph factory.

This module centralises the construction of the synthetic graphs that are used
as ground truth when generating datasets in :mod:`core.datasets`.
"""

import numpy as np
from pygsp.graphs import NNGraph, Graph


def random_geometric_graph(
    dim: int,
    seed: int,
    k: int = 5,
    binary_weights: bool = True,
) -> Graph:
    r"""Build a random geometric nearest-neighbour graph.

    The node coordinates are drawn uniformly in :math:`[0, \sqrt{5})^2` and a
    k-nearest-neighbour graph is built from them. By default the resulting
    adjacency weights are binarised.

    Parameters
    ----------
    dim : int
        Number of nodes (graph dimension).
    seed : int
        Random seed for reproducibility.
    k : int, optional
        Number of nearest neighbours (default 5).
    binary_weights : bool, optional
        If True (default), set all non-zero weights to 1.

    Returns
    -------
    graph : pygsp.graphs.Graph
        The constructed graph with node coordinates attached.
    """
    np.random.seed(seed)
    coord = np.sqrt(5) * np.random.rand(dim, 2)
    graph = NNGraph(coord, k=k)
    if binary_weights:
        graph.W[graph.W != 0] = 1
    return Graph(graph.W, coords=coord)
