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


def rbf_graph(
    dim: int,
    seed: int,
    sigma: float = 0.5,
    threshold: float = 0.75,
) -> Graph:
    r"""Build a random geometric graph with Gaussian RBF edge weights.

    This construction follows the first graph described in the paper: vertex
    coordinates are drawn uniformly at random in the unit square, edge weights
    are given by a Gaussian radial basis function

    .. math::

        W_{ij} = \exp\!\left(-\frac{d(i,j)^2}{2\sigma^2}\right),

    and all edges whose weight is smaller than ``threshold`` are removed.

    Parameters
    ----------
    dim : int
        Number of nodes (graph dimension).
    seed : int
        Random seed for reproducibility.
    sigma : float, optional
        Kernel width parameter of the RBF (default 0.5).
    threshold : float, optional
        Minimum edge weight to keep (default 0.75).

    Returns
    -------
    graph : pygsp.graphs.Graph
        The constructed graph with node coordinates attached.
    """
    np.random.seed(seed)
    coord = np.random.rand(dim, 2)

    # Pairwise Euclidean distances.
    diff = coord[:, np.newaxis, :] - coord[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)

    # Gaussian RBF weights.
    W = np.exp(-(dists ** 2) / (2.0 * sigma ** 2))

    # Remove weak edges and self-loops.
    W[W < threshold] = 0.0
    np.fill_diagonal(W, 0.0)

    # Symmetrise to guard against numerical noise.
    W = (W + W.T) / 2.0

    return Graph(W.astype(float), coords=coord)
