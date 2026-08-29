import os
import torch
import numpy as np
import pandas as pd
from os.path import join as pjoin
from scipy.io import loadmat
from pygsp.graphs import Graph
from pygsp.filters import Filter

from core.graphs import random_geometric_graph, rbf_graph
from matplotlib import pyplot as plt

class StatDataset(torch.utils.data.Dataset):

    def __init__(self, seed, dim, n, deg=3):
        super(StatDataset).__init__()

        # Random number generator seed
        np.random.seed(seed)

        # Generate graph
        graph = random_geometric_graph(dim, seed, k=5)

        h = np.random.rand(deg+1)
        h[0] = 1.

        def graph_kernel(e):
            poly = h[0]
            for p in range(1, 1 + deg):
                poly += h[p] * (e ** p)
            return 1. / poly

        graph_filter = Filter(graph, graph_kernel)
        H = graph_filter.compute_frame()
        X = np.random.randn(n, dim) @ H
        # X /= np.linalg.norm(X, axis=1).mean()

        # W = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), n)
        # W *= (10 ** (- snr / 20)) / np.linalg.norm(W, axis=1).mean()
        # X += W

        self.graph = graph
        self.X = torch.from_numpy(X).float()
        self.h = torch.from_numpy(h).float()

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.X.shape[0]

class PolyDataset(torch.utils.data.Dataset):

    def __init__(self, gnum, gtype, dim, n):
        super(PolyDataset).__init__()

        data_path = os.path.join("data", "main_exp1_data")
        graph_dirs = os.listdir(data_path)

        not_found = True
        for graph_dir in graph_dirs:
            if 'G%d' % gnum == graph_dir:
                graph_type_dirs = os.listdir(os.path.join(data_path, graph_dir))
                for graph_type_dir in graph_type_dirs:
                    if gtype == graph_type_dir:
                        dim_dirs = os.listdir(os.path.join(data_path, graph_dir, graph_type_dir))
                        for dim_dir in dim_dirs:
                            if 'D%d' % dim == dim_dir:
                                S0 = loadmat(os.path.join(data_path, graph_dir, graph_type_dir, dim_dir, 'S0.mat'))['S']
                                n_dirs = os.listdir(os.path.join(data_path, graph_dir, graph_type_dir, dim_dir))
                                for n_dir in n_dirs:
                                    if 'N%.0E' % n == n_dir:
                                        not_found = False
                                        found_path = os.path.join(data_path, graph_dir, graph_type_dir, dim_dir, n_dir)
                                        C = torch.from_numpy(loadmat(os.path.join(found_path, 'C.mat'))['C']).float()
                                        X = torch.from_numpy(loadmat(os.path.join(found_path, 'X.mat'))['X']).float()
                                        h = torch.from_numpy(loadmat(os.path.join(found_path, 'h.mat'))['h']).float()

        if not_found:
            raise FileNotFoundError('The requested dataset was not found.')

        self.graph = Graph(S0.astype(float))
        self.C = C
        self.X = X.T
        self.h = h

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.X.shape[0]

class FlankerDataset(torch.utils.data.Dataset):

    def __init__(self):

        super(FlankerDataset).__init__()

        roi_labels = np.load(pjoin("neuroemo", 'roi_labels.npy'))
        rois_df = pd.read_csv(pjoin("neuroemo", "labels_128_dictionary.csv"))
        roi_names = []
        for roi_label in roi_labels:
            roi_names.append(rois_df.iloc[roi_label-1]['Difumo_names'])
        self.roi_names = roi_names

        labels = np.load(pjoin("neuroemo", 'network_labels.npy'))
        unique_labels = np.unique(labels)
        labels_idx = []
        for label in unique_labels:
            labels_idx.append(np.where(labels == label)[0])
        self.labels = labels_idx

        self.vols = np.load(pjoin("neuroemo", 'vols.npy'))
        self.vols[self.vols > 0] = 100

        X = np.load(pjoin("neuroemo", 'X.npy'))
        X -= X.mean(1).reshape(-1, 1)
        X /= np.linalg.norm(X, axis=1).reshape(-1 ,1)

        self.dim = X.shape[1]
        self.n_signals = X.shape[0]

        # Convert to PyTorch tensor
        self.X = torch.from_numpy(X).float()

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.X.shape[0]

class SmoothDictDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            graph,
            n_samples,
            n_atoms,
            sparsity,
            seed,
            snr
    ):
        super(SmoothDictDataset).__init__()
        self.n_samples = n_samples
        self.n_atoms = n_atoms
        self.sparsity = sparsity
        self.seed = seed
        self.graph = graph
        self.snr = snr

        L = graph.L.todense().__array__()
        dL = np.random.rand(graph.N, graph.N)
        dL *= 10 ** (-snr / 20) * np.linalg.norm(L) / np.linalg.norm(dL)
        L += dL
        Sigma_x = np.linalg.pinv(L)

        D0 = np.random.randn(graph.N, n_atoms)
        D0 = np.linalg.inv(np.eye(graph.N) + L) @ D0
        D0 *= np.sqrt(n_atoms) / np.linalg.norm(D0)
        D0 = torch.from_numpy(D0).float()

        # Compute original dictionary
        Y = torch.zeros((n_samples, n_atoms)).scatter_(
            1, torch.randint(0, n_atoms, (n_samples, sparsity)), torch.randn((n_samples, sparsity))
        )
        self.X = Y @ D0.T
        self.X /= torch.linalg.norm(self.X, dim=1).unsqueeze(1)

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.n_samples


class SmoothDataset(torch.utils.data.Dataset):
    r"""Dataset of smooth graph signals.

    Generates a synthetic random geometric graph with Gaussian RBF edge weights
    (as described in the paper), computes its combinatorial Laplacian ``L``,
    and draws samples from

        x ~ N(ux, L^\dagger + sigma2 * I_n)

    where ``L^\dagger`` is the Laplacian pseudoinverse and ``sigma2`` is the
    noise power.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    dim : int
        Number of graph nodes.
    n : int
        Number of signal samples.
    k : int, optional
        Kept for API compatibility; ignored by the RBF graph construction
        (default 5).
    sigma2 : float, optional
        Noise power added to the diagonal of the covariance (default 0.0).
    ux : array-like or None, optional
        Mean vector. If None, a zero-mean vector is used.
    """

    def __init__(
        self,
        seed: int,
        dim: int,
        n: int,
        k: int = 5,
        sigma2: float = 0.0,
        ux=None,
    ):
        super(SmoothDataset).__init__()

        np.random.seed(seed)

        # Random geometric graph with Gaussian RBF edge weights.
        graph = rbf_graph(dim, seed)
        graph.compute_laplacian()

        L = np.asarray(graph.L.todense() if hasattr(graph.L, "todense") else graph.L)
        L = L.astype(float)

        # Laplacian pseudoinverse via eigen-decomposition for numerical stability.
        eigvals, eigvecs = np.linalg.eigh(L)
        inv_eigvals = np.where(np.abs(eigvals) > 1e-10, 1.0 / eigvals, 0.0)
        L_pinv = eigvecs @ np.diag(inv_eigvals) @ eigvecs.T

        # Covariance: L^\dagger + sigma2 * I.
        cov = L_pinv + float(sigma2) * np.eye(dim)
        cov = (cov + cov.T) / 2.0  # enforce exact symmetry

        # Mean vector.
        if ux is None:
            mean = np.zeros(dim)
        else:
            mean = np.asarray(ux, dtype=float)
            if mean.shape != (dim,):
                raise ValueError(f"ux must have shape ({dim},), got {mean.shape}")

        X = np.random.multivariate_normal(mean, cov, size=n)

        self.graph = graph
        self.X = torch.from_numpy(X).float()
        self.L = torch.from_numpy(L).float()
        self.sigma2 = float(sigma2)
        self.mean = torch.from_numpy(mean).float()

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.X.shape[0]
