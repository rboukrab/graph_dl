"""Refactored graph-learning experiment runners.

Each algorithm is encapsulated in a subclass of :class:`GraphLearningRunner`.
The common experiment loop, metric computation, and result saving live in the
base class; subclasses only provide algorithm-specific setup and per-iteration
updates.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from tqdm import tqdm

from core.config import ensure_log_dir, gl_experiment_path
from core.datasets import FlankerDataset, PolyDataset, SmoothDataset, StatDataset
from core.metrics import (
    compute_filterr_ggsr,
    compute_filterr_poly,
    evaluate_adjacency,
    evaluate_adjacency_weighted,
    evaluate_laplacian,
    evaluate_laplacian_weighted,
)
from core.utils import (
    heavy_tailedness,
    modularity,
    small_world_coefficient,
    weighted_max_matching,
)


class GraphLearningRunner(ABC):
    """Base runner for graph-learning experiments.

    Subclasses must define :attr:`name` (used for dispatch) and
    :attr:`method_name` (used for log-directory naming). They should also
    implement :meth:`add_args`, :meth:`setup`, :meth:`step`, and
    :meth:`get_graph_matrix`.
    """

    name: str = ""
    method_name: str = ""

    def __init__(self, args: Any, dataset: Any, device: torch.device):
        self.args = args
        self.dataset = dataset
        self.device = device

        self.dim: int = args.dim
        self.n_iters: int = args.n_iters
        self.sim_time: float = args.sim_time
        self.test_interval: int = args.test_interval
        self.seed: int = args.seed
        self.data_set: str = args.data_set

        self.graph = getattr(dataset, "graph", None)
        self.accum_time_i: float = 0.0
        self.metrics: Dict[str, np.ndarray] = {}
        self.sim_path: str = ""

    # ------------------------------------------------------------------
    # Algorithm-specific hooks
    # ------------------------------------------------------------------
    @classmethod
    @abstractmethod
    def add_args(cls, parser: Any) -> None:
        """Add algorithm-specific arguments to the parser."""
        raise NotImplementedError

    @abstractmethod
    def setup(self) -> None:
        """Initialize algorithm state. Called once before the main loop."""
        raise NotImplementedError

    @abstractmethod
    def step(self, iteration: int) -> None:
        """Execute one iteration of the algorithm."""
        raise NotImplementedError

    @abstractmethod
    def get_graph_matrix(self) -> torch.Tensor:
        """Return the current learned graph matrix for metric evaluation."""
        raise NotImplementedError

    def uses_adjacency_for_stat(self) -> bool:
        """Return True if :meth:`get_graph_matrix` returns an adjacency matrix.

        When False, the matrix is treated as a Laplacian and passed to
        :func:`evaluate_laplacian`. When True, it is passed to
        :func:`evaluate_adjacency`.
        """
        return False

    def extra_stat_metric_names(self) -> list:
        """Names of extra stat metrics returned by :meth:`extra_stat_metrics`."""
        return []

    def extra_stat_metrics(self, matrix: torch.Tensor) -> Dict[str, float]:
        """Return extra stat metrics for the current graph matrix."""
        return {}

    def cleanup(self) -> None:
        """Hook called after the main loop but before saving results."""
        pass

    # ------------------------------------------------------------------
    # Common experiment infrastructure
    # ------------------------------------------------------------------
    def _build_sim_path(self, param_x_name: str, param_x: Any,
                        param_y_name: str, param_y: Any) -> str:
        n_samples = getattr(self.args, "n_samples", 1e6)
        path = gl_experiment_path(
            method=self.method_name,
            data_set=self.data_set,
            dim=self.dim,
            n_samples=n_samples,
            param_x_name=param_x_name,
            param_x=param_x,
            param_y_name=param_y_name,
            param_y=param_y,
            seed=self.seed,
        )
        ensure_log_dir(path)
        return path

    def _allocate_metrics(self, n_metric_steps: int) -> None:
        self.metrics = {
            "sp": np.zeros(n_metric_steps, dtype=float),
            "accum_time": np.zeros(n_metric_steps, dtype=float),
        }
        if self.data_set in ("stat", "smooth"):
            self.metrics["nae"] = np.zeros(n_metric_steps, dtype=float)
            self.metrics["precision"] = np.zeros(n_metric_steps, dtype=float)
            self.metrics["recall"] = np.zeros(n_metric_steps, dtype=float)
            self.metrics["f1_score"] = np.zeros(n_metric_steps, dtype=float)
            for name in self.extra_stat_metric_names():
                self.metrics[name] = np.zeros(n_metric_steps, dtype=float)
        elif self.data_set == "fmri":
            self.metrics["acc"] = np.zeros(n_metric_steps, dtype=float)
            self.metrics["sig"] = np.zeros(n_metric_steps, dtype=float)
            self.metrics["mod"] = np.zeros(n_metric_steps, dtype=float)

    def _compute_metrics(self, i_test: int) -> None:
        self.metrics["accum_time"][i_test] = self.accum_time_i
        matrix = self.get_graph_matrix()

        if self.data_set in ("stat", "smooth"):
            self._compute_stat_metrics(i_test, matrix)
        elif self.data_set == "fmri":
            self._compute_fmri_metrics(i_test, matrix)

    def _compute_stat_metrics(self, i_test: int, matrix: torch.Tensor) -> None:
        metric_mode = getattr(self.args, "metric_mode", "binary")
        if self.uses_adjacency_for_stat():
            W = matrix.detach().clone().cpu().numpy()
            if metric_mode == "weighted":
                metrics = evaluate_adjacency_weighted(self.graph, W)
            else:
                metrics = evaluate_adjacency(self.graph, W)
        else:
            scale = np.linalg.norm(self.graph.L.todense()) / torch.norm(matrix)
            if metric_mode == "weighted":
                metrics = evaluate_laplacian_weighted(self.graph, scale * matrix)
            else:
                metrics = evaluate_laplacian(self.graph, scale * matrix)

        for key in ("nae", "precision", "recall", "f1_score", "sp"):
            if key in metrics:
                self.metrics[key][i_test] = metrics[key]

        extras = self.extra_stat_metrics(matrix)
        for key, value in extras.items():
            if key in self.metrics:
                self.metrics[key][i_test] = value

        desc = self._stat_description(i_test, metrics, extras)
        self.pbar.set_description(desc)

    def _stat_description(self, i_test: int, metrics: Dict[str, float],
                          extras: Dict[str, float]) -> str:
        return (
            f"f-score={100*metrics['f1_score']:.2f}, "
            f"nae={metrics['nae']:.5f}, sp={metrics['sp']}"
        )

    def _compute_fmri_metrics(self, i_test: int, matrix: torch.Tensor) -> None:
        def weight_func(a, b):
            intersect_ab = set(a) & set(b)
            return sum(self.dataset.vols[idx] for idx in intersect_ab)

        n_clusters = len(self.dataset.labels)
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            verbose=0,
        )

        W = matrix.detach().clone()
        if not self.uses_adjacency_for_stat():
            W = -W
        W[W < 0] = 0.0
        W += 1e-6 * torch.ones((self.dim, self.dim), device=self.device)
        W.fill_diagonal_(0)
        W = (W + W.T) / 2
        W_np = W.cpu().numpy()

        labels = clustering.fit_predict(W_np)
        matching, total = weighted_max_matching(
            self.dataset.labels,
            [np.where(labels == l)[0] for l in np.unique(labels)],
            weight_func,
        )
        acc = 100 * total / self.dataset.vols.sum()
        sp = (W_np[np.abs(W_np) > 1e-2] != 0).sum() / 2

        sig, mod = 0.0, 0.0
        try:
            W_thr = W_np.copy()
            W_thr[W_thr < 2e-6] = 0.0
            sig = small_world_coefficient(W_thr, n_random=1)
            mod = modularity(W_thr, labels)
        except (nx.NetworkXException, ZeroDivisionError):
            pass

        self.metrics["acc"][i_test] = acc
        self.metrics["sp"][i_test] = sp
        self.metrics["sig"][i_test] = sig
        self.metrics["mod"][i_test] = mod

        self.pbar.set_description(
            f"sp={sp}, clerr={acc:.2f}, sig={sig:.2f}, mod={mod:.2f}"
        )

    def _save_results(self) -> None:
        for key, value in self.metrics.items():
            np.save(os.path.join(self.sim_path, f"{key}.npy"), value)

    def run(self) -> None:
        """Run the standard experiment loop (used by GLAMA/Smooth/Poly)."""
        self.setup()
        n_metric_steps = int(np.ceil(self.n_iters / self.test_interval))
        self._allocate_metrics(n_metric_steps)
        self.pbar = tqdm(total=self.n_iters, position=0, leave=True)

        i = 0
        while i < self.n_iters and self.accum_time_i < self.sim_time:
            if i % self.test_interval == 0:
                self._compute_metrics(int(i / self.test_interval))

            start = time.time()
            self.step(i)
            self.accum_time_i += time.time() - start

            self.pbar.update(1)
            i += 1

        self.pbar.close()
        self.cleanup()
        self._save_results()


# ------------------------------------------------------------------------------
# GLAMA
# ------------------------------------------------------------------------------
class GlamaRunner(GraphLearningRunner):
    name = "glama"
    method_name = "glama"

    @classmethod
    def add_args(cls, parser: Any) -> None:
        parser.add_argument("--rho", type=float, default=1e-3)
        parser.add_argument("--gamma", type=float, default=1e-3)
        parser.add_argument("--delta", type=float, default=1e-1)

    def setup(self) -> None:
        self.gamma = self.args.gamma
        self.delta = self.args.delta
        self.rho = self.args.rho
        self.eta = self.args.rho

        self.sim_path = self._build_sim_path(
            "gamma", self.gamma, "delta", self.delta
        )

        X_train = self.dataset.X.to(self.device)
        self.L = torch.zeros((self.dim, self.dim), device=self.device)
        self.C = X_train.T @ X_train / X_train.shape[0] + self.delta * torch.eye(
            self.dim, device=self.device
        )
        self.Gamma = self.gamma * torch.eye(self.dim, device=self.device)
        self.Theta = torch.zeros((self.dim, self.dim), device=self.device)

    @staticmethod
    def _asym_soft_thresh(X, theta_1, theta_2):
        return torch.minimum(X + theta_1, torch.tensor(0)) + torch.maximum(
            X - theta_2, torch.tensor(0)
        )

    def step(self, iteration: int) -> None:
        Omega = torch.linalg.inv(self.Gamma)
        B = self.rho * Omega - self.C + self.Gamma
        L_old = self.L.detach().clone()
        L_diag = (1.0 / self.rho) * torch.diag(B)
        L_neg = (1.0 / self.rho) * self._asym_soft_thresh(B, self.gamma, self.gamma)
        L_neg[torch.where(L_neg > 0.0)[0], torch.where(L_neg > 0.0)[1]] = 0.0
        L_neg.fill_diagonal_(0)
        L_pos = (1.0 / (self.rho + self.eta)) * self._asym_soft_thresh(
            B, self.gamma, self.gamma + self.Theta
        )
        L_pos[torch.where(L_pos <= 0.0)[0], torch.where(L_pos <= 0.0)[1]] = 0.0
        L_pos.fill_diagonal_(0)
        self.L = torch.diag(L_diag) + L_pos + L_neg
        G = torch.maximum(torch.tensor(0), self.L)
        G.fill_diagonal_(0)
        self.Gamma += self.rho * (Omega - self.L)
        self.Theta += self.eta * G
        self.Theta.fill_diagonal_(0)
        self.rho = torch.sort(torch.abs(torch.linalg.eigvalsh(self.Gamma)))[0][0] ** 2
        self.eta = (1 + torch.sqrt(torch.tensor(2))) * self.rho

    def get_graph_matrix(self) -> torch.Tensor:
        return self.L

    def _stat_description(self, i_test, metrics, extras):
        return (
            f"gamma={self.gamma:.3E}, delta={self.delta:.3E}, "
            f"f-score={100*metrics['f1_score']:.2f}, "
            f"prec={100*metrics['precision']:.2f}, "
            f"recall={100*metrics['recall']:.2f}, "
            f"nae={metrics['nae']:.5f}, sp={metrics['sp']}"
        )


# ------------------------------------------------------------------------------
# Smoothness prior
# ------------------------------------------------------------------------------
class SmoothRunner(GraphLearningRunner):
    name = "smooth"
    method_name = "smooth"

    @classmethod
    def add_args(cls, parser: Any) -> None:
        parser.add_argument("--solver", type=str,
                            choices=["projadmm2", "ipm", "activeset"],
                            default="activeset",
                            help="Inner solver for the Laplacian projection")
        parser.add_argument("--admm_mode", type=str,
                            choices=["normal", "optimized"], default="optimized")
        parser.add_argument("--alpha", type=float, default=1e-3)
        parser.add_argument("--beta", type=float, default=100)
        parser.add_argument("--tol", type=float, default=1e-5)
        parser.add_argument("--max_iter", type=int, default=1000)
        parser.add_argument("--k_nn", type=int, default=5,
                            help="Number of nearest neighbours for --data_set smooth")
        parser.add_argument("--sigma2", type=float, default=0.0,
                            help="Noise power for --data_set smooth")
        parser.add_argument("--inner_metrics", action="store_true",
                            help="Compute metrics at every inner projection iteration")
        parser.add_argument("--inner_metric_interval", type=int, default=1,
                            help="Compute inner metrics every N inner iterations")

    def setup(self) -> None:
        from core.graph_learn import ProjADMM2, LaplacianIPM, LaplacianIPMSparse, LaplacianActiveSet

        self.alpha = self.args.alpha
        self.beta = self.args.beta
        self.tol = self.args.tol
        self.max_iter = self.args.max_iter

        self.sim_path = self._build_sim_path(
            "alpha", self.alpha, "beta", self.beta
        )

        if self.data_set in ("stat", "smooth"):
            graph_vol = self.graph.W.todense().sum()
        else:
            graph_vol = 100.0

        # Use the identity initial Laplacian for all inner solvers so that the
        # smoothness update starts from the same point.
        L0 = torch.eye(self.dim, device=self.device)

        if self.args.solver == "projadmm2":
            self.graph_learner = ProjADMM2(
                dim=self.dim,
                mode=self.args.admm_mode,
                L=L0,
                vol=graph_vol,
                device=self.device,
            )
        elif self.args.solver == "activeset":
            self.graph_learner = LaplacianActiveSet(
                dim=self.dim,
                L=L0,
                vol=graph_vol,
                device=self.device,
            )
        else:
            self.graph_learner = LaplacianIPMSparse(
                dim=self.dim,
                L=L0,
                vol=graph_vol,
                device=self.device,
                eps_in=1e-3,
                eps_out=1e-3,
                max_inner_iter=100,
                t0=1.0
            )
        self.X_train = self.dataset.X.to(self.device)

        self.inner_metrics = getattr(self.args, "inner_metrics", False)
        self.inner_metric_interval = getattr(self.args, "inner_metric_interval", 1)
        if self.inner_metrics and self.data_set not in ("stat", "smooth"):
            raise ValueError("--inner_metrics is only supported for stat/smooth datasets")
        if self.inner_metrics:
            self._gt_laplacian_norm = float(np.linalg.norm(self.graph.L.todense()))

    def _inner_metric_callback(self, inner_iter: int, L: torch.Tensor) -> None:
        if inner_iter % self.inner_metric_interval != 0:
            return
        scale = self._gt_laplacian_norm / torch.norm(L)
        scaled = scale * L
        if getattr(self.args, "metric_mode", "binary") == "weighted":
            metrics = evaluate_laplacian_weighted(self.graph, scaled)
        else:
            metrics = evaluate_laplacian(self.graph, scaled)
        for key in ("nae", "precision", "recall", "f1_score", "sp"):
            if key in metrics:
                self.metrics[key].append(metrics[key])
        desc = (
            f"inner_iter={inner_iter}, "
            f"f1-score={100*metrics.get('f1_score', 0):.2f}, "
            f"nae={metrics.get('nae', 1):.5f}, "
            f"sp={metrics.get('sp', 0)}"
        )
        self.pbar.set_description(desc)

    def _allocate_metrics(self, n_metric_steps: int) -> None:
        if self.inner_metrics:
            self.metrics = {
                "sp": [],
                "accum_time": [],
                "nae": [],
                "precision": [],
                "recall": [],
                "f1_score": [],
            }
        else:
            super()._allocate_metrics(n_metric_steps)

    def _compute_metrics(self, i_test: int) -> None:
        if self.inner_metrics:
            return
        super()._compute_metrics(i_test)

    def _save_results(self) -> None:
        if self.inner_metrics:
            for key, value in self.metrics.items():
                arr = np.asarray(value)
                if arr.size > 0:
                    np.save(os.path.join(self.sim_path, f"{key}.npy"), arr)
        else:
            super()._save_results()

    def step(self, iteration: int) -> None:
        L = self.graph_learner.L.detach().clone()
        Y_train = self.X_train @ torch.linalg.inv(
            torch.eye(self.dim, device=self.device) + self.alpha * L
        )
        C = Y_train.T @ Y_train
        H_k = -self.alpha * C / self.beta
        callback = self._inner_metric_callback if self.inner_metrics else None
        self.graph_learner.forward(
            F_t=H_k,
            tolerance=self.tol,
            max_iter=self.max_iter,
            callback=callback,
        )


    def get_graph_matrix(self) -> torch.Tensor:
        return self.graph_learner.L.detach().clone()

    def _stat_description(self, i_test, metrics, extras):
        return (
            f"alpha={self.alpha:.3E}, beta={self.beta:.3E}, "
            f"f1-score={100*metrics['f1_score']:.2f}, "
            f"prec={100*metrics['precision']:.2f}, "
            f"recall={100*metrics['recall']:.2f}, "
            f"nae={metrics['nae']:.5f}, sp={metrics['sp']}"
        )


# ------------------------------------------------------------------------------
# Polynomial graph learning
# ------------------------------------------------------------------------------
class PolyRunner(GraphLearningRunner):
    name = "poly"
    method_name = "poly"

    @classmethod
    def add_args(cls, parser: Any) -> None:
        parser.add_argument("--admm_mode", type=str,
                            choices=["normal", "optimized"], default="optimized")
        parser.add_argument("--deg", type=int, default=2)
        parser.add_argument("--alpha", type=float, default=4)
        parser.add_argument("--tol", type=float, default=1e-2)
        parser.add_argument("--max_iter", type=int, default=10)

    def extra_stat_metric_names(self) -> list:
        return ["filterr"]

    def setup(self) -> None:
        from core.graph_learn import ProjADMM2

        self.deg = self.args.deg
        self.alpha = self.args.alpha
        self.tol = self.args.tol
        self.max_iter = self.args.max_iter
        self.h = self.dataset.h if self.data_set == "stat" else None

        self.sim_path = self._build_sim_path(
            "deg", self.deg, "alpha", self.alpha
        )

        self.X_train = self.dataset.X.to(self.device)
        self.C = self.X_train.T @ self.X_train / self.X_train.shape[0]
        self.a = torch.rand(self.deg + 1, device=self.device)

        self.Lp = torch.zeros((self.deg + 1, self.dim, self.dim), device=self.device)
        self.Lp[0] = torch.eye(self.dim, device=self.device)
        self.Phi = torch.zeros((self.dim, self.dim), device=self.device)
        self.Omega = torch.zeros((self.dim, self.dim), device=self.device)
        self.U = torch.zeros((self.dim, self.dim), device=self.device)
        self.I = torch.eye(self.dim).bool().to(self.device)
        self.O = (torch.ones((self.dim, self.dim)) - torch.eye(self.dim)).bool().to(
            self.device
        )

        if self.data_set == "stat":
            graph_vol = self.graph.W.todense().sum()
        else:
            graph_vol = 100.0

        self.graph_learner = ProjADMM2(
            dim=self.dim,
            mode=self.args.admm_mode,
            L=(10 + 10 / 99) * torch.eye(self.dim, device=self.device)
            - (10 / 99) * torch.ones((self.dim, self.dim), device=self.device),
            vol=graph_vol,
            device=self.device,
        )

    def step(self, iteration: int) -> None:
        from torch.nn.functional import softshrink

        L = self.graph_learner.L.detach().clone()
        self.Lp[1] = L
        for p in range(2, self.deg + 1):
            self.Lp[p] = self.Lp[p - 1] @ L
            self.Lp[p] = 0.5 * (self.Lp[p] + self.Lp[p].T)

        Q = self.a[0] * self.Lp[0]
        for p in range(1, self.deg + 1):
            Q += self.a[p] * self.Lp[p]
        Q_inv = torch.linalg.inv(Q)

        grad_Q = torch.zeros(self.deg + 1, device=self.device)
        grad_C = torch.zeros(self.deg + 1, device=self.device)
        grad_Q[0] = torch.trace(Q_inv)
        grad_C[0] = torch.trace(self.C)
        for p in range(1, self.deg + 1):
            grad_Q[p] = -torch.trace(Q_inv @ self.Lp[p])
            grad_C[p] = torch.trace(self.C @ self.Lp[p])
        grad_a = grad_Q + grad_C
        grad_a /= torch.linalg.norm(grad_a)

        a_old = self.a.detach().clone()
        h_old = torch.trace(Q @ self.C) - torch.logdet(Q)
        a_t = 1000
        found = False
        idx = 0
        while not found and idx < 10:
            a_new = a_old - (1.0 / a_t) * grad_a
            Q_new = a_new[0] * torch.eye(self.dim, device=self.device)
            for p in range(1, self.deg + 1):
                Q_new += a_new[p] * self.Lp[p]
            h_new = torch.trace(Q_new @ self.C) - torch.logdet(Q_new)
            suff_dec = h_old + grad_a.dot(a_new - a_old) + 0.5 * a_t * torch.linalg.norm(
                a_new - a_old
            ) ** 2
            if h_new < suff_dec and (torch.linalg.eigvalsh(Q_new) >= 0).all():
                found = True
                self.a = a_new
            else:
                a_t *= 10
                idx += 1

        Q = Q_new
        Q_inv = torch.linalg.inv(Q)
        grad_Q = torch.zeros_like(L, device=self.device)
        grad_C = torch.zeros_like(L, device=self.device)
        for p in range(1, self.deg + 1):
            for r in range(p):
                grad_Q -= self.a[p] * self.Lp[r] @ Q_inv @ self.Lp[p - 1 - r]
                grad_C += self.a[p] * self.Lp[r] @ self.C @ self.Lp[p - 1 - r]
        grad_h_L = grad_Q + grad_C

        b_t = 100
        found = False
        L_old = L.detach().clone()
        h_old = torch.trace(Q @ self.C) - torch.logdet(Q)
        idx_L = 0
        while not found and idx_L < 10:
            F_t = b_t * L_old - grad_h_L
            rho = 1e-3
            for _ in range(self.max_iter):
                K = F_t + rho * (self.Omega + self.U)
                self.Phi = (softshrink(K, self.alpha) * self.O + K * self.I) / (
                    b_t + rho
                )
                self.graph_learner.forward(
                    F_t=-(self.Phi + self.U), tolerance=1e-5, max_iter=1000
                )
                self.Omega = self.graph_learner.L.detach().clone()
                self.U += self.Omega - self.Phi
                err = torch.linalg.norm(self.Phi - self.Omega)
                if err < self.tol:
                    break
            L_new = self.graph_learner.L.detach().clone()

            self.Lp[1] = L_new.detach().clone()
            for p in range(2, self.deg + 1):
                self.Lp[p] = self.Lp[p - 1] @ L_new
                self.Lp[p] = 0.5 * (self.Lp[p] + self.Lp[p].T)

            Q = self.a[0] * self.Lp[0]
            for p in range(1, self.deg + 1):
                Q += self.a[p] * self.Lp[p]

            h_new = torch.trace(Q @ self.C) - torch.logdet(Q)
            suff_dec = h_old + torch.trace(
                grad_h_L @ (L_new - L_old)
            ) + 0.5 * b_t * torch.linalg.norm(L_new - L_old) ** 2
            if h_new <= suff_dec and (torch.linalg.eigvalsh(Q) >= 0).all():
                found = True
            else:
                b_t *= 10
                self.graph_learner.L = L_old.detach().clone()
                idx_L += 1

    def get_graph_matrix(self) -> torch.Tensor:
        return self.Phi

    def extra_stat_metrics(self, matrix: torch.Tensor) -> Dict[str, float]:
        if self.data_set != "stat" or self.h is None:
            return {}
        return {"filterr": compute_filterr_poly(matrix, self.h, self.a)}

    def _stat_description(self, i_test, metrics, extras):
        return (
            f"deg={self.deg}, alpha={self.alpha:.2f}, "
            f"f-score={100*metrics['f1_score']:.2f}, "
            f"nae={metrics['nae']:.5f}, sp={metrics['sp']}, "
            f"filterr={extras.get('filterr', 0):.5f}"
        )


# ------------------------------------------------------------------------------
# Pairwise graph learning (PGL)
# ------------------------------------------------------------------------------
class PglRunner(GraphLearningRunner):
    name = "pgl"
    method_name = "pgl"

    @classmethod
    def add_args(cls, parser: Any) -> None:
        parser.add_argument("--admm_mode", type=str,
                            choices=["normal", "optimized"], default="optimized")
        parser.add_argument("--rho", type=float, default=1e-1)
        parser.add_argument("--eta", type=float, default=1e-1)
        parser.add_argument("--delta", type=float, default=1e-4)
        parser.add_argument("--S_iters", type=int, default=1000)

    def extra_stat_metric_names(self) -> list:
        return ["commerr", "filterr"]

    def setup(self) -> None:
        self.rho = self.args.rho
        self.eta = self.args.eta
        self.delta = self.args.delta
        self.S_iters = self.args.S_iters
        self.h = self.dataset.h.to(self.device) if self.data_set == "stat" else None

        self.sim_path = self._build_sim_path(
            "rho", self.rho, "eta", self.eta
        )

        self.X = self.dataset.X.to(self.device)
        self.C = self.X.T @ self.X / self.X.shape[0]

        self.P = torch.zeros((self.dim, self.dim), device=self.device)
        self.Z = torch.zeros((self.dim, self.dim), device=self.device)
        self.Q = torch.zeros((self.dim, self.dim), device=self.device)
        self.Y = torch.zeros((self.dim, self.dim), device=self.device)
        self.S = torch.zeros((self.dim, self.dim), device=self.device)
        self.T = torch.eye(self.dim, device=self.device)

    def step(self, iteration: int) -> None:
        # T update
        delta = 1e-4
        beta = 1e-4
        L2 = 1
        fT = (
            self.S @ self.S @ self.T
            + self.T @ self.S @ self.S
            - 2 * self.S @ self.T @ self.S
            + self.Q @ self.S
            - self.S @ self.Q
            + 1 / beta * (self.S @ self.Y - self.Y @ self.S)
        )
        W = self.T - 1 / L2 * (beta * fT + self.C)
        La, U = torch.linalg.eigh(W)
        I = torch.eye(self.dim).bool().to(self.device)
        self.T = U @ ((La.diag() + torch.sqrt(La.diag() ** 2 + 4 / L2 * I)) / 2) @ U.T
        Qp = self.S @ self.T - self.T @ self.S + (1 / beta) * self.Y
        nQp = torch.norm(Qp)
        if nQp > delta:
            self.Q = (delta / nQp) * Qp
        else:
            self.Q = Qp
        self.Y = self.Y + beta * (self.S @ self.T - self.T @ self.S - self.Q)

    def _step_inner(self, j: int) -> None:
        beta = 1e-4
        delta = 1e-4
        la_max = torch.max(torch.abs(torch.linalg.eigvalsh(self.T)))
        L1 = 4 * beta * la_max ** 2

        if self.S.sum() == 0.0:
            from core.utils import generate_graph_ER
            self.S = torch.from_numpy(generate_graph_ER(self.dim, 0.1)).to(
                self.device
            ).float()

        gS = (
            self.T @ self.T @ self.S
            + self.S @ self.T @ self.T
            - 2 * self.T @ self.S @ self.T
            + self.P @ self.T
            - self.T @ self.P
            + (1 / beta) * (self.T @ self.Z - self.Z @ self.T)
        )
        Sp = self.S - (1 / L1) * (self.rho * torch.eye(self.dim, device=self.device)
                                    + self.eta * self.S + beta * gS)
        Sp[Sp < 0] = 0
        self.S = (Sp + Sp.T) / 2
        self.S = self.S / torch.max(self.S)
        self.S.fill_diagonal_(0)

        Pp = self.T @ self.S - self.S @ self.T + (1 / beta) * self.Z
        nPp = torch.norm(Pp)
        if nPp > delta:
            self.P = (delta / nPp) * Pp
        else:
            self.P = Pp
        self.Z = self.Z + beta * (self.T @ self.S - self.S @ self.T - self.P)

    def get_graph_matrix(self) -> torch.Tensor:
        return self.S

    def uses_adjacency_for_stat(self) -> bool:
        return True

    def extra_stat_metrics(self, matrix: torch.Tensor) -> Dict[str, float]:
        extras = {"commerr": torch.norm(self.S @ self.T - self.T @ self.S).item()}
        if self.data_set == "stat" and self.h is not None:
            extras["filterr"] = compute_filterr_ggsr(self.S, self.T, self.h)
        return extras

    def _stat_description(self, i_test, metrics, extras):
        return (
            f"eta={self.eta:.3E}, rho={self.rho:.3E}, "
            f"f-score={100*metrics['f1_score']:.2f}, "
            f"nae={metrics['nae']:.5f}, sp={metrics['sp']}, "
            f"commerr={extras.get('commerr', 0):.5f}, "
            f"filterr={extras.get('filterr', 0):.5f}"
        )

    def run(self) -> None:
        """PGL has its own loop structure with inner S-updates."""
        self.setup()
        n_metric_steps = int(np.ceil(self.n_iters * self.S_iters / self.test_interval))
        self._allocate_metrics(n_metric_steps)
        self.pbar = tqdm(total=self.n_iters * self.S_iters, position=0, leave=True)

        i = 0
        while i < self.n_iters and self.accum_time_i < self.sim_time:
            self.step(i)
            for j in range(self.S_iters):
                start = time.time()
                self._step_inner(j)
                self.accum_time_i += time.time() - start

                k = self.S_iters * i + j
                if j % self.test_interval == 0:
                    self._compute_metrics(int(k / self.test_interval))

                self.pbar.update(1)
                if self.accum_time_i > self.sim_time:
                    break
            i += 1

        self.pbar.close()
        self.cleanup()
        self._save_results()


# Registry used by the dispatcher.
RUNNERS: Dict[str, type] = {
    cls.name: cls
    for cls in (GlamaRunner, SmoothRunner, PolyRunner, PglRunner)
}


def load_dataset(args: Any) -> Any:
    """Load the dataset selected by the common CLI args."""
    if args.data_set == "stat":
        if args.algorithm == "pgl":
            return PolyDataset(gnum=args.seed, gtype="NN", dim=args.dim, n=args.n_samples)
        return StatDataset(args.seed, args.dim, int(args.n_samples))
    elif args.data_set == "fmri":
        dataset = FlankerDataset()
        return dataset
    elif args.data_set == "smooth":
        return SmoothDataset(
            args.seed,
            args.dim,
            int(args.n_samples),
            k=getattr(args, "k_nn", 5),
            sigma2=getattr(args, "sigma2", 0.0),
        )
    else:
        raise ValueError(f"Unknown data_set: {args.data_set}")
