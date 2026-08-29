import pygsp
import torch
import torch.nn as nn
from core.graph_learn import BCD, GAMA, GLAMA
from core.sparse_coding import _AIHT
from torch.nn.functional import softshrink

class LapRegSparseNet(nn.Module):

    def __init__(self,
                 n_nodes: int,
                 n_atoms: int,
                 batch_size: int,
                 sparsity: int,
                 dictionary: torch.FloatTensor = None,
                 opt: str = 'gd',
                 lr: float = 0.01,
                 alpha: float = 0.001,
                 gamma: float = 1e-3,
                 rho: float = 1e-3,
                 device: str = 'cpu',
                 train: bool = True
                 ):
        """
        Initialize LapRegSparseNet instance.

        :param n_nodes: number of nodes in the graph.
        :param n_atoms: number of atoms in dictionary.
        :param batch_size: number of training signals to process in each iteration. 
        :param sparsity: number of non-zero elements in computed sparse codes. 
        :param graph_learn: graph learning method. 
        :param update_cov: flag to update covariance matrix. 
        :param cov_matrix: sample covariance matrix of data.
        :param sample_mean: sample mean vector of data.
        :param lr: learning rate of dictionary learning algorithm.
        :param alpha: Laplacian regularization parameter.
        :param gamma: learned Laplacian sparsity regularization parameter.
        :param rho: GL algorithm dual step.
        :param device: run model in CPU or GPU.
        """
        super(LapRegSparseNet, self).__init__()

        self.n_nodes = n_nodes
        self.n_atoms = n_atoms
        self.batch_size = batch_size
        self.sparsity = sparsity
        self.device = device
        self.opt = opt
        self.train = train

        self.lr = lr
        self.alpha = alpha

        self.gamma = gamma
        self.rho = rho

        if dictionary is None:
            self.D = torch.randn((n_nodes, n_atoms), device=device)
        else:
            self.D = dictionary
        self.normalize_weights()

        self.y = torch.zeros((batch_size, n_atoms), device=device)

    def normalize_weights(self) -> None:
        """
        Normalize the atoms of the dictionary to have unit energy.
        :return:
        """
        self.D *= self.n_atoms / torch.linalg.norm(self.D, dim=0).sum()

    def predict(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Predict the input x using the stored dictionary.
        :param x: input signal.
        :return: signal representation.
        """

        # Compute sparse code
        y = _AIHT(x, self.D, self.n_atoms, self.sparsity, 0.001, device=self.device)[0]

        # Return prediction
        return y @ self.D.T


    def forward(self, x: torch.FloatTensor, L: torch.FloatTensor) -> None:
        """
        Perform an iteration of the graph learning algorithm. Then, with the new graph estimation,
        compute the sparse codes corresponding to the input batch of signals and the prediction of these
        signals. Finally, update the dictionary using stochastic gradient descent (SGD).
        :param x: input signals.
        """

        y = _AIHT(x, self.D, self.n_atoms, self.sparsity, 0.001, device=self.device)[0]

        if self.train:
            if self.opt == 'gd':
                grad_R = (self.D @ y.T - x.T) @ y
                grad_L = self.alpha * L @ self.D
                self.D -= self.lr * (grad_L + grad_R)
            elif self.opt == 'ksvd':
                for j in range(self.n_atoms):
                    o_j = torch.ones((1, self.n_atoms), device=self.device)
                    o_j[0, j] = 0
                    E_j = x - ((self.D * o_j) @ (y * o_j).T).T
                    nnz_j = y[:, j].nonzero().squeeze()
                    if nnz_j.numel() > 0:
                        d_j_old = self.D[:, j].detach().clone()
                        v_j = E_j[nnz_j] @ self.D[:, j] / torch.linalg.norm(self.D[:, j]) ** 2
                        v_jI_alphaL = torch.linalg.norm(v_j) ** 2 * torch.eye(self.n_nodes, device=self.device) + self.alpha * L
                        if len(v_j.shape) == 0:
                            self.D[:, j] = torch.linalg.inv(v_jI_alphaL) @ E_j[nnz_j].T * v_j
                        else:
                            self.D[:, j] = torch.linalg.inv(v_jI_alphaL) @ E_j[nnz_j].T @ v_j
            self.normalize_weights()

class StationarySparseNet(nn.Module):

    def __init__(self,
                 n_nodes: int,
                 n_atoms: int,
                 batch_size: int,
                 sparsity: int,
                 dictionary: torch.FloatTensor = None,
                 H: torch.FloatTensor = None,
                 lr: float = 0.01,
                 alpha: float = 0.001,
                 deg: int = 2,
                 gamma: float = 1e-3,
                 rho: float = 1e-3,
                 device: str = 'cpu',
                 ):
        """
        Initialize StationarySparseNet instance.

        :param n_nodes: number of nodes in the graph.
        :param n_atoms: number of atoms in dictionary.
        :param batch_size: number of training signals to process in each iteration.
        :param sparsity: number of non-zero elements in computed sparse codes.
        :param graph_learn: graph learning method.
        :param update_cov: flag to update covariance matrix.
        :param cov_matrix: sample covariance matrix of data.
        :param sample_mean: sample mean vector of data.
        :param lr: learning rate of dictionary learning algorithm.
        :param alpha: Laplacian regularization parameter.
        :param gamma: learned Laplacian sparsity regularization parameter.
        :param rho: GL algorithm dual step.
        :param device: run model in CPU or GPU.
        """
        super(StationarySparseNet, self).__init__()

        self.n_nodes = n_nodes
        self.n_atoms = n_atoms
        self.batch_size = batch_size
        self.batch_size_y = 50
        self.sparsity = sparsity
        self.device = device

        self.lr = lr
        self.alpha = alpha
        self.deg = deg
        self.a = 0.1 * torch.rand(deg + 1, device=device)
        self.L = torch.zeros((deg + 1, n_nodes, n_nodes), device=device)
        self.L[0] = torch.eye(n_nodes, device=device)

        self.gamma = gamma
        self.rho = rho

        if dictionary is None:
            self.D = torch.randn((n_nodes, n_atoms), device=device)
        else:
            self.D = dictionary
        self.normalize_weights()

        self.y = torch.zeros((batch_size, n_atoms), device=device)

    def normalize_weights(self) -> None:
        """
        Normalize the atoms of the dictionary to have unit energy.
        :return:
        """
        self.D *= self.n_atoms / torch.linalg.norm(self.D, dim=0).sum()

    def predict(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Predict the input x using the stored dictionary.
        :param x: input signal.
        :return: signal representation.
        """

        # Compute sparse code
        y = _AIHT(x, self.D, self.n_atoms, self.sparsity, 0.001, device=self.device)[0]

        # Return prediction
        return y @ self.D.T

    def forward(self, x: torch.FloatTensor, L: torch.FloatTensor) -> None:
        """
        Perform an iteration of the graph learning algorithm. Then, with the new graph estimation,
        compute the sparse codes corresponding to the input batch of signals and the prediction of these
        signals. Finally, update the dictionary using stochastic gradient descent (SGD).
        :param x: input signals.
        """

        y = _AIHT(x, self.D, self.n_atoms, self.sparsity, 0.001, device=self.device)[0]

        grad_R = (self.D @ y.T - x.T) @ y
        self.L[1] = L.detach().clone()
        for p in range(2, self.deg + 1):
            self.L[p] = self.L[p-1] @ L
            self.L[p] = 0.5 * (self.L[p] + self.L[p].T)
        Q = self.a[0] * self.L[0]
        for p in range(1, self.deg + 1):
            Q += self.a[p] * self.L[p]
        grad_P = self.alpha * Q @ self.D
        self.D -= self.lr * (grad_P + grad_R)
        self.normalize_weights()

        C = (1. / self.n_atoms) * self.D @ self.D.T
        Q_inv = torch.linalg.inv(Q)
        grad_Q = torch.zeros(self.deg + 1, device=self.device)
        grad_C = torch.zeros(self.deg + 1, device=self.device)
        grad_Q[0] = torch.trace(Q_inv)
        grad_C[0] = torch.trace(C)
        for p in range(1, self.deg + 1):
            grad_Q[p] = torch.trace(Q_inv @ self.L[p])
            grad_C[p] = torch.trace(C @ self.L[p])
        grad_Q *= - 0.5 * self.n_atoms * self.alpha
        grad_C *= 0.5 * self.n_atoms * self.alpha
        grad_a = grad_Q + grad_C
        grad_a /= torch.linalg.norm(grad_a)

        a_old = self.a.detach().clone()
        h_old = 0.5 * self.alpha * self.n_atoms * (torch.trace(Q @ C) - torch.logdet(Q))
        b_k = 100
        found = False
        idx = 0
        while not found and idx < 10:
            a_new = a_old - (1. / b_k) * grad_a
            Q_new = a_new[0] * torch.eye(self.n_nodes, device=self.device)
            for p in range(1, self.deg + 1):
                Q_new += a_new[p] * self.L[p]
            h_new = 0.5 * self.alpha * self.n_atoms * (torch.trace(Q_new @ C) - torch.logdet(Q_new))
            suff_dec = h_old + grad_a.dot(a_new - a_old) + 0.5 * b_k * torch.linalg.norm(a_new - a_old) ** 2
            if h_new < suff_dec:
                found = True
                self.a = a_new
            else:
                b_k *= 10
                idx += 1

        Q = Q_new
        Q_inv = torch.linalg.inv(Q)
        grad_Q = torch.zeros_like(L, device=self.device)
        grad_C = torch.zeros_like(L, device=self.device)
        for p in range(1, self.deg + 1):
            for r in range(p):
                grad_Q += self.a[p] * self.L[r] @ Q_inv @ self.L[p - 1 - r]
                grad_C += self.a[p] * self.L[r] @ C @ self.L[p - 1 - r]
        grad_Q *= - 0.5 * self.alpha * self.n_atoms
        grad_C *= 0.5 * self.alpha * self.n_atoms
        return grad_Q + grad_C

class HeatDiffSparseNet(nn.Module):

    def __init__(self,
                 n_nodes: int,
                 n_filters: int,
                 batch_size: int,
                 laplacian: torch.Tensor = None,
                 lr: float = 0.01,
                 gamma: float = 0.001,
                 device: str = 'cpu'
                 ):
        """
        Initialize HeatDiffSparseNet instance.

        :param mode: run in 'batch' or 'online' mode.
        :param n_nodes: number of nodes in the graph.
        :param n_filters: number of atoms in dictionary.
        :param batch_size: number of training signals to process in each iteration.
        :param graph_learn: graph learning method.
        :param update_cov: flag to update covariance matrix.
        :param cov_matrix: sample covariance matrix of data.
        :param sample_mean: sample mean vector of data.
        :param lr: learning rate of dictionary learning algorithm.
        :param gamma: Laplacian regularization parameter.
        :param n_cov: number of signals to compute initial covariance matrix.
        :param gamma: learned Laplacian sparsity regularization parameter.
        :param rho: GL algorithm dual step.
        :param device: run model in CPU or GPU.
        """
        super(HeatDiffSparseNet, self).__init__()

        self.n_nodes = n_nodes
        self.n_filters = n_filters
        self.batch_size = batch_size
        self.device = device

        self.lr = lr
        self.gamma = gamma
        self.k = 1

        self.y = torch.zeros((batch_size, n_filters * n_nodes), device=device)
        self.D = torch.empty((n_nodes, n_filters * n_nodes), device=device)
        self.tau = torch.rand((n_filters,), device=device)
        for f in range(n_filters):
            self.D[:, f * n_nodes:(f + 1) * n_nodes] = torch.linalg.matrix_exp(-self.tau[f] * laplacian)

    def predict(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Predict the input x using the stored dictionary.
        :param x: input signal.
        :return: signal representation.
        """

        y = torch.zeros((x.shape[0], self.n_filters * self.n_nodes), device=self.device)

        lr = 1 / torch.linalg.norm(2 * self.D.T @ self.D)

        idx = 0
        err = 99999.
        while idx < 1000 and err > 1e-4:
            y_old = y.detach().clone()
            grad_y = -2 * (x - y @ self.D.T) @ self.D
            y = softshrink(y - lr * grad_y, self.gamma * lr)
            err = torch.linalg.norm(y - y_old)
            idx += 1

        # Return prediction
        return y @ self.D.T

    def forward(self, x: torch.FloatTensor, L: torch.FloatTensor) -> None:
        """
        Perform an interation of the graph learning algorithm. Then, with the new graph estimation,
        compute the sparse codes corresponding to the input batch of signals and the prediction of these
        signals. Finally, update the dictionary using stochastic gradient descent (SGD).
        :param x: input signals.
        """

        # Update time scales
        e_k = torch.zeros(self.n_filters, device=self.device)
        for f in range(self.n_filters):
            y_f = self.y[:, (f * self.n_nodes):((f + 1) * self.n_nodes)]
            e_kf_acum = 2 * torch.linalg.norm(y_f) * torch.linalg.norm(x)
            for p in range(self.n_filters):
                y_p = self.y[:, (p * self.n_nodes):((p + 1) * self.n_nodes)]
                e_kf_acum += 4 * torch.linalg.norm(y_p) * torch.linalg.norm(y_f)
            e_k[f] = (torch.linalg.eigvalsh(L).abs().max() ** 2) * e_kf_acum
        e_k = e_k.max()

        for f in range(self.n_filters):
            y_f = self.y[:, (f * self.n_nodes):((f + 1) * self.n_nodes)]
            Ayx_f = y_f.T @ x
            grad_f = 2 * torch.trace(Ayx_f @ L @ torch.matrix_exp(- self.tau[f] * L))
            for p in range(self.n_filters):
                y_p = self.y[:, (p * self.n_nodes):((p + 1) * self.n_nodes)]
                Ayy_pf = y_p.T @ y_f
                grad_f -= 2 * torch.trace(Ayy_pf @ L @ torch.matrix_exp(- (self.tau[p] + self.tau[f]) * L))
            if e_k > 0:
                self.tau[f] = torch.maximum(- (grad_f - e_k * (self.tau[f] ** self.k)) / e_k, torch.tensor(0))

        lr = 1 / torch.linalg.norm(2 * self.D.T @ self.D)
        grad_y = -2 * (x - self.y @ self.D.T) @ self.D
        self.y = softshrink(self.y - lr * grad_y, self.gamma * lr)
        vals_L, vecs_L = torch.linalg.eigh(L)
        grad_L = torch.zeros_like(L, device=self.device)
        if L.max() != 0:
            for f in range(self.n_filters):
                y_f = self.y[:, (f * self.n_nodes):((f + 1) * self.n_nodes)]
                vals_f = - self.tau[f] * vals_L
                exp_vals_f = torch.exp(vals_f)
                Ayx_f = y_f.T @ x
                Byx_f1 = exp_vals_f.repeat(self.n_nodes, 1).T - exp_vals_f.repeat(self.n_nodes, 1)
                Byx_f2 = vals_f.repeat(self.n_nodes, 1).T - vals_f.repeat(self.n_nodes, 1)
                Byx_f = Byx_f1 / Byx_f2
                Byx_f.fill_diagonal_(0.)
                Byx_f += torch.diag(exp_vals_f)
                grad_L += 2 * self.tau[f] * vecs_L @ ((vecs_L.T @ Ayx_f.T @ vecs_L) * Byx_f) @ vecs_L.T
                for p in range(self.n_filters):
                    y_p = self.y[:, (p * self.n_nodes):((p + 1) * self.n_nodes)]
                    vals_pf = - (self.tau[p] + self.tau[f]) * vals_L
                    exp_vals_pf = torch.exp(vals_pf)
                    Ayy_pf = y_p.T @ y_f
                    Byy_pf1 = exp_vals_pf.repeat(self.n_nodes, 1).T - exp_vals_pf.repeat(self.n_nodes, 1)
                    Byy_pf2 = vals_pf.repeat(self.n_nodes, 1).T - vals_pf.repeat(self.n_nodes, 1)
                    Byy_pf = Byy_pf1 / Byy_pf2
                    Byy_pf.fill_diagonal_(0.)
                    Byy_pf += torch.diag(exp_vals_pf)
                    grad_L -= (self.tau[p] + self.tau[f]) * vecs_L @ ((vecs_L.T @ Ayy_pf.T @ vecs_L) * Byy_pf) @ vecs_L.T

        return grad_L

class ApproximateKSVD(nn.Module):

    def __init__(self,
                 n_nodes: int,
                 n_atoms: int,
                 batch_size: int,
                 sparsity: int,
                 graph: pygsp.graphs.Graph = None,
                 graph_learn: str = 'none',
                 update_cov: bool = False,
                 dictionary: torch.FloatTensor = None,
                 cov_matrix: torch.FloatTensor = None,
                 sample_mean: torch.FloatTensor = None,
                 fit_intercept: bool = False,
                 lr: float = 0.01,
                 alpha: float = 0.001,
                 n_cov: int = 100,
                 gamma: float = 1e-3,
                 rho: float = 1e-3,
                 device: str = 'cpu'
                 ):
        """
        Initialize SparseNet instance.

        :param n_nodes: number of nodes in the graph.
        :param n_atoms: number of atoms in dictionary.
        :param batch_size: number of training signals to process in each iteration.
        :param sparsity: number of non-zero elements in computed sparse codes.
        :param graph_learn: graph learning method.
        :param update_cov: flag to update covariance matrix.
        :param cov_matrix: sample covariance matrix of data.
        :param sample_mean: sample mean vector of data.
        :param lr: learning rate of dictionary learning algorithm.
        :param alpha: Laplacian regularization parameter.
        :param n_cov: number of signals to compute initial covariance matrix.
        :param gamma: learned Laplacian sparsity regularization parameter.
        :param rho: GL algorithm dual step.
        :param device: run model in CPU or GPU.
        """
        super(ApproximateKSVD, self).__init__()

        assert graph_learn in ['none', 'known', 'bcd', 'gama', 'glama'], \
            "graph_learn must be one of the following values ['none', 'known', 'bcd', 'gama', 'glama']"

        if graph_learn == 'known' and graph is None:
            raise ValueError("Provide the graph.")

        if graph_learn in ['bcd', 'gama', 'glama'] and cov_matrix is None:
            raise ValueError("Provide the covariance matrix of the data.")

        self.n_nodes = n_nodes
        self.n_atoms = n_atoms
        self.batch_size = batch_size
        self.batch_size_y = 50
        self.sparsity = sparsity
        self.device = device

        self.graph = graph
        self.graph_learn = graph_learn
        self.update_cov = update_cov
        self.fit_intercept = fit_intercept

        self.S = cov_matrix.to(device)
        self.sample_mean = sample_mean

        self.L = None
        if self.graph:
            self.L = torch.from_numpy(graph.L.todense()).float().to(device)

        self.lr = lr
        self.alpha = alpha
        self.c = 10

        self.n_cov = n_cov
        self.gamma = gamma
        self.rho = rho

        if dictionary is None:
            self.D = torch.randn((n_nodes, n_atoms), device=device)
        else:
            self.D = dictionary
        self.normalize_weights()

        # Graph learner
        self.graph_learner = None
        if graph_learn == 'bcd':
            self.graph_learner = BCD(
                cov_matrix=self.S,
                n_cov=n_cov,
                batch_size=batch_size,
                update_cov=self.update_cov,
                device=device
            )
        elif graph_learn == 'gama':
            self.graph_learner = GAMA(
                cov_matrix=self.S,
                gamma=gamma,
                rho=rho,
                n_cov=n_cov,
                batch_size=batch_size,
                update_cov=self.update_cov,
                device=device
            )
        elif graph_learn == 'glama':
            self.graph_learner = GLAMA(
                cov_matrix=self.S,
                gamma=self.gamma,
                rho_1=self.rho,
                rho_2=self.rho,
                n_cov=self.n_cov,
                batch_size=batch_size,
                update_cov=self.update_cov,
                device=device
            )

    def normalize_weights(self) -> None:
        """
        Normalize the atoms of the dictionary to have unit energy.
        :return:
        """
        norm_D = torch.linalg.norm(self.D, dim=0)
        self.D[:, (norm_D > 0)] /= norm_D[norm_D > 0]

    def AIHT(self, x) -> torch.FloatTensor:
        """
        Compute the sparse code of input signals using the Accelerated Iterative Hard Thresholding (AIHT) algorithm.

        :param x: Input batch of signals.
        :return: Output batch of sparse codes.
        """

        return _AIHT(x, self.D, self.n_atoms, self.sparsity, 0.001, device=self.device)[0]

    def sylvester(self, A, B, C):
        m = B.shape[-1]
        n = A.shape[-1]
        R, U = torch.linalg.eig(A)
        S, V = torch.linalg.eig(B)
        F = torch.linalg.solve(U, (C + 0j) @ V)
        W = R[..., :, None] - S[..., None, :]
        Y = F / W
        X = U[..., :n, :n] @ Y[..., :n, :m] @ torch.linalg.inv(V)[..., :m, :m]
        return X.real if all(torch.isreal(x.flatten()[0])
                             for x in [A, B, C]) else X

    def _get_adjacency_matrix(self) -> torch.FloatTensor:
        """
        Compute adjacency matrix of estimated graph.
        :return: adjacency matrix.
        """
        if self.graph_learner:
            W = - self.graph_learner.Phi.detach().clone()
            W = (W + W.T) / 2.
            W.fill_diagonal_(0)
            return W
        else:
            return None

    def _get_laplacian_matrix(self) -> torch.FloatTensor:
        """
        Compute Laplacian matrix of estimated graph.
        :return: Laplacian matrix.
        """
        if self.graph_learner:
            W = self._get_adjacency_matrix()
            Deg = torch.diag(torch.sqrt(W.__abs__().sum(dim=0)))
            L = Deg - W
            return L
        else:
            return None

    def forward(self, x):

        # Graph learning step
        if self.graph_learn == 'bcd' or self.graph_learn == 'gama' or self.graph_learn == 'glama':
            self.graph_learner.forward(x, self.D, self.alpha)

        y = self.AIHT(x)

        # Update the dictionary
        if self.graph_learn == 'none':
            self.D = x.T @ torch.linalg.pinv(y).T
        elif self.graph_learn == 'known':
            self.D = self.sylvester(self.alpha * self.L, -y.T @ y, x.T @ y)
        elif self.graph_learn == 'bcd' or self.graph_learn == 'gama' or self.graph_learn == 'glama':
            L_n = self._get_laplacian_matrix()
            if self.graph:
                L0 = torch.from_numpy(self.graph.L.todense()).float().to(self.device)
                L_n *= torch.linalg.norm(L0) / torch.linalg.norm(L_n)
            self.D = self.sylvester(self.alpha * L_n, -y.T @ y, x.T @ y)

    def predict(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Predict the input x using the stored dictionary.
        :param x: input signal.
        :return: signal representation.
        """
        if self.fit_intercept:
            # Remove mean
            x_mean = x.mean(dim=1).unsqueeze(1)
            x -= x_mean

        # Compute sparse code
        y = self.AIHT(x)

        # Return prediction
        return y @ self.D.T
