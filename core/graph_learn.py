import time

import numpy as np
import torch
from qpth.qp import QPFunction
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from torch.autograd import Variable

from core.utils import duplication_matrix
from torch.nn.functional import softshrink


def _dense_equality_matrix(dim: int, m: int, row_pos: torch.Tensor,
                           diag_pos: torch.Tensor, device: str) -> torch.Tensor:
    """Dense (dim+1) x m equality matrix A (row sums + trace = vol).

    Built by scatter from the index structure instead of converting the sparse
    tensor with ``to_dense`` (which is unreliable on some sparse-tensor builds).
    """
    A_dense = torch.zeros(dim + 1, m, dtype=torch.float64, device=device)
    A_dense[:dim].scatter_(1, row_pos, 1.0)
    A_dense[dim, diag_pos] = 1.0
    return A_dense

class BCD:

    def __init__(
            self,
            cov_matrix: torch.FloatTensor,
            device: str = 'cpu'
    ):
        super().__init__()

        self.device = device

        self.S = cov_matrix.to(device)
        self.num_nodes = cov_matrix.shape[0]
        self.Phi = torch.diag(1 / torch.diag(self.S))

    def forward(self, S_d: torch.FloatTensor, tolerance: float, max_iter: int):
        for i in range(self.num_nodes):
            P = self.Phi.detach().clone()
            P = torch.vstack((P[:i], P[(i + 1):]))
            P = torch.hstack((P[:, :i], P[:, (i + 1):]))

            k_i = S_d[:, i].detach().clone()
            k_i = torch.cat((k_i[:i], k_i[(i + 1):]))
            k_ii = S_d[i, i].detach().clone()

            Q = 2 * P
            p = 2 * P @ k_i
            G = - torch.eye(self.num_nodes - 1, device=self.device)
            h = torch.zeros(self.num_nodes - 1, device=self.device)
            e = Variable(torch.Tensor())
            x = QPFunction(verbose=0)(Q, p, G, h, e, e).squeeze()

            p_i = - (1 / k_ii) * P @ (k_i + x)
            p_ii = (1 / k_ii) * (1 - p_i.dot(k_i))
            p_i = torch.cat((p_i[:i], p_ii.unsqueeze(0), p_i[i:]))
            self.Phi[i] = p_i
            self.Phi[:, i] = p_i


class GAMA:

    def __init__(
            self,
            cov_matrix: torch.FloatTensor,
            gamma: float,
            rho: float,
            mode: str = "dl",
            device: str = 'cpu'
    ):
        super().__init__()

        self.S = cov_matrix
        self.num_nodes = cov_matrix.shape[0]
        self.Phi = torch.zeros((self.num_nodes, self.num_nodes), device=device)
        if mode == "dl":
            self.Gamma = gamma * torch.eye(self.num_nodes, device=device)
        else:
            self.Gamma = self.S + gamma * torch.eye(self.num_nodes, device=device)

        self.gamma = gamma
        self.rho = rho

    def soft_thresh(self, X, theta):
        return torch.minimum(X + theta, torch.tensor(0)) + torch.maximum(X - theta, torch.tensor(0))

    def forward(self, S_d: torch.FloatTensor, tolerance: float, max_iter: int):

        Omega_n = torch.linalg.inv(self.Gamma)
        self.Phi = (1. / self.rho) * self.soft_thresh(self.rho * Omega_n - S_d + self.Gamma, self.gamma)
        self.Gamma += self.rho * (Omega_n - self.Phi)
        self.rho = torch.abs(torch.linalg.eigvalsh(self.Gamma)).sort()[0][0] ** 2


class GLAMA:

    def __init__(
            self,
            cov_matrix: torch.FloatTensor,
            gamma: float,
            rho: float,
            eta: float,
            mode: str = "dl",
            device: str = 'cpu'
    ):
        super().__init__()

        self.device = device

        self.S = cov_matrix
        self.num_nodes = cov_matrix.shape[0]
        self.L = torch.zeros((self.num_nodes, self.num_nodes), device=device)
        if mode == "dl":
            self.Gamma = gamma * torch.eye(self.num_nodes, device=device)
        else:
            self.Gamma = self.S + gamma * torch.eye(self.num_nodes, device=device)
        self.Theta = torch.zeros((self.num_nodes, self.num_nodes), device=device)
        self.lambd = torch.zeros((self.num_nodes,), device=device)

        self.gamma = gamma
        self.rho = rho
        self.eta = eta

    def asym_soft_thresh(self, X, theta_1, theta_2):
        return torch.minimum(X + theta_1, torch.tensor(0)) + torch.maximum(X - theta_2, torch.tensor(0))

    def forward(self, S_d: torch.FloatTensor, tolerance: float, max_iter: int):

        for _ in range(max_iter):
            Omega = torch.linalg.inv(self.Gamma)
            B = self.rho * Omega - S_d + self.Gamma
            L_old = self.L.detach().clone()
            L_diag = (1. / self.rho) * torch.diag(B)
            L_neg = (1. / self.rho) * self.asym_soft_thresh(
                B,
                theta_1=self.gamma,
                theta_2=self.gamma
            )
            L_neg[torch.where(L_neg > 0.)[0], torch.where(L_neg > 0.)[1]] = 0.
            L_neg.fill_diagonal_(0)
            L_pos = (1. / (self.rho + self.eta)) * self.asym_soft_thresh(
                B,
                theta_1=self.gamma,
                theta_2=self.gamma + self.Theta
            )
            L_pos[torch.where(L_pos <= 0.)[0], torch.where(L_pos <= 0.)[1]] = 0.
            L_pos.fill_diagonal_(0)
            self.L = torch.diag(L_diag) + L_pos + L_neg
            G = torch.maximum(torch.tensor(0), self.L)
            G.fill_diagonal_(0)
            self.Gamma += self.rho * (Omega - self.L)
            self.Theta += self.eta * G
            self.Theta.fill_diagonal_(0)
            self.rho = torch.sort(torch.abs(torch.linalg.eigvalsh(self.Gamma)))[0][0] ** 2
            self.eta = (1 + torch.sqrt(torch.tensor(2))) * self.rho
            # self.eta = 0.001

            err = torch.linalg.norm(L_old - self.L)
            if err < tolerance:
                break

class HomogeneousSelfDualEmbedding:

    def __init__(
            self,
            dim: int,
            beta: float,
            device: str = 'cpu'
    ):
        super().__init__()

        self.device = device

        dim = 3

        self.dim = dim
        self.dim_min = int(dim * (dim - 1) / 2)
        self.dim_max = int(dim * (dim + 1) / 2)

        gamma1 = (dim - 2) / (1 + 4 * beta) + 1 / (1 + 2 * beta)
        gamma2 = 1 / (1 + 4 * beta)
        delta1 = 1 / gamma1
        delta2 = gamma2 / ((gamma1 ** 2) * (1 + gamma2 * dim / gamma1))
        self.mu1 = 1 / (2 + delta1)
        self.mu2 = (self.mu1 ** 2) * (delta1 - 4 * delta2) / (1 - self.mu1 * (delta1 - 4 * delta2) * self.dim_min)

        self.L = torch.zeros((dim, dim), device=device)

        index_dup, value_dup = duplication_matrix(dim)
        self.M_dup = torch.sparse_coo_tensor(
            index_dup, value_dup,
            size=(int(dim * (dim + 1) / 2), dim ** 2),
            device=device
        ).T.coalesce()
        index_dup = self.M_dup.indices().cpu()
        value_dup = self.M_dup.values().cpu()

        # Matrix A
        index_eq = torch.tensor([
            [i for i in range(dim) for j in range(dim)],
            [i for i in range(dim ** 2)]
        ])
        value_eq = torch.tensor([1 for i in range(dim ** 2)]).float()
        C_eq = torch.sparse_coo_tensor(indices=index_eq, values=value_eq)
        A = C_eq @ self.M_dup.cpu()
        A0 = torch.hstack([A, torch.sparse_coo_tensor(size=(A.shape[0], 1))])


        index_Q = A.indices()
        index_Q[0] += int(dim * (dim + 1) / 2) + 1
        value_Q = A.values()

        # Matrix B
        ineq_tril_indices = torch.tril_indices(dim, dim, -1)
        col_indices_ineq = torch.tensor(
            [ineq_tril_indices[0, i] + torch.arange(dim, dim - ineq_tril_indices[1, i], -1).sum() - \
             ineq_tril_indices[1, i] for i in range(int(dim * (dim - 1) / 2))]
        )
        index_ineq = torch.tensor([
            [i for i in range(int(dim * (dim - 1) / 2))],
            [col_indices_ineq[i] for i in range(int(dim * (dim - 1) / 2))]
        ])
        # index_ineq = torch.hstack((index_ineq, torch.tensor([[index_ineq[0].max() + 1], [index_ineq[1].max() + 1]])))
        # index_ineq[0] += int(dim * (dim + 1) / 2) + 1 + dim
        value_ineq = torch.tensor([1 for i in range(int(dim * (dim - 1) / 2))])
        # value_ineq = torch.cat((value_ineq, torch.tensor([-1])))
        B = torch.sparse_coo_tensor(
            indices=index_ineq,
            values=value_ineq,
            size=(self.dim_min, self.dim_max),
            device=self.device
        ).coalesce()
        B0 = torch.vstack([
            torch.hstack([B, torch.sparse_coo_tensor(size=(B.shape[0], 1), device=self.device)]),
            torch.hstack([
                torch.sparse_coo_tensor(size=(1, B.shape[1]), device=self.device),
                torch.sparse_coo_tensor(indices=[[0], [0]], values=[-1], device=self.device)
            ])
        ])

        C0 = torch.vstack([
            torch.hstack([
                -torch.sqrt(torch.tensor(2 * beta)) * self.M_dup,
                torch.sparse_coo_tensor(size=(self.M_dup.shape[0], 1), device=self.device)
            ]),
            torch.hstack([
                torch.sparse_coo_tensor(size=(2, self.M_dup.shape[1]), device=self.device),
                torch.sparse_coo_tensor(indices=[[0, 1], [0, 0]], values=[1, -1], device=self.device)
            ])
        ])

        index_Q = torch.hstack((index_Q, index_ineq))
        value_Q = torch.cat((value_Q, value_ineq))

        # Matrix A_c
        self.index_A_c = torch.hstack((index_dup, torch.tensor([[index_dup[0].max() + 1], [index_dup[1].max() + 1]])))
        self.index_A_c[0] += index_Q[0].max() + 1
        value_A_c = - torch.cat((value_dup, torch.tensor([-1])))
        index_Q = torch.hstack((index_Q, self.index_A_c))
        value_Q = torch.cat((value_Q, value_A_c))

        self.index_c_c = torch.tensor([
            [index_Q[0].max() + 1 for i in range(self.index_A_c[1].max() + 1)],
            [i for i in range(self.index_A_c[1].max() + 1)]
        ])
        value_c_c = torch.tensor([0 for i in range(self.index_A_c[1].max() + 1)])
        value_c_c[self.index_A_c[1].max()] = - 1

        index_Q = torch.hstack((index_Q, self.index_c_c))
        value_Q = torch.cat((value_Q, value_c_c))

        index_b = torch.tensor([
            [index_Q[0].max() + 1 for i in range(2)],
            [i + dim + int(dim * (dim + 1) / 2) + dim ** 2 + int(dim * (dim - 1) / 2) + 2 for i in range(2)]
        ])
        value_b = torch.tensor([1 for i in range(2)])

        index_Q = torch.hstack((index_Q, index_b))
        value_Q = torch.cat((value_Q, value_b))

        self.index_c = torch.tensor([
            [index_Q[0].max() for i in range(self.index_A_c[1].max() + 1)],
            [i for i in range(self.index_A_c[1].max() + 1)]
        ])
        value_c = torch.tensor([1 for i in range(self.index_A_c[1].max() + 1)])

        index_Q = torch.hstack((index_Q, self.index_c))
        value_Q = torch.cat((value_Q, value_c))
        value_Q *= -1

        self.Q = torch.sparse_coo_tensor(
            indices=index_Q,
            values=value_Q,
            size=(index_Q.max() + 1, index_Q.max() + 1),
            device=self.device
        ).coalesce()

        self.d_idx = torch.zeros(dim, dtype=int, device=device)
        accum_idx = 0
        for i in range(dim):
            self.d_idx[i] = accum_idx
            accum_idx += dim - i

        vw_dim = self.Q.shape[0]
        self.v = torch.zeros(vw_dim, device=device)
        self.v_tilde = torch.zeros(vw_dim, device=device)
        self.w = torch.zeros(vw_dim, device=device)

    def forward(self, c_k: float, H_k: torch.FloatTensor, tolerance: float, max_iter: int):

        # Multiply M_dup by sqrt(c_k) in A_c
        index_Q_A_c_row = (self.Q.indices()[0] >= self.index_A_c[0].min()) * (self.Q.indices()[0] <= self.index_A_c[0].max() - 1)
        index_Q_A_c_col = (self.Q.indices()[1] >= self.index_A_c[1].min()) * (self.Q.indices()[1] <= self.index_A_c[1].max() - 1)
        index_Q_A_c = index_Q_A_c_row * index_Q_A_c_col
        value_Q = self.Q.values()
        value_Q[index_Q_A_c] *= torch.sqrt(2 * c_k) / value_Q[index_Q_A_c][0]

        h_k = H_k.reshape((self.dim ** 2,))
        q = - self.M_dup.T @ h_k
        c = torch.cat((q, torch.tensor(1, device=self.device).reshape(-1)))

        # Replace c in R
        index_Qc_row = (self.Q.indices()[0] >= self.index_c[0].min()) * (self.Q.indices()[0] <= self.index_c[0].max())
        index_Qc_col = (self.Q.indices()[1] >= self.index_c[1].min()) * (self.Q.indices()[1] <= self.index_c[1].max())
        index_Qc = index_Qc_row * index_Qc_col
        value_Q[index_Qc] = -c

        R = self.Q - self.Q.T
        R = R.coalesce()
        IR = torch.eye(self.Q.shape[0], device=self.device).to_sparse_coo() + R
        IR = IR.coalesce()
        IR_sp = csr_matrix(
            (IR.values().cpu().numpy(), (IR.indices()[0].cpu().numpy(), IR.indices()[1].cpu().numpy())),
            shape=(R.shape[0], R.shape[0])
        )
        B = splu(IR_sp, permc_spec='MMD_ATA')

        for n in range(max_iter):

            v_old = self.v.detach().clone()

            b = self.v + self.w
            # r = b - IR @ self.v_tilde
            # p = r.detach().clone()
            # if torch.linalg.norm(r) > 0:
            #     a_k = torch.linalg.norm(r) ** 2 / (torch.linalg.norm(p) ** 2 + torch.linalg.norm(R @ p) ** 2)
            #     b_k = 1 - a_k
            #     for k in range(R.shape[0]):
            #         self.v_tilde += a_k * p
            #         r -= a_k * IR @ p
            #         p = r - b_k * p
            #         a_k = torch.linalg.norm(r) ** 2 / (torch.linalg.norm(p) ** 2 + torch.linalg.norm(R @ p) ** 2)
            #         b_k = 1 - a_k
            #         # print(torch.linalg.norm(r))
            #         if torch.linalg.norm(r) < 1e-5: break

            self.v_tilde = torch.from_numpy(B.solve(b.cpu().numpy())).cuda()
            # print(torch.linalg.norm(self.v_tilde - torch.from_numpy(v_tilde_cp).cuda()))

            ix_min = 0
            ix_max = ix_min + int(self.dim * (self.dim + 1) / 2 + 1)
            r_k = self.w[ix_min:ix_max]
            x_tilde = self.v_tilde[ix_min:ix_max]
            y_k = x_tilde - r_k
            lambd = (1 / self.dim) * (y_k[self.d_idx].sum() - 1000)
            x = y_k
            x[self.d_idx] -= lambd

            imu_min = ix_max
            imu_max = imu_min + self.dim
            mu = self.v_tilde[imu_min:imu_max]

            inu_min = imu_max
            inu_max = inu_min + int(self.dim * (self.dim - 1) / 2 + 1)
            t_k = self.w[inu_min:inu_max]
            nu_tilde = self.v_tilde[inu_min:inu_max]
            nu = torch.maximum(nu_tilde - t_k, torch.tensor(0))

            ieta_min = inu_max
            ieta_max = ieta_min + self.dim ** 2 + 2
            u_k = self.w[ieta_min:ieta_max]
            eta_tilde = self.v_tilde[ieta_min:ieta_max]
            x_cone_k = (eta_tilde - u_k)[:self.dim ** 2 + 2 - 1]
            t_cone_k = (eta_tilde - u_k)[self.dim ** 2 + 2 - 1]
            norm_x_cone_k = torch.linalg.norm(x_cone_k)
            if torch.linalg.norm(x_cone_k).item() == 0:
                eta = torch.cat((x_cone_k, t_cone_k.reshape(1,)))
            else:
                eta = (t_cone_k + norm_x_cone_k) / (2 * norm_x_cone_k) * torch.cat((x_cone_k, norm_x_cone_k.reshape(1,)))

            itau_min = ieta_max
            itau_max = itau_min + 1
            kappa = self.w[itau_min:itau_max]
            tau_tilde = self.v_tilde[itau_min:itau_max]
            tau = torch.maximum(tau_tilde - kappa, torch.tensor(0)).reshape(1,)

            self.v = torch.cat((x, mu, nu, eta, tau))
            self.w = self.w - self.v_tilde + self.v

            # end_i = time.time()
            # print("ADMM iteration: %f" % (end_i - start_i))

            err = torch.linalg.norm(self.v - v_old) / self.v.shape[0]
            # print(err)
            if(err < tolerance):
                break

        l = self.M_dup @ x[:len(x) - 1]
        self.L = l.reshape(self.dim, self.dim).to(self.device)

class StationaryADMM:

    def __init__(
            self,
            dim: int,
            L: torch.Tensor = None,
            mode: str = 'optimized',
            device: str = 'cpu'
    ):
        super().__init__()

        self.device = device
        self.mode = mode

        self.dim = dim
        if L is None:
            self.L = torch.eye(dim, device=device)
        else:
            self.L = L

        index_dup, value_dup = duplication_matrix(dim)
        self.M_dup = torch.sparse_coo_tensor(index_dup, value_dup, size=(int(dim * (dim + 1) / 2), dim ** 2)).T.coalesce().to(device)
        index_dup = self.M_dup.indices()
        value_dup = self.M_dup.values()

        # Matrix A
        index_eq = torch.tensor([
            [i for i in range(dim) for j in range(dim)],
            [i for i in range(dim ** 2)]
        ])
        # index_tr = torch.tensor([
        #     [dim for i in range(dim)],
        #     [i * (dim + 1) for i in range(dim)]
        # ])
        # value_tr = torch.tensor([1 for i in range(dim)])
        # index_eq = torch.hstack((index_eq, index_tr))
        value_eq = torch.tensor([1 for i in range(dim ** 2)]).float()
        # value_eq = torch.cat((value_eq, value_tr))
        C_eq = torch.sparse_coo_tensor(indices=index_eq, values=value_eq).to(device)
        self.A = C_eq @ self.M_dup
        self.P = self.M_dup.T @ self.M_dup

        # Q_inv = torch.sparse_coo_tensor(indices=self.P.indices(), values=1 / (self.P.values() + self.rho))
        #
        # a = (self.dim - 2) / torch.tensor(2 + self.rho) + 1 / torch.tensor(1 + self.rho)
        # b = 1 / torch.tensor(2 + self.rho)
        #
        # c = - 1 / a
        # d = b / (a * (a + b * self.dim))
        #
        # R_inv = c * torch.eye(dim) + d * torch.ones(dim, dim)
        # M_inv = Q_inv.to_dense() - self.rho * Q_inv @ self.A.T @ R_inv @ self.A @ Q_inv

        # Matrix B
        ineq_tril_indices = torch.tril_indices(dim, dim, -1)
        col_indices_ineq = torch.tensor(
            [ineq_tril_indices[0, i] + torch.arange(dim, dim - ineq_tril_indices[1, i], -1).sum() - \
             ineq_tril_indices[1, i] for i in range(int(dim * (dim - 1) / 2))]
        )
        self.index_ineq = torch.tensor([col_indices_ineq[i] for i in range(int(dim * (dim - 1) / 2))]).sort().values
        # index_ineq = torch.tensor([
        #     [i for i in range(int(dim * (dim - 1) / 2))],
        #     [col_indices_ineq[i] for i in range(int(dim * (dim - 1) / 2))]
        # ])
        # index_ineq = torch.hstack((index_ineq, torch.tensor([[index_ineq[0].max() + 1], [index_ineq[1].max() + 1]])))
        # value_ineq = torch.tensor([1 for i in range(int(dim * (dim - 1) / 2))])
        # value_ineq = torch.cat((value_ineq, torch.tensor([-1])))
        # self.B = torch.sparse_coo_tensor(
        #     indices=index_ineq,
        #     values=value_ineq,
        #     size=(int(dim * (dim - 1) / 2), int(dim * (dim + 1) / 2))
        # )

        self.d_idx = torch.zeros(dim, dtype=int)
        accum_idx = 0
        for i in range(dim):
            self.d_idx[i] = accum_idx
            accum_idx += dim - i

        self.u = torch.zeros(int(self.dim * (self.dim + 1) / 2), device=device)
        self.x = torch.zeros(int(self.dim * (self.dim + 1) / 2), device=device)
        self.z = torch.zeros(int(self.dim * (self.dim + 1) / 2), device=device)

    @staticmethod
    def pos_threshold(x, gamma):

        return torch.minimum(x, torch.tensor(0)) + torch.maximum(x, gamma)

    def forward(self, b_t: float, F_t: torch.FloatTensor, tolerance: float, max_iter: int):

        rho = b_t
        h_k = F_t.reshape((self.dim ** 2,))
        q = - self.M_dup.T @ h_k

        if self.mode == 'normal':
            Mk = torch.hstack((
                torch.vstack((torch.sparse_coo_tensor(indices=self.P.indices(), values=(b_t * self.P.values() + rho)), self.A)),
                torch.vstack((self.A.T, torch.sparse_coo_tensor(size=(self.dim, self.dim), device=self.device)))
            ))
            Mk_inv = torch.linalg.inv(Mk.to_dense())
        else:
            Rk_inv = torch.sparse_coo_tensor(indices=self.P.indices(), values=1 / (b_t * self.P.values() + rho))

            ak = (self.dim - 2) / (2 * b_t + rho) + 1 / (b_t + rho)
            bk = 1 / (2 * b_t + rho)

        # max_iter = 100
        # tol = 1e-4
        for _ in range(max_iter):

            x_old = self.x.detach().clone()

            if self.mode == 'normal':
                self.x = Mk_inv @ torch.cat((rho * (self.z + self.u) - q, torch.zeros(self.dim, device=self.device)))
                self.x = self.x[:int(self.dim * (self.dim + 1) / 2)]
            else:
                r_n = Rk_inv @ (rho * (self.z + self.u) - q)

                x1_n = - (1 / ak) * Rk_inv @ self.A.T @ self.A @ r_n
                x2_n = (bk / (ak ** 2 + ak * bk * self.dim)) * (self.A @ r_n).sum() * (Rk_inv @ self.A.T).sum(1)

                self.x = r_n + x1_n + x2_n

            v = self.x - self.u
            self.z[self.index_ineq] = torch.minimum(v[self.index_ineq], torch.tensor(0.))
            lambd = (1 / self.dim) * (v[self.d_idx].sum() - 10 * self.dim)
            self.z[self.d_idx] = v[self.d_idx] - lambd
            self.u += self.z - self.x

            # end_i = time.time()
            # print("ADMM iteration: %f" % (end_i - start_i))

            err = torch.linalg.norm(self.x - x_old) / self.x.shape[0]
            if(err < tolerance):
                break

        l = self.M_dup @ self.x
        self.L = l.reshape(self.dim, self.dim).to(self.device)


class ProjADMM:

    def __init__(
            self,
            dim: int,
            L: torch.Tensor = None,
            vol: float = 100.,
            mode: str = 'optimized',
            device: str = 'cpu'
    ):
        super().__init__()

        self.device = device
        self.mode = mode
        self.rho = 2

        self.dim = dim
        if L is None:
            self.L = torch.eye(dim, device=device)
        else:
            self.L = L

        index_dup, value_dup = duplication_matrix(dim)
        self.M_dup = torch.sparse_coo_tensor(index_dup, value_dup, size=(int(dim * (dim + 1) / 2), dim ** 2)).T.coalesce().to(device)

        # Matrix A
        index_eq1 = torch.tensor([
            [i for i in range(dim) for j in range(dim)],
            [i for i in range(dim ** 2)]
        ])
        value_eq1 = torch.tensor([1 for i in range(dim ** 2)]).float()
        C1_eq = torch.sparse_coo_tensor(indices=index_eq1, values=value_eq1).to(device)
        self.A1 = C1_eq @ self.M_dup

        index_eq2 = torch.tensor([
            [0 for _ in range(dim)],
            [i * (dim + 1) for i in range(dim)]
        ])
        value_eq2 = torch.tensor([1 for _ in range(dim)]).float()
        # index_eq = torch.hstack((index_eq1, index_eq2))
        # value_eq = torch.cat((value_eq1, value_eq2))
        c2_eq = torch.sparse_coo_tensor(indices=index_eq2, values=value_eq2).to(device)
        self.a2 = c2_eq @ self.M_dup

        self.A = torch.vstack((self.A1, self.a2))
        self.P = self.M_dup.T @ self.M_dup
        self.Rinv = torch.sparse_coo_tensor(
            indices=self.P.indices(),
            values=1. / (self.P.values() + self.rho)
        ).to(device)

        # Dense equality matrix A (built manually to avoid the unreliable
        # ``torch.sparse.Tensor.to_dense()`` implementation on this PyTorch
        # build).  It is only (dim+1) x m, so keeping it dense is cheap and
        # lets us pre-compute (A R^{-1} A^T)^{-1} exactly.
        m = dim * (dim + 1) // 2
        A_dense = torch.zeros((dim + 1, m), dtype=torch.float64, device=device)
        for i in range(dim):
            for j in range(dim):
                if j >= i:
                    pos = i * dim + j - i * (i + 1) // 2
                else:
                    pos = j * dim + i - j * (j + 1) // 2
                A_dense[i, pos] = 1.0
        for i in range(dim):
            A_dense[dim, i * (2 * dim - i + 1) // 2] = 1.0

        Rinv_c = self.Rinv.coalesce()
        Rinv_dense = torch.zeros(m, dtype=torch.float64, device=device)
        Rinv_dense[Rinv_c.indices()[1]] = Rinv_c.values().to(torch.float64)

        ARinvAT = (A_dense * Rinv_dense) @ A_dense.T
        self._A_Rinv_AT_inv = torch.from_numpy(
            np.linalg.inv(ARinvAT.detach().cpu().numpy())
        ).to(device=device, dtype=torch.float32)

        # Matrix B
        index_ineq = torch.tensor([
            [i for i in range(int(dim * (dim - 1) / 2))],
            [j + 1 + i * dim for i in range(dim - 1) for j in range(i, dim - 1)]
        ])
        value_ineq = torch.tensor([1 for _ in range(int(dim * (dim - 1) / 2))]).float()
        C_ineq = torch.sparse_coo_tensor(
            indices=index_ineq,
            values=value_ineq,
            size=(int(dim * (dim - 1) / 2), dim ** 2)
        ).to(device)
        self.B = C_ineq @ self.M_dup

        # Vector b
        self.b = torch.sparse_coo_tensor(
            indices=torch.tensor([[dim]]),
            values=torch.tensor([vol]).float()
        ).to(device)

        self.idx_off = self.B.indices()[1]
        self.idx_on = self.a2.indices()[1]

        # Outer loop variables
        self.l = torch.zeros(int(self.dim * (self.dim + 1) / 2), device=device)
        self.z = torch.zeros(int(self.dim * (self.dim + 1) / 2), device=device)
        self.v = torch.zeros(int(self.dim * (self.dim + 1) / 2), device=device)

        # Constants
        self.delta1 = ((2 + self.rho)**2 - self.dim * (1 + self.rho) * (2 + self.rho)) / (self.dim * (self.dim + self.rho) * ((2 * self.dim + self.rho) * (1 + self.rho) - (2 + self.rho)))
        #self.delta1 = (2 + self.rho) / (self.dim * (1 + self.rho) - self.rho)
        self.delta2 = -((2 + self.rho) * (1 + self.rho)) / (self.dim * ((2 * self.dim + self.rho) * (1 + self.rho) - (2 + self.rho)))
        # self.delta2 = (1 + self.rho) * self.delta1 / (2 * self.dim * (1 + self.rho) - self.rho)
        self.delta3 = ((2 * self.dim + self.rho) * (1 + self.rho)**2) / (self.dim * ((2 * self.dim + self.rho) * (1 + self.rho) - (2 + self.rho)))
        # self.delta3 = self.delta1 - self.delta2 * self.dim
        self.s = self.dim - self.delta1 * self.dim + self.delta2 * (self.dim ** 2)
        self.th = self.P.values()

        if self.mode == 'naive':
            # Dense KKT matrix M = [[P + rho I, A^T], [A, 0]], built
            # analytically (to_dense on sparse tensors is unreliable on some
            # sparse-tensor builds).
            m = int(self.dim * (self.dim + 1) / 2)
            diag_pos = torch.tensor(
                [i * (2 * dim - i + 1) // 2 for i in range(dim)],
                dtype=torch.long, device=device
            )
            row_pos = torch.zeros((dim, dim), dtype=torch.long, device=device)
            for i in range(dim):
                for j in range(dim):
                    pos = (i * dim + j - i * (i + 1) // 2 if j >= i
                           else j * dim + i - j * (j + 1) // 2)
                    row_pos[i, j] = pos
            A_dense = _dense_equality_matrix(
                dim, m, row_pos, diag_pos, device
            ).to(dtype=torch.float32)
            P_full = torch.full((m,), 2.0, dtype=torch.float32, device=device)
            P_full[diag_pos] = 1.0
            self.M = torch.zeros((m + dim + 1, m + dim + 1),
                                 dtype=torch.float32, device=device)
            self.M[:m, :m] = torch.diag(P_full + self.rho)
            self.M[:m, m:] = A_dense.T
            self.M[m:, :m] = A_dense

    @staticmethod
    def pos_threshold(x, gamma):

        return torch.minimum(x, torch.tensor(0)) + torch.maximum(x, gamma)

    def forward(self,
        F_t: torch.FloatTensor,
        # delta: float,
        # tolerance_out: float,
        # max_iter_out: int,
        tolerance: float,
        max_iter: int,
        callback=None,
        max_time: float = None,
    ):

        f_t = F_t.reshape((self.dim ** 2,))
        q = self.M_dup.T @ f_t

        if max_time is not None:
            max_iter = max(max_iter, 10 ** 9)

        t0 = time.perf_counter() if max_time is not None else None

        # Projection step
        self.iter = 0
        if self.mode == 'naive':
            for _ in range(max_iter):
                if max_time is not None and (time.perf_counter() - t0) > max_time:
                    break
                l_old = self.l.detach().clone()
                l_mu = torch.linalg.solve(self.M, torch.cat((
                    q + self.rho * (self.z + self.v),
                    self.b.to_dense()
                )))
                self.l = l_mu[:int(self.l.shape[0])]
                w = self.l - self.v
                self.z[self.idx_on] = w[self.idx_on]
                self.z[self.idx_off] = torch.minimum(w[self.idx_off], torch.tensor(0.))
                self.v += self.z - self.l
                err = torch.linalg.norm(self.l - l_old) / self.l.shape[0]
                self.iter = _ + 1
                if callback is not None:
                    L_iter = (self.M_dup @ self.l).reshape(self.dim, self.dim).to(self.device)
                    callback(self.iter, L_iter)
                if max_time is None and err < tolerance:
                    break

        if self.mode == 'optimized':
            for _ in range(max_iter):
                if max_time is not None and (time.perf_counter() - t0) > max_time:
                    break
                l_old = self.l.detach().clone()
                r = q + self.rho * (self.z + self.v)
                s = self.A @ self.Rinv @ r - self.b
                # Dual variable update using the precomputed exact inverse of
                # A R^{-1} A^T (the previous analytic formula did not match the
                # true inverse, which left equality constraints unsatisfied).
                t = self._A_Rinv_AT_inv @ s
                self.l = self.Rinv @ (r - self.A.T @ t)
                w = self.l - self.v
                self.z[self.idx_on] = w[self.idx_on]
                self.z[self.idx_off] = torch.minimum(w[self.idx_off], torch.tensor(0.))
                self.v += self.z - self.l
                err = torch.linalg.norm(self.l - l_old) / self.l.shape[0]
                self.iter = _ + 1
                if callback is not None:
                    L_iter = (self.M_dup @ self.l).reshape(self.dim, self.dim).to(self.device)
                    callback(self.iter, L_iter)
                if max_time is None and err < tolerance:
                    break

        l = self.M_dup @ self.l
        self.L = l.reshape(self.dim, self.dim).to(self.device)


class ProjIPM:
    """Interior-point Laplacian projection solver with sparse tensors.

    ``D`` and ``A`` are kept as sparse COO tensors (constructed the same way
    as in ``ProjADMM``) and the inner Newton step uses sparse matrix-vector
    products.  This avoids building the dense duplication matrix in memory.

    With ``mode='naive'`` the inner Newton step instead assembles the dense
    KKT system ``[[H, A^T], [A, 0]]`` and solves it directly with
    ``torch.linalg.solve``, mirroring the naive mode of ``ProjADMM``.
    """

    def __init__(
            self,
            dim: int,
            L: torch.Tensor = None,
            vol: float = 100.,
            t0: float = 1.0,
            mu: float = 2.0,
            eps_out: float = 1e-6,
            eps_in: float = 1e-6,
            max_inner_iter: int = 100,
            alpha: float = 0.25,
            beta_ls: float = 0.5,
            mode: str = 'optimized',
            device: str = 'cpu'
    ):
        self.device = device
        self.dim = dim
        self.mode = mode
        self.vol = float(vol)
        self.t = float(t0)
        self.mu = float(mu)
        self.eps_out = float(eps_out)
        self.eps_in = float(eps_in)
        self.max_inner_iter = int(max_inner_iter)
        self.alpha = float(alpha)
        self.beta_ls = float(beta_ls)
        self.D_minus = int(dim * (dim - 1) / 2)

        # Sparse duplication matrix D: vec(L) = D l (same construction as
        # ProjADMM, but with float64 values).
        index_dup, value_dup = duplication_matrix(dim)
        self.m = dim * (dim + 1) // 2
        self.D = torch.sparse_coo_tensor(
            index_dup, value_dup.to(dtype=torch.float64),
            size=(self.m, dim * dim),
            device=device
        ).T.coalesce()

        # Sparse equality matrix A: row sums zero + trace = vol.
        # Built directly in vech space to avoid the slow sparse @ sparse
        # product used in ProjADMM.
        a1_rows, a1_cols = [], []
        for i in range(dim):
            for j in range(dim):
                if j >= i:
                    pos = i * dim + j - i * (i + 1) // 2
                else:
                    pos = j * dim + i - j * (j + 1) // 2
                a1_rows.append(i)
                a1_cols.append(pos)
        self.A1 = torch.sparse_coo_tensor(
            indices=torch.tensor([a1_rows, a1_cols], dtype=torch.long,
                                 device=device),
            values=torch.ones(dim * dim, dtype=torch.float64, device=device),
            size=(dim, self.m)
        ).coalesce()

        a2_rows = [0] * dim
        a2_cols = [i * (2 * dim - i + 1) // 2 for i in range(dim)]
        self.a2 = torch.sparse_coo_tensor(
            indices=torch.tensor([a2_rows, a2_cols], dtype=torch.long,
                                 device=device),
            values=torch.ones(dim, dtype=torch.float64, device=device),
            size=(1, self.m)
        ).coalesce()
        self.A = torch.vstack((self.A1, self.a2)).coalesce()

        self.b = torch.zeros((dim + 1,), dtype=torch.float64, device=device)
        self.b[dim] = self.vol

        # Diagonal entries are allowed to be positive; off-diagonals must be <=0.
        self.diag_mask = torch.zeros(self.m, dtype=torch.bool, device=device)
        self.diag_mask[self.a2.indices()[1]] = True
        self.off_mask = ~self.diag_mask

        # P = D^T D  (diagonal with 1's for diagonals, 2's for off-diags).
        self.P_diag = self.diag_mask.to(torch.float64) + 2.0 * self.off_mask.to(torch.float64)

        # Precompute index structures used to build the Schur complement S
        # and A^T temp without forming dense matrices in the inner loop.
        self._diag_pos = torch.tensor(
            [i * (2 * dim - i + 1) // 2 for i in range(dim)],
            dtype=torch.long, device=device
        )
        self._row_pos = torch.zeros((dim, dim), dtype=torch.long,
                                    device=device)
        off_i, off_j, off_pos = [], [], []
        for i in range(dim):
            for j in range(i + 1, dim):
                pos = i * dim + j - i * (i + 1) // 2
                off_i.append(i)
                off_j.append(j)
                off_pos.append(pos)
            for j in range(dim):
                if j >= i:
                    pos = i * dim + j - i * (i + 1) // 2
                else:
                    pos = j * dim + i - j * (j + 1) // 2
                self._row_pos[i, j] = pos
        self._off_i = torch.tensor(off_i, dtype=torch.long, device=device)
        self._off_j = torch.tensor(off_j, dtype=torch.long, device=device)
        self._off_pos = torch.tensor(off_pos, dtype=torch.long, device=device)

        # Dense equality matrix for the 'naive' mode (direct KKT solves).
        self.A_dense = None
        if mode == 'naive':
            self.A_dense = _dense_equality_matrix(
                self.dim, self.m, self._row_pos, self._diag_pos, device
            )

        # Precompute A A^T (dense, small) for the initial projection.
        self._AAT = self._build_AAT(dim, device)

        # Strictly feasible initial point.
        if L is not None:
            l0 = (self.D.T @ L.reshape(-1).to(device=device,
                                               dtype=torch.float64))
            # Project onto A l = b exactly.
            err = self.A @ l0 - self.b
            corr_np = np.linalg.solve(self._AAT.detach().cpu().numpy(),
                                      err.detach().cpu().numpy())
            l0 = l0 - self.A.T @ torch.from_numpy(corr_np).to(self.device)
        else:
            d = self.vol / dim
            c = self.vol / (dim * (dim - 1))
            l0 = torch.full((self.m,), -c, dtype=torch.float64, device=device)
            l0[self.diag_mask] = d
        l0[self.diag_mask] = l0[self.diag_mask].abs().clamp(min=1e-6)
        l0[self.off_mask] = -l0[self.off_mask].abs().clamp(min=1e-6)

        # The uniform point already satisfies A l = b; re-project only if L
        # was supplied, otherwise a second projection is unnecessary.
        if L is not None:
            err = self.A @ l0 - self.b
            corr_np = np.linalg.solve(self._AAT.detach().cpu().numpy(),
                                      err.detach().cpu().numpy())
            l0 = l0 - self.A.T @ torch.from_numpy(corr_np).to(self.device)
        self.l = l0

        self.L = torch.eye(dim, device=device, dtype=torch.float32)

    @staticmethod
    def _build_AAT(dim: int, device: str) -> torch.Tensor:
        """Dense matrix A A^T for the equality constraints (values are 1)."""
        AAT = torch.zeros((dim + 1, dim + 1), dtype=torch.float64,
                          device=device)
        for i in range(dim):
            AAT[i, i] = float(dim)
            for j in range(i + 1, dim):
                AAT[i, j] = 1.0
                AAT[j, i] = 1.0
            AAT[i, dim] = 1.0
            AAT[dim, i] = 1.0
        AAT[dim, dim] = float(dim)
        return AAT

    def _objective(self, l: torch.Tensor, t: float, q: torch.Tensor
                   ) -> torch.Tensor:
        """Barrier objective g(l; t) = t f(l) + phi(l)."""
        off = l[self.off_mask]
        barrier = (-torch.log(-off)).sum()
        return (0.5 * t * (l * self.P_diag * l).sum()
                - t * (q * l).sum() + barrier)

    def forward(self,
                F_t: torch.FloatTensor,
                tolerance: float = 1e-6,
                max_iter: int = 50,
                callback=None,
                max_time: float = None):
        """Run the barrier interior-point algorithm."""
        if max_time is not None:
            max_iter = max(max_iter, 10 ** 9)

        q = self.D.T @ F_t.reshape(-1).to(dtype=torch.float64,
                                           device=self.device)
        t = float(self.t)
        eps_out = max(tolerance, self.eps_out)

        t0 = time.perf_counter() if max_time is not None else None

        self.iter = 0
        for _ in range(max_iter):
            self.iter = _ + 1
            if max_time is not None and (time.perf_counter() - t0) > max_time:
                break
            if max_time is None and self.D_minus / t <= eps_out:
                if callback is not None:
                    L_iter = (self.D @ self.l).reshape(self.dim, self.dim).to(
                        device=self.device, dtype=torch.float32
                    )
                    callback(self.iter, L_iter)
                break

            for _ in range(self.max_inner_iter):
                l = self.l
                off = l[self.off_mask]

                H_diag = t * self.P_diag.clone()
                g = t * (self.P_diag * l) - t * q
                H_diag[self.off_mask] = H_diag[self.off_mask] + 1.0 / (off * off)
                g[self.off_mask] = g[self.off_mask] + 1.0 / (-off)

                if self.mode == 'naive':
                    # Solve the dense KKT system directly:
                    # [[H, A^T], [A, 0]] [delta_l; nu] = [-g; 0].
                    m = self.m
                    kkt = torch.zeros((m + self.dim + 1, m + self.dim + 1),
                                      dtype=torch.float64, device=self.device)
                    kkt[:m, :m] = torch.diag(H_diag)
                    kkt[:m, m:] = self.A_dense.T
                    kkt[m:, :m] = self.A_dense
                    rhs = torch.cat((
                        -g,
                        torch.zeros(self.dim + 1, dtype=torch.float64,
                                    device=self.device)
                    ))
                    delta_l = torch.linalg.solve(kkt, rhs)[:m]
                else:
                    H_inv = 1.0 / H_diag

                    # Build the Schur complement S = -A H^{-1} A^T directly from
                    # the precomputed index structure (no dense/sparse matmul).
                    S = torch.zeros((self.dim + 1, self.dim + 1),
                                    dtype=torch.float64, device=self.device)
                    arange_d = torch.arange(self.dim, device=self.device)
                    S[arange_d, arange_d] = -H_inv[self._row_pos].sum(dim=1)
                    S[self._off_i, self._off_j] = -H_inv[self._off_pos]
                    S[self._off_j, self._off_i] = -H_inv[self._off_pos]
                    S[:self.dim, self.dim] = -H_inv[self._diag_pos]
                    S[self.dim, :self.dim] = -H_inv[self._diag_pos]
                    S[self.dim, self.dim] = -H_inv[self._diag_pos].sum()

                    S_inv = torch.from_numpy(
                        np.linalg.inv(S.detach().cpu().numpy())
                    ).to(device=self.device, dtype=torch.float64)

                    # A H^{-1} g computed from the sparse structure of A.
                    Hg = H_inv * g
                    Ahinv_g = torch.zeros(self.dim + 1, dtype=torch.float64,
                                          device=self.device)
                    Ahinv_g[:self.dim] = Hg[self._row_pos].sum(dim=1)
                    Ahinv_g[self.dim] = Hg[self._diag_pos].sum()
                    temp = S_inv @ Ahinv_g

                    # A^T temp, again using the structure of A.
                    AT_temp = torch.zeros(self.m, dtype=torch.float64,
                                          device=self.device)
                    AT_temp[self._off_pos] = temp[self._off_i] + temp[self._off_j]
                    AT_temp[self._diag_pos] = temp[:self.dim] + temp[self.dim]

                    # Newton direction for the equality-constrained centering step.
                    # delta_l = -H^{-1} g + H^{-1} A^T (R g), with
                    # R = -S^{-1} A H^{-1}, so A^T (R g) = -A^T temp.
                    delta_l = -H_inv * g - H_inv * AT_temp
                omega_sq = (delta_l * H_diag * delta_l).sum()

                s = 1.0
                g_curr = self._objective(l, t, q)
                for _ in range(50):
                    l_cand = l + s * delta_l
                    if (l_cand[self.off_mask] >= 0).any():
                        s *= self.beta_ls
                        continue
                    g_cand = self._objective(l_cand, t, q)
                    if g_cand <= g_curr - self.alpha * s * omega_sq:
                        break
                    s *= self.beta_ls

                self.l = l + s * delta_l
                if omega_sq.sqrt() <= self.eps_in:
                    break

            if callback is not None:
                L_iter = (self.D @ self.l).reshape(self.dim, self.dim).to(
                    device=self.device, dtype=torch.float32
                )
                callback(self.iter, L_iter)

            t *= self.mu

        self.L = (self.D @ self.l).reshape(self.dim, self.dim).to(
            device=self.device, dtype=torch.float32
        )



class ProjActiveSet:
    """Laplacian projection using a dual active-set (Goldfarb--Idnani) method.

    Solves
        min_{l}  (1/2) l^T P l - q^T l
        s.t.     A l = b,  B l <= 0
    where l is the half-vectorization of a valid Laplacian matrix.  The
    inequality constraints force the off-diagonal entries to be non-positive.

    This implementation maintains the inverse of the KKT Schur-complement
    matrix N = M P0^{-1} M^T, where M = [A; B_W] collects the equality and the
    currently active inequality constraints.  Adding or removing an active
    inequality is performed with O((D+|W|)^2) rank-1 updates of N^{-1} rather
    than refactorizing the working-set KKT system.

    With ``mode='naive'`` the working-set KKT system is instead assembled
    densely and solved directly with ``torch.linalg.solve`` at every update,
    mirroring the naive modes of ``ProjADMM`` and ``ProjIPM``.
    """

    def __init__(
            self,
            dim: int,
            L: torch.Tensor = None,
            vol: float = 100.,
            max_iter: int = None,
            mode: str = 'optimized',
            device: str = 'cpu'
    ):
        self.device = device
        self.dim = dim
        self.mode = mode
        self.vol = float(vol)
        self.D_minus = int(dim * (dim - 1) / 2)
        if max_iter is None:
            max_iter = 10 * self.D_minus
        self.max_iter = int(max_iter)

        # Sparse duplication matrix D: vec(L) = D l.
        index_dup, value_dup = duplication_matrix(dim)
        self.m = dim * (dim + 1) // 2
        self.D = torch.sparse_coo_tensor(
            index_dup, value_dup.to(dtype=torch.float64),
            size=(self.m, dim * dim),
            device=device
        ).T.coalesce()

        # Sparse equality matrix A: row sums zero + trace = vol.
        a1_rows, a1_cols = [], []
        for i in range(dim):
            for j in range(dim):
                if j >= i:
                    pos = i * dim + j - i * (i + 1) // 2
                else:
                    pos = j * dim + i - j * (j + 1) // 2
                a1_rows.append(i)
                a1_cols.append(pos)
        self.A1 = torch.sparse_coo_tensor(
            indices=torch.tensor([a1_rows, a1_cols], dtype=torch.long,
                                 device=device),
            values=torch.ones(dim * dim, dtype=torch.float64, device=device),
            size=(dim, self.m)
        ).coalesce()

        a2_rows = [0] * dim
        a2_cols = [i * (2 * dim - i + 1) // 2 for i in range(dim)]
        self.a2 = torch.sparse_coo_tensor(
            indices=torch.tensor([a2_rows, a2_cols], dtype=torch.long,
                                 device=device),
            values=torch.ones(dim, dtype=torch.float64, device=device),
            size=(1, self.m)
        ).coalesce()
        self.A = torch.vstack((self.A1, self.a2)).coalesce()

        self.b = torch.zeros((dim + 1,), dtype=torch.float64, device=device)
        self.b[dim] = self.vol

        # Diagonal entries are allowed to be positive; off-diagonals must be <=0.
        self.diag_mask = torch.zeros(self.m, dtype=torch.bool, device=device)
        self.diag_mask[self.a2.indices()[1]] = True
        self.off_mask = ~self.diag_mask
        self.off_positions = torch.where(self.off_mask)[0]
        self._off_positions_list = [int(p) for p in self.off_positions]

        # P = D^T D  (diagonal with 1's for diagonals, 2's for off-diags).
        self.P_diag = self.diag_mask.to(torch.float64) + 2.0 * self.off_mask.to(torch.float64)
        self.P0_inv = 1.0 / self.P_diag

        # Precompute index structures for A * vector and A^T * vector.
        self._diag_pos = torch.tensor(
            [i * (2 * dim - i + 1) // 2 for i in range(dim)],
            dtype=torch.long, device=device
        )
        self._row_pos = torch.zeros((dim, dim), dtype=torch.long,
                                    device=device)
        off_i, off_j, off_pos = [], [], []
        for i in range(dim):
            for j in range(i + 1, dim):
                pos = i * dim + j - i * (i + 1) // 2
                off_i.append(i)
                off_j.append(j)
                off_pos.append(pos)
            for j in range(dim):
                if j >= i:
                    pos = i * dim + j - i * (i + 1) // 2
                else:
                    pos = j * dim + i - j * (j + 1) // 2
                self._row_pos[i, j] = pos
        self._off_i = torch.tensor(off_i, dtype=torch.long, device=device)
        self._off_j = torch.tensor(off_j, dtype=torch.long, device=device)
        self._off_pos = torch.tensor(off_pos, dtype=torch.long, device=device)

        # Precompute N^{-1} for the equality-only working set.
        self._N_eq_inv = self._build_N_eq_inv(dim, device)

        # Dense constraint matrices for the 'naive' mode (direct KKT solves).
        if mode == 'naive':
            self.A_dense = _dense_equality_matrix(
                self.dim, self.m, self._row_pos, self._diag_pos, device
            )
            self.B_dense = torch.zeros(self.D_minus, self.m,
                                       dtype=torch.float64, device=device)
            edge_idx = torch.arange(self.D_minus, dtype=torch.long,
                                    device=device)
            self.B_dense[edge_idx, self.off_positions] = 1.0

        # Initial Laplacian.
        if L is not None:
            self.L = L.to(device=device, dtype=torch.float32)
        else:
            self.L = torch.eye(dim, device=device, dtype=torch.float32)

    @staticmethod
    def _build_N_eq_inv(dim: int, device: str) -> torch.Tensor:
        """Inverse of N_eq = A P0^{-1} A^T (computed analytically)."""
        N_eq = torch.full((dim + 1, dim + 1), 0.5,
                          dtype=torch.float64, device=device)
        N_eq[:dim, :dim] += (dim / 2.0) * torch.eye(dim, dtype=torch.float64,
                                                    device=device)
        N_eq[:dim, dim] = 1.0
        N_eq[dim, :dim] = 1.0
        N_eq[dim, dim] = float(dim)
        return torch.from_numpy(
            np.linalg.inv(N_eq.detach().cpu().numpy())
        ).to(device=device, dtype=torch.float64)

    def _A_mul_vec(self, vec: torch.Tensor) -> torch.Tensor:
        """Compute A @ vec using the sparsity pattern of A."""
        out = torch.zeros(self.dim + 1, dtype=torch.float64,
                          device=self.device)
        out[:self.dim] = vec[self._row_pos].sum(dim=1)
        out[self.dim] = vec[self._diag_pos].sum()
        return out

    def _A_T_mul_vec(self, vec: torch.Tensor) -> torch.Tensor:
        """Compute A^T @ vec using the sparsity pattern of A."""
        out = torch.zeros(self.m, dtype=torch.float64, device=self.device)
        out[self._off_pos] = vec[self._off_i] + vec[self._off_j]
        out[self._diag_pos] = vec[:self.dim] + vec[self.dim]
        return out

    def _p1_for_edge(self, edge_idx: int) -> torch.Tensor:
        """Column of A P0^{-1} corresponding to edge ``edge_idx``."""
        pos = self._off_positions_list[edge_idx]
        p1 = torch.zeros(self.dim + 1, dtype=torch.float64,
                         device=self.device)
        val = self.P0_inv[pos]
        p1[self._off_i[edge_idx]] = val
        p1[self._off_j[edge_idx]] = val
        return p1

    def _solve_wkkt(self, W: list, g: torch.Tensor) -> tuple:
        """Solve the working-set KKT system directly (naive mode).

        Builds N = M diag(P0^{-1}) M^T for M = [A; B_W] and solves
        N y = M g - b_ext, then recovers the primal
        l = g - P0^{-1} M^T y.
        """
        if W:
            idx = torch.tensor(W, dtype=torch.long, device=self.device)
            M = torch.vstack((self.A_dense, self.B_dense[idx]))
        else:
            M = self.A_dense
        N = (M * self.P0_inv.unsqueeze(0)) @ M.T
        b_ext = torch.zeros(M.shape[0], dtype=torch.float64,
                            device=self.device)
        b_ext[:self.dim + 1] = self.b
        y = torch.linalg.solve(N, M @ g - b_ext)
        l = g - self.P0_inv * (M.T @ y)
        return y, l

    def _add_edge_naive(
        self,
        edge_idx: int,
        W: list,
        y: torch.Tensor,
        l: torch.Tensor,
        g: torch.Tensor
    ) -> tuple:
        """Naive update: refactorize the KKT system after adding an edge."""
        W = W + [edge_idx]
        y, l = self._solve_wkkt(W, g)
        return W, y, None, l

    def _remove_edge_naive(
        self,
        j: int,
        W: list,
        y: torch.Tensor,
        l: torch.Tensor,
        g: torch.Tensor
    ) -> tuple:
        """Naive update: refactorize the KKT system after removing an edge."""
        W = W[:j] + W[j + 1:]
        y, l = self._solve_wkkt(W, g)
        return W, y, None, l

    def _add_edge(
        self,
        edge_idx: int,
        W: list,
        y: torch.Tensor,
        N_inv: torch.Tensor,
        l: torch.Tensor,
        g: torch.Tensor
    ) -> tuple:
        """Rank-1 KKT update when edge ``edge_idx`` enters the working set."""
        dim = self.dim
        pos_new = self._off_positions_list[edge_idx]

        # p = M P0^{-1} a_new^T (only equality block is non-zero here because
        # the new edge is not yet in the working set).
        p1 = self._p1_for_edge(edge_idx)
        p2 = torch.zeros(len(W), dtype=torch.float64, device=self.device)
        p = torch.cat([p1, p2])

        t = N_inv @ p
        gamma = float(self.P0_inv[pos_new].item())
        sigma = gamma - (p @ t).item()
        if abs(sigma) < 1e-18:
            sigma = 1e-18 if sigma >= 0 else -1e-18

        tau = g[pos_new].item() - (p @ y).item()
        scale = tau / sigma

        # Update multipliers and N^{-1} (Sherman-Morrison for a bordering row).
        y_new = torch.cat([y - scale * t,
                           torch.tensor([scale], dtype=torch.float64,
                                        device=self.device)])
        m0 = len(y)
        new_N_inv = torch.zeros((m0 + 1, m0 + 1), dtype=torch.float64,
                                device=self.device)
        new_N_inv[:m0, :m0] = N_inv + (1.0 / sigma) * torch.outer(t, t)
        new_N_inv[:m0, m0] = -(1.0 / sigma) * t
        new_N_inv[m0, :m0] = -(1.0 / sigma) * t
        new_N_inv[m0, m0] = 1.0 / sigma

        # Update primal: l_new = l - scale * P0^{-1} * (a_new - M^T t).
        bw_t = torch.zeros(self.m, dtype=torch.float64, device=self.device)
        for k, idx in enumerate(W):
            bw_t[self._off_positions_list[idx]] = t[dim + 1 + k]
        correction = -self._A_T_mul_vec(t[:dim + 1]) - bw_t
        correction[pos_new] += 1.0
        l_new = l - scale * self.P0_inv * correction

        W.append(edge_idx)
        return W, y_new, new_N_inv, l_new

    def _remove_edge(
        self,
        j: int,
        W: list,
        y: torch.Tensor,
        N_inv: torch.Tensor,
        l: torch.Tensor,
        g: torch.Tensor
    ) -> tuple:
        """Rank-1 KKT downdate when the active edge at position ``j`` leaves."""
        dim = self.dim
        m = len(y)

        # Move the leaving constraint to the last position, downdate, and
        # permute back.
        perm = list(range(m))
        perm[j], perm[m - 1] = perm[m - 1], perm[j]
        inv_perm = [0] * m
        for i, p in enumerate(perm):
            inv_perm[p] = i

        N_inv_perm = N_inv[perm][:, perm]
        y_perm = y[perm]

        K = N_inv_perm[:m - 1, :m - 1]
        u = N_inv_perm[:m - 1, m - 1]
        d = N_inv_perm[m - 1, m - 1].item()
        alpha = y_perm[m - 1].item()

        if abs(d) < 1e-18:
            d = 1e-18 if d >= 0 else -1e-18

        N_inv_red = K - (1.0 / d) * torch.outer(u, u)
        y_red = y_perm[:m - 1] - (alpha / d) * u

        N_inv = N_inv_red[inv_perm[:m - 1]][:, inv_perm[:m - 1]]
        y = y_red[inv_perm[:m - 1]]
        W = W[:j] + W[j + 1:]

        # Recompute the primal from the reduced multipliers.
        bw = torch.zeros(self.m, dtype=torch.float64, device=self.device)
        for k, idx in enumerate(W):
            bw[self._off_positions_list[idx]] = y[dim + 1 + k]
        l = g - self.P0_inv * (self._A_T_mul_vec(y[:dim + 1]) + bw)

        return W, y, N_inv, l

    def forward(self, F_t: torch.FloatTensor, max_iter: int = None,
                tolerance: float = None, callback=None,
                max_time: float = None):
        """Run the dual active-set algorithm.

        Parameters
        ----------
        F_t : torch.Tensor
            Target matrix whose half-vectorization defines q = D^T vec(F_t).
        max_iter : int, optional
            Maximum number of working-set updates.  Defaults to the value
            passed to the constructor.
        tolerance : float, optional
            Ignored; present for API compatibility with other projection
            solvers called by the runners.
        callback : callable, optional
            If given, called as ``callback(iteration, L)`` after each
            working-set update with the current Laplacian estimate.
        max_time : float, optional
            Maximum wall time in seconds. If set, the algorithm stops after
            the first working-set update that exceeds this limit.
        """
        if max_iter is None:
            max_iter = self.max_iter
        if max_time is not None:
            max_iter = max(max_iter, 10 ** 9)

        q = self.D.T @ F_t.reshape(-1).to(dtype=torch.float64,
                                           device=self.device)
        g = self.P0_inv * q

        t0 = time.perf_counter() if max_time is not None else None

        # Try to warm-start from the previous active set if available.
        warm_started = False
        prev_W = getattr(self, "W", None)
        prev_l = getattr(self, "l", None)
        if prev_W is not None and prev_l is not None and len(prev_W) > 0:
            try:
                A_dense = _dense_equality_matrix(
                    self.dim, self.m, self._row_pos, self._diag_pos,
                    self.device
                )
                rows = [A_dense]
                for edge_idx in prev_W:
                    row = torch.zeros(self.m, dtype=torch.float64,
                                      device=self.device)
                    row[self._off_positions_list[edge_idx]] = 1.0
                    rows.append(row)
                M = torch.vstack(rows)
                # N = M @ diag(P0_inv) @ M.T
                N = (M * self.P0_inv.unsqueeze(0)) @ M.T
                N_inv = torch.from_numpy(
                    np.linalg.inv(N.detach().cpu().numpy())
                ).to(device=self.device, dtype=torch.float64)
                b_ext = torch.zeros(M.shape[0], dtype=torch.float64,
                                    device=self.device)
                b_ext[:self.dim + 1] = self.b
                y = N_inv @ (M @ g - b_ext)
                l = g - self.P0_inv * (M.T @ y)
                if (l[self.off_positions] <= 1e-12).all():
                    W = list(prev_W)
                    warm_started = True
            except Exception:
                warm_started = False

        if not warm_started:
            # Equality-constrained minimizer (cold start).
            if self.mode == 'naive':
                N_eq = (self.A_dense * self.P0_inv.unsqueeze(0)) @ self.A_dense.T
                y = torch.linalg.solve(N_eq, self._A_mul_vec(g) - self.b)
            else:
                y = self._N_eq_inv @ (self._A_mul_vec(g) - self.b)
            l = g - self.P0_inv * self._A_T_mul_vec(y)
            W = []
            N_inv = self._N_eq_inv.clone()

        self.iter = 0

        for _ in range(max_iter):
            self.iter = _ + 1
            if max_time is not None and (time.perf_counter() - t0) > max_time:
                break
            off_vals = l[self.off_positions]

            if callback is not None:
                L_iter = (self.D @ l).reshape(self.dim, self.dim).to(
                    device=self.device, dtype=torch.float32
                )
                callback(self.iter, L_iter)

            # Add the most violated off-diagonal not yet in the working set.
            if len(W) < self.D_minus:
                in_W = torch.zeros(self.D_minus, dtype=torch.bool,
                                   device=self.device)
                if W:
                    in_W[torch.tensor(W, dtype=torch.long,
                                      device=self.device)] = True
                feasible_idx = torch.where(~in_W)[0]
                feasible_off = off_vals[feasible_idx]
                if feasible_off.numel() > 0 and feasible_off.max().item() > 0:
                    i_star = int(feasible_idx[feasible_off.argmax()])
                    if self.mode == 'naive':
                        W, y, N_inv, l = self._add_edge_naive(
                            i_star, W, y, l, g
                        )
                    else:
                        W, y, N_inv, l = self._add_edge(
                            i_star, W, y, N_inv, l, g
                        )
                    continue

            # Remove the active edge with the most negative multiplier.
            if W:
                active_mult = y[self.dim + 1:]
                j_star = int(active_mult.argmin())
                if float(active_mult[j_star]) < -1e-12:
                    if self.mode == 'naive':
                        W, y, N_inv, l = self._remove_edge_naive(
                            j_star, W, y, l, g
                        )
                    else:
                        W, y, N_inv, l = self._remove_edge(
                            j_star, W, y, N_inv, l, g
                        )
                    continue

            # KKT conditions satisfied.
            break

        self.l = l
        self.W = W
        self.L = (self.D @ l).reshape(self.dim, self.dim).to(
            device=self.device, dtype=torch.float32
        )
