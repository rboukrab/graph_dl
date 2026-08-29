import argparse
import numpy as np
from quadprog import solve_qp
import torch
from scipy.optimize import linear_sum_assignment
import networkx as nx
from scipy.stats import kurtosis

def print_parameters(args: argparse.Namespace):
    args_dict = vars(args)
    param_list = {
        'mode': "Algorithm mode",
        'data_set': "Dataset: Synthetic data or Real fMRI data",
        'graph_learn': "Graph learning algorithm",
        'n_runs': "Number of runs",
        'dim': "Dimension of signals",
        'n_atoms': "Number of atoms",
        'n_samples': "Number of samples in dataset",
        'n_train': "Number of samples used for training",
        'n_test': "Number of samples used for testing",
        'n_cov': "Number of samples used to initialize covariance matrix",
        'n_epochs': "Number of epochs",
        'batch_size': "Batch size",
        'test_interval': "Test interval",
        'lr': "Dictionary SGD learning rate",
        'sp': "Non-zero elements of sparse codes",
        'alpha': "Laplacian regularization parameter",
        'snr': "Signal-to-Noise Ratio",
        'gamma': "GL algorithm sparsity regularization parameter",
        'rho': "GL algorithm dual step size"
    }
    summary_str = "---------------------------------------------------------------------------------" + "\n"
    line_new = "{:<12} {:>10}  {:<100}".format(
        "Param #", "Value", "Description")
    summary_str += line_new + "\n"
    summary_str += "=================================================================================" + "\n"
    for arg in args_dict:
        line_new = "{:<12} {:>10}  {:<100}".format(arg, args_dict[arg], param_list[arg])
        summary_str += line_new + "\n"
    summary_str += "=================================================================================" + "\n"
    print(summary_str)

def asym_soft_thresh(X, left_thresh, right_thresh):
    return np.minimum(X + left_thresh, 0) + np.maximum(X - right_thresh, 0)


def bcd_glasso(kappa, dim, K):
    Q = np.diag(1. / np.diag(K))
    for k in range(kappa):
        for j in range(dim):
            i_t = np.arange(dim).tolist()
            i_t.pop(j)
            Q_j = Q[np.ix_(i_t, i_t)]
            k_j = K[:, j]
            k_j = k_j[np.ix_(i_t)]
            k_jj = K[j, j]
            lmda_j = solve_qp(2 * Q_j, - 2 * Q_j.T.dot(k_j), np.eye(dim - 1), np.zeros(dim - 1))[0]
            q_j = - (1. / k_jj) * Q_j.dot(lmda_j + k_j)
            q_jj = (1. / k_jj) * (1 - q_j.dot(k_j))
            Q[np.ix_(i_t)[0], j] = q_j
            Q[j, np.ix_(i_t)[0]] = q_j
            Q[j, j] = q_jj
    W = - Q
    W = (W + W.T) / 2.
    np.fill_diagonal(W, 0.)
    W[W.__abs__() < 1e-9] = 0.
    D = np.diag(W.sum(axis=0))
    L = (D - W) / D.max()
    return L

def duplication_matrix(n):
    index = torch.zeros((2, 0), dtype=torch.int)
    value = torch.ones(n ** 2, dtype=torch.float)
    for j in range(n):
        for i in range(j, n):
            row_idx = int(j * n + i - ((j + 1) * j) / 2)
            col_idx1 = i * n + j
            col_idx2 = j * n + i
            if col_idx1 == col_idx2:
                index = torch.hstack((
                    index,
                    torch.tensor([[row_idx], [col_idx1]])
                ))
            else:
                index = torch.hstack((
                    index,
                    torch.tensor([[row_idx], [col_idx1]]),
                    torch.tensor([[row_idx], [col_idx2]])
                ))
    return index, value

def generate_graph_ER(dim, p):
    flag_connected = False
    while not flag_connected:
        S = np.random.rand(dim, dim) < p
        S = np.triu(S, 1)
        S = S + S.T
        if np.sum(S.sum(0) == 0) == 0:
            flag_connected = True
    return S

def weighted_max_matching(list1, list2, weight_func):
    n = len(list1)
    assert len(list2) == n, "Both lists must contain the same number of subsets"

    # Construct the cost matrix using the negative of the weight function
    cost_matrix = np.zeros((n, n), dtype=float)
    for i, a in enumerate(list1):
        for j, b in enumerate(list2):
            cost_matrix[i, j] = -weight_func(a, b)  # negate for maximization

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matching = [(i, j, -cost_matrix[i, j]) for i, j in zip(row_ind, col_ind)]
    total_weight = -cost_matrix[row_ind, col_ind].sum()

    return matching, total_weight

def small_world_coefficient(adj_matrix, n_random):
    G = nx.Graph(adj_matrix)
    L = nx.average_shortest_path_length(G)
    C = nx.average_clustering(G)

    # Random graph with the same number of nodes and edges
    sigma = 0
    for n in range(n_random):
        random_graph = nx.erdos_renyi_graph(
            G.number_of_nodes(),
            G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2),
            seed=n+1
        )
        L_random = nx.average_shortest_path_length(random_graph)
        C_random = nx.average_clustering(random_graph)

        # Calculate small-world coefficient (sigma)
        sigma += (C / C_random) / (L / L_random)
    sigma /= n_random

    return sigma

def modularity(adj_matrix, labels):
    G = nx.Graph(adj_matrix)
    unique_labels = np.unique(labels)
    comms = [np.where(labels == u)[0].tolist() for u in unique_labels]
    return nx.community.modularity(G, comms)


def heavy_tailedness(adj_matrix):
    """Simple heavy-tailedness proxy: excess kurtosis of upper-triangle weights."""
    W = np.asarray(adj_matrix)
    iu = np.triu_indices_from(W, k=1)
    weights = W[iu]
    if weights.size == 0 or np.std(weights) == 0:
        return 0.0
    return float(kurtosis(weights, fisher=True))
