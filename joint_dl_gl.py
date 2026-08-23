import os
import time
import argparse

from joint_ol_gl.config import joint_experiment_path, ensure_log_dir
import torch
import numpy as np
from pygsp.graphs import Graph
from tqdm import tqdm
import networkx as nx

from joint_ol_gl.models import LapRegSparseNet, StationarySparseNet, HeatDiffSparseNet
from joint_ol_gl.datasets import StatDataset, FlankerDataset
from joint_ol_gl.metrics import compute_mse, compute_nae, compute_scores
from joint_ol_gl.graph_learn import BCD, GAMA, GLAMA, StationaryADMM

# Training settings
parser = argparse.ArgumentParser(
    description='PyTorch fMRI Online Graph Learning under Smoothness Priors'
)
parser.add_argument('--device', type=str, default=None)
parser.add_argument('--data_set', type=str,
                    choices=['stat', 'fmri'], default='fmri',
                    help='Dataset to use')
parser.add_argument('--method', type=str,
                    choices=['smooth', 'stationary', 'heatdiff'], default='smooth',
                    help='Dictionary learning method')
parser.add_argument('--graph_learn', type=str,
                    choices=['known', 'bcd', 'gama', 'glama', 'admm'], default='glama',
                    help='Graph learning method')
parser.add_argument('--admm_mode', type=str,
                    choices=['normal', 'optimized'], default='optimized',
                    help='ADMM computing method')
parser.add_argument('--dim', type=int, default=100,
                    help='dimensionality of data')
parser.add_argument('--n_atoms', type=int, default=200,
                    help='number of atoms in the dictionary')
parser.add_argument('--n_cov', type=int, default=1000,
                    help='number of samples to compute initial covariance')
parser.add_argument('--n_iters', type=int, default=10000,
                    help='number of iterations')
parser.add_argument('--batch_size', type=int, default=2500)
parser.add_argument('--sim_time', type=float, default=9999999,
                    help='simulation time in seconds')
parser.add_argument('--test_interval', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.01, help='Dictionary SGD learning rate')
parser.add_argument('--sp', type=int, default=6, help='sparsity of the representation')
parser.add_argument('--alpha', type=float, default=0.001,
                    help='parameter to control graph smoothness of the solution')
parser.add_argument('--rho', type=float, default=1e-2,
                    help='parameter to control graph smoothness of the solution')
parser.add_argument('--gamma', type=float, default=1e-3,
                    help='parameter to control graph smoothness of the solution')
parser.add_argument('--graph_learn_tol', type=float, default=1e-5,
                    help='Graph learning algorithm tolerance')
parser.add_argument('--graph_learn_max_iter', type=int, default=1000,
                    help='Graph learning algorithm max iters')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
parser.add_argument('--thresh', type=float, default=0.5,
                    help='Learned Laplacian threshold')

def find_max_intersection_pairs(list1, list2):
    used_in_list1 = set()
    used_in_list2 = set()
    pairs = []

    while len(used_in_list1) < len(list1) and len(used_in_list2) < len(list2):
        max_intersection = 0
        max_pair = (None, None)

        # Iterate through all possible pairs
        for i, set1 in enumerate(list1):
            if i in used_in_list1:
                continue
            for j, set2 in enumerate(list2):
                if j in used_in_list2:
                    continue
                intersection = len(set1 & set2)
                if intersection > max_intersection:
                    max_intersection = intersection
                    max_pair = (i, j)

        # If a valid pair is found, add to result and mark sets as used
        if max_pair[0] is not None and max_pair[1] is not None:
            used_in_list1.add(max_pair[0])
            used_in_list2.add(max_pair[1])
            pairs.append((list1[max_pair[0]], list2[max_pair[1]]))
        else:
            break

    return pairs

if __name__ == '__main__':

    # Parse input arguments
    args = parser.parse_args()
    device = args.device
    data_set = args.data_set
    method = args.method
    graph_learn = args.graph_learn
    admm_mode = args.admm_mode
    dim = args.dim
    n_atoms = args.n_atoms
    n_cov = args.n_cov
    n_iters = args.n_iters
    batch_size = args.batch_size
    sim_time = args.sim_time
    test_interval = args.test_interval
    lr = args.lr
    sp = args.sp
    graph_learn_tol = args.graph_learn_tol
    graph_learn_max_iter = args.graph_learn_max_iter
    seed = args.seed
    thresh = args.thresh

    alpha = args.alpha
    gamma = args.gamma
    rho = args.rho

    # Device
    if device is None:
        if graph_learn == 'bcd':
            device = 'cpu'
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Constants
    if data_set == "stat":
        n_samples = 3500
        n_train = 2500
        n_test = 1000
        dataset = StatDataset(seed, dim, n_samples)
        graph = dataset.graph
    else:
        n_samples = 3500
        n_train = 2500
        n_test = 1000
        dataset = FlankerDataset()
        dim = dataset.dim
        graph = None

    # Create directories if they do not exist
    sim_path = joint_experiment_path(
        data_set=data_set,
        method=method,
        graph_learn=graph_learn,
        batch_size=batch_size,
        sp=sp,
        alpha=alpha,
        seed=seed,
    )
    ensure_log_dir(sim_path)

    alpha *= batch_size / n_train
    gamma *= batch_size / n_train

    # Performance metrics
    mse = np.zeros(int(n_iters / test_interval), dtype=float)
    clerr = np.zeros(int(n_iters / test_interval), dtype=float)
    lap_sparsity = np.zeros(int(n_iters / test_interval), dtype=float)
    lap_transitivity = np.zeros(int(n_iters / test_interval), dtype=float)
    lap_clustering = np.zeros(int(n_iters / test_interval), dtype=float)
    accum_time = np.zeros(int(n_iters / test_interval), dtype=float)
    if data_set == "stat":
        nae = np.zeros((int(n_iters / test_interval)), dtype=float)
        tp = np.zeros((int(n_iters / test_interval)), dtype=float)
        fp = np.zeros((int(n_iters / test_interval)), dtype=float)
        tn = np.zeros((int(n_iters / test_interval)), dtype=float)
        fn = np.zeros((int(n_iters / test_interval)), dtype=float)

    # Reproducibility
    torch.manual_seed(seed)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
    X_train = dataset.X[train_dataset.indices].to(device)
    X_test = dataset.X[test_dataset.indices].to(device)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False
    )

    # Mean and covariance matrix
    S_n = dataset.X[train_dataset.indices[:n_cov]].T @ dataset.X[train_dataset.indices[:n_cov]] / n_cov
    S_n = S_n.to(device)
    L = torch.zeros((dim, dim), device=device)

    if method == 'smooth':
        sparsenet = LapRegSparseNet(
            n_nodes=dim,
            n_atoms=n_atoms,
            batch_size=n_train,
            dictionary=X_train[:n_atoms].T,
            sparsity=sp,
            lr=lr,
            n_cov=n_cov,
            alpha=alpha,
            device=device
        )
    elif method == 'stationary':
        if data_set == 'fmri':
            D0 = torch.zeros(n_atoms, dim)
            D0[:, :1] = 1
            perm = torch.stack([torch.randperm(dim) for _ in range(len(D0))])
            D0 = D0.gather(dim=1, index=perm).to(device)
        sparsenet = StationarySparseNet(
            n_nodes=dim,
            n_atoms=n_atoms,
            batch_size=n_train,
            sparsity=sp,
            lr=lr,
            n_cov=n_cov,
            alpha=alpha,
            device=device
        )
    elif method == 'heatdiff':
        W = torch.rand((dim, dim), device=device)
        W.fill_diagonal_(0)
        W = (W + W.T) / 2
        L = torch.diag(W.sum(0)) - W
        L *= 20 / dim
        sparsenet = HeatDiffSparseNet(
            n_nodes=dim,
            n_filters=int(n_atoms / dim),
            batch_size=n_train,
            laplacian=L,
            sparsity=sp,
            lr=lr,
            alpha=alpha,
            device=device
        )
    # Graph learner
    graph_learner = None
    if graph_learn == 'bcd':
        graph_learner = BCD(
            cov_matrix=S_n,
            device=device
        )
    elif graph_learn == 'gama':
        graph_learner = GAMA(
            cov_matrix=S_n,
            gamma=gamma,
            rho=rho,
            device=device
        )
    elif graph_learn == 'glama':
        graph_learner = GLAMA(
            cov_matrix=S_n,
            gamma=gamma,
            rho=rho,
            eta=rho,
            device=device
        )
    elif graph_learn == 'admm':
        graph_learner = StationaryADMM(
            dim=dim,
            alpha=alpha,
            L=L,
            mode=admm_mode,
            device=device
        )

    pbar = tqdm(total=n_iters, position=0, leave=True)

    accum_time_i = 0.
    i_test = 0

    # Run the algorithm
    i = 0
    while i < n_iters and accum_time_i < sim_time:
        train_dataloader_iter = iter(train_dataloader)

        # Compute performance metrics
        if i % test_interval == 0:
            i_test = int(i / test_interval)
            mse[i_test] = compute_mse(sparsenet, X_test).item()
            accum_time[i_test] = accum_time_i
            if data_set == "stat":
                if graph_learn in ['gama', 'glama', 'admm'] and i_test > 0:

                    # Get adjacency matrix
                    W = - graph_learner.L.detach().clone()

                    W.fill_diagonal_(0)
                    W = (W + W.T) / 2

                    tp[i_test], fp[i_test], tn[i_test], fn[i_test] = (
                        compute_scores(W, graph, thresh)
                    )
                    W[W.__abs__() < thresh] = 0.
                    nae[i_test] = compute_nae(W, graph)

            if i_test > 0:
                if graph_learner:
                    L = graph_learner.L.detach().clone()
                else:
                    L = torch.from_numpy(graph.L.todense()).to(device).float()

                if data_set == "stat":
                    f_score = 2 * tp[i_test] / (2 * tp[i_test] + fn[i_test] + fp[i_test])
                    precision = tp[i_test] / (tp[i_test] + fp[i_test])
                    recall = tp[i_test] / (tp[i_test] + fn[i_test])
                    pbar.set_description(
                        "alpha = %.3E, mse = %.2f, prec = %.2f, recall = %.2f, f_score = %.2f, thresh = %.2f" % \
                        (alpha, mse[i_test], 100 * precision, 100 * recall, 100 * f_score.max(), thresh))
                else:
                    keep_rois = np.array([8,  19,  25,  34,  40,  44,  46,  51,  54,  61,  63,  72,  81,
                        86,  96, 101, 102, 106, 107, 121, 122, 127, 138, 144, 149, 150,
                       158, 174, 175, 176, 198, 205, 209, 210, 211, 216, 227, 228, 239,
                       257, 264, 272, 287, 295, 296, 300, 305, 313, 317, 318, 319, 320,
                       325, 330, 337, 339, 347, 361, 363, 373, 388, 395, 409, 411, 413,
                       416, 418, 419, 423, 426, 434, 436, 437, 439, 445, 447, 450, 458,
                       465, 486, 492, 494, 501, 510])
                    gs = [{8,  25,  51,  54,  61,  81,  86, 106, 121, 122, 127, 144, 149,
                           150, 158, 174, 176, 209, 216, 227, 239, 264, 287, 295, 300, 318,
                           320, 325, 330, 339, 361, 363, 418, 423, 426, 434, 445, 447, 450,
                           486, 494, 510},
                          {19,  34,  40,  44,  46,  63,  72,  96, 101, 102, 107, 138, 175,
                           198, 205, 210, 211, 228, 257, 272, 296, 305, 313, 317, 319, 337,
                           347, 373, 388, 395, 409, 411, 413, 416, 419, 436, 437, 439, 458,
                           465, 492, 501}
                    ]
                    L_print = L.detach().clone()
                    L_print[L_print.__abs__() < 1e-2] = 0
                    W = - L_print
                    W = (W + W.T) / 2.
                    W.fill_diagonal_(0)
                    G = nx.from_numpy_array(W.cpu().numpy())
                    c = nx.community.kernighan_lin_bisection(G)
                    c_roi = [set(keep_rois[np.array(list(c[i]))]) for i in range(2)]
                    pairs = find_max_intersection_pairs(gs, c_roi)
                    n_intersect = 0
                    for pair in pairs:
                        n_intersect += len(pair[0].intersection(pair[1]))

                    clerr[i_test] = n_intersect / len(keep_rois)
                    lap_sparsity[i_test] = (L_print != 0).sum()
                    lap_clustering[i_test] = nx.average_clustering(G)
                    lap_transitivity[i_test] = nx.transitivity(G)
                    pbar.set_description("mse = %.2f, sp = %d, clerr = %.2f" % (mse[i_test], lap_sparsity[i_test], 100 * n_intersect / len(keep_rois)))

        # noinspection PyUnboundLocalVariable
        for _ in range(int(n_train / batch_size)):

            if graph_learner:
                L = graph_learner.L.detach().clone()
            elif graph:
                L = torch.from_numpy(graph.L.todense()).float().to(device)
            else:
                L = torch.zeros((dim, dim), device=device)

            start = time.time()

            # Update steps
            X_train_i = next(train_dataloader_iter).to(device)

            # Update steps
            grad_h_L = sparsenet.forward(X_train_i, L)

            # Graph learning step
            if graph_learner:
                if method == "stationary":
                    c_k = 1.
                    found = False
                    L_old = L.detach().clone()
                    L_p = L_old
                    Q = sparsenet.a[0] * torch.eye(dim, device=device)
                    for p in range(1, sparsenet.deg + 1):
                        Q += sparsenet.a[p] * L_p
                        L_p = L @ L_p
                    C = sparsenet.D @ sparsenet.D.T
                    h_old = 0.5 * alpha * (torch.trace(Q @ C) - n_atoms * torch.logdet(Q))
                    while not found:
                        H_k = c_k * L_old - grad_h_L
                        graph_learner.L = L_old.detach().clone()
                        graph_learner.forward(
                            c_k=c_k,
                            H_k=H_k,
                            tolerance=graph_learn_tol,
                            max_iter=graph_learn_max_iter
                        )
                        L_new = graph_learner.L.detach().clone()
                        L_p = L_new
                        Q = sparsenet.a[0] * torch.eye(dim, device=device)
                        for p in range(1, sparsenet.deg + 1):
                            Q += sparsenet.a[p] * L_p
                            L_p = L @ L_p
                        h_new = 0.5 * alpha * (torch.trace(Q @ C) - n_atoms * torch.logdet(Q))
                        suff_dec = h_old + torch.trace(grad_h_L @ (L_new - L_old)) + 0.5 * c_k * torch.linalg.norm(L_new - L_old) ** 2
                        if h_new <= suff_dec:
                            found = True
                        else:
                            c_k *= 10
                elif method == "diffusion":
                    c_k = 0.
                    H_k = c_k * L - grad_h_L
                    graph_learner.forward(
                        c_k=c_k,
                        H_k=H_k,
                        tolerance=graph_learn_tol,
                        max_iter=graph_learn_max_iter
                    )
                elif method == "heatdiff":
                    eta = 2
                    c_k = 1e-1
                    D_old = sparsenet.D.detach().clone()
                    y = sparsenet.y
                    L_old = L.detach().clone()
                    h_k_old = torch.linalg.norm(X_train_i - y @ D_old.T) ** 2 + alpha * y.abs().sum()

                    found = False
                    while not found:
                        H_k = c_k * L_old - grad_h_L
                        graph_learner.forward(
                            c_k=c_k,
                            H_k=H_k,
                            tolerance=graph_learn_tol,
                            max_iter=graph_learn_max_iter
                        )
                        L = graph_learner.L
                        D = torch.empty((dim, n_atoms), device=device)
                        for f in range(sparsenet.n_filters):
                            D[:, f * dim:(f + 1) * dim] = torch.linalg.matrix_exp(-sparsenet.tau[f] * L)
                        h_k = torch.linalg.norm(X_train_i - y @ D.T) ** 2 + alpha * y.abs().sum()
                        grad_h_L_L_old = torch.trace(grad_h_L.T @ (L - L_old))
                        norm_L_L_old = 0.5 * c_k * torch.linalg.norm(L - L_old) ** 2
                        if not h_k < h_k_old + grad_h_L_L_old + norm_L_L_old:
                            c_k *= eta
                        else:
                            found = True
                else:
                    graph_learner.forward(
                        S_d=S_n,
                        tolerance=graph_learn_tol,
                        max_iter=graph_learn_max_iter
                    )

            end = time.time()
            accum_time_i += (end - start)

        pbar.update(1)
        i += 1

    # Save performance metrics
    np.save(os.path.join(sim_path, "accum_time.npy"), accum_time)
    np.save(os.path.join(sim_path, "mse.npy"), mse)
    np.save(os.path.join(sim_path, "clerr.npy"), clerr)
    np.save(os.path.join(sim_path, "lap_mat.npy"), L.cpu().numpy())
    np.save(os.path.join(sim_path, "lap_sparsity.npy"), lap_sparsity)
    np.save(os.path.join(sim_path, "lap_clustering.npy"), lap_clustering)
    np.save(os.path.join(sim_path, "lap_transitivity.npy"), lap_transitivity)

    if data_set == "stat":
        np.save(os.path.join(sim_path, "nae.npy"), nae)
        np.save(os.path.join(sim_path, "tp.npy"), tp)
        np.save(os.path.join(sim_path, "fp.npy"), fp)
        np.save(os.path.join(sim_path, "tn.npy"), tn)
        np.save(os.path.join(sim_path, "fn.npy"), fn)
