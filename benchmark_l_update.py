import os
import time
import argparse

from core.config import benchmark_experiment_path, ensure_log_dir
import torch
import numpy as np
from core.datasets import StatDataset
from core.graph_learn import ProjADMM
from torch.nn.functional import softshrink

# Training settings
parser = argparse.ArgumentParser(
    description='Benchmark L-update step in polynomial graph learning'
)
parser.add_argument('--device', type=str, default=None)
parser.add_argument('--dim', type=int, default=100, help='Graph dimension')
parser.add_argument('--deg', type=int, default=2, help='Laplacian polynomial degree')
parser.add_argument('--alpha', type=float, default=4)
parser.add_argument('--mode', type=str, choices=['naive', 'optimized'], default='naive',
                    help='ADMM computing method')
parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples for covariance')
parser.add_argument('--n_iters', type=int, default=100, help='Number of benchmark iterations')
parser.add_argument('--tol', type=float, default=1e-2, help='Graph learning algorithm tolerance')
parser.add_argument('--max_iter', type=int, default=10, help='Graph learning algorithm max iters')
parser.add_argument('--seed', type=int, default=2)

if __name__ == '__main__':

    # Parse input arguments
    args = parser.parse_args()
    device = args.device
    dim = args.dim
    deg = args.deg
    alpha = args.alpha
    mode = args.mode
    n_samples = args.n_samples
    n_iters = args.n_iters
    tol = args.tol
    max_iter = args.max_iter
    seed = args.seed

    # Device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Load dataset to get known filter coefficients
    dataset = StatDataset(seed, dim, n_samples, deg=deg)
    h = dataset.h.to(device)  # Known filter coefficients from dataset
    graph = dataset.graph

    # Use known filter coefficients (fix a to h)
    a = h.clone()  # Fix a to the known filter coefficients
    print(f"Fixed polynomial coefficients a (from dataset.h): {a.cpu().numpy()}")

    # Prepare data
    X_train = dataset.X.to(device)
    C = X_train.T @ X_train / X_train.shape[0]

    # Initialize Laplacian powers
    Lp = torch.zeros((deg + 1, dim, dim), device=device)
    Lp[0] = torch.eye(dim, device=device)

    # Initialize ADMM variables
    Phi = torch.zeros((dim, dim), device=device)
    Omega = torch.zeros((dim, dim), device=device)
    U = torch.zeros((dim, dim), device=device)

    I = torch.eye(dim).bool().to(device)
    O = (torch.ones((dim, dim)) - torch.eye(dim)).bool().to(device)

    # Initialize graph learner
    graph_vol = graph.W.todense().sum()
    graph_learner = ProjADMM(
        dim=dim,
        mode=mode,
        L=(10 + 10 / 99) * torch.eye(dim, device=device) - (10 / 99) * torch.ones((dim, dim), device=device),
        vol=graph_vol,
        device=device
    )

    # Initialize L from graph learner
    L = graph_learner.L.detach().clone()

    # Initialize Laplacian powers with initial L
    Lp[1] = L.detach().clone()
    for p in range(2, deg + 1):
        Lp[p] = Lp[p - 1] @ L
        Lp[p] = 0.5 * (Lp[p] + Lp[p].T)

    # Storage for timing results
    l_update_times = []

    print(f"\nBenchmarking L-update step for {n_iters} iterations...")
    print(f"Dimension: {dim}, Degree: {deg}, Alpha: {alpha}, Mode: {mode}")
    print("-" * 60)

    # Benchmark loop
    for i in range(n_iters):
        # Graph learning step (L-update)
        start = time.time()

        # Compute Q and Q_inv with fixed a
        Q = a[0] * Lp[0]
        for p in range(1, deg + 1):
            Q += a[p] * Lp[p]
        Q_inv = torch.linalg.inv(Q)

        # Compute gradient w.r.t. L
        grad_Q = torch.zeros_like(L, device=device)
        grad_C = torch.zeros_like(L, device=device)
        for p in range(1, deg + 1):
            for r in range(p):
                grad_Q -= a[p] * Lp[r] @ Q_inv @ Lp[p - 1 - r]
                grad_C += a[p] * Lp[r] @ C @ Lp[p - 1 - r]
        grad_h_L = grad_Q + grad_C

        # L-update with backtracking line search
        b_t = 100
        found = False
        L_old = L.detach().clone()
        h_old = torch.trace(Q @ C) - torch.logdet(Q)
        idx_L = 0

        while not found and idx_L < 10:
            F_t = b_t * L_old - grad_h_L
            rho = 1e-3
            for _ in range(max_iter):
                Phi_old = Phi.detach().clone()
                K = F_t + rho * (Omega + U)
                Phi = (softshrink(K, alpha) * O + K * I) / (b_t + rho)
                graph_learner.forward(
                    F_t=-(Phi+U),
                    tolerance=1e-5,
                    max_iter=1000
                )
                Omega = graph_learner.L.detach().clone()
                U += Omega - Phi
                err = torch.linalg.norm(Phi - Omega)
                if err < tol:
                    break
            L_new = graph_learner.L.detach().clone()

            # Update Laplacian powers
            Lp[1] = L_new.detach().clone()
            for p in range(2, deg + 1):
                Lp[p] = Lp[p - 1] @ L_new
                Lp[p] = 0.5 * (Lp[p] + Lp[p].T)

            # Compute new Q and inverse
            Q = a[0] * Lp[0]
            for p in range(1, deg + 1):
                Q += a[p] * Lp[p]

            h_new = torch.trace(Q @ C) - torch.logdet(Q)
            suff_dec = h_old + torch.trace(grad_h_L @ (L_new - L_old)) + 0.5 * b_t * torch.linalg.norm(
                L_new - L_old) ** 2
            if h_new <= suff_dec and (torch.linalg.eigvalsh(Q) >= 0).all():
                found = True
                L = L_new
            else:
                b_t *= 10
                graph_learner.L = L_old.detach().clone()
                idx_L += 1

        end = time.time()
        l_update_time = end - start
        l_update_times.append(l_update_time)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"Iteration {i+1:3d}/{n_iters}: L-update time = {l_update_time*1000:.3f} ms")

    # Statistics
    l_update_times = np.array(l_update_times)
    mean_time = np.mean(l_update_times)
    std_time = np.std(l_update_times)
    min_time = np.min(l_update_times)
    max_time = np.max(l_update_times)

    print("-" * 60)
    print(f"\nL-update Benchmark Results (dim={dim}, deg={deg}):")
    print(f"  Mean time per iteration: {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")
    print(f"  Min time: {min_time*1000:.3f} ms")
    print(f"  Max time: {max_time*1000:.3f} ms")
    print(f"  Total time: {np.sum(l_update_times):.3f} s")

    # Save results
    sim_path = benchmark_experiment_path(dim=dim, deg=deg, alpha=alpha, seed=seed)
    ensure_log_dir(sim_path)

    np.save(os.path.join(sim_path, "l_update_times.npy"), l_update_times)
    print(f"\nResults saved to: {sim_path}")
