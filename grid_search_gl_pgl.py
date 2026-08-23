import argparse
import subprocess
import sys

import numpy as np

# Training settings
parser = argparse.ArgumentParser(
    description='PyTorch fMRI Online Graph Learning under Smoothness Priors'
)
parser.add_argument('--device', type=str, default="cuda:0", help='Device')
parser.add_argument('--data_set', type=str,
                    choices=['stat', 'fmri'], default='fmri',
                    help='Dataset to use')
parser.add_argument('--n_samples', type=int, default=1000000,
                    help='number of samples')
parser.add_argument('--n_iters', type=int, default=10,
                    help='number of iterations')
parser.add_argument('--sim_time', type=float, default=60,
                    help='simulation time in seconds')
parser.add_argument('--test_interval', type=int, default=10)
parser.add_argument('--min_rho_exp', type=float, default=-9)
parser.add_argument('--max_rho_exp', type=float, default=5)
parser.add_argument('--n_rho', type=int, default=8)
parser.add_argument('--min_eta_exp', type=float, default=-5.5)
parser.add_argument('--max_eta_exp', type=float, default=-2)
parser.add_argument('--n_eta', type=int, default=8)

if __name__ == '__main__':

    args = parser.parse_args()
    data_set = args.data_set
    n_samples = args.n_samples
    n_iters = args.n_iters
    sim_time = args.sim_time
    test_interval = args.test_interval
    device = args.device

    rhos = np.logspace(args.min_rho_exp, args.max_rho_exp, args.n_rho)
    etas = np.logspace(args.min_eta_exp, args.max_eta_exp, args.n_eta)

    for rho in rhos:
        for eta in etas:
            for seed in range(1, 11):
                cmd = [
                    sys.executable, 'run_gl.py',
                    '--algorithm', 'pgl',
                    '--device', str(device),
                    '--data_set', str(data_set),
                    '--n_samples', str(n_samples),
                    '--n_iters', str(n_iters),
                    '--sim_time', str(sim_time),
                    '--test_interval', str(test_interval),
                    '--seed', str(seed),
                    '--rho', f'{rho:.3E}',
                    '--eta', f'{eta:.3E}',
                ]
                subprocess.run(cmd, check=False)
