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
parser.add_argument('--n_iters', type=int, default=1,
                    help='number of iterations')
parser.add_argument('--max_iter', type=int, default=50000,
                    help='number of iterations')
parser.add_argument('--sim_time', type=float, default=9999999,
                    help='simulation time in seconds')
parser.add_argument('--test_interval', type=int, default=10)
parser.add_argument('--min_deg', type=int, default=5)
parser.add_argument('--max_deg', type=int, default=5)
parser.add_argument('--min_alpha_exp', type=float, default=0.833338389)
parser.add_argument('--max_alpha_exp', type=float, default=0.833338389)
parser.add_argument('--n_alpha', type=int, default=1)

if __name__ == '__main__':

    args = parser.parse_args()
    data_set = args.data_set
    n_iters = args.n_iters
    sim_time = args.sim_time
    test_interval = args.test_interval
    device = args.device

    degs = np.arange(args.min_deg, args.max_deg + 1)
    alphas = np.logspace(args.min_alpha_exp, args.max_alpha_exp, args.n_alpha)

    for deg in degs:
        for alpha in alphas:
            for seed in range(1, 301):
                cmd = [
                    sys.executable, 'run_gl.py',
                    '--algorithm', 'poly',
                    '--device', str(device),
                    '--data_set', str(data_set),
                    '--n_iters', str(n_iters),
                    '--sim_time', str(sim_time),
                    '--test_interval', str(test_interval),
                    '--seed', str(seed),
                    '--deg', str(deg),
                    '--alpha', f'{alpha:.3E}',
                ]
                subprocess.run(cmd, check=False)
