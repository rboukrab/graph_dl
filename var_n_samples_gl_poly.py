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
                    choices=['stat', 'fmri'], default='stat',
                    help='Dataset to use')
parser.add_argument('--n_iters', type=int, default=1,
                    help='number of iterations')
parser.add_argument('--sim_time', type=float, default=9999999,
                    help='simulation time in seconds')
parser.add_argument('--test_interval', type=int, default=10)
parser.add_argument('--alpha', type=float, default=3)
parser.add_argument('--deg', type=int, default=1)
parser.add_argument('--min_n_exp', type=int, default=2)
parser.add_argument('--max_n_exp', type=int, default=6)
parser.add_argument('--n_n', type=int, default=9)

if __name__ == '__main__':

    args = parser.parse_args()
    data_set = args.data_set
    n_iters = args.n_iters
    sim_time = args.sim_time
    test_interval = args.test_interval
    device = args.device

    deg = args.deg
    alpha = args.alpha

    ns = np.logspace(args.min_n_exp, args.max_n_exp, args.n_n).astype(int)

    for n_samples in ns:
        for seed in range(1, 101):
            cmd = [
                sys.executable, 'gl_poly.py',
                '--device', str(device),
                '--data_set', str(data_set),
                '--n_iters', str(n_iters),
                '--n_samples', str(n_samples),
                '--sim_time', str(sim_time),
                '--test_interval', str(test_interval),
                '--seed', str(seed),
                '--deg', str(deg),
                '--alpha', f'{alpha:.3E}',
            ]
            subprocess.run(cmd, check=False)
