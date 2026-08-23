import os
import math
import numpy as np
import matplotlib
from scipy import interpolate
import re

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage

plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "times",
    "font.size": 17,
    # "legend.fontsize": 10,
    # "figure.figsize": [9.50, 5.50],
    "figure.autolayout": True
})
# plt.rc('text.latex', preamble=r'\usepackage{pifont}')

def getImage(path):
   return OffsetImage(plt.imread(path, format="jpg"), zoom=.01)

if __name__ == '__main__':

    dim = 34
    method = 'poly'
    dataset = 'fmri'

    vmin = {
        'f-score': 50,
        'filterr': 0,
        'nae': 0,
        'sp': 0,
        'acc': 50,
        'mod': 0,
        'sig': 0,
    }
    vmax = {
        'f-score': 100,
        'filterr': 0.5,
        'nae': 1.,
        'sp': 200,
        'acc': 100,
        'mod': 0.6,
        'sig': 3,
    }

    if method == 'smooth':
        param_x_name = 'alpha'
        param_y_name = 'beta'
    elif method == 'glama':
        param_x_name = 'gamma'
        param_y_name = 'delta'
    elif method == 'pgl':
        param_x_name = 'rho'
        param_y_name = 'eta'
    elif method == 'poly':
        param_x_name = 'deg'
        param_y_name = 'alpha'

    if method == 'poly':
        pattern_dir = ('gl_%s' % method) + '-%s-dim%s-n%d-%s%d-%s%.3E-s%d'
    else:
        pattern_dir = ('gl_%s' % method) + '-%s-dim%s-n%d-%s%.3E-%s%.3E-s%d'
    if dataset == 'fmri':
        base_dir = "gl_fmri_%s" % method
    else:
        base_dir = "gl_%s" % method
    base_path = os.path.join("logs", base_dir)
    sim_dirs = os.listdir(os.path.join("logs", base_dir))
    pattern_float = r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?"
    pattern_int_s = r"s([-+]?\d+)"
    pattern_int_n = r"n([-+]?\d+)"

    params_x = []
    params_y = []
    seeds = []
    for sim_dir in sim_dirs:
        numbers = re.findall(pattern_float, sim_dir)
        if method == 'poly':
            params_x.append(int(re.findall(r"deg([-+]?\d+)", sim_dir)[0]))
            params_y.append(float(numbers[0]))
        else:
            params_x.append(float(numbers[0]))
            params_y.append(float(numbers[1]))
        seed = re.findall(pattern_int_s, sim_dir)
        seeds.append(int(seed[0]))
    n = 1e6

    params_x = np.sort(np.unique(np.array(params_x)))
    params_y = np.sort(np.unique(np.array(params_y)))
    seeds = np.sort(np.unique(np.array(seeds)))

    metrics_load = {
        'nae': {},
        'f-score': {},
        'sp': {},
        'precision': {},
        'recall': {},
        'time': {},
        'filterr': {},
        'acc': {},
        'mod': {},
        'sig': {},
    }
    for param_x in params_x:
        for param_y in params_y:
            metrics_load['sp'][(param_x, param_y)] = []
            if dataset == 'stat':
                metrics_load['nae'][(param_x, param_y)] = []
                metrics_load['filterr'][(param_x, param_y)] = []
                metrics_load['f-score'][(param_x, param_y)] = []
                metrics_load['sp'][(param_x, param_y)] = []
                metrics_load['precision'][(param_x, param_y)] = []
                metrics_load['recall'][(param_x, param_y)] = []
            elif dataset == 'fmri':
                metrics_load['acc'][(param_x, param_y)] = []
                metrics_load['sig'][(param_x, param_y)] = []
                metrics_load['mod'][(param_x, param_y)] = []
            for seed in seeds:
                sim_path = os.path.join(
                    base_path, pattern_dir % (dataset, dim, n, param_x_name, param_x, param_y_name, param_y, seed)
                )
                if os.path.exists(sim_path):
                    accum_time = np.load(os.path.join(sim_path, "accum_time.npy"))
                    time_indices = np.nonzero(accum_time)[0]
                    if time_indices.size > 0:
                        last_time = accum_time[time_indices[-1]]
                    else:
                        last_time = 0.
                    last_time = 25
                    i_time = np.abs(accum_time - last_time).argmin()

                    sp = np.load(os.path.join(sim_path, "sp.npy"))
                    sp = np.nan_to_num(sp, nan=0.)
                    metrics_load['sp'][(param_x, param_y)].append(sp[i_time])

                    if dataset == 'stat':
                        nae = np.load(os.path.join(sim_path, 'nae.npy'))
                        nae = np.nan_to_num(nae, nan=vmax['nae'])
                        metrics_load['nae'][(param_x, param_y)].append(nae[i_time])

                        if method in ['poly', 'pgl']:
                            filterr = np.load(os.path.join(sim_path, "filterr.npy"))
                            filterr = np.nan_to_num(filterr, nan=vmax['filterr'])
                            metrics_load['filterr'][(param_x, param_y)].append(filterr[i_time])

                        precision = np.load(os.path.join(sim_path, "precision.npy"))
                        recall = np.load(os.path.join(sim_path, "recall.npy"))
                        f1_score = np.load(os.path.join(sim_path, "f1_score.npy"))

                        metrics_load['f-score'][(param_x, param_y)].append(100 * f1_score[i_time])
                        metrics_load['precision'][(param_x, param_y)].append(100 * precision[i_time])
                        metrics_load['recall'][(param_x, param_y)].append(100 * recall[i_time])
                    elif dataset == 'fmri':
                        acc = np.load(os.path.join(sim_path, "acc.npy"))
                        metrics_load['acc'][(param_x, param_y)].append(acc[i_time])

                        sig = np.load(os.path.join(sim_path, "sig.npy"))
                        metrics_load['sig'][(param_x, param_y)].append(sig[i_time])

                        mod = np.load(os.path.join(sim_path, "mod.npy"))
                        metrics_load['mod'][(param_x, param_y)].append(mod[i_time])

    if dataset == 'stat':
        metrics = {
            'nae': {},
            'filterr': {},
            'sp': {},
            'f-score': {},
            'precision': {},
            'recall': {}
        }
    elif dataset == 'fmri':
        metrics = {
            'sp': {},
            'acc': {},
            'sig': {},
            'mod': {},
        }
    for metric in metrics.keys():
        for param_x in params_x:
            for param_y in params_y:
                metrics[metric][(param_x, param_y)] = {
                    'mean': np.array(metrics_load[metric][(param_x, param_y)]).mean(),
                    'std': np.array(metrics_load[metric][(param_x, param_y)]).std()
                }
    if method == 'poly':
        A, B = np.meshgrid(params_y, params_x, indexing='ij')
    else:
        A, B = np.meshgrid(params_x, params_y, indexing='ij')

    if dataset == 'stat':
        plot_metrics = ['nae', 'f-score', 'sp']
        if method in ['poly', 'pgl']:
            plot_metrics.append('filterr')
    elif dataset == 'fmri':
        plot_metrics = ['acc', 'mod', 'sig']
    for metric in plot_metrics:
        if method == 'poly':
            fig = plt.figure(metric, figsize=(4, 2))
        else:
            fig = plt.figure(metric, figsize=(4, 3))
        fig.tight_layout()
        ax = plt.gca()
        # ax.set_title(metric)
        ax.set_axisbelow(True)
        ax.set_xscale("log")
        if method != 'poly':
            ax.set_yscale("log")
            ax.set_xlim([1e-1, 1e2])
        # ax.set_xlabel("$\\%s$" % param_x_name)
        # ax.set_ylabel("$\\%s$" % param_y_name)
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
        ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)

        if method == 'poly':
            C = np.zeros((A.shape[0], A.shape[1]), dtype=float)
            for i, param_y in enumerate(params_y):
                for j, param_x in enumerate(params_x):
                    C[i, j] = metrics[metric][(param_x, param_y)]['mean']

            # sc = ax.scatter(A, B, c=C, cmap='plasma', marker='s', s=10 * (1/A)**0.5, vmin=0, vmax=vmax[metric])
            sc = ax.pcolormesh(A, B, C, shading='auto', cmap='plasma', vmin=vmin[metric], vmax=vmax[metric])
            ax.set_xlim([A.min(), A.max()])
        else:
            C = np.zeros((A.shape[0], A.shape[1]), dtype=float)
            for i, param_x in enumerate(params_x):
                for j, param_y in enumerate(params_y):
                    C[i, j] = metrics[metric][(param_x, param_y)]['mean']

            sc = ax.scatter(A, B, c=C, cmap='plasma', marker='s', s=500, vmin=vmin[metric], vmax=vmax[metric])
            plt.xlim(np.min(A), np.max(A))
            plt.ylim(np.min(B), np.max(B))
            # sc = ax.pcolormesh(A, B, C, shading='auto', cmap='plasma', vmin=0, vmax=vmax[metric])

        # Add colorbar
        cbar = fig.colorbar(sc)
        # cbar.set_label('Magnitude')

    plt.show(block=True)
