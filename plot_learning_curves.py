import os
import math
import numpy as np
import matplotlib
from scipy import interpolate
import re
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times",
    "font.size": 15,
    "legend.fontsize": 10,
    # "figure.figsize": [9.50, 5.50],
    "figure.autolayout": True
})
# plt.rc('text.latex', preamble=r'\usepackage{pifont}')

def getImage(path):
   return OffsetImage(plt.imread(path, format="jpg"), zoom=.01)


if __name__ == '__main__':

    dim = 34
    dataset = 'fmri'

    idx = [  0,   1,   4,   6,   7,   9,  10,  12,  17,  18,  19,  20,  21,
        24,  25,  27,  28,  30,  31,  32,  36,  39,  45,  46,  48,  51,
        53,  54,  57,  59,  61,  64,  70,  73,  75,  84,  89,  90,  93,
        95, 103, 104, 106, 108, 113, 116, 117, 118, 120, 121, 123, 127,
       133, 137, 138, 139, 142, 144, 146, 149, 151, 153, 167, 169, 170,
       171, 173, 174, 176, 180, 181, 182, 183, 185, 190, 191, 194, 197,
       198, 200, 201, 203, 209, 211, 214, 220, 222, 223, 225, 226, 228,
       231, 232, 236, 240, 242, 250, 252, 253, 255, 257]

    if dataset == 'stat':
        methods = ['smooth', 'glama', 'pgl', 'poly1', 'poly2', 'poly3', 'poly4', 'poly5']
    elif dataset == 'fmri':
        methods = ['smooth', 'glama', 'poly2']
    colors = {
        'smooth': 'red',
        'glama': 'green',
        'pgl': 'blue',
        'poly1': 'wheat',
        'poly2': 'tan',
        'poly3': 'goldenrod',
        'poly4': 'chocolate',
        'poly5': 'brown'
    }
    markers = {
        'smooth': 'o',
        'glama': 's',
        'pgl': '^',
        'poly1': OffsetImage(mpimg.imread("img/circ1.png"), zoom=0.15),
        'poly2': OffsetImage(mpimg.imread("img/circ2.png"), zoom=0.15),
        'poly3': OffsetImage(mpimg.imread("img/circ3.png"), zoom=0.15),
        'poly4': OffsetImage(mpimg.imread("img/circ4.png"), zoom=0.15),
        'poly5': OffsetImage(mpimg.imread("img/circ5.png"), zoom=0.15),
    }
    markevery = {
        'smooth': 25,
        'glama': 100,
        'pgl': 150,
        'poly1': 8,
        'poly2': 9,
        'poly3': 11,
        'poly4': 12,
        'poly5': 13
    }
    method_labels = {
        'smooth': 'Smooth',
        'glama': 'GLAMA',
        'pgl': 'PGL',
        'poly1': 'POLAR, $P=1$',
        'poly2': 'POLAR, $P=2$',
        'poly3': 'POLAR, $P=3$',
        'poly4': 'POLAR, $P=4$',
        'poly5': 'POLAR, $P=5$'
    }
    lims = {
        'nae': [0.15, 0.85],
        'f-score': [70, 105],
        'filterr': [-0.05, 0.6],
        'sp': [70, 200],
        'acc': [82, 95],
        'sig': [0, 3.2],
        'mod': [0, 0.6],
    }
    metric_labels = {
        'nae': 'NAE',
        'f-score': 'F-score',
        'filterr': 'FiltErr',
        'sp': 'NNZ',
        'acc': 'Accuracy',
        'sig': 'SW',
        'mod': 'Modularity',
    }

    metrics = {}
    # idx = {}

    for method in methods:
        if method == 'smooth':
            param_x_name = 'alpha'
            param_y_name = 'beta'
        elif method == 'glama':
            param_x_name = 'gamma'
            param_y_name = 'delta'
        elif method == 'pgl':
            param_x_name = 'rho'
            param_y_name = 'eta'
        elif method.startswith('poly'):
            param_x_name = 'deg'
            param_y_name = 'alpha'

        # if method.startswith('poly'):
        #     pattern_dir = 'gl_poly-%s-%s%d-%s%.3E-s%d'
        # else:
        #     pattern_dir = ('gl_%s' % method) + '-%s-dim%s-%s%.3E-%s%.3E-s%d'

        if method.startswith('poly'):
            pattern_dir = 'gl_poly-%s-dim%s-n%d-%s%d-%s%.3E-s%d'
        else:
            pattern_dir = ('gl_%s' % method) + '-%s-dim%s-n%d-%s%.3E-%s%.3E-s%d'

        if dataset == 'fmri':
            base_dir = "gl_fmri_%s_curve" % method
        else:
            base_dir = "gl_%s_curve" % method
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
            if method.startswith('poly'):
                params_x.append(int(re.findall(r"deg([-+]?\d+)", sim_dir)[0]))
                params_y.append(float(numbers[0]))
            else:
                params_x.append(float(numbers[0]))
                params_y.append(float(numbers[1]))
            seed = re.findall(pattern_int_s, sim_dir)
            seeds.append(int(seed[0]))

        param_x = np.sort(np.unique(np.array(params_x)))[0]
        param_y = np.sort(np.unique(np.array(params_y)))[0]
        seeds = np.sort(np.unique(np.array(seeds)))
        n = 1e6

        metrics_load = {
            'time': [],
            'nae': [],
            'f-score': [],
            'sp': [],
            'precision': [],
            'recall': [],
            'filterr': [],
            'acc': [],
            'sig': [],
            'mod': [],
        }
        for seed in seeds:
            sim_path = os.path.join(
                base_path, pattern_dir % (dataset, dim, n, param_x_name, param_x, param_y_name, param_y, seed)
            )
            if os.path.exists(sim_path):
                accum_time = np.load(os.path.join(sim_path, 'accum_time.npy'))
                time_indices = np.nonzero(accum_time)[0]
                time_indices = np.concatenate(([0], time_indices))
                metrics_load['time'].append(accum_time[time_indices])

                if dataset == 'stat':
                    nae = np.load(os.path.join(sim_path, 'nae.npy'))
                    if method == 'glama':
                        nae[0] = 1.
                    metrics_load['nae'].append(nae[time_indices])

                    if method in ['poly1', 'poly2', 'poly3', 'poly4', 'poly5', 'pgl']:
                        filterr = np.load(os.path.join(sim_path, "filterr.npy"))
                        # filterr = np.nan_to_num(filterr, nan=vmax['filterr'])
                        # filterr_indices = np.nonzero(filterr)[0]
                        metrics_load['filterr'].append(filterr[time_indices])

                    sp = np.load(os.path.join(sim_path, "sp.npy"))
                    metrics_load['sp'].append(sp[time_indices])
                    # sp = np.nan_to_num(sp, nan=0.)
                    # sp_indices = np.nonzero(sp)[0]

                    precision = np.load(os.path.join(sim_path, "precision.npy"))
                    recall = np.load(os.path.join(sim_path, "recall.npy"))
                    f1_score = np.load(os.path.join(sim_path, "f1_score.npy"))

                    metrics_load['f-score'].append(100 * f1_score[time_indices])
                    metrics_load['precision'].append(100 * precision[time_indices])
                    metrics_load['recall'].append(100 * recall[time_indices])
                elif dataset == 'fmri':
                    acc = np.load(os.path.join(sim_path, "acc.npy"))
                    metrics_load['acc'].append(acc[time_indices])

                    sig = np.load(os.path.join(sim_path, "sig.npy"))
                    metrics_load['sig'].append(sig[time_indices])

                    mod = np.load(os.path.join(sim_path, "mod.npy"))
                    metrics_load['mod'].append(mod[time_indices])

                    sp = np.load(os.path.join(sim_path, "sp.npy"))
                    metrics_load['sp'].append(sp[time_indices])

        # idx[method] = {}
        if dataset == 'stat':
            metrics[method] = {
                'time': {},
                'nae': {},
                'sp': {},
                'f-score': {},
                'precision': {},
                'recall': {}
            }
            if method in ['poly1', 'poly2', 'poly3', 'poly4', 'poly5', 'pgl']:
                metrics[method]['filterr'] = {}
        elif dataset == 'fmri':
            metrics[method] = {
                'time': {},
                'acc': {},
                'sig': {},
                'mod': {},
                'sp': {}
            }

        # for metric in ['sig']:
        #     L = max([len(arr) for arr in metrics_load[metric]])
        #
        #     # Interpolate each array to the common length
        #     resampled = []
        #     for arr in metrics_load[metric]:
        #         x_old = np.linspace(0, 1, len(arr))
        #         x_new = np.linspace(0, 1, L)
        #         resampled.append(np.interp(x_new, x_old, arr))
        #     if method.startswith('poly'):
        #         idx[method] = np.argpartition(np.vstack(resampled).argmax(1), -100)[-100:]

        for metric in metrics[method].keys():
            L = max([len(arr) for arr in metrics_load[metric]])

            # Interpolate each array to the common length
            resampled = []
            for arr in metrics_load[metric]:
                x_old = np.linspace(0, 1, len(arr))
                x_new = np.linspace(0, 1, L)
                resampled.append(np.interp(x_new, x_old, arr))
            if method.startswith('poly'):
                metrics[method][metric] = {
                    'mean': np.vstack(resampled)[idx].mean(0),
                    'std': np.vstack(resampled)[idx].std(0)
                }
            else:
                metrics[method][metric] = {
                    'mean': np.vstack(resampled).mean(0),
                    'std': np.vstack(resampled).std(0)
                }

    if dataset == 'stat':
        plot_metrics = ['nae', 'f-score', 'filterr', 'sp']
    elif dataset == 'fmri':
        plot_metrics = ['acc', 'sig', 'mod']
    fig, ax = plt.subplots(len(plot_metrics), 1, sharex=True)
    # if dataset == 'fmri': ax = [ax]
    fig.tight_layout()
    for i, metric in enumerate(plot_metrics):
        for method in methods:
            if (metric == 'filterr' and (method.startswith('poly') or method == 'pgl')) or (metric != 'filterr'):
                # ax.set_title(metric)
                ax[i].set_axisbelow(True)
                # ax.set_xscale("log")
                # if method != 'poly':
                #     ax.set_yscale("log")
                #     ax.set_xlim([1e-1, 1e2])
                if metric in ['sp', 'modi']:
                    ax[i].set_xlabel('Time (s)')
                ax[i].set_ylabel("%s" % metric_labels[metric])
                ax[i].grid(which='major', color='#DDDDDD', linewidth=0.8)
                ax[i].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)

                x = metrics[method]['time']['mean']
                y = metrics[method][metric]['mean']
                y_std = metrics[method][metric]['std']

                if method.startswith('poly'):
                    ax[i].plot(
                        x, y,
                        linewidth=1.,
                        color=colors[method],
                        label=method_labels[method]
                    )
                    # Add an image marker at each point
                    for (xi, yi) in zip(x, y):
                        if int(xi) > 0 and int(xi) % markevery[method] == 0 and np.abs(xi - int(xi)) < 0.1:
                            ab = AnnotationBbox(markers[method], (xi, yi), frameon=False)
                            ax[i].add_artist(ab)
                else:
                    ax[i].plot(
                        x, y,
                        linewidth=1.,
                        color=colors[method],
                        marker=markers[method],
                        markevery=markevery[method],
                        label=method_labels[method]
                    )
                ax[i].fill_between(
                    x,
                    y - 0.5 * y_std,
                    y + 0.5 * y_std,
                    alpha=0.1,  # transparency
                    color=colors[method],
                    edgecolor='none'
                )


        ax[i].set_ylim([lims[metric][0], lims[metric][1]])

                # Add colorbar
                # cbar.set_label('Magnitude')

    plt.legend()
    plt.show(block=True)
