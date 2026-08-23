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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times",
    "font.size": 15,
    # "legend.fontsize": 10,
    # "figure.figsize": [9.50, 5.50],
    "figure.autolayout": True
})
# plt.rc('text.latex', preamble=r'\usepackage{pifont}')

def getImage(path):
   return OffsetImage(plt.imread(path, format="jpg"), zoom=.01)

if __name__ == '__main__':

    dim = 20
    dataset = 'stat'

    methods = ['smooth', 'glama', 'pgl', 'poly1', 'poly2', 'poly3', 'poly4', 'poly5']
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
    lims = {
        'nae': [0.12, 0.75],
        'f-score': [74, 103],
        'filterr': [0, 0.5],
        'sp': [75, 185],
    }
    metric_labels = {
        'nae': 'NAE',
        'f-score': 'F-score',
        'filterr': 'FiltErr',
        'sp': 'NNZ',
    }
    method_labels = {
        'smooth': 'Smooth',
        'glama': 'GLAMA',
        'pgl': 'PGL',
        'poly1': 'Poly, $P=1$',
        'poly2': 'Poly, $P=2$',
        'poly3': 'Poly, $P=3$',
        'poly4': 'Poly, $P=4$',
        'poly5': 'Poly, $P=5$'
    }

    metrics = {}

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

        if method.startswith('poly'):
            pattern_dir = 'gl_poly-%s-dim%s-n%d-%s%d-%s%.3E-s%d'
        else:
            pattern_dir = ('gl_%s' % method) + '-%s-dim%s-n%d-%s%.3E-%s%.3E-s%d'
        base_dir = "var_%s" % method
        base_path = os.path.join("logs", base_dir)
        sim_dirs = os.listdir(os.path.join("logs", base_dir))
        pattern_float = r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?"
        pattern_int = r"([-+]?\d+)"

        params_x = []
        params_y = []
        seeds = []
        ns = []
        for sim_dir in sim_dirs:
            numbers = re.findall(pattern_float, sim_dir)
            if method.startswith('poly'):
                params_x.append(int(re.findall(r"deg([-+]?\d+)", sim_dir)[0]))
                params_y.append(float(numbers[0]))
            else:
                params_x.append(float(numbers[0]))
                params_y.append(float(numbers[1]))
            ints = re.findall(pattern_int, sim_dir)
            seeds.append(int(ints[-1]))
            ns.append(int(ints[1]))

        param_x = np.sort(np.unique(np.array(params_x)))[0]
        param_y = np.sort(np.unique(np.array(params_y)))[0]
        seeds = np.sort(np.unique(np.array(seeds)))
        ns = np.sort(np.unique(np.array(ns)))

        metrics_load = {
            'nae': {},
            'f-score': {},
            'sp': {},
            'filterr': {}
        }
        for n in ns:
            metrics_load['nae'][n] = []
            metrics_load['f-score'][n] = []
            metrics_load['sp'][n] = []
            metrics_load['filterr'][n] = []
            for seed in seeds:
                if dataset == 'stat':
                    sim_path = os.path.join(
                        base_path, pattern_dir % (dataset, dim, n, param_x_name, param_x, param_y_name, param_y, seed)
                    )
                    if os.path.exists(sim_path):
                        accum_time = np.load(os.path.join(sim_path, "accum_time.npy"))
                        time_indices = np.nonzero(accum_time)[0]
                        i_time = time_indices[-1]

                        nae = np.load(os.path.join(sim_path, 'nae.npy'))
                        metrics_load['nae'][n].append(nae[i_time])

                        if method in ['poly1', 'poly2', 'poly3', 'poly4', 'poly5', 'pgl']:
                            filterr = np.load(os.path.join(sim_path, "filterr.npy"))
                            metrics_load['filterr'][n].append(filterr[i_time])

                        sp = np.load(os.path.join(sim_path, "sp.npy"))
                        metrics_load['sp'][n].append(sp[i_time])

                        f1_score = np.load(os.path.join(sim_path, "f1_score.npy"))
                        metrics_load['f-score'][n].append(100 * f1_score[i_time])

        metrics[method] = {
            'nae': {},
            'sp': {},
            'filterr': {},
            'f-score': {},
        }
        for metric in metrics[method].keys():
            metrics[method][metric] = {
                'mean': [],
                'std': []
            }
            for n in ns:
                metrics[method][metric]['mean'].append(np.array(metrics_load[metric][n]).mean())
                metrics[method][metric]['std'].append(np.array(metrics_load[metric][n]).std())

    plot_metrics = ['nae', 'f-score', 'filterr']
    fig, ax = plt.subplots(len(plot_metrics), 1, sharex=True)
    fig.tight_layout()
    for i, metric in enumerate(plot_metrics):
        for method in methods:
            if (metric == 'filterr' and (method.startswith('poly') or method == 'pgl')) or (metric != 'filterr'):
                # ax.set_title(metric)
                ax[i].set_axisbelow(True)
                ax[i].set_xscale("log")
                # if method != 'poly':
                #     ax.set_yscale("log")
                #     ax.set_xlim([1e-1, 1e2])
                if metric == 'sp':
                    ax[i].set_xlabel('Number of samples ($N$)')
                ax[i].set_ylabel("%s" % metric_labels[metric])
                ax[i].grid(which='major', color='#DDDDDD', linewidth=0.8)
                ax[i].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)

                x = ns
                y = np.array(metrics[method][metric]['mean'])
                y_std = np.array(metrics[method][metric]['std'])

                if method.startswith('poly'):
                    ax[i].plot(
                        x, y,
                        linewidth=1.,
                        color=colors[method],
                        label=method_labels[method]
                    )
                    # Add an image marker at each point
                    for (xi, yi) in zip(x, y):
                        # if int(xi) > 0 and int(xi) % markevery[method] == 0 and np.abs(xi - int(xi)) < 0.1:
                        ab = AnnotationBbox(markers[method], (xi, yi), frameon=False)
                        ax[i].add_artist(ab)
                else:
                    ax[i].plot(
                        x, y,
                        linewidth=1.,
                        color=colors[method],
                        marker=markers[method],
                        # markevery=markevery[method],
                        label=method_labels[method]
                    )
                ax[i].fill_between(
                    x,
                    y - 0.5 * y_std,
                    y + 0.5 * y_std,
                    alpha=0.1,
                    color=colors[method],
                    edgecolor='none'
                )

        ax[i].set_ylim([lims[metric][0], lims[metric][1]])

        if metric == 'nae':
            ax[i].legend()

        # Add colorbar
        # cbar.set_label('Magnitude')

    plt.show(block=True)
