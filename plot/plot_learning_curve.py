"""Plot learning curves from a single graph-learning experiment log directory.

Loads the .npy metric files saved by the runners and plots F-score, precision,
recall, and NAE as a function of metric step (iteration / test interval).

Example
-------
python plot/plot_learning_curve.py --log_dir logs/gl_smooth-smooth-dim20-n10000-alpha1.000E-02-beta3.162E+00-s1
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot F-score, precision, recall and NAE from a log directory"
    )
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Path to the experiment log directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: <log_dir>/learning_curve.png)")
    parser.add_argument("--x_axis", type=str, default="step",
                        choices=["step", "time"],
                        help="Plot against metric step or accumulated time")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.log_dir, "learning_curve.png")

    metric_files = {
        "F-score": "f1_score.npy",
        "Precision": "precision.npy",
        "Recall": "recall.npy",
        "NAE": "nae.npy",
    }

    x_label = "Metric step"
    x = None
    if args.x_axis == "time":
        time_path = os.path.join(args.log_dir, "accum_time.npy")
        if os.path.exists(time_path):
            x = np.load(time_path)
            x_label = "Time (s)"

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    for ax, (label, fname) in zip(axes, metric_files.items()):
        path = os.path.join(args.log_dir, fname)
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue

        y = np.load(path)
        steps = np.arange(len(y)) if x is None else x[: len(y)]
        ax.plot(steps, y, linewidth=1.5)
        ax.set_ylabel(label)
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        if label in ("F-score", "Precision", "Recall"):
            ax.set_ylim([0.0, 1.05])

    axes[-2].set_xlabel(x_label)
    axes[-1].set_xlabel(x_label)
    fig.suptitle(os.path.basename(args.log_dir))
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(args.output, dpi=150)
    print(f"Saved learning curve to {args.output}")


if __name__ == "__main__":
    main()
