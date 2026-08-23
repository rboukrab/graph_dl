"""
Plot wavelet atoms at different vertices and scales using PyGSP.

Requirements:
    pip install pygsp matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters

plt.rcParams.update({
    "text.usetex": True,              # Enable LaTeX
    "font.family": "serif",           # Use serif fonts
    "font.serif": ["Computer Modern"],# Default LaTeX font
})

def plot_wavelet_atoms(graph, filter_bank, vertices, out_file="wavelet_atoms.png"):
    """
    graph          : pygsp.graphs.Graph object (with coordinates for plotting)
    filter_bank    : pygsp.filters.Filter instance (e.g., MexicanHat)
    vertices       : list of vertex indices where to localize the wavelet
    out_file       : filename to save the figure
    """
    Nf = filter_bank.Nf  # number of filters (scales)
    nverts = len(vertices)

    # Prepare figure: rows = vertices, cols = scales
    fig, axes = plt.subplots(nverts, Nf, figsize=(3 * Nf, 3 * nverts),
                             squeeze=False)

    for r, v in enumerate(vertices):
        # localize returns an (N, Nf) array: each column is the atom for that scale
        atoms = filter_bank.localize(v, method='chebyshev')  # efficient approx

        for c in range(Nf):
            ax = axes[r, c]
            # Plot the wavelet atom values on the graph
            # G.plot accepts a signal of size G.N to color nodes
            title = f"v={v}  scale={c+1}"
            graph.plot_signal(atoms[:, c], ax=ax, colorbar=True, vertex_size=20)
            ax.set_title("")
            ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    print(f"Saved {out_file}")
    plt.show()


if __name__ == "__main__":
    # 1) choose a graph that has coordinates for plotting
    #    Bunny is a nice 3D example if you installed the data; sensor is simpler.
    G = graphs.Sensor(100, seed=42)   # Use graphs.Sensor(100) or graphs.Grid2d(x, y) if you prefer

    # 2) design a wavelet filter bank. Nf = number of scales
    wavelets = filters.MexicanHat(G, Nf=3)

    # 3) pick some vertices to visualize (indices)
    #    Choose a few nodes spread across the graph
    verts_to_show = [10, 20, 30]

    # 4) plot
    plot_wavelet_atoms(G, wavelets, verts_to_show, out_file="pygsp_wavelet_atoms.png")
