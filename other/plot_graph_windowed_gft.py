"""
Windowed Graph Fourier Transform (WGFT) example with PyGSP
"""

import matplotlib.pyplot as plt
import numpy as np
from pygsp import graphs, filters

plt.rcParams.update({
    "text.usetex": True,              # Enable LaTeX
    "font.family": "serif",           # Use serif fonts
    "font.serif": ["Computer Modern"],# Default LaTeX font
})

# 1) Graph
G = graphs.Sensor(100, seed=42)
G = graphs.StochasticBlockModel(N=100, k=3, p=0.5, q=0.01, seed=42)
G.set_coordinates(seed=42)
heat = filters.Heat(G, tau=5)
signal = heat.localize(20) * G.U[:, 2] + heat.localize(55) * G.U[:, 7] + heat.localize(80) * G.U[:, 5]
# G.plot_signal(signal)
# plt.show(block=True)
G.compute_fourier_basis()

# 2) Signal (something oscillatory on the graph)
# signal = np.sin(3 * np.linspace(0, np.pi, G.N))

# 3) Define a window (heat kernel centered at vertex v0)
window = filters.Heat(G, tau=30)    # heat kernel filter

# 4) Windowed Graph Fourier Transform
# Multiply signal by the shifted window (localization at each vertex)
# Then project into graph Fourier basis
WGFT = np.zeros((G.N, G.N))  # rows: vertex, cols: frequency
for v in range(G.N):
    wv = window.localize(v)
    localized = signal * wv
    # Project into Fourier basis
    spec = G.U.T @ localized   # G.U are eigenvectors
    WGFT[v, :] = np.abs(spec)  # magnitude

# 5) Plot Vertex–Frequency representation
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
G.plot_signal(signal, ax=ax[0])
ax[0].set_title("")
ax[0].axis('off')
plt.imshow(WGFT.T, aspect='auto', origin='lower',
           extent=[0, G.N-1, 0, G.N-1], cmap='viridis', axes=ax[1])
plt.colorbar(label="")
plt.xlabel("Vertex index")
plt.ylabel("Graph frequency index")
# plt.title("Windowed Graph Fourier Transform (WGFT)")
plt.tight_layout()
plt.show()
