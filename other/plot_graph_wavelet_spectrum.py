"""
Plot the spectral responses of a wavelet filterbank in PyGSP.

Requirements:
    pip install pygsp matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np
from pygsp import graphs, filters

plt.rcParams.update({
    "text.usetex": True,              # Enable LaTeX
    "font.family": "serif",           # Use serif fonts
    "font.serif": ["Computer Modern"],# Default LaTeX font
})

# 1) Build a graph
G = graphs.Sensor(100, seed=42)   # Random geometric graph in 2D
G.compute_fourier_basis()  # needed to access Laplacian eigenvalues

# 2) Design a wavelet filter bank
Nf = 6
wavelets = filters.MexicanHat(G, Nf=Nf)

# Get eigenvalues (spectrum)
#lambdas = G.e
lambdas = np.linspace(0, G.e.max(), 1000)
response = wavelets.evaluate(lambdas)

# Evaluate each filter at the spectrum points
plt.figure(figsize=(10, 3))
plt.plot(lambdas, response[0], label=f"$\phi(\lambda)$")
for i in range(1, Nf):
    plt.plot(lambdas, response[i], label=f"$\psi_{i}(\lambda)$")

# 3) Plot the filter responses in spectral domain
# plt.title("Graph Wavelet Filterbank Spectrum")
plt.xlabel("Graph Laplacian eigenvalue $\lambda$")
plt.ylabel("Filter response")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
