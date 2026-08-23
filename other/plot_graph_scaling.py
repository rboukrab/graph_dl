import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting

# 1) Path graph of 30 nodes
n = 100
G = graphs.Sensor(n, seed=42)
G.compute_fourier_basis()  # compute eigenvalues (G.e) and eigenvectors (G.U)

# 2) Delta impulse at center node
center = 15
delta = np.zeros(n)
delta[center] = 1.0

# 3) Define two scales s1, s2 and a heat‐kernel filter g(u)=exp(-u)
scales = [0.3, 0.6, 1.5]  # small s→wide in vertex domain, large s→narrow
signals = []

for s in scales:
    # 3a) filter in the spectral domain:  g(s·λ) = exp(-s*λ)
    g_lambda = np.exp(-s * G.e)  # G.e holds all eigenvalues λ_i

    # 3b) apply:  f_s = U [ g(s·λ) * (Uᵀ δ) ]
    x_hat = G.U.T.dot(delta)  # graph Fourier transform of δ
    y_hat = g_lambda * x_hat
    f_s = G.U.dot(y_hat)  # inverse transform back to vertices

    signals.append(f_s)

# 4) Plot side by side on the path layout
pos = np.vstack([np.arange(n), np.zeros(n)]).T

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, f_s, s in zip(axes, signals, scales):
    G.plot_signal(f_s, ax=ax, colorbar=True, limits=[0., 0.08])
    ax.set_title(rf"Scale $s={s}$")
    ax.axis("off")

plt.tight_layout()
plt.show()