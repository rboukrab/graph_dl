import numpy           as np
import matplotlib.pyplot as plt
from pygsp            import graphs, filters, plotting

plt.rcParams.update({
    "text.usetex": True,              # Enable LaTeX
    "font.family": "serif",           # Use serif fonts
    "font.serif": ["Computer Modern"],# Default LaTeX font
})

# 1) Load graph and Fourier basis
np.random.seed(42)
G = graphs.Sensor(100, seed=42)
G.compute_fourier_basis()

# 2) Heat-diffuse an impulse at idx0
idx0 = 10

tau    = 10.0
heat   = filters.Heat(G, tau)   # H_tau = exp(-tau L)
f_heat = heat.evaluate(G.e)[0]    # f = H_tau * delta0
original_signal = G.U @ f_heat

# 3) Translate f_heat to a new node idx1
idx1   = 10
u_j    = G.U[idx1, :]           # eigenvector values u_ℓ(idx1)
t_hat  = f_heat * u_j            # multiply each coefficient by u_ℓ(idx1)
f_translated = G.igft(t_hat)    # inverse GFT → translated signal

idx2   = 40
u_j    = G.U[idx2, :]           # eigenvector values u_ℓ(idx1)
t_hat  = f_heat * u_j            # multiply each coefficient by u_ℓ(idx1)
f_translated2 = G.igft(t_hat)    # inverse GFT → translated signal

idx2   = 55
u_j    = G.U[idx2, :]           # eigenvector values u_ℓ(idx1)
t_hat  = f_heat * u_j            # multiply each coefficient by u_ℓ(idx1)
f_translated3 = G.igft(t_hat)    # inverse GFT → translated signal

# 4) Plot original and translated signals
fig, axes = plt.subplots(1, 3, figsize=(12,5))

G.plot_signal(f_translated, ax=axes[0])
axes[0].axis('off')
axes[0].set_title('')
G.plot_signal(f_translated2, ax=axes[1])
axes[1].axis('off')
axes[1].set_title('')
G.plot_signal(f_translated3, ax=axes[2])
axes[2].axis('off')
axes[2].set_title('')

plt.tight_layout()
plt.show()