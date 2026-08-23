import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters

# 1) Build a simple graph and compute its spectrum
G = graphs.Path(50)        # 50-node path graph (good proxy for spectrum)
G.compute_fourier_basis()  # computes G.e (eigenvalues)

# 2) Collect eigenvalues
lam = G.e                    # array of length N

# 3) Define low-pass filters
λc    = 1.0                  # cutoff for ideal & Butterworth
tau   = 1.5                  # diffusion time for heat
order = 4                    # order for Butterworth & polynomial
K     = 10                   # degree for polynomial & Chebyshev

# Ideal low-pass
h_ideal = np.array([1 if l <= λc else 0 for l in lam])

# Heat (diffusion) low-pass: h(l) = exp(-tau * l)
h_heat = np.exp(-tau * lam)

# Butterworth low-pass: 1 / (1 + (l/λc)^(2*order))
h_butt = 1.0 / (1.0 + (lam/λc)**(2*order))

# Polynomial low-pass: approximate ideal by simple truncated Taylor
# here we take h_poly(l) = sum_{k=0}^K (-λc)^(-k) * l^k  (toy example)
a = [(-1)**k / (λc**k) for k in range(K+1)]
h_poly = sum(a[k] * lam**k for k in range(K+1))

# Chebyshev low-pass: compute Chebyshev coefficients to approximate ideal
# use PyGSP helper
h_cheb = filters.Filter(G, lambda x: np.interp(x, G.e, h_ideal))

# 4) Plot all low-pass responses
plt.figure(figsize=(8,5))
plt.plot(lam, h_ideal, 'k-',    label='Ideal LPF')
plt.plot(lam, h_heat,  'C1--',  label='Heat LPF')
plt.plot(lam, h_butt,  'C2-.',  label=f'Butterworth (n={order})')
plt.plot(lam, h_poly,  'C3:',   label=f'Poly deg={K}')
plt.ylim([0, 1.1])
# plt.plot(lam, h_cheb,  'C4-',   label=f'Chebyshev deg={K}')

plt.xlabel(r'$\lambda$ (eigenvalue)')
plt.ylabel(r'$h(\lambda)$ (gain)')
plt.title('Low-Pass Graph Filters (Spectral Response)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()