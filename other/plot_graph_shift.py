import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting

plotting.BACKEND = 'matplotlib'
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times",
    "font.size": 12,
    "legend.fontsize": 10,
    "figure.figsize": (10, 5),
    "figure.autolayout": True
})

G2 = graphs.Ring(N=20)
G2.compute_fourier_basis()
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
G2.plot_signal(G2.U[:, 4], ax=axes[0])
G2.plot_signal(np.roll(G2.U[:, 4], 5), ax=axes[1])
G2.set_coordinates('line1D')
axes[2].stem(G2.U[:, 4], label='$\mathbf{x}$')
markerline, stemlines, baseline = axes[2].stem(np.roll(G2.U[:, 4], 5), label='$\mathbf{S}^5\mathbf{x}$')
plt.setp(stemlines, color='orange')
plt.setp(markerline, color='orange', marker='s')
fig.tight_layout()
axes[2].legend()
plt.show()