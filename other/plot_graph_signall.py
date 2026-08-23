from pygsp import graphs, filters
G = graphs.Logo()
G.estimate_lmax()
g = filters.Heat(G, tau=1000)

import numpy as np
import matplotlib.pyplot as plt
lambdas = [0, 1, 5, 10, 20, 500]
for lambd in lambdas:
    fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    # DELTAS = [20, 500, 1000]
    # s = np.zeros(G.N)
    # s[DELTAS] = 1
    # s = g.filter(s)
    G.plot_signal(G.U[:, lambd], ax=axes, vertex_size=30, limits=[0.02, 0.04])
    plt.show()