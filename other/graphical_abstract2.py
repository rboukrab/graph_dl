import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting

# 1. Create a random sensor graph
G = graphs.Sensor(50, seed=7)
G.compute_laplacian()

# 2. Generate a random noise signal
np.random.seed(7)
signal = np.random.randn(G.N)

# 3. Define a low-pass filter (e.g., heat kernel or exponential window)
g = filters.Heat(G, tau=10)

# 4. Filter the signal
filtered_signal = g.filter(signal)

# 5. Plot original vs filtered signals
G.plot_signal(signal)
plt.title("Original Random Noise")

G.plot_signal(filtered_signal)
plt.title("Filtered Signal (Low-pass)")

plt.show()
