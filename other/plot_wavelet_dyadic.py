import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import morlet, ricker

plt.rcParams.update({
    "text.usetex": True,              # Enable LaTeX
    "font.family": "serif",           # Use serif fonts
    "font.serif": ["Computer Modern"],# Default LaTeX font
})

# 1. Morlet wavelet (continuous)
#    Use scipy.signal.morlet: generates a complex Morlet; take the real part.
width = 5.0
num_points = 400
t_morlet = np.linspace(-1, 1, num_points)
morlet_wv = np.real(morlet(num_points, w=width))

# 2. Mexican Hat (Ricker) wavelet
#    Use scipy.signal.ricker: its "points" param is the length, and "a" is the width.
a = 12.0
mexhat_wv = ricker(num_points, a)
t_mexhat = np.linspace(-1, 1, num_points)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(8, 2), constrained_layout=True)

axes[0].plot(t_morlet, morlet_wv, lw=2, color='C1')
axes[0].set_title(f'Morlet Wavelet (w={width})')
axes[0].set_xlim(t_morlet.min(), t_morlet.max())
axes[0].set_ylim(-0.8, 0.8)
axes[0].grid(True)

axes[1].plot(t_mexhat, 2.5 * mexhat_wv, lw=2, color='C2')
axes[1].set_title(f'Mexican Hat (Ricker) Wavelet (a={a})')
axes[1].set_xlim(t_mexhat.min(), t_mexhat.max())
axes[1].set_ylim(-0.8, 0.8)
axes[1].grid(True)

plt.show()