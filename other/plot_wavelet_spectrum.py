import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,              # Enable LaTeX
    "font.family": "serif",           # Use serif fonts
    "font.serif": ["Computer Modern"],# Default LaTeX font
})

# Haar filters
h = np.concatenate((
    np.array([1/np.sqrt(2)] * 2),
    np.array([1/np.sqrt(2)] * 2)
))
g = np.concatenate((
    np.array([1/np.sqrt(2)] * 2),
    np.array([-1/np.sqrt(2)] * 2)
))

nfft = 4096
freqs = np.linspace(0, 0.5, nfft//2)  # normalized frequency [0,1]

# Frequency responses
H = np.fft.fft(h, n=nfft)[:nfft//2]
G = np.fft.fft(g, n=nfft)[:nfft//2]
H_mag = np.abs(H)
G_mag = np.abs(G)

levels = 3  # number of levels for decomposition
fig, (ax1, ax2) = plt.subplots(nrows=2,
                               ncols=1,
                               figsize=(10, 4),
                               sharex=True)   # share the x‐axis if you like

# Scaling response at final level
resp = H_mag.copy()
for j in range(1, levels):
    resp = np.interp(np.linspace(0,1,len(resp)),
                     np.linspace(0,1,len(resp)), resp)
ax1.fill_between(np.linspace(0,0.5 * 0.5**(levels-1),len(resp)//2),
                 0, resp[:len(resp)//2] / max(resp), color='skyblue', alpha=0.5,
                 label=f'Scaling (Level {levels})')

# Wavelet responses at each level
for j in range(1, levels+1):
    resp = G_mag.copy()
    for k in range(1, j):
        resp = np.interp(np.linspace(0,1,len(resp)//2),
                         np.linspace(0,1,len(resp)), resp)
    if j == 1:
        freq_axis = np.linspace(0, 0.5, len(resp)//2)
        ax1.fill_between(freq_axis, 0, resp[:len(resp)//2] / max(resp), alpha=0.4, label=f'Wavelet (Level {j})')
    else:
        freq_axis = np.linspace(0, 0.5 ** (j - 1), len(resp))
    #freq_axis = np.linspace(0,0.5**(j-1),len(resp))
        ax1.fill_between(freq_axis, 0, resp / max(resp), alpha=0.4, label=f'Wavelet (Level {j})')

ax1.set_ylabel('Normalized magnitude')
ax1.set_title('Multilevel Magnitude Frequency Responses (Haar Wavelet)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.5)

# Scaling responses at each level
for j in range(1, levels+1):
    resp = H_mag.copy()
    for k in range(1, j):
        resp = np.interp(np.linspace(0,1,len(resp)//2),
                         np.linspace(0,1,len(resp)), resp)
    freq_axis = np.linspace(0,0.5 * 0.5**(j-1),len(resp)//2)
    ax2.fill_between(freq_axis, 0, resp[:len(resp)//2] / max(resp), alpha=0.4, label=f'Scaling (Level {j})')

ax2.set_xlabel('Normalized Frequency ($\\times\\pi$ rad/sample)')
ax2.set_ylabel('Normalized magnitude')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.5)
plt.tight_layout()

plt.show()
