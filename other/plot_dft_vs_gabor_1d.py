import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

plt.rcParams.update({
    "text.usetex": True,              # Enable LaTeX
    "font.family": "serif",           # Use serif fonts
    "font.serif": ["Computer Modern"],# Default LaTeX font
})

# --- Generate a test signal ---
fs = 500          # Sampling frequency
t = np.linspace(0, 2, 2*fs, endpoint=False)  # 2 seconds
# Signal: low frequency (10 Hz) for first second, high frequency (40 Hz) for second
signal = np.concatenate([np.sin(2*np.pi*10*t[:int(0.75*fs)]) * np.exp(-3*t[:int(0.75*fs)]), np.sin(2*np.pi*40*t[int(0.75*fs):]) * np.exp(-3 * t[:int(1.25*fs)])])

# --- Discrete Fourier Transform (global frequency content) ---
freqs = np.fft.fftfreq(len(signal), 1/fs)
spectrum = np.fft.fft(signal)

# --- Short-Time Fourier Transform (Gabor transform) ---
f, tt, Zxx = stft(signal, fs=fs, nperseg=128)

# --- Plot ---
plt.figure(figsize=(12, 8))

# # Original signal
# plt.subplot(3,1,1)
# plt.plot(t, signal)
# plt.title("Original Signal")
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
#
# # DFT (no time localization)
# plt.subplot(3,1,2)
# plt.plot(freqs[:len(freqs)//2], np.abs(spectrum[:len(spectrum)//2]))
# plt.title("DFT: Global Frequency Spectrum")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Magnitude")

# Gabor/STFT (time-frequency localization)
plt.subplot(3,1,3)
plt.pcolormesh(tt, f, np.abs(Zxx), shading='gouraud')
# plt.title("Gabor Transform (STFT): Time-Frequency Representation")
# plt.ylabel("Frequency [Hz]")
# plt.xlabel("Time [s]")
# plt.colorbar(label="Magnitude")
plt.axis('off')

plt.tight_layout()
plt.show()
