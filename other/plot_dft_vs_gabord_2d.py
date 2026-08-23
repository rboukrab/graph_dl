import numpy as np
import matplotlib.pyplot as plt
from skimage.data import camera
from skimage.color import rgb2gray
from skimage.filters import gabor
from numpy.fft import fftshift, fft2

plt.rcParams.update({
    "text.usetex": True,              # Enable LaTeX
    "font.family": "serif",           # Use serif fonts
    "font.serif": ["Computer Modern"],# Default LaTeX font
})

# --- Load an example image ---
image = camera()  # grayscale test image (512x512)

# --- Global Fourier Transform (DFT) ---
F = fftshift(fft2(image))
magnitude_spectrum = np.log1p(np.abs(F))  # log for visibility

# --- Gabor Transform (local frequency filters) ---
# Apply Gabor filters at a few orientations and frequencies
frequencies = [0.1, 0.2, 0.3]  # spatial frequencies
thetas = [0, np.pi / 4, np.pi / 2]  # orientation in radians (0 = horizontal)
thetas_labels = ['0', '45º', '90º']
gabor_results = [gabor(image, frequency=frequencies[i], theta=thetas[i])[0] for i in range(3)]

# --- Plot ---
plt.figure(figsize=(12, 8))

# Original image
plt.subplot(1, len(frequencies)+2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Fourier spectrum
plt.subplot(1, len(frequencies)+2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("DFT (Spectrum)")
plt.axis("off")

# Gabor filter responses
for i, f in enumerate(frequencies):
    plt.subplot(1, len(frequencies)+2, i+3)
    plt.imshow(gabor_results[i], cmap='gray')
    plt.title(f"Gabor $f$={f}, $\\theta$={thetas_labels[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
