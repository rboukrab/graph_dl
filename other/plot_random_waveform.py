import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility (optional)
np.random.seed(7)

# Generate time axis
duration = 2.0  # seconds
sample_rate = 1000  # samples per second
t = np.linspace(0, duration, int(duration * sample_rate))

# Generate random waveform by combining multiple random sine waves
num_components = 5
waveform = np.zeros_like(t)

for _ in range(num_components):
    # Random frequency between 1 and 20 Hz
    frequency = np.random.uniform(1, 8)
    # Random amplitude between 0.1 and 1.0
    amplitude = np.random.uniform(0.1, 1.0)
    # Random phase
    phase = np.random.uniform(0, 2 * np.pi)
    
    waveform += amplitude * np.sin(2 * np.pi * frequency * t + phase)

# Add some random noise
noise = np.random.normal(0, 0.1, len(t))
waveform += noise

# Create the plot
plt.figure(figsize=(12, 6))
plt.fill_between(t, waveform, alpha=0.3, color='#ed839e')
plt.plot(t, waveform, linewidth=1.5, color='#ed839e')
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.title('Randomly Generated Waveform', fontsize=14, fontweight='bold')
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()

# Show the plot
plt.show()

