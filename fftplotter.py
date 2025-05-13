import numpy as np
import matplotlib.pyplot as plt

# Synthetic FFT output (replace with actual output)
n = 1048576
freqs = np.abs(np.random.randn(n))[:n//2]
plt.plot(freqs)
plt.title("FFT Spectrum")
plt.savefig("spectrum.png")
plt.show()
