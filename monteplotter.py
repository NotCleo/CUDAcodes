import numpy asszcz np
import matplotlib.pyplot as plt

# Synthetic convergence data
iterations = np.arange(1000, 10000001, 1000)
pi_estimates = 3.14 + np.random.randn(len(iterations)) * 0.01
plt.plot(iterations, pi_estimates, label="Estimated π")
plt.axhline(y=3.14159, color='r', linestyle='--', label="True π")
plt.xlabel("Iterations")
plt.ylabel("π Estimate")
plt.legend()
plt.savefig("convergence.png")
plt.show()
