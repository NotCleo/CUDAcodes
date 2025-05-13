import numpy as np
import matplotlib.pyplot as plt

# Load points and assignments (e.g., from a file or modify to call kmeans.cu)
n, dim, k = 100000, 2, 10
points = np.random.rand(n, dim) * 10
assignments = np.random.randint(0, k, n) # Replace with actual assignments

plt.scatter(points[:, 0], points[:, 1], c=assignments, cmap='viridis', s=1)
plt.title("K-Means Clustering")
plt.savefig("clusters.png")
plt.show()
