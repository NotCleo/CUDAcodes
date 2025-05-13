import numpy as np
import matplotlib.pyplot as plt

# Synthetic image (replace with actual output)
width, height = 1920, 1080
image = np.random.rand(height, width)
plt.imshow(image, cmap='gray')
plt.title("Sharpened Image")
plt.savefig("sharpened.png")
plt.show()
