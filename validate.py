import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('model.h5')

# Synthetic test data (replace with actual spectrograms)
n_test = 10
test_spectrograms = np.random.rand(n_test, 128, 128, 1)

# Predict
probs = model.predict(test_spectrograms)

# Apply thresholding (call threshold.cu via PyCUDA)
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

n = len(probs)
probs = probs.astype(np.float32)
labels = np.zeros(n, dtype=np.int32)

# Load and compile CUDA kernel
with open('threshold.cu', 'r') as f:
    mod = SourceModule(f.read())
threshold = mod.get_function("threshold")

# Allocate device memory
d_input = cuda.mem_alloc(probs.nbytes)
d_output = cuda.mem_alloc(labels.nbytes)
cuda.memcpy_htod(d_input, probs)

# Launch kernel
threads = 256
blocks = (n + threads - 1) // threads
threshold(d_input, d_output, np.int32(n), np.float32(0.5), block=(threads,1,1), grid=(blocks,1))

# Copy results back
cuda.memcpy_dtoh(labels, d_output)

print("Predicted labels:", labels)
