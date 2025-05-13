import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import os

# Synthetic dataset (replace with ASVspoof or real data)
n_samples = 100
spectrograms = np.random.rand(n_samples, 128, 128, 1) # From spectrogram.cu
labels = np.random.randint(0, 2, n_samples) # 0=real, 1=fake

# Define CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(spectrograms, labels, epochs=5, batch_size=32)

# Save model
model.save('model.h5')
