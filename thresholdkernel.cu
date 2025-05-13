#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void threshold(float *input, int *output, int n, float thresh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] > thresh ? 1 : 0; // 1 = fake, 0 = real
    }
}

int main() {
    int n = 1000; // Number of predictions
    float *h_input, *h_output, *d_input;
    int *d_output;

    // Allocate host memory
    h_input = (float*)malloc(n * sizeof(float));
    h_output = (int*)malloc(n * sizeof(int));
    if (!h_input || !h_output) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    // Initialize synthetic probabilities
    srand(time(NULL));
    for (int i = 0; i < n; i++) h_input[i] = rand() / (float)RAND_MAX;

    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc(&d_input, n * sizeof(float));
    err |= cudaMalloc(&d_output, n * sizeof(int));
    if (err != cudaSuccess) {
        printf("Device memory allocation failed: %s\n", cudaGetErrorString(err));
        free(h_input); free(h_output);
        return 1;
    }

    // Copy data to device
    err = cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Data transfer failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input); cudaFree(d_output); free(h_input); free(h_output);
        return 1;
    }

    // Launch kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    threshold<<<blocks, threads>>>(d_input, d_output, n, 0.5);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel time: %f ms\n", ms);

    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    // Copy results back
    err = cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Result transfer failed: %s\n", cudaGetErrorString(err));
    }

    // Print sample results
    for (int i = 0; i < 10; i++) {
        printf("Prediction %d: %d\n", i, h_output[i]);
    }

    // Free memory
    cudaFree(d_input); cudaFree(d_output);
    free(h_input); free(h_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
