#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void estimatePi(unsigned int *count, int points_per_thread, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, idx, 0, &state);
    unsigned int localCount = 0;
    for (int i = 0; i < points_per_thread; i++) {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if (x * x + y * y <= 1.0f) localCount++;
    }
    count[idx] = localCount;
}

int main() {
    int n = 10000000, threads = 256, blocks = 100;
    int points_per_thread = n / (blocks * threads);
    unsigned int *h_count, *d_count;
    unsigned int total = 0;

    // Allocate memory
    h_count = (unsigned int*)malloc(blocks * threads * sizeof(unsigned int));
    cudaError_t err = cudaMalloc(&d_count, blocks * threads * sizeof(unsigned int));
    if (!h_count || err != cudaSuccess) {
        printf("Memory allocation failed: %s\n", cudaGetErrorString(err));
        free(h_count); cudaFree(d_count);
        return 1;
    }

    // Launch kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    estimatePi<<<blocks, threads>>>(d_count, points_per_thread, (unsigned int)time(NULL));
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
    err = cudaMemcpy(h_count, d_count, blocks * threads * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Result transfer failed: %s\n", cudaGetErrorString(err));
    }

    // Sum counts
    for (int i = 0; i < blocks * threads; i++) total += h_count[i];
    printf("Pi ≈ %f\n", 4.0 * total / (float)n);

    // Free memory
    cudaFree(d_count);
    free(h_count);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
