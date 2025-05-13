#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA kernel to assign points to nearest centroid
__global__ void assignClusters(float *points, float *centroids, int *assignments, int n, int k, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float minDist = 1e10;
        int minIdx = 0;
        for (int c = 0; c < k; c++) {
            float dist = 0;
            for (int d = 0; d < dim; d++) {
                float diff = points[idx * dim + d] - centroids[c * dim + d];
                dist += diff * diff;
            }
            if (dist < minDist) {
                minDist = dist;
                minIdx = c;
            }
        }
        assignments[idx] = minIdx;
    }
}

int main() {
    // Parameters
    int n = 100000, k = 10, dim = 2; // 100K 2D points, 10 clusters
    float *points, *centroids, *d_points, *d_centroids;
    int *assignments, *d_assignments;

    // Allocate host memory
    points = (float*)malloc(n * dim * sizeof(float));
    centroids = (float*)malloc(k * dim * sizeof(float));
    assignments = (int*)malloc(n * sizeof(int));
    if (!points || !centroids || !assignments) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    // Initialize random data
    srand(time(NULL));
    for (int i = 0; i < n * dim; i++) points[i] = rand() / (float)RAND_MAX * 10;
    for (int i = 0; i < k * dim; i++) centroids[i] = rand() / (float)RAND_MAX * 10;

    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc(&d_points, n * dim * sizeof(float));
    err |= cudaMalloc(&d_centroids, k * dim * sizeof(float));
    err |= cudaMalloc(&d_assignments, n * sizeof(int));
    if (err != cudaSuccess) {
        printf("Device memory allocation failed: %s\n", cudaGetErrorString(err));
        free(points); free(centroids); free(assignments);
        return 1;
    }

    // Copy data to device
    err = cudaMemcpy(d_points, points, n * dim * sizeof(float), cudaMemcpyHostToDevice);
    err |= cudaMemcpy(d_centroids, centroids, k * dim * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Data transfer failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_points); cudaFree(d_centroids); cudaFree(d_assignments);
        free(points); free(centroids); free(assignments);
        return 1;
    }

    // Launch kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    assignClusters<<<blocks, threads>>>(d_points, d_centroids, d_assignments, n, k, dim);
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
    err = cudaMemcpy(assignments, d_assignments, n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Result transfer failed: %s\n", cudaGetErrorString(err));
    }

    // Print sample results
    for (int i = 0; i < 10; i++) {
        printf("Point %d assigned to cluster %d\n", i, assignments[i]);
    }

    // Free memory
    cudaFree(d_points); cudaFree(d_centroids); cudaFree(d_assignments);
    free(points); free(centroids); free(assignments);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
