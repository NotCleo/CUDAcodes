#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUFFT(err) if (err != CUFFT_SUCCESS) { printf("cuFFT error: %d\n", err); exit(1); }

int main() {
    int n = 1048576; // 1M points
    cufftComplex *h_data, *d_data;
    cufftHandle plan;

    // Allocate host memory
    h_data = (cufftComplex*)malloc(n * sizeof(cufftComplex));
    if (!h_data) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    // Initialize synthetic signal
    for (int i = 0; i < n; i++) {
        h_data[i].x = sin(2 * M_PI * i / 1024.0);
        h_data[i].y = 0;
    }

    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc(&d_data, n * sizeof(cufftComplex));
    if (err != cudaSuccess) {
        printf("Device memory allocation failed: %s\n", cudaGetErrorString(err));
        free(h_data);
        return 1;
    }

    // Copy data to device
    err = cudaMemcpy(d_data, h_data, n * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Data transfer failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data); free(h_data);
        return 1;
    }

    // Create cuFFT plan and execute
    CHECK_CUFFT(cufftPlan1d(&plan, n, CUFFT_C2C, 1));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("cuFFT time: %f ms\n", ms);

    // Copy results back
    err = cudaMemcpy(h_data, d_data, n * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Result transfer failed: %s\n", cudaGetErrorString(err));
    }

    // Print sample results
    for (int i = 0; i < 10; i++) {
        printf("Frequency %d: %f + %fi\n", i, h_data[i].x, h_data[i].y);
    }

    // Free memory
    CHECK_CUFFT(cufftDestroy(plan));
    cudaFree(d_data);
    free(h_data);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
