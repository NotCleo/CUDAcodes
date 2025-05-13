#include <cufft.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUFFT(err) if (err != CUFFT_SUCCESS) { printf("cuFFT error: %d\n", err); exit(1); }

int main() {
    int n = 1024; // Window size for STFT
    int num_windows = 1000; // Number of segments
    cufftComplex *h_data, *d_data;
    cufftHandle plan;

    // Allocate host memory
    h_data = (cufftComplex*)malloc(n * num_windows * sizeof(cufftComplex));
    if (!h_data) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    // Initialize synthetic audio
    for (int i = 0; i < n * num_windows; i++) {
        h_data[i].x = sin(2 * M_PI * i / 256.0); // Replace with audio samples
        h_data[i].y = 0;
    }

    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc(&d_data, n * num_windows * sizeof(cufftComplex));
    if (err != cudaSuccess) {
        printf("Device memory allocation failed: %s\n", cudaGetErrorString(err));
        free(h_data);
        return 1;
    }

    // Copy data to device
    err = cudaMemcpy(d_data, h_data, n * num_windows * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Data transfer failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data); free(h_data);
        return 1;
    }

    // Create batched cuFFT plan
    CHECK_CUFFT(cufftPlan1d(&plan, n, CUFFT_C2C, num_windows));
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
    err = cudaMemcpy(h_data, d_data, n * num_windows * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Result transfer failed: %s\n", cudaGetErrorString(err));
    }

    // Save spectrogram (magnitude)
    FILE *fp = fopen("spectrogram.dat", "w");
    for (int i = 0; i < num_windows; i++) {
        for (int j = 0; j < n / 2; j++) {
            float mag = sqrt(h_data[i * n + j].x * h_data[i * n + j].x + h_data[i * n + j].y * h_data[i * n + j].y);
            fprintf(fp, "%f ", mag);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    // Free memory
    CHECK_CUFFT(cufftDestroy(plan));
    cudaFree(d_data);
    free(h_data);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
