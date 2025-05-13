#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Constant memory for 3x3 kernel
__constant__ float cnn_kernel[9] = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};

__global__ void cnnConvolution(float *input, float *output, int width, int height, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float sum = 0;
        for (int ky = -ksize/2; ky <= ksize/2; ky++) {
            for (int kx = -ksize/2; kx <= ksize/2; kx++) {
                int imgX = x + kx, imgY = y + ky;
                if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
                    sum += input[imgY * width + imgX] * cnn_kernel[(ky + ksize/2) * ksize + (kx + ksize/2)];
                }
            }
        }
        output[y * width + x] = sum;
    }
}

int main() {
    int width = 128, height = 128; // Spectrogram size
    int ksize = 3;
    float *h_input, *h_output, *d_input, *d_output;

    // Allocate host memory
    h_input = (float*)malloc(width * height * sizeof(float));
    h_output = (float*)malloc(width * height * sizeof(float));
    if (!h_input || !h_output) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    // Initialize synthetic spectrogram
    srand(time(NULL));
    for (int i = 0; i < width * height; i++) h_input[i] = rand() / (float)RAND_MAX;

    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc(&d_input, width * height * sizeof(float));
    err |= cudaMalloc(&d_output, width * height * sizeof(float));
    if (err != cudaSuccess) {
        printf("Device memory allocation failed: %s\n", cudaGetErrorString(err));
        free(h_input); free(h_output);
        return 1;
    }

    // Copy data to device
    err = cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Data transfer failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input); cudaFree(d_output); free(h_input); free(h_output);
        return 1;
    }

    // Launch kernel
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    cnnConvolution<<<blocks, threads>>>(d_input, d_output, width, height, ksize);
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
    err = cudaMemcpy(h_output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Result transfer failed: %s\n", cudaGetErrorString(err));
    }

    // Save output
    FILE *fp = fopen("cnn_output.dat", "w");
    for (int i = 0; i < width * height; i++) {
        fprintf(fp, "%f\n", h_output[i]);
    }
    fclose(fp);

    // Free memory
    cudaFree(d_input); cudaFree(d_output);
    free(h_input); free(h_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
