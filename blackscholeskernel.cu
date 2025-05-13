#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

#define NUM_OPTIONS 1000000
#define THREADS_PER_BLOCK 256

// Approximate Normal CDF
__device__ float normal_cdf(float x) {
    const float b1 = 0.319381530f;
    const float b2 = -0.356563782f;
    const float b3 = 1.781477937f;
    const float b4 = -1.821255978f;
    const float b5 = 1.330274429f;
    const float p = 0.2316419f;
    const float c = 0.39894228f;

    float t = 1.0f / (1.0f + p * fabsf(x));
    float poly = ((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t;
    float result = 1.0f - c * expf(-0.5f * x * x) * poly;
    return (x >= 0.0f) ? result : (1.0f - result);
}

// CUDA Kernel
__global__ void black_scholes_kernel(float *S0, float *K, float *T, float r, float sigma,
                                     float *call_prices, float *put_prices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float s = S0[idx];
    float k = K[idx];
    float t = T[idx];

    float sqrtT = sqrtf(t);
    float d1 = (logf(s / k) + (r + 0.5f * sigma * sigma) * t) / (sigma * sqrtT);
    float d2 = d1 - sigma * sqrtT;

    float nd1 = normal_cdf(d1);
    float nd2 = normal_cdf(d2);
    float neg_nd1 = normal_cdf(-d1);
    float neg_nd2 = normal_cdf(-d2);

    call_prices[idx] = s * nd1 - k * expf(-r * t) * nd2;
    put_prices[idx] = k * expf(-r * t) * neg_nd2 - s * neg_nd1;
}

int main() {
    float r = 0.05f;
    float sigma = 0.2f;
    int n = NUM_OPTIONS;

    float *h_S0 = (float *)malloc(n * sizeof(float));
    float *h_K = (float *)malloc(n * sizeof(float));
    float *h_T = (float *)malloc(n * sizeof(float));
    float *h_call_prices = (float *)malloc(n * sizeof(float));
    float *h_put_prices = (float *)malloc(n * sizeof(float));

    if (!h_S0 || !h_K || !h_T || !h_call_prices || !h_put_prices) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        h_S0[i] = 100.0f;
        h_K[i] = 80.0f + ((float)rand() / RAND_MAX) * 40.0f;
        h_T[i] = 0.5f + ((float)rand() / RAND_MAX) * 1.5f;
    }

    float *d_S0, *d_K, *d_T, *d_call_prices, *d_put_prices;
    cudaMalloc(&d_S0, n * sizeof(float));
    cudaMalloc(&d_K, n * sizeof(float));
    cudaMalloc(&d_T, n * sizeof(float));
    cudaMalloc(&d_call_prices, n * sizeof(float));
    cudaMalloc(&d_put_prices, n * sizeof(float));

    cudaMemcpy(d_S0, h_S0, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T, h_T, n * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    black_scholes_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_S0, d_K, d_T, r, sigma, d_call_prices, d_put_prices, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_call_prices, d_call_prices, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_put_prices, d_put_prices, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Compute Time: %.4f ms\n", milliseconds);

    float avg_call = 0.0f, avg_put = 0.0f;
    for (int i = 0; i < n; i++) {
        avg_call += h_call_prices[i];
        avg_put += h_put_prices[i];
    }
    avg_call /= n;
    avg_put /= n;
    printf("Average Call Price: %.4f\n", avg_call);
    printf("Average Put Price: %.4f\n", avg_put);

    FILE *fp = fopen("black_scholes_prices.dat", "w");
    if (fp) {
        for (int i = 0; i < n; i++) {
            fprintf(fp, "%f %f %f %f %f\n", h_S0[i], h_K[i], h_T[i], h_call_prices[i], h_put_prices[i]);
        }
        fclose(fp);
    } else {
        printf("Failed to open output file\n");
    }

    cudaFree(d_S0);
    cudaFree(d_K);
    cudaFree(d_T);
    cudaFree(d_call_prices);
    cudaFree(d_put_prices);
    free(h_S0);
    free(h_K);
    free(h_T);
    free(h_call_prices);
    free(h_put_prices);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
