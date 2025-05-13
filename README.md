# CUDAcodes

This Repo contains all the CUDA kernels I have written

1) Monte Carlo Simulation for Option Pricing
2) Parallel Reduction
3) Image Convolution
4) Parallel Scan
5) Parallel Compact
6) Matrix Transpose 
7) Parallel Reductions

Note : Have covered loop unrolling, shared memory, warp divergence and plenty more techniques to above algorithms

BONUS : I have also added the performance comparisons of the reduction kernels. 

Performed some of these on the H100 (super fun), which I was provided access to via a Docker container, the rest have been done on the T4 GPU.

# Future work
The repository will be updated with following CUDA kernels (later)

1) Radix sorting
2) Merge sorting
3) Matrix transpose
4) Convolution (1D/2D) with constant memory masking and with global memory access
5) Thread/Block organization
6) FP performance comparison 

# Note
nvcc filename.cu -o whatever
./whatever
python filename.py


Do not forget to check the nvcc compiler version with the driver versions, the mismatch will require to step to; nvcc -arch=sm_75 -o whatever filename.cu

