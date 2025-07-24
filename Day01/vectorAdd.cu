#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 10;
    float A[N], B[N], C[N];

    // Fill A and B with numbers
    for (int i = 0; i < N; ++i) {
        A[i] = i;
        B[i] = i * 2;
    }

    // Allocate memory on GPU
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // Copy A and B to GPU
    cudaMemcpy(d_a, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel
    int blocksize = 256;
    int gridsize = (N + blocksize - 1) / blocksize;
    vectorAdd<<<gridsize, blocksize>>>(d_a, d_b, d_c, N);

    // Copy result back to CPU
    cudaMemcpy(C, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result of vector addition:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
    }

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
