#include <iostream>
#include <cuda_runtime.h>

#define M 512 // Rows in A and C
#define K 512 // Columns in A and rows in B
#define N 512 // Columns in B and C

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(const float *A, const float *B, float *C, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row in C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Col in C

    if (row < m && col < n)
    {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i)
        {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Error check macro
#define cudaCheckError(ans)                   \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " at " << file << ":" << line << std::endl;
        if (abort)
            exit(code);
    }
}

int main()
{
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    // Initialize A and B
    for (int i = 0; i < M * K; ++i)
        h_A[i] = 1.0f; // A filled with 1s
    for (int i = 0; i < K * N; ++i)
        h_B[i] = 2.0f; // B filled with 2s

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaCheckError(cudaMalloc(&d_A, size_A));
    cudaCheckError(cudaMalloc(&d_B, size_B));
    cudaCheckError(cudaMalloc(&d_C, size_C));

    // Copy A and B to device
    cudaCheckError(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // Define block and grid size
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);

    // Launch kernel
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    // Print a few sample results
    std::cout << "Sample C values:\n";
    for (int i = 0; i < 10; ++i)
        std::cout << "C[" << i << "] = " << h_C[i] << " (Expected: " << K * 2.0f << ")\n";

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
