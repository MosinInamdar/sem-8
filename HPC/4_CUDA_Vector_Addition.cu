#include <iostream>
#include <cuda_runtime.h>

#define N 1000000 // Number of elements in the vectors

// CUDA Kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
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
    size_t size = N * sizeof(float);

    // Allocate input vectors in host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Allocate vectors in device memory
    float *d_A, *d_B, *d_C;
    cudaCheckError(cudaMalloc(&d_A, size));
    cudaCheckError(cudaMalloc(&d_B, size));
    cudaCheckError(cudaMalloc(&d_C, size));

    // Copy vectors from host memory to device memory
    cudaCheckError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaCheckError(cudaGetLastError());      // Check launch error
    cudaCheckError(cudaDeviceSynchronize()); // Wait for GPU to finish

    // Copy result from device to host
    cudaCheckError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify result
    std::cout << "Sample results:\n";
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "C[" << i << "] = " << h_C[i] << "  (Expected: " << h_A[i] + h_B[i] << ")\n";
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaCheckError(cudaFree(d_A));
    cudaCheckError(cudaFree(d_B));
    cudaCheckError(cudaFree(d_C));

    return 0;
}
