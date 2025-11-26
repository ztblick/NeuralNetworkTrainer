#include "activations.cuh"
#include <cuda_runtime.h>

#define DEFAULT_THREADS_PER_BLOCK   256

__global__ void reluKernel(float* data, int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (data[idx] < 0) data[idx] = 0.0f;
}

void relu(const Matrix& m) {

    // We will be applying the ReLU function to each resulting calculation
    // So, with a weights array that is [ N x K] and a previous activations
    // vector that is [K x 1], we will need to repeat the ReLu function
    // on the resulting N floating point values.

    // With this in mind, it would make sense to prefetch this data, then
    // break it into blocks where each thread will ReLU their one value.
    int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    
    // 2. Calculate how many blocks we need
    int blocks = (m.size + threadsPerBlock - 1) / threadsPerBlock;
    
    // 3. Launch kernel - each thread handles ONE element
    reluKernel<<<blocks, threadsPerBlock>>>(m.data, m.size);
    
    cudaDeviceSynchronize();
}

void test_relu() {
    int size = 1e3;
    Matrix m(size / 10, 10);
    
    // Initialize with some negative and positive values
    for (int i = 0; i < m.size; i++) {
        m.data[i] = i - size/2;
    }
    
    printf("Before ReLU:\n");
    for (int i = 0; i < size; i++) {
        printf("%.1f ", m.data[i]);
    }
    printf("\n");
    
    // Apply ReLU
    relu(m);
    
    printf("After ReLU:\n");
    for (int i = 0; i < size; i++) {
        printf("%.1f ", m.data[i]);
    }
    printf("\n");
    
    // Expected: [0, 0, 0, 0, 0, 0, 1, 2, 3, 4]
}