#include "activations.cuh"
#include <cuda_runtime.h>

#define DEFAULT_THREADS_PER_BLOCK   256

__global__ void reluKernel(float* data, int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (data[idx] < 0) data[idx] = 0.0f;
}

void relu(float* data, int size) {

    // We will be applying the ReLU function to each resulting calculation
    // So, with a weights array that is [ N x K] and a previous activations
    // vector that is [K x 1], we will need to repeat the ReLu function
    // on the resulting N floating point values.

    // With this in mind, it would make sense to prefetch this data, then
    // break it into blocks where each thread will ReLU their one value.
    int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    
    // 2. Calculate how many blocks we need
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    // 3. Launch kernel - each thread handles ONE element
    reluKernel<<<blocks, threadsPerBlock>>>(data, size);
    
    cudaDeviceSynchronize();
}

void crossEntropyLoss() {

}

void softMax() {
    
}