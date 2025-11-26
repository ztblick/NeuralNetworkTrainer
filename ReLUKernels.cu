#include "ReLUKernels.cuh"

__global__ void relu_forward_kernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
    }
}

__global__ void relu_backward_kernel(const float* grad_output, const float* output,
                                     float* grad_input, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = output[idx] > 0.0f ? grad_output[idx] : 0.0f;
    }
}

void launch_relu_forward(const float* d_input, float* d_output,
                         size_t size, size_t batch_size) {
    size_t total = size * batch_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    relu_forward_kernel<<<blocks, threads>>>(d_input, d_output, total);
}

void launch_relu_backward(const float* d_grad_output, const float* d_output,
                          float* d_grad_input, size_t size, size_t batch_size) {
    size_t total = size * batch_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(d_grad_output, d_output, 
                                              d_grad_input, total);
}