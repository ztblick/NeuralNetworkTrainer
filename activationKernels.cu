#include "activationKernels.cuh"

__global__ void relu_forward_kernel(const float* input, float* output, int size) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
}

__global__ void relu_backward_kernel(const float* grad_output, const float* output,
                                     float* grad_input, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = output[idx] > 0.0f ? grad_output[idx] : 0.0f;
    }
}

void launch_relu_forward(const Matrix& d_input, Matrix& d_output, size_t batch_size) {
    
    size_t total = d_input.rows * batch_size;
    int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    relu_forward_kernel<<<blocks, threadsPerBlock>>>(   d_input.data,
                                                        d_output.data,
                                                        total);

    cudaDeviceSynchronize();
}

void launch_relu_backward(const Matrix& d_grad_output, const Matrix& d_output,
                          Matrix& d_grad_input, size_t batch_size) {

    size_t total = d_grad_input.rows * batch_size;  // TODO this may need to be transposed
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    relu_backward_kernel<<<blocks, threads>>>(  d_grad_output.data,
                                                d_output.data, 
                                                d_grad_input.data,
                                                total);
}