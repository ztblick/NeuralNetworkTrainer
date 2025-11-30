#include "activationKernels.cuh"

__global__ void relu_forward_kernel(    const float* input,
                                        float* output,
                                        int size) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
}

void launch_relu_forward(   const Matrix& d_input,
                            Matrix& d_output,
                            size_t batch_size) {
    
    size_t total = d_input.rows * batch_size;
    int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    relu_forward_kernel<<<blocks, threadsPerBlock>>>(   d_input.data,
                                                        d_output.data,
                                                        total);

    cudaDeviceSynchronize();
}

__global__ void relu_backward_kernel(   const float* grad_output,
                                        const float* output,
                                        float* grad_input,
                                        size_t size) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = output[idx] > 0.0f ? grad_output[idx] : 0.0f;
    }
}

void launch_relu_backward(  const Matrix& d_grad_output,
                            const Matrix& d_output,
                            Matrix& d_grad_input,
                            size_t batch_size) {

    size_t total = d_grad_input.rows * batch_size;  // TODO this may need to be transposed
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    relu_backward_kernel<<<blocks, threads>>>(  d_grad_output.data,
                                                d_output.data, 
                                                d_grad_input.data,
                                                total);

    cudaDeviceSynchronize();
}

__global__ void softmax_cross_entropy_kernel(   const float* input,      // [batch_size, num_classes]
                                                float* output,           // [batch_size, num_classes] - will hold gradients
                                                float* losses,           // [batch_size] - per-sample losses
                                                const int* true_classes, // [batch_size]
                                                size_t num_classes,
                                                size_t batch_size) {
    
    // Each block of threads will process one test case
    int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) return;
    
    // ================================================ //
    // Implement softmax + gradient computation         //
    // ================================================ //

    // Step 1: Find max (reduction)
    // This will access the two floats of memory that is shared between all threads in this block!
    extern __shared__ float shared[];
    float* s_max = &shared[0];
    float* s_sum = &shared[1];

    // key idea: each thread compares the its two values, then writes the max to its base index
    // Then the next thread will use that value in its next comparison
    // And the final max value will be written by thread 0 into shared
    int tid = threadIdx.x;
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            input[tid] = fmaxf(input[tid], input[tid + s]);
        }
        __syncthreads();
    }

    // TODO -- CONTINUE ON REDUCTION!

    // Step 2-5: Softmax computation
    // Step 6: Gradient (need true_class_idx)
    // Step 7: Write output
    // Step 8: Compute and write loss
}

void launch_output_forward( const Matrix& d_input,
                            Matrix& d_output,
                            float* d_loss,
                            size_t batch_size,
                            const int* d_true_class_indices) {

    // We will need to split this across blocks to run the softmax and cross-entropy loss
    // functions across each set of data for each example in the batch.

    // Input: the logits (z1 .... zn) produced by the final hidden layer (dense + ReLU).
    // Output: the probabilites assigned to each class

    size_t num_classes = d_input.cols;

    // Strategy: One block per sample
    // Each block handles one sample's softmax + gradient computation
    dim3 blocks(batch_size);
    dim3 threadsPerBlock(DEFAULT_THREADS_PER_BLOCK);  // Tune this - need enough threads for reduction
    
    // Shared memory: store per-block max value and sum of exponentials
    size_t shared_mem = 2 * sizeof(float);  // max_val, exp_sum
    
    softmax_cross_entropy_kernel<<<blocks, threadsPerBlock, shared_mem>>>(
        d_input.data,
        d_output.data,
        d_loss,
        d_true_class_indices,
        num_classes,
        batch_size
    );

    cudaDeviceSynchronize();
}

void launch_output_backward(    const Matrix& d_grad_output,
                                const Matrix& d_output,
                                Matrix& d_grad_input,
                                size_t batch_size) {

    // Input: the probabilites assigned to each class (as well as the correct class for each test case).
    // Output: the gradient vector to send to the next layer during the backward pass.
}
