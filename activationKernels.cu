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
                            size_t batch_size,
                            size_t features) {

    size_t total = features * batch_size;
    int threads = DEFAULT_THREADS_PER_BLOCK;
    int blocks = (total + threads - 1) / threads;

    relu_backward_kernel<<<blocks, threads>>>(  d_grad_output.data,
                                                d_output.data, 
                                                d_grad_input.data,
                                                total);
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

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    const float* sample_input = input + sample_idx * num_classes;
    float* sample_output = output + sample_idx * num_classes;
    int true_class = true_classes[sample_idx];

    // ===== Step 1: Find max logit =====

    // First, we will do a grid-stride loop
    // For MNIST, this doesn't really make sense, as we have only 10 classes.
    // But this will allow us to expand later on.
    float thread_max = -INFINITY;
    for (int i = tid; i < num_classes; i += num_threads) {
        thread_max = fmaxf(thread_max, sample_input[i]);
    }

    // Now, we will read these local maximums into shared data, then do a parallel reduction on it.
    // This will access the two floats of memory that is shared between all threads in this block!
    extern __shared__ float s_data[];
    s_data[tid] = thread_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }
    float max_logit = s_data[0];
    __syncthreads();

    // ===== Step 2: Compute exp(z - max) and sum =====
    // Again, we start with a grid-stride loop to find the sum of all the logits that this thread should consider on its own
    float thread_sum = 0.0f;
    for (int i = tid; i < num_classes; i += num_threads) {
        thread_sum += expf(sample_input[i] - max_logit);
    }
    
    // Now we do a parallel reduction to get total sum of exponentials
    s_data[tid] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = s_data[tid] + s_data[tid + s];
        }
        __syncthreads();
    }
    float exp_sum = s_data[0];
    
    // ===== Step 3: Compute softmax probabilities and gradients =====
    // Grid-stride loop in case the number of classes exceeds the number of threads
    for (int i = tid; i < num_classes; i += num_threads) {
        float prob = expf(sample_input[i] - max_logit) / exp_sum;
        
        // Compute gradient, which is simply the softmax value...
        float gradient = prob;

        // ...Except in the case of the true class!
        if (i == true_class) {
            gradient -= 1.0f;

            // Compute loss:
            float loss = -logf(prob);
            losses[sample_idx] = loss;
        }
        sample_output[i] = gradient;
    }
}

void launch_output_forward( const Matrix& d_input,
                            Matrix& d_output,
                            int batch_size,
                            float* d_loss,
                            const int* d_true_class_indices) {

    // We will need to split this across blocks to run the softmax and cross-entropy loss
    // functions across each set of data for each example in the batch.

    // Input: the logits (z1 .... zn) produced by the final hidden layer (dense + ReLU).
    // Output: the gradient (assigned to each class)

    size_t num_classes = d_input.cols;
    ASSERT(d_input.rows == batch_size);
    ASSERT(d_input.rows == d_output.rows && d_input.cols == d_output.cols);

    // Strategy: One block per sample
    // Each block handles one sample's softmax + gradient computation
    dim3 blocks(batch_size);
    dim3 threadsPerBlock(DEFAULT_THREADS_PER_BLOCK);  // Tune this - need enough threads for reduction
    
    // Shared memory: store per-block array to allow for parallel reductions to be done
    size_t shared_mem = threadsPerBlock.x * sizeof(float);

    softmax_cross_entropy_kernel<<<blocks, threadsPerBlock, shared_mem>>>(
        d_input.data,
        d_output.data,
        d_loss,
        d_true_class_indices,
        num_classes,
        batch_size
    );
}
