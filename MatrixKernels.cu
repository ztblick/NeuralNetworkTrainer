
#include "MatrixKernels.cuh"

__global__ void tiledMatrixMultiplyKernel(const float* A, const float* B, float* C,
                                          const float *bias,
                                            int M, int N, int K) {

    // Allocate shared memory for tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global row and column this thread computes
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;

    // Find the number of tiles we need to load for each full row/col multiplication operation
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Add the relevant components from each tile to the entire sum
    for (int tile = 0; tile < num_tiles; tile++) {

        // Here, we will load each value in the tile from A
        // This is split across ALL threads, so each thread adds one value
        int aCol = tile * TILE_SIZE + tx;
        if (row < M && aCol < K) {
            tileA[ty][tx] = A[row * K + aCol];
        } else {
            tileA[ty][tx] = 0.0f;  // Padding for boundary
        }

        // We do the same for B
        int bRow = tile * TILE_SIZE + ty;
        if (bRow < K && col < N) {
            tileB[ty][tx] = B[bRow * N + col];
        } else {
            tileB[ty][tx] = 0.0f;  // Padding for boundary
        }

        // Wait for all threads to finish loading
        __syncthreads();

        // Now that we've loaded the tile into shared memory, each thread will calculate their
        // chunk of their sum
        for (int idx = 0; idx < TILE_SIZE; idx++)
            sum += tileA[ty][idx] * tileB[idx][tx];

        // Wait for all threads to finish multiplying, then move on to the next tile!
        __syncthreads();
    }

    // Finally, add the bias before writing the result into the output matrix
    if (row < M && col < N) {
        C[row * N + col] = sum + bias[col];
    }
}

/**
 * d_input.data      A: [M, K] = [batch_size, input_features]
 * weights.data      B: [K, N] = [input_features, output_features]
 * d_output.data     C: [M, N] = [batch_size, output_features]
 * bias.data      bias: [1, N] = [1, output_features]           
**/
void launch_dense_forward(  const Matrix& weights,
                            const Matrix& d_input,
                            Matrix& d_output,
                            const Matrix& bias,
                            size_t batch_size,
                            size_t input_features,
                            size_t output_features) {

    int M = batch_size;
    int K = input_features;
    int N = output_features;

    // Define block dimensions and grid dimensions                        
    dim3 threads(TILE_SIZE, TILE_SIZE);

    // Here, we ensure that we will not have too few blocks in our grid.
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Toggle these on and off to test performance
    cudaMemPrefetchAsync(weights.data, weights.size * sizeof(float), 0, 0);
    cudaMemPrefetchAsync(d_input.data, d_input.size * sizeof(float), 0, 0);
    cudaMemPrefetchAsync(d_output.data, d_output.size * sizeof(float), 0, 0);

    // Launch our kernel!
    tiledMatrixMultiplyKernel<<<blocks, threads>>>(d_input.data,
                                                    weights.data,
                                                    d_output.data,
                                                    bias.data,
                                                    M, N, K);
}


__global__ void tiledMatMulTranspose(const float* A, const float* B, float* C,
                                     int M, int N, int K,
                                     bool transposeA, bool transposeB) {

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        // Load tile from A
        if (transposeA) {
            // A is stored as [K, M], we want A^T which is [M, K]
            // row is in [0, M), we're reading from column 'row' of original A
            int aCol = tile * TILE_SIZE + tx;  // Position in K dimension
            if (aCol < K && row < M) {
                tileA[ty][tx] = A[aCol * M + row];  // Read down column of A
            } else {
                tileA[ty][tx] = 0.0f;
            }
        } else {
            // A is stored as [M, K], normal read
            int aCol = tile * TILE_SIZE + tx;
            if (row < M && aCol < K) {
                tileA[ty][tx] = A[row * K + aCol];
            } else {
                tileA[ty][tx] = 0.0f;
            }
        }
        
        // Load tile from B
        if (transposeB) {
            int bRow = tile * TILE_SIZE + ty;  // Position in K dimension
            if (bRow < K && col < N) {
                tileB[ty][tx] = B[col * K + bRow];  // Read down column of B
            } else {
                tileB[ty][tx] = 0.0f;
            }
        } else {
            // B is stored as [K, N], normal read
            int bRow = tile * TILE_SIZE + ty;
            if (bRow < K && col < N) {
                tileB[ty][tx] = B[bRow * N + col];
            } else {
                tileB[ty][tx] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void bias_gradient_kernel(
        const float* d_grad_output,
        float* grad_bias,
        int batch_size,
        int output_features)  {

    int tid = threadIdx.x;
    // Each block handles one output feature (bias)
    int feature_idx = blockIdx.x;
    extern __shared__ float s_data[];

    // Load this thread's contribution: sum across batch for this feature
    float grad_value = 0.0f;
    for (int sample = tid; sample < batch_size; sample += blockDim.x) {
        grad_value += d_grad_output[sample * output_features + feature_idx];
    }

    s_data[tid] = grad_value;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // Finally: thread 0 writes out the sum to the bias gradient
    if (tid == 0) {
        grad_bias[feature_idx] = s_data[0];
    }
}

/**
    const Matrix& d_grad_output,  // [batch, output_features] - gradient from next layer
    Matrix& d_grad_input,         // [batch, input_features] - OUTPUT: gradient to previous layer
    Matrix& grad_weights,         // [input_features, output_features] - OUTPUT: weight gradients
    Matrix& grad_bias,            // [1, output_features] - OUTPUT: bias gradients
    const Matrix& weights,        // [input_features, output_features] - from forward pass
    const Matrix& d_cached_input, // [batch, input_features] - from forward pass           
**/
void launch_dense_backward(
    const Matrix& d_grad_output,  
    Matrix& d_grad_input,         
    Matrix& grad_weights,         
    Matrix& grad_bias,           
    const Matrix& weights,       
    const Matrix& d_cached_input, 
    size_t batch_size,
    size_t input_features,
    size_t output_features) {

    dim3 threads(TILE_SIZE, TILE_SIZE);

    // Launch kernel to calculate weight gradients
    // 1. grad_weights = input^T * grad_output
    //    input: [batch, input_features] stored as [K=batch, M=input_features]
    //    grad_output: [batch, output_features] stored as [K=batch, N=output_features]
    //    Result: [input_features, output_features] = [M, N]

    int M1 = input_features;
    int N1 = output_features;
    int K1 = batch_size;
    
    dim3 blocks1((N1 + TILE_SIZE - 1) / TILE_SIZE,
                 (M1 + TILE_SIZE - 1) / TILE_SIZE);
    
    tiledMatMulTranspose<<<blocks1, threads>>>(
        d_cached_input.data,
        d_grad_output.data,
        grad_weights.data,
        M1, N1, K1,
        true,   // transposeA
        false); // transposeB

    // Launch kernel to perform input layer gradient calculation
    // 2. grad_input = grad_output * weights^T
    //    grad_output: [batch, output_features] stored as [M=batch, K=output_features]
    //    weights: [input_features, output_features] stored as [N=input_features, K=output_features]
    //    Result: [batch, input_features] = [M, N]
    int M2 = batch_size;
    int N2 = input_features;
    int K2 = output_features;
    
    dim3 blocks2((N2 + TILE_SIZE - 1) / TILE_SIZE,
                 (M2 + TILE_SIZE - 1) / TILE_SIZE);
    
    tiledMatMulTranspose<<<blocks2, threads>>>(
        d_grad_output.data,
        weights.data,
        d_grad_input.data,
        M2, N2, K2,
        false,  // transposeA
        true);  // transposeB
    
#if DEBUG
    cudaDeviceSynchronize();  // Force completion
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA ERROR in grad_input matmul: %s\n", cudaGetErrorString(err));
    }

    // Check if anything was written
    printf("  After grad_input matmul: d_grad_input[0] = %.6f\n", d_grad_input.data[0]);
#endif


    // Launch kernel to calculate bias gradients 
    // 3. grad_bias = sum(grad_output for each batch (a.k.a. axis=0)) =====
    //  Input: [batch, output_features]
    //  Output: [output_features]

    // Strategy: One block per bias.
    //      Threads within block will reduce the sum for that bias, then write the sum into the gradient.
    dim3 blocks(output_features);
    dim3 threadsPerBlock(DEFAULT_THREADS_PER_BLOCK);  // Tune this - need enough threads for reduction
    
    // Shared memory: store per-block array to allow for parallel reductions to be done
    size_t shared_mem = threadsPerBlock.x * sizeof(float);
    
    bias_gradient_kernel<<<blocks, threadsPerBlock, shared_mem>>>(
        d_grad_output.data,
        grad_bias.data,
        batch_size,
        output_features);
}

__global__ void weight_update_kernel(
    float* weights,
    const float* gradients,
    float learning_rate,
    size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

void launch_weight_update(Matrix& weights, const Matrix& gradients, float learning_rate) {
    size_t total = weights.rows * weights.cols;
    int threads = DEFAULT_THREADS_PER_BLOCK;
    int blocks = (total + threads - 1) / threads;
    
    weight_update_kernel<<<blocks, threads>>>(
        weights.data,
        gradients.data,
        learning_rate,
        total
    );
}


__global__ void scale_matrix(float* data, float scale, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= scale;
    }
}


void launch_scale_matrix(
    Matrix& grad,
    float scale) {

    size_t total = grad.rows * grad.cols;
    int threads = DEFAULT_THREADS_PER_BLOCK;
    int blocks = (total + threads - 1) / threads;
    
    scale_matrix<<<blocks, threads>>>(
        grad.data,
        scale,
        total
    );
}