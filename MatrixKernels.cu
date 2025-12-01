
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

    // Define and launch kernel to calculate weight gradients

    // Define and launch kernel to calculate gradient to previous layer

    // Define and launch kernel to calculate bias gradient

}