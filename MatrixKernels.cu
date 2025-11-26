
#include "MatrixKernels.cuh"

__global__ void tiledMatrixMultiplyKernel(const float* A, const float* B, float* C,
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


    // Finally, insert the sum to our resulting matrix!
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// GPU Tiled wrapper function
void tiledMatrixMultiplyGPU(const Matrix& A, const Matrix& B, Matrix& C) {

    int M = A.rows;
    int K = B.rows;
    int N = B.cols;

    // Define block dimensions and grid dimensions                        
    dim3 threads(TILE_SIZE, TILE_SIZE);

    // Here, we ensure that we will not have too few blocks in our grid.
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Toggle these on and off to test performance
    cudaMemPrefetchAsync(A.data, A.size * sizeof(float), 0, 0);
    cudaMemPrefetchAsync(B.data, B.size * sizeof(float), 0, 0);
    cudaMemPrefetchAsync(C.data, C.size * sizeof(float), 0, 0);

    // Launch our kernel!
    tiledMatrixMultiplyKernel<<<blocks, threads>>>(A.data, B.data, C.data, M, N, K);

    cudaDeviceSynchronize();
}

// Naive GPU kernel
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (!(row < M && col < N)) return;

    float sum = 0.0;

    for (int idx = 0; idx < K; idx++)
        sum += A[row * K + idx] * B[N * idx + col];
    
    C[row * N + col] = sum;
}

// GPU wrapper function
void matrixMultiplyGPU(const Matrix& A, const Matrix& B, Matrix& C) {

    int M = A.rows;
    int K = B.rows;
    int N = B.cols;

    // Define block dimensions and grid dimensions                        
    dim3 threads(THREAD_X, THREAD_Y);

    // Here, we ensure that we will not have too few blocks in our grid.
    dim3 blocks((N + THREAD_X - 1) / THREAD_X,
                (M + THREAD_Y - 1) / THREAD_Y);
    
    // Toggle these on and off to test performance
    // cudaMemPrefetchAsync(d_A, M*K*sizeof(float), 0, 0);
    // cudaMemPrefetchAsync(d_B, K*N*sizeof(float), 0, 0);
    // cudaMemPrefetchAsync(d_C, M*N*sizeof(float), 0, 0);

    // Launch our kernel!
    matrixMultiplyKernel<<<blocks, threads>>>(A.data, B.data, C.data, M, N, K);

    cudaDeviceSynchronize();
}