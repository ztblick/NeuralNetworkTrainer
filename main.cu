#include "matrix.cuh"
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    // Test dimensions
    int M = 12;
    int K = 8;
    int N = 10;
    
    printf("Matrix Multiplication: (%d x %d) * (%d x %d)\n", M, K, K, N);
    
    // First: create all the matrices
    float* A = createMatrix(M, K, FILL_RANDOM);
    float* B = createMatrix(K, N, FILL_RANDOM);
    float* C = createMatrix(M, N, FILL_ZEROES);

    float* d_A = createCudaMatrix(M, K, FILL_RANDOM);
    float* d_B = createCudaMatrix(K, N, FILL_RANDOM);
    float* d_C = createCudaMatrix(M, N, FILL_ZEROES);
    
    // Now we will run the CPU version for a benchmark and for correctness
    printMatrix(A, M, K, "CPU_A");
    printMatrix(B, K, N, "CPU_B");
    printMatrix(C, M, N, "CPU_C");
    matrixMultiplyCPU(A, B, C, M, N, K);
    printMatrix(C, M, N, "CPU_C");

    // Now we will run the GPU version
    matrixMultiplyGPU(d_A, d_B, d_C, M, N, K);

    // Let's double-check the results of the GPU test...

    // Let's update performance statistics
    // TODO

    // Done! Let's free our data and return
    freeMatrix(A);
    freeMatrix(B);
    freeMatrix(C);

    freeCudaMatrix(d_A);
    freeCudaMatrix(d_B);
    freeCudaMatrix(d_C);
    
    return 0;
}