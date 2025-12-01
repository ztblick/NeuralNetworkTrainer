#include "Matrix.h"
#include "MatrixKernels.cuh"
#include <math.h>
#include <random>
#include <cuda_runtime.h>
#include "timer.cuh"

Matrix::Matrix(int rows, int cols)
        : rows(rows), cols(cols) {
        ASSERT(rows > 0 && cols > 0);
        size = rows * cols;
        cudaError_t err = cudaMallocManaged(&data, rows * cols * sizeof(float));
        ASSERT(err == cudaSuccess);
}
    
Matrix::~Matrix() {
    cudaFree(data);
}

/*
    This will perform matrix multiplication on the CPU.
    I'll use this to validate my GPU results as well as to create
    a benchmark for the GPU performance.
*/
void matrixMultiplyCPU(const float* A, const float* B, float* C,
                       int M, int N, int K) {

    float sum = 0.0;

    // A: M x K
    // B: K x N
    // C: M x N
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            sum = 0.0;
            for (int idx = 0; idx < K; idx++) {
                sum += A[row * K + idx] * B[N * idx + col];
            }
            C[row * N + col] = sum;
        }
    }
}