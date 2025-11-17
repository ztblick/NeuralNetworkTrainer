#include "matrix.cuh"
#include "debug.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

// Naive GPU kernel
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    // TODO: Implement
}

// GPU wrapper function
void matrixMultiplyGPU(const float* d_A, const float* d_B, float* d_C,
                       int M, int N, int K) {
    // TODO: Implement kernel launch
}

#include <random>

void fillMatrixRandom(float* matrix, int rows, int cols, 
                      float min = 0.0f, float max = 1.0f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = dis(gen);
    }
}

// Initialize and fill the matrix
float* createMatrix(int rows, int cols, bool fill_option) {
    float* matrix = (float *) malloc(sizeof(float) * rows * cols);
    ASSERT(matrix);

    if (fill_option == FILL_RANDOM)
        fillMatrixRandom(matrix, rows, cols);
    else /*fill_option == FILL_ZEROES*/
        memset(matrix, 0, rows * cols * sizeof(float));
    return matrix;
}

float* createCudaMatrix(int rows, int cols, bool random) {
    // TODO: Implement
    return nullptr;
}

void freeCudaMatrix(float* matrix) {
    cudaFree(matrix);
}

void freeMatrix(float* matrix) {
    free(matrix);
}

bool verifyResults(const float* A, const float* B, int size, float epsilon) {
    // TODO: Implement
    return true;
}

void printMatrix(const float* matrix, int rows, int cols, const char* name) {
    printf("\n%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%7.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}