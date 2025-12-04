// Matrix.h


/**
 * Matrix multiplication: C = A * B
 * 
 * A: M x K
 *      M rows
 *      K cols
 * 
 * B: K x N
 *      K rows
 *      N cols
 * 
 * C: M x N
 *      M rows
 *      N cols
 * 
 * So, in our cuda block structure, our x values (indicating cols) should exceed N
 * and our y values (indicating rows) should exceed M.
 * 
 **/


#pragma once
#include "config.h"
#include <cuda_runtime.h>

#define CPU_TEST    false
#define GPU_TEST    true

class Matrix {
public:
    float* data;
    int rows, cols, size;
    
    Matrix() : rows(0), cols(0), size(0), data(nullptr) {}

    Matrix(int rows, int cols)
        : rows(rows), cols(cols) {
        ASSERT(rows > 0 && cols > 0);
        size = rows * cols;
        cudaError_t err = cudaMallocManaged(&data, rows * cols * sizeof(float));
        ASSERT(err == cudaSuccess);
    }

    ~Matrix() {
        cudaFree(data);
    }

    Matrix(const Matrix&) = delete;           
    Matrix& operator=(const Matrix&) = delete; 

    // Allow move
    Matrix(Matrix&& other) noexcept
        : rows(other.rows), cols(other.cols), size(other.size), data(other.data) {
        other.data = nullptr;
        other.rows = other.cols = other.size = 0;
    }
    
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            if (data) cudaFree(data);  // Free old resource
            
            rows = other.rows;
            cols = other.cols;
            size = other.size;
            data = other.data;
            
            other.data = nullptr;
            other.rows = other.cols = other.size = 0;
        }
        return *this;
    }
};

void printMatrix(const float* matrix, int rows, int cols, const char* name = "Matrix");

struct BenchmarkResult {
    float time_ms;
    float gflops;
    float bandwidth_gb_s;
};

BenchmarkResult benchmarkMatrixMultiply(
    void (*func)(const Matrix&, const Matrix&, Matrix&),
    int M, int N, int K,
    const char* name,
    bool is_gpu = false
);

void writeBenchmarkToFile(const char* filename,
                          int M, int N, int K,
                          BenchmarkResult cpu_result,
                          BenchmarkResult gpu_result
);

void test_gemm();