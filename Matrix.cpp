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

void fillMatrixRandom(Matrix& m, 
                      float min = 0.0f, float max = 1.0f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    
    for (int i = 0; i < m.size; i++) {
        m.data[i] = dis(gen);
    }
}

void fillMatrixZeros(Matrix& m) {
    cudaMemset(m.data, 0, m.rows * m.cols * sizeof(float));
}

// Benchmarking and timing code from Claude
BenchmarkResult benchmarkMatrixMultiply(
    void (*func)(const Matrix&, const Matrix&, Matrix&),
                int M, int N, int K,
                const char* name,
                bool is_gpu) {

    // Allocate matrices & fill them with initial values
    Matrix A(M, K);
    fillMatrixRandom(A);
    Matrix B(K, N);
    fillMatrixRandom(B);
    Matrix C(M, N);
    fillMatrixZeros(A);

    // Warmup run
    func(A, B, C);
    if (is_gpu) cudaDeviceSynchronize();
    
    // Benchmark
    const int num_runs = 10;
    float total_time = 0.0f;
    
    if (is_gpu) {
        GPUTimer timer;
        for (int i = 0; i < num_runs; i++) {
            fillMatrixZeros(C);
            timer.start();
            func(A, B, C);
            total_time += timer.stop();
        }
    } else {
        CPUTimer timer;
        for (int i = 0; i < num_runs; i++) {
            fillMatrixZeros(C);
            timer.start();
            func(A, B, C);
            total_time += timer.stop();
        }
    }
    
    float avg_time = total_time / num_runs;
    
    // Calculate GFLOPS
    float num_ops = 2.0f * M * N * K;
    float gflops = (num_ops / avg_time) / 1e6;
    
    // Calculate bandwidth
    float bytes_accessed = (M*K + K*N + M*N) * sizeof(float);
    float bandwidth = (bytes_accessed / avg_time) / 1e6;
    
    printf("\n%s Performance:\n", name);
    printf("  Matrix size: [ %d x %d ] * [ %d x %d ]\n", M, K, K, N);
    printf("  Time: %.3f ms\n", avg_time);
    printf("  GFLOPS: %.2f\n", gflops);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth);

    BenchmarkResult result = {avg_time, gflops, bandwidth};
    return result;
}

void printMatrix(const float* matrix, int rows, int cols, const char* name) {
#ifdef NDEBUG
    return;
#endif
    printf("\n%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%7.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void test_gemm() {
    srand(time(NULL));
    
    printf("=== Matrix Multiplication Benchmark ===\n");
    
    // Test different sizes
    int sizes[] = {512, 1024, 2048, 4096};
    int num_sizes = 4;
    
    for (int i = 0; i < num_sizes; i++) {
        int size = sizes[i];
        
        printf("\n\n========================================\n");
        printf("=== Testing Size: [%d x %d] ===\n", size, size);
        printf("========================================\n");
        
        // Benchmark GPU
        BenchmarkResult gpu = benchmarkMatrixMultiply(
            matrixMultiplyGPU,
            size, size, size,
            "GPU (Naive)",
            GPU_TEST
        );

        
        // Benchmark GPU Tiled
        BenchmarkResult gpu_tiled = benchmarkMatrixMultiply(
            tiledMatrixMultiplyGPU,
            size, size, size,
            "GPU (Tiled)",
            GPU_TEST
        );

        
        // Calculate speedup
        float speedup = gpu.time_ms / gpu_tiled.time_ms;
        printf("\n>>> Speedup: %.2fx <<<\n", speedup);
    }
    
    printf("\n\n=== Benchmark Complete ===\n");    
}