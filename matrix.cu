#include "matrix.cuh"
#include "debug.cuh"
#include "timer.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>

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
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (!(row < M && col < N)) return;

    float sum = 0.0;

    for (int idx = 0; idx < K; idx++)
        sum += A[row * K + idx] * B[N * idx + col];
    
    C[row * N + col] = sum;
}

// GPU wrapper function
void matrixMultiplyGPU(const float* d_A, const float* d_B, float* d_C,
                       int M, int N, int K) {

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
    matrixMultiplyKernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);

    cudaDeviceSynchronize();
}

void fillMatrixRandom(float* matrix, int rows, int cols, 
                      float min = 0.0f, float max = 1.0f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = dis(gen);
    }
}

void fillMatrixZeros(float* matrix, int rows, int cols) {
    memset(matrix, 0, rows * cols * sizeof(float));
}

// Initialize and fill the matrix
float* createMatrix(int rows, int cols, bool fill_option) {

    // Allocate host memory
    float* matrix = (float *) malloc(sizeof(float) * rows * cols);
    ASSERT(matrix);

    // Initialize data
    if (fill_option == FILL_RANDOM)
        fillMatrixRandom(matrix, rows, cols);
    else /*fill_option == FILL_ZEROES*/
        fillMatrixZeros(matrix, rows, cols);
    return matrix;
}

float* createCudaMatrix(int rows, int cols, bool fill_option) {
    float* matrix;
    
    // Allocate unified memory
    cudaError_t err = cudaMallocManaged(&matrix, rows * cols * sizeof(float));
    ASSERT(err == cudaSuccess);

    // Initialize data
    if (fill_option == FILL_RANDOM)
        fillMatrixRandom(matrix, rows, cols);
    else /*fill_option == FILL_ZEROES*/
        fillMatrixZeros(matrix, rows, cols);
    return matrix;
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


// Benchmarking and timing code from Claude

BenchmarkResult benchmarkMatrixMultiply(
    void (*func)(const float*, const float*, float*, int, int, int),
                int M, int N, int K,
                const char* name,
                bool is_gpu) {
    // Allocate matrices
    float *A, *B, *C;
    cudaMallocManaged(&A, M * K * sizeof(float));
    cudaMallocManaged(&B, K * N * sizeof(float));
    cudaMallocManaged(&C, M * N * sizeof(float));
    
    // Initialize using your functions
    fillMatrixRandom(A, M, K);
    fillMatrixRandom(B, K, N);
    fillMatrixZeros(C, M, N);
    
    // Warmup run
    func(A, B, C, M, N, K);
    if (is_gpu) cudaDeviceSynchronize();
    
    // Benchmark
    const int num_runs = 10;
    float total_time = 0.0f;
    
    if (is_gpu) {
        GPUTimer timer;
        for (int i = 0; i < num_runs; i++) {
            fillMatrixZeros(C, M, N);
            timer.start();
            func(A, B, C, M, N, K);
            total_time += timer.stop();
        }
    } else {
        CPUTimer timer;
        for (int i = 0; i < num_runs; i++) {
            fillMatrixZeros(C, M, N);
            timer.start();
            func(A, B, C, M, N, K);
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
    printf("  Matrix size: %dx%d * %dx%d\n", M, K, K, N);
    printf("  Time: %.3f ms\n", avg_time);
    printf("  GFLOPS: %.2f\n", gflops);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth);
    
    // Cleanup
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    
    BenchmarkResult result = {avg_time, gflops, bandwidth};
    return result;
}

void writeBenchmarkToFile(const char* filename,
                          int M, int N, int K,
                          BenchmarkResult cpu_result,
                          BenchmarkResult gpu_result) {
    FILE* f = fopen(filename, "a");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        return;
    }
    
    // Write header if file is empty
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0) {
        fprintf(f, "M,N,K,CPU_ms,CPU_GFLOPS,GPU_ms,GPU_GFLOPS,Speedup\n");
    }
    
    float speedup = cpu_result.time_ms / gpu_result.time_ms;
    
    fprintf(f, "%d,%d,%d,%.3f,%.2f,%.3f,%.2f,%.2fx\n",
            M, N, K,
            cpu_result.time_ms, cpu_result.gflops,
            gpu_result.time_ms, gpu_result.gflops,
            speedup);
    
    fclose(f);
    printf("Results written to %s\n", filename);
}