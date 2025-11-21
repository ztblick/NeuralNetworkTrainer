#include "matrix.cuh"
#include "activations.cuh"
#include "timer.cuh"
#include <stdio.h>
#include <time.h>

void test_gemm() {
    srand(time(NULL));
    
    printf("=== Matrix Multiplication Benchmark ===\n");
    
    // Test different sizes
    int sizes[] = {512, 1024, 2048, 4096};
    int num_sizes = 4;
    ASSERT(sizes[num_sizes] > 0);
    
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

void test_relu() {
    int size = 1e3;
    float* data;
    
    cudaMallocManaged(&data, size * sizeof(float));
    
    // Initialize with some negative and positive values
    for (int i = 0; i < size; i++) {
        data[i] = i - size/2;  // [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
    }
    
    printf("Before ReLU:\n");
    for (int i = 0; i < size; i++) {
        printf("%.1f ", data[i]);
    }
    printf("\n");
    
    // Apply ReLU
    relu(data, size);
    
    printf("After ReLU:\n");
    for (int i = 0; i < size; i++) {
        printf("%.1f ", data[i]);
    }
    printf("\n");
    
    // Expected: [0, 0, 0, 0, 0, 0, 1, 2, 3, 4]
    
    cudaFree(data);
}

int main() {
    // test_gemm();
    test_relu();
}