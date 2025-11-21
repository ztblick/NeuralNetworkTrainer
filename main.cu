#include "matrix.cuh"
#include "timer.cuh"
#include <stdio.h>
#include <time.h>

int main() {
    srand(time(NULL));
    
    printf("=== Matrix Multiplication Benchmark ===\n");
    
    // Test different sizes
    int sizes[] = {512, 1024, 2048};
    int num_sizes = 3;
    
    for (int i = 0; i < num_sizes; i++) {
        int size = sizes[i];
        
        printf("\n\n========================================\n");
        printf("=== Testing Size: %d x %d ===\n", size, size);
        printf("========================================\n");
        
        // Benchmark GPU
        BenchmarkResult gpu = benchmarkMatrixMultiply(
            matrixMultiplyGPU,
            size, size, size,
            "GPU (Naive)",
            true
        );

        
        // Benchmark GPU Tiled
        BenchmarkResult gpu_tiled = benchmarkMatrixMultiply(
            tiledMatrixMultiplyGPU,
            size, size, size,
            "GPU (Tiled)",
            true
        );

        
        // Calculate speedup
        float speedup = gpu.time_ms / gpu_tiled.time_ms;
        printf("\n>>> Speedup: %.2fx <<<\n", speedup);
    }
    
    printf("\n\n=== Benchmark Complete ===\n");
    printf("Results saved to benchmark_results.csv\n");
    
    return 0;
}