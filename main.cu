#include "matrix.cuh"
#include "timer.cuh"
#include <stdio.h>
#include <time.h>

int main() {
    srand(time(NULL));
    
    printf("=== Matrix Multiplication Benchmark ===\n");
    
    // Test different sizes
    int sizes[] = {128, 256, 512, 1024};
    int num_sizes = 4;
    
    for (int i = 0; i < num_sizes; i++) {
        int size = sizes[i];
        
        printf("\n\n========================================\n");
        printf("=== Testing Size: %d x %d ===\n", size, size);
        printf("========================================\n");
        
        // Benchmark CPU
        BenchmarkResult cpu = benchmarkMatrixMultiply(
            matrixMultiplyCPU,
            size, size, size,
            "CPU (Naive)",
            false
        );
        
        // Benchmark GPU
        BenchmarkResult gpu = benchmarkMatrixMultiply(
            matrixMultiplyGPU,
            size, size, size,
            "GPU (Naive)",
            true
        );
        
        // Calculate speedup
        float speedup = cpu.time_ms / gpu.time_ms;
        printf("\n>>> Speedup: %.2fx <<<\n", speedup);
        
        // Write to CSV
        writeBenchmarkToFile("benchmark_results.csv", 
                            size, size, size, cpu, gpu);
    }
    
    printf("\n\n=== Benchmark Complete ===\n");
    printf("Results saved to benchmark_results.csv\n");
    
    return 0;
}