#ifndef MATRIX_CUH
#define MATRIX_CUH

#define FILL_RANDOM true
#define FILL_ZEROES false

#define THREAD_X    16
#define THREAD_Y    16

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
void matrixMultiplyCPU(const float* A, const float* B, float* C, 
                       int M, int N, int K);

void matrixMultiplyGPU(const float* d_A, const float* d_B, float* d_C,
                       int M, int N, int K);

// Helper to allocate and initialize matrices
float* createMatrix(int rows, int cols, bool random);
void freeMatrix(float* matrix);
float* createCudaMatrix(int rows, int cols, bool random);
void freeCudaMatrix(float* matrix);

// Verification
bool verifyResults(const float* A, const float* B, int size, float epsilon = 1e-4);
void printMatrix(const float* matrix, int rows, int cols, const char* name = "Matrix");

struct BenchmarkResult {
    float time_ms;
    float gflops;
    float bandwidth_gb_s;
};

BenchmarkResult benchmarkMatrixMultiply(
    void (*func)(const float*, const float*, float*, int, int, int),
    int M, int N, int K,
    const char* name,
    bool is_gpu = false
);

void writeBenchmarkToFile(const char* filename,
                          int M, int N, int K,
                          BenchmarkResult cpu_result,
                          BenchmarkResult gpu_result);

#endif // MATRIX_CUH