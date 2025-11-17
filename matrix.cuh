#ifndef MATRIX_CUH
#define MATRIX_CUH

#define FILL_RANDOM true
#define FILL_ZEROES false

// Matrix multiplication: C = A * B
// A: M x K, B: K x N, C: M x N
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

#endif // MATRIX_CUH