#pragma once
#include "config.h"
#include "Matrix.h"
#define TILE_SIZE   16
#define THREAD_X    32
#define THREAD_Y    8

void tiledMatrixMultiplyGPU(const Matrix& A, const Matrix& B, Matrix& C);

void matrixMultiplyGPU(const Matrix& A, const Matrix& B, Matrix& C);

/**
 * These functions are called by the dense layer objects. They launch the dense layer's
 * forward or backward passes. They call the cuda kernels defined in the kernels files.
 */
void launch_dense_forward(  const Matrix& weights,
                            const Matrix& d_input,
                            Matrix& d_output,
                            const Matrix& bias,
                            size_t batch_size,
                            size_t input_features,
                            size_t output_features);
