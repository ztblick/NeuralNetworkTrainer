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

void launch_dense_backward(
    const Matrix& d_grad_output,  // [batch, output_features] - gradient from next layer
    Matrix& d_grad_input,         // [batch, input_features] - OUTPUT: gradient to previous layer
    Matrix& grad_weights,         // [input_features, output_features] - OUTPUT: weight gradients
    Matrix& grad_bias,            // [1, output_features] - OUTPUT: bias gradients
    const Matrix& weights,        // [input_features, output_features] - from forward pass
    const Matrix& d_cached_input, // [batch, input_features] - from forward pass
    size_t batch_size,
    size_t input_features,
    size_t output_features);


void launch_weight_update(
    Matrix& weights, 
    const Matrix& gradients, 
    float learning_rate);
