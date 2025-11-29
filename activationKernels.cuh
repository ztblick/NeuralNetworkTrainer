#pragma once
#include "Matrix.h"

/**
 * These kernel launchers act as the bridge between the CPU and the GPU.
 * The Launch functions use CPU-based structs, like the Matix class.
 * But they will pass only fundamental data (float*) to the GPU.
 * This data, for now, is declared using CudaMallocManaged. Later on, for optimization,
 * it can be modified to copy memory between the host and the device.
 */

 /**
  * These functions are called my the ReLU layer class -- when forward() or backward()
  * are called, these functions are called, which in turn call their respective GPU kernels.
  */
void launch_relu_forward(const Matrix& d_input, Matrix& d_output, size_t batch_size);

void launch_relu_backward(const Matrix& d_grad_output, const Matrix& d_output, Matrix& d_grad_input, size_t batch_size);

/**
 * These functions are called by the loss layer class, which applies the softmax and cross-entropy loss
 * functions to create the probabilities of each classification and the intial gradient.
 */
void launch_output_forward(const Matrix& d_input, Matrix& d_output, size_t batch_size, int* true_class_indices);
void launch_output_backward(const Matrix& d_grad_output, const Matrix& d_output, Matrix& d_grad_input, size_t batch_size);
