// activationKernels.cuh - CUDA interface
#pragma once
#include "Matrix.h"

// Kernel launchers - these are the bridge between CPU and GPU
void launch_relu_forward(const Matrix& d_input, Matrix& d_output, size_t batch_size);

void launch_relu_backward(const Matrix& d_grad_output, const Matrix& d_output, Matrix& d_grad_input, size_t batch_size);

void launch_softmax_forward();
