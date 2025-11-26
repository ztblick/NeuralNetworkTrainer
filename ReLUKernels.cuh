// relu_kernels.cuh - CUDA interface
#pragma once
#include "config.h"

// Kernel launchers - these are the bridge between CPU and GPU
void launch_relu_forward(const float* d_input, float* d_output, 
                         size_t size, size_t batch_size);

void launch_relu_backward(const float* d_grad_output, const float* d_output,
                          float* d_grad_input, size_t size, size_t batch_size);