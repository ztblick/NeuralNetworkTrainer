// ReLULayer.cpp - CPU implementation
#include "ReLULayer.h"
#include "ReLUKernels.cuh"
#include <cuda_runtime.h>

ReLULayer::ReLULayer(size_t batch_size, size_t input_size)
    : input_size(input_size) {
    this->batch_size = batch_size;
    cudaMallocManaged(&d_output, batch_size * input_size * sizeof(float));
}

ReLULayer::~ReLULayer() {
    cudaFree(d_output);
}

void ReLULayer::forward(const float* d_input) {
    launch_relu_forward(d_input, d_output, input_size, batch_size);
}

void ReLULayer::backward(const float* d_grad_output, float* d_grad_input) {
    launch_relu_backward(d_grad_output, d_output, d_grad_input, 
                         input_size, batch_size);
}