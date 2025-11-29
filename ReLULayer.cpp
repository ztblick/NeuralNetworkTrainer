// ReLULayer.cpp - CPU implementation
#include "ReLULayer.h"
#include "activationKernels.cuh"
#include <cuda_runtime.h>

ReLULayer::ReLULayer(size_t batch_size, size_t input_size)
    : d_output(batch_size * input_size, 1),  // Initialize in initializer list
      input_size(input_size) {

    this->batch_size = batch_size;
}

// Define the destructor -- Matrix destructor handles cleanup
// ReLULayer::~ReLULayer() = default;

void ReLULayer::forward(const Matrix& d_input) {
    launch_relu_forward(d_input, d_output, batch_size);
}

void ReLULayer::backward(const Matrix& d_grad_output, Matrix& d_grad_input) {
    launch_relu_backward(d_grad_output, d_output, d_grad_input, batch_size);
}

const Matrix& ReLULayer::getOutput() const {
    return d_output;  // Return reference to member
}