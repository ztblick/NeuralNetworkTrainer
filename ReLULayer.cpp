// ReLULayer.cpp - CPU implementation
#include "ReLULayer.h"
#include "activationKernels.cuh"
#include <cuda_runtime.h>

ReLULayer::ReLULayer(size_t batch_size, size_t features)
    : features(features),
      batch_size(batch_size) {

      d_output = Matrix(batch_size, features);
      d_grad_input = Matrix(batch_size, features);
    }
// Define the destructor -- Matrix destructor handles cleanup
// ReLULayer::~ReLULayer() = default;

void ReLULayer::forward(const Matrix& d_input) {
    launch_relu_forward(d_input, d_output, batch_size);
}

void ReLULayer::backward(const Matrix& d_grad_output) {
    launch_relu_backward(d_grad_output, d_output, d_grad_input, batch_size, features);
}

void ReLULayer::updateWeights(const float learningRate) {

}

const Matrix& ReLULayer::getOutput() const {
    return d_output;  // Return reference to member
}