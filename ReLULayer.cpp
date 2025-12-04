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

#if DEBUG
    printf("ReLU Layer::backward called\n");
    printf("  grad_output[0] = %.6f\n", d_grad_output.data[0]);
    printf("  d_output[0] = %.6f\n", d_output.data[0]);
#endif

    launch_relu_backward(d_grad_output, d_output, d_grad_input, batch_size, features);

#if DEBUG
    printf("  grad_input[0] = %.6f\n", d_grad_input.data[0]);
#endif

}

void ReLULayer::updateWeights(const float learningRate) {

}

const Matrix& ReLULayer::getOutput() const {
    return d_output;  // Return reference to member
}