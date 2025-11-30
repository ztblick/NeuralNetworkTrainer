// ReLULayer.cpp - CPU implementation
#include "OutputLayer.h"
#include "activationKernels.cuh"
#include <cuda_runtime.h>

OutputLayer::OutputLayer(size_t batch_size, size_t num_classes) 
    : batch_size(batch_size),
      num_classes(num_classes),
      d_output(batch_size, num_classes) {

    cudaMallocManaged(&d_loss, batch_size * sizeof(float));
}

// Define the destructor -- Matrix destructor handles cleanup
OutputLayer::~OutputLayer() {
    cudaFree(d_loss);
}

void OutputLayer::forward(const Matrix& d_input) {
    ASSERT(false && "Use forward_with_labels for OutputLayer");
}

void OutputLayer::forward_with_labels(const Matrix& d_input, const int* d_true_classes) {
    launch_output_forward(d_input, d_output, batch_size, d_loss, d_true_classes);
}

void OutputLayer::backward(const Matrix& d_grad_output, Matrix& d_grad_input) {
    launch_output_backward(d_grad_output, d_output, d_grad_input, batch_size);
}

const Matrix& OutputLayer::getOutput() const {
    return d_output;  // Return reference to member
}

float* OutputLayer::getLoss() {
    return d_loss;
}