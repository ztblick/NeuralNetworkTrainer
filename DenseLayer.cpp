#include <random>
#include "DenseLayer.h"
#include "MatrixKernels.cuh"
#include <cuda_runtime.h>

void initialize_he(float* data, size_t size, size_t fan_in) {
    std::random_device rd;
    std::mt19937 gen(rd());
    float std = sqrtf(2.0f / fan_in);
    std::normal_distribution<float> dist(0.0f, std);
    
    for (size_t i = 0; i < size; i++) {
        data[i] = dist(gen);
    }
}

DenseLayer::DenseLayer(size_t batch_size, size_t input_features, size_t output_features)
    : batch_size(batch_size),
      input_features(input_features),
      output_features(output_features),
      weights(input_features, output_features),
      bias(1, output_features), 
      d_output(batch_size, output_features),
      grad_weights(input_features, output_features),
      grad_bias(1, output_features),              
      d_cached_input(batch_size, input_features) {
    
    // Initialize weights with He initialization
    initialize_he(weights.data, input_features * output_features, input_features);
    
    // Initialize bias to zero
    cudaMemset(bias.data, 0, output_features * sizeof(float));
}

void DenseLayer::forward(const Matrix& d_input) {
    // Cache the inputs for the later backpropagation calculation
    cudaMemcpy(d_cached_input.data, d_input.data, 
               batch_size * input_features * sizeof(float),
               cudaMemcpyDeviceToDevice);

    launch_dense_forward(weights, d_input, d_output, bias, batch_size, input_features, output_features);
}

void DenseLayer::backward(const Matrix& d_grad_output, Matrix& d_grad_input) {
    launch_dense_backward(d_grad_output,
                         d_grad_input,
                         grad_weights,      
                         grad_bias,         
                         weights,
                         d_cached_input,
                         batch_size,
                         input_features,
                         output_features);
}

void DenseLayer::updateWeights(const float learningRate) {

}

const Matrix& DenseLayer::getOutput() const {
    return d_output;  // Return reference to member
}