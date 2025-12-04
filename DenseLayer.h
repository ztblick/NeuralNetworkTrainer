// DenseLayer.h
#pragma once
#include "Layer.h"

class DenseLayer : public Layer {
private:
    size_t batch_size;
    size_t input_features;
    size_t output_features;
    Matrix d_cached_input;


public:
    Matrix weights;
    Matrix bias;
    Matrix grad_weights;  // ∂L/∂W
    Matrix grad_bias;     // ∂L/∂b

    DenseLayer(size_t batch_size, size_t input_features, size_t output_features);
    // ~DenseLayer();
    
    void forward(const Matrix& d_input) override;
    void backward(const Matrix& d_grad_output) override;
    void updateWeights(const float learningRate) override;
    const Matrix& getOutput() const override;
};