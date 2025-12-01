// DenseLayer.h
#pragma once
#include "Layer.h"

class DenseLayer : public Layer {
private:
    size_t batch_size;
    size_t input_features;
    size_t output_features;
    
public:
    Matrix d_output;
    Matrix weights;
    Matrix bias;

    DenseLayer(size_t batch_size, size_t input_features, size_t output_features);
    // ~DenseLayer();
    
    void forward(const Matrix& d_input) override;
    void backward(const Matrix& d_grad_output, Matrix& d_grad_input) override;
    const Matrix& getOutput() const override;
};