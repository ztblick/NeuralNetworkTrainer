// ReLULayer.h
#pragma once
#include "Layer.h"

class ReLULayer : public Layer {
private:
    size_t batch_size;
    size_t features;
    
public:
    ReLULayer(size_t batch_size, size_t features);
    // ~ReLULayer();   // Matrix destructor handles cleanup
    
    void forward(const Matrix& d_input) override;
    void backward(const Matrix& d_grad_output) override;
    void updateWeights(const float learningRate) override;
    const Matrix& getOutput() const override;
};