// ReLULayer.h
#pragma once
#include "Layer.h"

class ReLULayer : public Layer {
private:
    size_t input_size;
    
public:
    ReLULayer(size_t batch_size, size_t input_size);
    ~ReLULayer();
    
    void forward(const float* d_input) override;
    void backward(const float* d_grad_output, float* d_grad_input) override;
};