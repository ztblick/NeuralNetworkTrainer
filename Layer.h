// Layer.h - Pure CPU interface
#pragma once
#include "config.h"
#include <cstddef>

class Layer {
protected:
    float* d_output;  // GPU pointer
    size_t batch_size;
    
public:
    virtual ~Layer() = default;
    virtual void forward(const float* d_input) = 0;
    virtual void backward(const float* d_grad_output, float* d_grad_input) = 0;
    virtual float* getOutput() const { return d_output; }
};