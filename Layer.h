// Layer.h - Pure CPU interface
#pragma once
#include "Matrix.h"
#include <cstddef>

class Layer {    
public:
    Matrix d_output;       
    Matrix d_grad_input;  

    virtual ~Layer() = default;
    virtual void forward(const Matrix& d_input) = 0;
    virtual void backward(const Matrix& d_grad_output) = 0;  
    virtual void updateWeights(float learning_rate) {} 
    virtual const Matrix& getOutput() const = 0;
    virtual const Matrix& getGradInput() const { return d_grad_input; } 
};