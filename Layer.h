// Layer.h - Pure CPU interface
#pragma once
#include "Matrix.h"
#include <cstddef>

class Layer {
public:
    virtual ~Layer() = default;
    virtual void forward(const Matrix& input) = 0;
    virtual void backward(const Matrix& d_grad_output, Matrix& d_grad_input) = 0;
    virtual const Matrix& getOutput() const = 0;
};