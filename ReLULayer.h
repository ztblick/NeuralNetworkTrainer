// ReLULayer.h
#pragma once
#include "Layer.h"

class ReLULayer : public Layer {
private:
    size_t batch_size;
    size_t input_size;
    Matrix d_output;
    
public:
    ReLULayer(size_t batch_size, size_t input_size);
    // ~ReLULayer();   // Matrix destructor handles cleanup
    
    void forward(const Matrix& d_input) override;
    void backward(const Matrix& d_grad_output, Matrix& d_grad_input) override;
    const Matrix& getOutput() const override;
};