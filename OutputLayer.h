// OutputLayer.h
#pragma once
#include "Layer.h"

class OutputLayer : public Layer {
private:
    size_t batch_size;
    size_t num_classes;
    float* d_loss;
    
public:
    OutputLayer(size_t batch_size, size_t num_classes);
    ~OutputLayer();
    
    void forward(const Matrix& d_input) override;
    void forward_with_labels(const Matrix& d_input, const int* d_true_classes);
    void backward(const Matrix& d_grad_output) override;
    const Matrix& getOutput() const override;
    void updateWeights(const float learningRate) override;
    float* getLoss();
    float getAverageLoss();
};