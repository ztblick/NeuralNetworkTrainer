#include "activationKernels.cuh"
#include "ReLULayer.h"
#include <vector>
#include <cuda_runtime.h>

int run_neural_network() {
    // --- SETUP ---    
    // 1. Create the Stack
    std::vector<Layer*> network;
    // network.push_back(new DenseLayer(784, 128));             // Input -> Hidden 1
    network.push_back(new ReLULayer(BATCH_SIZE, 128));          // Activation
    // network.push_back(new DenseLayer(128, 64));              // Hidden 1 -> Hidden 2
    // network.push_back(new ReLULayer());                      // Activation
    // network.push_back(new DenseLayer(64, 10));               // Hidden 2 -> Logits
    
    // The Loss Layer (Softmax + CrossEntropy) sits at the end
    // SoftmaxCrossEntropy* loss_layer = new SoftmaxCrossEntropy();

    // // 2. Pre-allocate Tensors (Memory)
    // // You create 'intermediate' tensors to hold data between layers
    // Tensor input_batch(64, 784);
    // Tensor h1_out(64, 128);
    // Tensor h2_out(64, 64);
    // Tensor final_logits(64, 10);
    // Tensor probs(64, 10);

    // // --- TRAINING LOOP ---
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {

            // Initialize data for this particular training run
            setup_batch_labels(BATCH_SIZE);
        
    //     // A. Forward Pass (The Chain Reaction)
    //     network[0]->forward(input_batch, h1_out);
    //     network[1]->forward(h1_out, h1_out); // ReLU applies in-place or to new buffer
    //     network[2]->forward(h1_out, h2_out);
    //     network[3]->forward(h2_out, h2_out);
    //     network[4]->forward(h2_out, final_logits);

    //     // B. Loss Calculation (Your new kernel)
    //     loss_layer->set_targets(current_batch_targets);
    //     loss_layer->forward(final_logits, probs);

    //     // C. Backward Pass (Reverse Chain Reaction)
    //     // Start with the loss layer
    //     Tensor grad_flow(64, 10); 
    //     loss_layer->backward(probs, grad_flow); // Gradients start here!

    //     // Pass gradients back up the stack
    //     network[4]->backward(grad_flow, h2_grad);
    //     network[3]->backward(h2_grad, h2_grad); // ReLU backward
    //     network[2]->backward(h2_grad, h1_grad);
    //     ... and so on ...
        
    //     // D. Update Weights
    //     for (auto layer : network) {
    //         layer->update_weights(0.01f); // SGD
    //     }
    }
    return 0;
}

void setup_batch_labels(int batch_size) {
    int h_labels[batch_size] = {0};  // TODO -- add MNIST labels for this test
    int* d_labels;
    cudaMalloc(&d_labels, batch_size * sizeof(int));
    cudaMemcpy(d_labels, h_labels, batch_size * sizeof(int), cudaMemcpyHostToDevice);
}


void test_relu() {
    int size = 1e3;
    Matrix input(size / 10, 10);
    
    // Initialize with some negative and positive values
    for (int i = 0; i < input.size; i++) {
        input.data[i] = i - size/2;
    }
    
    printf("Before ReLU:\n");
    for (int i = 0; i < size; i++) {
        printf("%.1f ", input.data[i]);
    }
    printf("\n");
    
    // Apply ReLU
    Layer* l = new ReLULayer(BATCH_SIZE, 128);
    l->forward(input);
    const Matrix& output = l->getOutput();

    printf("After ReLU:\n");
    for (int i = 0; i < size; i++) {
        printf("%.1f ", output.data[i]);
    }
    printf("\n");
}

int main() {
    // test_gemm();
    test_relu();
}