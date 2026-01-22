#include "activationKernels.cuh"
#include "ReLULayer.h"
#include "OutputLayer.h"
#include "DenseLayer.h"
#include "mnistLoader.h"
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

void print_mnist_sample(const float* image, int label) {
    printf("Label: %d\n", label);
    for (int row = 0; row < 28; row++) {
        for (int col = 0; col < 28; col++) {
            float pixel = image[row * 28 + col];
            if (pixel > 0.5) printf("██");
            else if (pixel > 0.2) printf("▓▓");
            else if (pixel > 0.1) printf("░░");
            else printf("  ");
        }
        printf("\n");
    }
}

int train_neural_network() {
    // --- SETUP ---    
    // 0. Load MNIST data
    MNISTDataset mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

    // 1. Create the Stack
    std::vector<Layer*> network;
    network.push_back(new DenseLayer(BATCH_SIZE, 784, 256, 0));
    network.push_back(new ReLULayer(BATCH_SIZE, 256));          
    network.push_back(new DenseLayer(BATCH_SIZE, 256, 128, 1)); 
    network.push_back(new ReLULayer(BATCH_SIZE, 128));          
    network.push_back(new DenseLayer(BATCH_SIZE, 128, 64, 2));              
    network.push_back(new ReLULayer(BATCH_SIZE, 64));                      
    network.push_back(new DenseLayer(BATCH_SIZE, 64, NUM_CLASSES, 3));               
    
    // The Loss Layer (Softmax + CrossEntropy) sits at the end
    OutputLayer* output_layer = new OutputLayer(BATCH_SIZE, NUM_CLASSES);

    // // 2. Pre-allocate Memory
    Matrix d_batch_images(BATCH_SIZE, 784);
    int* d_batch_labels;
    cudaMallocManaged(&d_batch_labels, BATCH_SIZE * sizeof(int));

    // // --- TRAINING LOOP ---
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {

        int num_batches = mnist.num_samples / BATCH_SIZE;

        for (int batch = 0; batch < num_batches; batch++) {
            
            // Load batch data from MNIST
            mnist.get_batch(batch, BATCH_SIZE, d_batch_images.data, d_batch_labels);
            
            // Forward pass
            const Matrix* input = &d_batch_images;
            for (auto layer : network) {
                layer->forward(*input);
                input = &layer->getOutput();
            }
            
            // Loss calculation
            output_layer->forward_with_labels(*input, d_batch_labels);

            // Backward pass
            const Matrix* grad_input = &output_layer->getOutput();

            for (int i = network.size() - 1; i >= 0; i--) {
                network[i]->backward(*grad_input);
                grad_input = &network[i]->getGradInput();
            }

            // Update weights
            for (auto layer : network) {
                layer->updateWeights(LEARNING_RATE);
            }
            
            // Print loss every PRINT_FREQUENCY batches
            if (batch % PRINT_FREQUENCY == 0) {
                float avg_loss = output_layer->getAverageLoss();
                printf("Epoch %d, Batch %d, Loss: %.4f\n", epoch, batch, avg_loss);
            }
        }
    }
    // Cleanup
    for (auto layer : network) delete layer;
    delete output_layer;
    cudaFree(d_batch_labels);
    
    return 0;
}

int main() {
    // test_gemm();
    // test_relu();
    // test_output_layer();
    // test_dense_layer();
    // test_forward_pass_with_mnist();
    train_neural_network();
}