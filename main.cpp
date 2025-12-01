#include "activationKernels.cuh"
#include "ReLULayer.h"
#include "OutputLayer.h"
#include "DenseLayer.h"
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

void setup_batch_labels(int batch_size) {
    int h_labels[batch_size] = {0};  // TODO -- add MNIST labels for this test
    int* d_labels;
    cudaMalloc(&d_labels, batch_size * sizeof(int));
    cudaMemcpy(d_labels, h_labels, batch_size * sizeof(int), cudaMemcpyHostToDevice);
}

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

void test_output_layer() {
    printf("\n=== Testing Output Layer ===\n");
    
    // Simple case: 1 sample, 3 classes
    size_t batch_size = 1;
    size_t num_classes = 3;
    
    // Create logits: [2.0, 1.0, 0.1]
    float h_logits[] = {2.0f, 1.0f, 0.1f};
    int h_true_class[] = {0};  // True class is index 0
    
    // Create input matrix and copy logits
    Matrix d_input(batch_size, num_classes);
    cudaMemcpy(d_input.data, h_logits, num_classes * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create true class labels on device
    int* d_true_classes;
    cudaMallocManaged(&d_true_classes, batch_size * sizeof(int));
    cudaMemcpy(d_true_classes, h_true_class, batch_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Create output layer
    OutputLayer output_layer(batch_size, num_classes);
    
    // Run forward pass
    output_layer.forward_with_labels(d_input, d_true_classes);
    cudaDeviceSynchronize();
    
    // Get results
    const Matrix& d_output = output_layer.getOutput();
    float* d_loss = output_layer.getLoss();  // You'll need to add a getter or make it public
    
    // Copy results to host
    float h_gradients[3];
    float h_loss;
    cudaMemcpy(h_gradients, d_output.data, num_classes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute expected values
    float max_logit = 2.0f;
    float exp_vals[3] = {
        expf(2.0f - 2.0f),  // = 1.0
        expf(1.0f - 2.0f),  // = 0.368
        expf(0.1f - 2.0f)   // = 0.150
    };
    float sum = exp_vals[0] + exp_vals[1] + exp_vals[2];  // = 1.518
    
    float expected_probs[3] = {
        exp_vals[0] / sum,  // = 0.659
        exp_vals[1] / sum,  // = 0.242
        exp_vals[2] / sum   // = 0.099
    };
    
    float expected_gradients[3] = {
        expected_probs[0] - 1.0f,  // True class gets -1
        expected_probs[1] - 0.0f,
        expected_probs[2] - 0.0f
    };
    
    float expected_loss = -logf(expected_probs[0]);
    
    // Print results
    printf("Logits: [%.1f, %.1f, %.1f]\n", h_logits[0], h_logits[1], h_logits[2]);
    printf("Expected probs: [%.3f, %.3f, %.3f] (sum=%.3f)\n", 
           expected_probs[0], expected_probs[1], expected_probs[2],
           expected_probs[0] + expected_probs[1] + expected_probs[2]);
    printf("Actual gradients: [%.3f, %.3f, %.3f]\n", 
           h_gradients[0], h_gradients[1], h_gradients[2]);
    printf("Expected gradients: [%.3f, %.3f, %.3f]\n",
           expected_gradients[0], expected_gradients[1], expected_gradients[2]);
    printf("Expected loss: %.3f\n", expected_loss);
    printf("Actual loss: %.3f\n", h_loss);
    
    // Verify
    float tolerance = 1e-4;
    bool passed = true;
    for (int i = 0; i < 3; i++) {
        if (fabsf(h_gradients[i] - expected_gradients[i]) > tolerance) {
            printf("ERROR: Gradient mismatch at index %d\n", i);
            passed = false;
        }
    }
    if (fabsf(h_loss - expected_loss) > tolerance) {
        printf("ERROR: Loss mismatch\n");
        passed = false;
    }
    
    if (passed) {
        printf("✓ Test passed!\n");
    }
    
    // Cleanup
    cudaFree(d_true_classes);
}

void test_dense_layer() {
    printf("\n=== Testing Dense Layer ===\n");
    
    // Small test: 2 samples, 3 input features, 2 output features
    size_t batch_size = 2;
    size_t input_features = 3;
    size_t output_features = 2;
    
    // Create simple input: [[1,2,3], [4,5,6]]
    float h_input[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    
    Matrix d_input(batch_size, input_features);
    cudaMemcpy(d_input.data, h_input, 6 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create dense layer
    DenseLayer dense(batch_size, input_features, output_features);
    
    // Manually set weights for predictable output
    // weights = [[0.1, 0.2],
    //            [0.3, 0.4],
    //            [0.5, 0.6]]
    float h_weights[] = {
        0.1f, 0.2f,
        0.3f, 0.4f,
        0.5f, 0.6f
    };
    cudaMemcpy(dense.weights.data, h_weights, 6 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set bias = [0.1, 0.2]
    float h_bias[] = {0.1f, 0.2f};
    cudaMemcpy(dense.bias.data, h_bias, 2 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run forward pass
    dense.forward(d_input);
    cudaDeviceSynchronize();
    
    // Get output
    float h_output[4];  // 2 samples * 2 features
    cudaMemcpy(h_output, dense.d_output.data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute expected output manually
    // Sample 0: [1,2,3] * [[0.1,0.2],[0.3,0.4],[0.5,0.6]] + [0.1,0.2]
    //         = [1*0.1 + 2*0.3 + 3*0.5, 1*0.2 + 2*0.4 + 3*0.6] + [0.1,0.2]
    //         = [0.1 + 0.6 + 1.5, 0.2 + 0.8 + 1.8] + [0.1,0.2]
    //         = [2.2, 2.8] + [0.1,0.2]
    //         = [2.3, 3.0]
    
    // Sample 1: [4,5,6] * weights + bias
    //         = [4*0.1 + 5*0.3 + 6*0.5, 4*0.2 + 5*0.4 + 6*0.6] + [0.1,0.2]
    //         = [0.4 + 1.5 + 3.0, 0.8 + 2.0 + 3.6] + [0.1,0.2]
    //         = [4.9, 6.4] + [0.1,0.2]
    //         = [5.0, 6.6]
    
    float expected[] = {2.3f, 3.0f, 5.0f, 6.6f};
    
    printf("Input:\n");
    printf("  [%.1f, %.1f, %.1f]\n", h_input[0], h_input[1], h_input[2]);
    printf("  [%.1f, %.1f, %.1f]\n", h_input[3], h_input[4], h_input[5]);
    
    printf("\nOutput:\n");
    printf("  [%.1f, %.1f]  (expected: [%.1f, %.1f])\n", 
           h_output[0], h_output[1], expected[0], expected[1]);
    printf("  [%.1f, %.1f]  (expected: [%.1f, %.1f])\n",
           h_output[2], h_output[3], expected[2], expected[3]);
    
    // Verify
    float tolerance = 1e-4;
    bool passed = true;
    for (int i = 0; i < 4; i++) {
        if (fabsf(h_output[i] - expected[i]) > tolerance) {
            printf("ERROR: Output mismatch at index %d: got %.3f, expected %.3f\n", 
                   i, h_output[i], expected[i]);
            passed = false;
        }
    }
    
    if (passed) {
        printf("✓ Dense layer test passed!\n");
    }
}

int main() {
    // test_gemm();
    // test_relu();
    // test_output_layer();
    test_dense_layer();
}