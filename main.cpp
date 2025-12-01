#include "activationKernels.cuh"
#include "ReLULayer.h"
#include "OutputLayer.h"
#include "DenseLayer.h"
#include "mnistLoader.h"
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

int train_neural_network() {
    // --- SETUP ---    

    // 0. Load MNIST data (TODO: implement this)
    MNISTDataset mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

    // 1. Create the Stack
    std::vector<Layer*> network;
    network.push_back(new DenseLayer(BATCH_SIZE, 784, 128));
    network.push_back(new ReLULayer(BATCH_SIZE, 128));          
    network.push_back(new DenseLayer(BATCH_SIZE, 128, 64));              
    network.push_back(new ReLULayer(BATCH_SIZE, 64));                      
    network.push_back(new DenseLayer(BATCH_SIZE, 64, NUM_CLASSES));               
    
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
            
            // TODO: Backward pass
            // TODO: Weight updates
            
            // Print loss every 100 batches
            if (batch % 100 == 0) {
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

void test_forward_pass_with_mnist() {
    printf("\n=== Testing Forward Pass with MNIST ===\n");
    
    // Load MNIST
    MNISTDataset mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    
    // Small batch for testing
    size_t batch_size = 4;
    
    // Create network
    std::vector<Layer*> network;
    network.push_back(new DenseLayer(batch_size, 784, 128));
    network.push_back(new ReLULayer(batch_size, 128));
    network.push_back(new DenseLayer(batch_size, 128, 64));
    network.push_back(new ReLULayer(batch_size, 64));
    network.push_back(new DenseLayer(batch_size, 64, 10));
    
    OutputLayer* output_layer = new OutputLayer(batch_size, 10);
    
    // Allocate batch memory
    Matrix d_batch_images(batch_size, 784);
    int* d_batch_labels;
    cudaMallocManaged(&d_batch_labels, batch_size * sizeof(int));
    
    // Load first batch
    mnist.get_batch(0, batch_size, d_batch_images.data, d_batch_labels);
    cudaDeviceSynchronize();
    
    // Print what we loaded
    printf("\n1. Data Loading Test:\n");
    int h_labels[4];
    cudaMemcpy(h_labels, d_batch_labels, 4 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("   Batch labels: [%d, %d, %d, %d]\n", 
           h_labels[0], h_labels[1], h_labels[2], h_labels[3]);
    
    // Check a few pixel values
    float h_pixels[10];
    cudaMemcpy(h_pixels, d_batch_images.data, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("   First 10 pixels of image 0: ");
    for (int i = 0; i < 10; i++) printf("%.2f ", h_pixels[i]);
    printf("\n");
    
    // Forward pass
    printf("\n2. Forward Pass:\n");
    const Matrix* input = &d_batch_images;
    for (size_t i = 0; i < network.size(); i++) {
        network[i]->forward(*input);
        input = &network[i]->getOutput();
        printf("   After layer %ld: shape [%d, %d]\n", 
               i, input->rows, input->cols);
    }
    
    output_layer->forward_with_labels(*input, d_batch_labels);
    printf("   After output layer: gradients computed\n");
    
    // Check output probabilities (before gradient subtraction, we need raw softmax)
    // For this test, let's check the gradients instead
    printf("\n3. Output Layer Validation:\n");
    float h_gradients[40];  // 4 samples * 10 classes
    cudaMemcpy(h_gradients, output_layer->getOutput().data, 
               40 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // For each sample, check:
    // - Gradient at true class should be (prob - 1), so negative
    // - Gradients at other classes should be prob, so positive
    // - Sum of (gradients + one-hot) should be ~0 (numerical check)
    for (int sample = 0; sample < 4; sample++) {
        printf("   Sample %d (label=%d):\n", sample, h_labels[sample]);
        printf("     Gradients: [");
        
        float sum_probs = 0.0f;
        for (int c = 0; c < 10; c++) {
            float grad = h_gradients[sample * 10 + c];
            // Recover probability from gradient
            float prob = (c == h_labels[sample]) ? grad + 1.0f : grad;
            sum_probs += prob;
            
            if (c < 3 || c == h_labels[sample]) {
                printf("%.3f%s", grad, c < 9 ? ", " : "");
            } else if (c == 3) {
                printf("..., ");
            }
        }
        printf("]\n");
        printf("     Sum of probabilities: %.4f (should be ~1.0)\n", sum_probs);
        
        // Check gradient at true class is negative
        float true_class_grad = h_gradients[sample * 10 + h_labels[sample]];
        if (true_class_grad >= 0.0f) {
            printf("     WARNING: Gradient at true class should be negative!\n");
        }
    }
    
    // Check losses
    printf("\n4. Loss Validation:\n");
    float h_losses[4];
    cudaMemcpy(h_losses, output_layer->getLoss(), 4 * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < 4; i++) {
        printf("   Sample %d loss: %.4f\n", i, h_losses[i]);
        if (h_losses[i] < 0 || h_losses[i] > 10) {
            printf("     WARNING: Loss seems unreasonable!\n");
        }
    }
    
    float avg_loss = output_layer->getAverageLoss();
    printf("   Average loss: %.4f\n", avg_loss);
    printf("   (Random guessing would give ~2.3 for 10 classes)\n");
    
    // Test with different batch to ensure different outputs
    printf("\n5. Variety Test (different inputs → different outputs):\n");
    mnist.get_batch(1, batch_size, d_batch_images.data, d_batch_labels);
    
    input = &d_batch_images;
    for (auto layer : network) {
        layer->forward(*input);
        input = &layer->getOutput();
    }
    output_layer->forward_with_labels(*input, d_batch_labels);
    
    float avg_loss2 = output_layer->getAverageLoss();
    printf("   Batch 0 avg loss: %.4f\n", avg_loss);
    printf("   Batch 1 avg loss: %.4f\n", avg_loss2);
    
    if (fabsf(avg_loss - avg_loss2) < 1e-5) {
        printf("   WARNING: Losses are identical - network might not be using input!\n");
    } else {
        printf("   ✓ Different inputs produce different outputs\n");
    }
    
    printf("\n✓ Forward pass test complete!\n");
    
    // Cleanup
    for (auto layer : network) delete layer;
    delete output_layer;
    cudaFree(d_batch_labels);
}

int main() {
    // test_gemm();
    // test_relu();
    // test_output_layer();
    // test_dense_layer();
    test_forward_pass_with_mnist();
    // train_neural_network();
}