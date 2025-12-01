#include "config.h"
#include <cstdint>

struct MNISTDataset {
    float* h_images;      // Host: all images [num_samples, 784]
    uint8_t* h_labels;    // Host: all labels [num_samples]
    size_t num_samples;
    
    MNISTDataset(const char* image_file, const char* label_file);
    ~MNISTDataset();
    
    void get_batch(int batch_idx, size_t batch_size, 
                   float* d_images, int* d_labels);
};