#include "mnistLoader.h"
#include <cuda_runtime.h>

uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0xff) |
           ((val >> 8) & 0xff00) |
           ((val << 8) & 0xff0000) |
           ((val << 24) & 0xff000000);
}

MNISTDataset::MNISTDataset(const char* image_file, const char* label_file) {
    // Read images
    FILE* img_fp = fopen(image_file, "rb");
    if (!img_fp) {
        fprintf(stderr, "Cannot open %s\n", image_file);
        exit(1);
    }
    
    uint32_t magic, num_images, rows, cols;
    fread(&magic, 4, 1, img_fp);
    fread(&num_images, 4, 1, img_fp);
    fread(&rows, 4, 1, img_fp);
    fread(&cols, 4, 1, img_fp);
    
    // Swap from big-endian
    magic = swap_endian(magic);
    num_images = swap_endian(num_images);
    rows = swap_endian(rows);
    cols = swap_endian(cols);
    
    printf("Loading MNIST: %d images of %dx%d\n", num_images, rows, cols);
    
    num_samples = num_images;
    size_t image_size = rows * cols;  // 784
    
    // Allocate and read image data
    uint8_t* raw_images = new uint8_t[num_samples * image_size];
    fread(raw_images, 1, num_samples * image_size, img_fp);
    fclose(img_fp);
    
    // Convert to float and normalize to [0, 1]
    h_images = new float[num_samples * image_size];
    for (size_t i = 0; i < num_samples * image_size; i++) {
        h_images[i] = raw_images[i] / 255.0f;
    }
    delete[] raw_images;
    
    // Read labels
    FILE* lbl_fp = fopen(label_file, "rb");
    if (!lbl_fp) {
        fprintf(stderr, "Cannot open %s\n", label_file);
        exit(1);
    }
    
    uint32_t num_labels;
    fread(&magic, 4, 1, lbl_fp);
    fread(&num_labels, 4, 1, lbl_fp);
    
    magic = swap_endian(magic);
    num_labels = swap_endian(num_labels);
    
    h_labels = new uint8_t[num_labels];
    fread(h_labels, 1, num_labels, lbl_fp);
    fclose(lbl_fp);
    
    printf("Loaded %zu samples\n", num_samples);
}

MNISTDataset::~MNISTDataset() {
    delete[] h_images;
    delete[] h_labels;
}

void MNISTDataset::get_batch(int batch_idx, size_t batch_size,
                              float* d_images, int* d_labels) {
    size_t offset = batch_idx * batch_size;
    
    // Copy images: [batch_size, 784]
    cudaMemcpy(d_images, 
               h_images + offset * 784,
               batch_size * 784 * sizeof(float),
               cudaMemcpyHostToDevice);
    
    // Copy labels and convert to int
    int h_batch_labels[batch_size];
    for (size_t i = 0; i < batch_size; i++) {
        h_batch_labels[i] = h_labels[offset + i];
    }
    
    cudaMemcpy(d_labels,
               h_batch_labels,
               batch_size * sizeof(int),
               cudaMemcpyHostToDevice);
}