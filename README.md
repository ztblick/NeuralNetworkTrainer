# NeuralNetworkTrainer: a GPU-Accelerated Application for Neural Network Training

In this project, I will explore the mathematics behind neural networks. I will implement matrix math on my GPU to perform all the calculations necessary to train my own neural network. This will initially be done on a simple dataset — probably the MNIST dataset. My goal is to learn and understand all the internals of the existing Deep Learning libraries like PyTorch and TensorFlow — simply put, I want to understand how they work\!

Once I have a basic model working, I will turn my attention to optimization on my hardware, an NVIDIA RTX 3060 GPU. I intend to optimize the code for the hardware. I have a strong understanding of virtual memory already, so I think the memory management (and the optimizations we can make to speed up the multithreaded training) should be pretty easy to understand and fun to implement\!

**Phase 1: Matrix Operations**
- [x] CPU reference implementation
- [x] Naive GPU kernel
- [x] Tiled matrix multiplication with shared memory
- [x] Performance benchmarking

**Phase 2: Neural Network Operations**
- [ ] ReLU activation
- [ ] Softmax
- [ ] Cross-entropy loss

| Test: 1024x1024 \* 1024x1024 |  |  |  |
| :---- | :---- | :---- | :---- |
| ***Version*** | ***Runtime (ms)*** | ***Speedup from prev.*** | ***Total Speedup*** |
| **CPU only** | **2360.013** | **—** | **—** |
| **Simple GPU kernel** | **3.736**  | **631.72x** | **631.72x** |
| **Tiled GPU kernel** | **2.512**  | **1.27x** | **939.5x** |
|  |  |  |  |