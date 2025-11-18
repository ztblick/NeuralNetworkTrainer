NVCC = nvcc
# My machine has an NVIDIA RTX 3060 GPU, so I set the architecture to sm_86
GPU_ARCH = 86
PERFORMANCE_FLAGS = -O3 -DNDEBUG -arch=sm_$(GPU_ARCH) -std=c++11
DEBUG_FLAGS = -g -G -O0 -DDEBUG -arch=sm_$(GPU_ARCH) -std=c++11

TARGET = matrix_test

SRCS = main.cu matrix.cu
HEADERS = matrix.cuh debug.cuh timer.cuh

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(PERFORMANCE_FLAGS) $(SRCS) -o $(TARGET)

debug: $(SRCS) $(HEADERS)
	$(NVCC) $(DEBUG_FLAGS) $(SRCS) -o $(TARGET)_debug

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all debug clean run