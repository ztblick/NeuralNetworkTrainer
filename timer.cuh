#ifndef TIMER_CUH
#define TIMER_CUH

#include <cuda_runtime.h>
#include <stdio.h>

// CPU timer
class CPUTimer {
private:
    struct timespec start_time, end_time;
    
public:
    void start() {
        clock_gettime(CLOCK_MONOTONIC, &start_time);
    }
    
    float stop() {
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        float elapsed = (end_time.tv_sec - start_time.tv_sec) * 1000.0f;
        elapsed += (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0f;
        return elapsed; // milliseconds
    }
};

// GPU timer (more accurate for GPU operations)
class GPUTimer {
private:
    cudaEvent_t start_event, stop_event;
    
public:
    GPUTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    
    ~GPUTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        cudaEventRecord(start_event, 0);
    }
    
    float stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        float elapsed;
        cudaEventElapsedTime(&elapsed, start_event, stop_event);
        return elapsed; // milliseconds
    }
};

#endif // TIMER_CUH