// config.h
#pragma once
#include "debug.h"

#define DEBUG           0

#if DEBUG
#define NUM_EPOCHS      1
#define BATCH_SIZE      64
#else
#define NUM_EPOCHS      100
#define BATCH_SIZE      64
#endif


#define LEARNING_RATE   0.05

#define NUM_CLASSES     10

#define DEFAULT_THREADS_PER_BLOCK   256

#define PRINT_FREQUENCY 100

