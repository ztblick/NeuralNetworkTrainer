// config.h
#pragma once
#include "debug.h"

#define DEBUG           1

#if DEBUG
#define NUM_EPOCHS      5
#define BATCH_SIZE      64
#elif
#define NUM_EPOCHS      10
#define BATCH_SIZE      64
#endif


#define LEARNING_RATE   0.01

#define NUM_CLASSES     10

#define DEFAULT_THREADS_PER_BLOCK   256

#define PRINT_FREQUENCY 100

