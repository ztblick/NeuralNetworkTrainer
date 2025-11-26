#ifndef ACTIVATIONS_CUH
#define ACTIVATIONS_CUH

#include "Matrix.h"

// Wrapper function: applies ReLU activation kernel to the data for the given layer.
void relu(
    const Matrix& input
);


void test_relu();

#endif // ACTIVATIONS_CUH