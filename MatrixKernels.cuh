#pragma once
#include "config.h"
#include "Matrix.h"
#define TILE_SIZE   16
#define THREAD_X    32
#define THREAD_Y    8

void tiledMatrixMultiplyGPU(const Matrix& A, const Matrix& B, Matrix& C);

void matrixMultiplyGPU(const Matrix& A, const Matrix& B, Matrix& C);