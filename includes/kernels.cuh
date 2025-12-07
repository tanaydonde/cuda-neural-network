#pragma once

#define TILE 16

void warmup_cuda();

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K);

__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K);

__global__ void relu_kernel(float* x, int n);

