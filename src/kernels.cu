#include "kernels.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < M && col < N) {
        float sum = 0.0f;
        for(int i = 0; i < K; ++i) {
              sum += A[row*K + i] * B[i*N + col];
        }
        C[row*N + col] = sum;
    }
}

__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    
    float sum = 0.0f;
    
    int num_tiles = (K + (TILE-1))/TILE;
    
    for(int t = 0; t < num_tiles; ++t) {
        int aRow = row;
        int aCol = t*TILE + threadIdx.x;
        
        int bRow = t*TILE + threadIdx.y;
        int bCol = col;
        
        if(aRow < M && aCol < K) {
            As[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if(bRow < K && bCol < N) {
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for(int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();

    }
    if(row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void relu_kernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

void warmup_cuda() {
    cudaFree(0);
}