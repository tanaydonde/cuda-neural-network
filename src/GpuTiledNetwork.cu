#include "GpuTiledNetwork.h"

void GpuTiledNetwork::matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim2D(TILE, TILE);
    dim3 gridDim2D(
        (N + blockDim2D.x - 1)/blockDim2D.x, // gets ceil(N/blockDim2D.x)
        (M + blockDim2D.y - 1)/blockDim2D.y //gets ceil(M/blockDim2D.y)
    );
    matmul_tiled<<<gridDim2D, blockDim2D>>>(A, B, C, M, N, K);
}