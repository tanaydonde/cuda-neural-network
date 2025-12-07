#include "GpuNetwork.h"

GpuNetwork::GpuNetwork(int input_dim, int hidden_dim, int output_dim):
                        input_dim_(input_dim),
                        hidden_dim_(hidden_dim),
                        output_dim_(output_dim) {
    std::vector<float> W1(input_dim * hidden_dim);
    std::vector<float> W2(hidden_dim * output_dim);

    for(int i = 0; i < input_dim * hidden_dim; ++i) {
        W1[i] = 0.001f * (i+1) * (i % 2 == 0 ? -1 : 1);
    }

    for(int i = 0; i < hidden_dim * output_dim; ++i) {
        W2[i] = 0.002f * (i+1) * (i % 3 == 0 ? -1 : 1);
    }

    cudaMalloc(&d_W1_, input_dim * hidden_dim * sizeof(float));
    cudaMalloc(&d_W2_, hidden_dim * output_dim * sizeof(float));

    cudaMemcpy(d_W1_, W1.data(), input_dim * hidden_dim  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2_, W2.data(), hidden_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice);
}

GpuNetwork::~GpuNetwork(){
    cudaFree(d_W1_);
    cudaFree(d_W2_);
}

std::vector<float> GpuNetwork::forward(const std::vector<float>& input, int M) {
    int input_size = input_dim_*M;
    int hidden_size = hidden_dim_*M;
    int output_size = output_dim_*M;

    std::vector<float> output(output_size);
    
    float* d_input = nullptr;
    float* d_hidden = nullptr;
    float* d_output = nullptr;
    
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_hidden, hidden_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);

    //layer 1
    matmul(d_input, d_W1_, d_hidden, M, hidden_dim_, input_dim_);
    relu(d_hidden, hidden_size);

    //layer 2
    matmul(d_hidden, d_W2_, d_output, M, output_dim_, hidden_dim_);

    cudaDeviceSynchronize();
    
    cudaMemcpy(output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    
    return output;
}

void GpuNetwork::matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim2D(16, 16);
    dim3 gridDim2D(
        (N + blockDim2D.x - 1)/blockDim2D.x, // gets ceil(N/blockDim2D.x)
        (M + blockDim2D.y - 1)/blockDim2D.y //gets ceil(M/blockDim2D.y)
    );
    matmul_kernel<<<gridDim2D, blockDim2D>>>(A, B, C, M, N, K);
}

void GpuNetwork::relu(float* x, int n) {
    int blockSize1D = 256;
    int gridSize1D = (n + blockSize1D-1)/blockSize1D; //gets ceil(n/blockSize1D)

    relu_kernel<<<gridSize1D, blockSize1D>>>(x, n);
}