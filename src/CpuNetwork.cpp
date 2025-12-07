#include "CpuNetwork.h"

CpuNetwork::CpuNetwork(int input_dim, int hidden_dim, int output_dim):
                        input_dim_(input_dim),
                        hidden_dim_(hidden_dim),
                        output_dim_(output_dim) {
                        
    W1_ = new float[input_dim * hidden_dim];
    W2_ = new float[hidden_dim * output_dim];

    for(int i = 0; i < input_dim * hidden_dim; ++i) {
        W1_[i] = 0.001f * (i+1) * (i % 2 == 0 ? -1 : 1);
    }

    for(int i = 0; i < hidden_dim * output_dim; ++i) {
        W2_[i] = 0.002f * (i+1) * (i % 3 == 0 ? -1 : 1);
    }
}

CpuNetwork::~CpuNetwork() {
    delete[] W1_;
    delete[] W2_;
}

std::vector<float> CpuNetwork::forward(const std::vector<float>& input, int M) {
    std::vector<float> hidden(M*hidden_dim_);
    std::vector<float> output(M*output_dim_);
    
    //layer 1
    matmul(input.data(), W1_, hidden.data(), M, hidden_dim_, input_dim_);
    relu(hidden.data(), M*hidden_dim_);
    
    //layer 2
    matmul(hidden.data(), W2_, output.data(), M, output_dim_, hidden_dim_);

    return output;
}

void CpuNetwork::matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for(int k = 0; k < K; ++k) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

void CpuNetwork::relu(float* x, int n) {
    for(int i = 0; i < n; ++i) {
        x[i] = std::max(0.0f, x[i]);
    }
}
