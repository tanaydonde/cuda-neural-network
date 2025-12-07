#pragma once
#include "INetwork.h"
#include "kernels.cuh"

class GpuNetwork : public INetwork {
    public:
        GpuNetwork(int input_dim, int hidden_dim, int output_dim);
        ~GpuNetwork() override;
        std::vector<float> forward(const std::vector<float>& input, int M) override;
    
    protected:
        int input_dim_, hidden_dim_, output_dim_;
        float* d_W1_ = nullptr;
        float* d_W2_ = nullptr;

        void matmul(const float* A, const float* B, float* C, int M, int N, int K) override;
        void relu(float* x, int n) override;
};