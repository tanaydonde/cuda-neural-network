#pragma once
#include "INetwork.h"
#include <algorithm>

class CpuNetwork : public INetwork {
    public:
        CpuNetwork(int input_dim, int hidden_dim, int output_dim);
        ~CpuNetwork() override;
        std::vector<float> forward(const std::vector<float>& input, int M) override;

    protected:
        void matmul(const float* A, const float* B, float* C, int M, int N, int K) override;
        void relu(float* x, int n) override;
    
    private:
        int input_dim_, hidden_dim_, output_dim_;
        float* W1_ = nullptr;
        float* W2_ = nullptr;
};