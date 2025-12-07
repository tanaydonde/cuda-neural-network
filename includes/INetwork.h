#pragma once
#include <vector>

class INetwork {
    public:
        virtual ~INetwork() = default;
        virtual std::vector<float> forward(const std::vector<float>& input, int M) = 0;

    protected:
        virtual void matmul(const float* A, const float* B, float* C, int M, int N, int K) = 0;
        virtual void relu(float* x, int n) = 0;
};