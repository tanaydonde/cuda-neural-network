#pragma once
#include "INetwork.h"
#include "GpuNetwork.h"

class GpuTiledNetwork : public GpuNetwork {
    public:
        using GpuNetwork::GpuNetwork;
        ~GpuTiledNetwork() override = default;
    
    protected:
        void matmul(const float* A, const float* B, float* C, int M, int N, int K) override;
};