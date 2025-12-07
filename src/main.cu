#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cassert>
#include <cctype>

#include "CpuNetwork.h"
#include "GpuNetwork.h"
#include "GpuTiledNetwork.h"

enum class Mode { CPU, GPU, GPU_TILED };

void print_help(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --mode cpu|gpu|gpu_tiled|all  (you can specify multiple)\n"
              << "     e.g., --mode cpu gpu\n"
              << "  --M <int>\n"
              << "  --input_dim <int>\n"
              << "  --hidden_dim <int>\n"
              << "  --output_dim <int>\n"
              << "  --help\n";
}

std::string to_lower(const std::string& m){
    std::string r;
    for(unsigned char c : m) {
        r += static_cast<char>(std::tolower(c));
    }
    return r;
}

Mode parse_mode(const std::string& m) {
    if (m == "cpu") return Mode::CPU;
    if (m == "gpu") return Mode::GPU;
    if (m == "gpu_tiled") return Mode::GPU_TILED;
    throw std::runtime_error("Unknown mode: " + m);
}

bool contains(std::vector<Mode>& modes, Mode m) {
    for(Mode val : modes) {
        if(val == m){
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv){
    std::vector<Mode> modes;
    int M = 1024;
    int input_dim = 784;
    int hidden_dim = 128;
    int output_dim = 10;

    for (int i = 1; i < argc; i++) {
        std::string arg = to_lower(argv[i]);

        if (arg == "--help") {
            print_help(argv[0]);
            return 0;
        }

        else if (arg == "--mode") {
            while (i + 1 < argc && argv[i+1][0] != '-') {
                std::string m = to_lower(argv[++i]);
                if (m == "all") {
                    modes = { Mode::CPU, Mode::GPU, Mode::GPU_TILED };
                } else {
                    modes.push_back(parse_mode(m));
                }
            }
        }

        else if (arg == "--m" && i + 1 < argc) {
            M = std::stoi(argv[++i]);
        }

        else if (arg == "--input_dim" && i + 1 < argc) {
            input_dim = std::stoi(argv[++i]);
        }

        else if (arg == "--hidden_dim" && i + 1 < argc) {
            hidden_dim = std::stoi(argv[++i]);
        }

        else if (arg == "--output_dim" && i + 1 < argc) {
            output_dim = std::stoi(argv[++i]);
        }

        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_help(argv[0]);
            return 1;
        }
    }

    if (modes.empty()) {
        modes.push_back(Mode::GPU);
        modes.push_back(Mode::GPU_TILED);
    }

    std::cout << "Running with:\n";
    std::cout << "M=" << M << "\n";
    std::cout << "input_dim=" << input_dim << "\n";
    std::cout << "hidden_dim=" << hidden_dim << "\n";
    std::cout << "output_dim=" << output_dim << "\n\n";

    warmup_cuda();
    
    std::vector<float> input(input_dim * M);
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < input_dim; ++j) {
            input[i*input_dim + j] = (i*input_dim + j + 1)/(1000000.0f);
        }
    }

    CpuNetwork cpu_net(input_dim, hidden_dim, output_dim);
    GpuNetwork gpu_net(input_dim, hidden_dim, output_dim);
    GpuTiledNetwork gpu_tiled(input_dim, hidden_dim, output_dim);

    const int NUM_RUNS = 50;

    // =========================================CPU TESTING=========================================

    if(contains(modes, Mode::CPU)){
        float time_sum_cpu = 0;
        std::vector<float> last_output_cpu;

        for(int i = 0; i < NUM_RUNS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            last_output_cpu = cpu_net.forward(input, M);
            auto end  = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(end - start).count();
            time_sum_cpu += ms;
        }
        std::cout << "CPU forward pass output (first 8):\n";
        for(int i = 0; i < 8; ++i) {
            std::cout << last_output_cpu[i] << " ";
        }
        std::cout << "\nAverage Runtime Over " << NUM_RUNS << " Runs: " << time_sum_cpu/NUM_RUNS << " ms\n\n";
    }

    // =========================================GPU TESTING=========================================

    if(contains(modes, Mode::GPU)){
        float time_sum_gpu = 0;
        std::vector<float> last_output_gpu;

        for(int i = 0; i < NUM_RUNS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            last_output_gpu = gpu_net.forward(input, M);
            auto end  = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(end - start).count();
            time_sum_gpu += ms;
        }
        std::cout << "GPU forward pass output (first 8):\n";
        for(int i = 0; i < 8; ++i) {
            std::cout << last_output_gpu[i] << " ";
        }
        std::cout << "\nAverage Runtime Over " << NUM_RUNS << " Runs: " << time_sum_gpu/NUM_RUNS << " ms\n\n";
    }

    // =========================================GPU TILED TESTING=========================================

    if(contains(modes, Mode::GPU_TILED)){
        float time_sum_tiled = 0;
        std::vector<float> last_output_tiled;

        for(int i = 0; i < NUM_RUNS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            last_output_tiled = gpu_tiled.forward(input, M);
            auto end  = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(end - start).count();
            time_sum_tiled += ms;
        }
        std::cout << "GPU tiled forward pass output (first 8):\n";
        for(int i = 0; i < 8; ++i) {
            std::cout << last_output_tiled[i] << " ";
        }
        std::cout << "\nAverage Runtime Over " << NUM_RUNS << " Runs: " << time_sum_tiled/NUM_RUNS << " ms\n\n";
    }
    
    return 0;
}