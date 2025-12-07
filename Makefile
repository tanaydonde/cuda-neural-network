ARCH ?= sm_75

NVCC       := nvcc
CXXFLAGS   := -std=c++17
INCLUDES   := -Iincludes

NVCCFLAGS  := $(CXXFLAGS) $(INCLUDES) -arch=$(ARCH)

CPP_SRCS := src/CpuNetwork.cpp
CU_SRCS  := src/main.cu src/GpuNetwork.cu src/GpuTiledNetwork.cu src/kernels.cu

OBJS := $(CPP_SRCS:.cpp=.o) $(CU_SRCS:.cu=.o)

TARGET := cuda_nn

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# compile c++ sources with nvcc (it’ll call the host compiler)
src/%.o: src/%.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# compile cuda sources
src/%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean