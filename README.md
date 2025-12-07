# CUDA Neural Network

Simple CPU, GPU, and GPU-tiled implementations of a neural-network forward pass.

## **Build**

Requires the NVIDIA CUDA Toolkit.

### **Default build**
```bash
make
```

### **Build for a different GPU architecture**
```bash
make ARCH=sm_86     # RTX 30-series
make ARCH=sm_89     # RTX 40-series
make ARCH=sm_80     # A100
```

(Default ARCH is `sm_75`.)

## **Run**

```bash
./cuda_nn [options]
```

Examples:

```bash
./cuda_nn               
./cuda_nn --mode cpu
./cuda_nn --mode gpu gpu_tiled
./cuda_nn --mode all --M 20000
./cuda_nn --help     # shows all available flags
```

## **Kaggle**

Kaggle inputs are read-only. Copy the project before building:

```bash
cp -r /kaggle/input/<project_folder> /kaggle/working/
cd /kaggle/working/<project_folder>
make
./cuda_nn
```