# Quick Start Guide

Get the UCT Device API Hello World example running in 5 minutes!

## Prerequisites

```bash
# Install build tools
pip3 install meson ninja

# Verify CUDA is installed
nvcc --version

# Verify UCX is installed (with CUDA support)
ucx_info -v
```

## Option 1: Using the Build Script (Easiest)

```bash
# Navigate to example directory
cd examples/uct_device_hello_world

# Build
./build.sh

# Run
./builddir/uct_device_hello
```

### Build Script Options

```bash
# Clean build
./build.sh --clean

# Debug build
./build.sh --debug

# Specify CUDA architecture
./build.sh --cuda-arch sm_80

# Custom UCX installation
./build.sh --ucx-prefix /opt/ucx

# All options
./build.sh --clean --debug --cuda-arch sm_86 --ucx-prefix /usr/local
```

## Option 2: Manual Meson Build

```bash
# Configure
meson setup builddir

# Compile
meson compile -C builddir

# Run
./builddir/uct_device_hello
```

## Option 3: With Custom UCX Installation

```bash
# Set PKG_CONFIG_PATH to find UCX
export PKG_CONFIG_PATH=/path/to/ucx/lib/pkgconfig:$PKG_CONFIG_PATH

# Build
meson setup builddir
meson compile -C builddir

# Run
./builddir/uct_device_hello
```

## Expected Output

```
=== UCT Device API Hello World ===

Found 1 CUDA device(s)
Using device: NVIDIA A100-SXM4-40GB

Allocated device buffers:
  Source:      0x7f1234567000
  Destination: 0x7f1234568000

UCT Device API Concepts:
------------------------
1. Device API allows GPU kernels to initiate communication
2. Operations are called from __device__ code (CUDA kernels)
3. Supported operations:
   - uct_device_ep_put_single()      : Single PUT
   ...

Launching simple device kernel...
Device Kernel: Hello from GPU thread 0!
Device Kernel: Performing device-side PUT operation...
Device Kernel: Device API not supported on this transport

Kernel execution completed

=== UCT Device API Hello World Complete ===
```

## Troubleshooting

### "meson: command not found"
```bash
pip3 install --user meson ninja
export PATH="$HOME/.local/bin:$PATH"
```

### "UCX not found"
```bash
# Find UCX installation
find /usr -name "ucx.pc" 2>/dev/null

# Set PKG_CONFIG_PATH
export PKG_CONFIG_PATH=/usr/lib/pkgconfig:$PKG_CONFIG_PATH

# Or install UCX
# See: https://github.com/openucx/ucx
```

### "CUDA not found"
```bash
# Check CUDA installation
ls /usr/local/cuda

# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Build fails with architecture error
```bash
# Check your GPU architecture
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Build for your specific architecture
# Example for compute capability 8.0 (A100)
./build.sh --cuda-arch sm_80
```

## Next Steps

1. Read the full [README.md](README.md) for detailed API documentation
2. Study the example code in `uct_device_hello.cu`
3. Explore advanced tests in `test/gtest/uct/cuda/test_cuda_ipc_device.cc`
4. Integrate Device API into your own application

## Common Use Cases

### Intra-node GPU Communication
```bash
# Build UCX with CUDA IPC
./configure --with-cuda
make install

# Use cuda_ipc transport for same-node GPU-to-GPU
```

### Inter-node GPU Direct RDMA
```bash
# Build UCX with DOCA GPUNetIO
./configure --with-cuda --with-doca-gpunetio=/opt/mellanox/doca
make install

# Use rc_gda transport for cross-node GPU RDMA
```

## Performance Tips

- Use warp/block cooperation for better throughput
- Batch operations with `put_multi` when possible  
- Balance GPU compute with communication overlap
- Profile with nsight-systems to understand performance

## Getting Help

- UCX Documentation: https://openucx.readthedocs.io
- UCX GitHub: https://github.com/openucx/ucx
- UCX Mailing List: https://elist.ornl.gov/mailman/listinfo/ucx-group
- CUDA Documentation: https://docs.nvidia.com/cuda/

