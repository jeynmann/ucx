# UCT Device API Hello World Example

This example demonstrates the UCT Device API, which allows GPU kernels to directly initiate communication operations from device code.

## Overview

The UCT Device API enables GPU-initiated communication without explicit CPU involvement:
- **Device-side operations**: Communication calls made directly from CUDA kernels
- **Asynchronous progress**: Operations can progress independently on the GPU
- **Multiple cooperation levels**: Support for thread, warp, block, and grid-level operations

## Supported Transports

The Device API works with:
1. **CUDA IPC** (`cuda_ipc`): For intra-node GPU-to-GPU communication
2. **rc_gda** (GPU Direct Async): For inter-node RDMA with Mellanox adapters (requires DOCA GPUNetIO)

## Building with Meson

### Prerequisites

```bash
# Install Meson and Ninja
pip3 install meson ninja

# Ensure CUDA toolkit is installed
# Ensure UCX is installed with CUDA support
```

### Build Steps

```bash
cd examples/uct_device_hello_world

# Configure the build
meson setup builddir

# Compile
meson compile -C builddir

# Run the example
./builddir/uct_device_hello
```

### Build Options

```bash
# Specify CUDA architecture
meson setup builddir -Dcuda_arch=sm_80

# Use custom UCX installation
PKG_CONFIG_PATH=/path/to/ucx/lib/pkgconfig meson setup builddir

# Debug build
meson setup builddir --buildtype=debug
```

## Code Structure

```
uct_device_hello.cu
├── main()                    # Host code: Setup and initialization
│   ├── Initialize CUDA
│   ├── Initialize UCT
│   ├── Allocate device memory
│   └── Launch kernel
└── device_hello_kernel()     # Device code: GPU kernel
    └── uct_device_ep_put_single()  # Device API call
```

## Device API Functions

### Main Operations

```cuda
// Single PUT operation
ucs_status_t uct_device_ep_put_single<level>(
    uct_device_ep_h device_ep,
    const uct_device_mem_element_t *mem_elem,
    const void *address,
    uint64_t remote_address,
    size_t length,
    uint64_t flags,
    uct_device_completion_t *comp);

// Batched PUT operations
ucs_status_t uct_device_ep_put_multi<level>(...);

// Atomic increment
ucs_status_t uct_device_ep_atomic_add<level>(...);

// Progress communications
void uct_device_ep_progress<level>(uct_device_ep_h device_ep);

// Check completion status
ucs_status_t uct_device_ep_check_completion<level>(
    uct_device_ep_h device_ep,
    uct_device_completion_t *comp);
```

### Cooperation Levels

```cuda
UCS_DEVICE_LEVEL_THREAD  // Single thread operates independently
UCS_DEVICE_LEVEL_WARP    // Warp (32 threads) cooperates
UCS_DEVICE_LEVEL_BLOCK   // Thread block cooperates
UCS_DEVICE_LEVEL_GRID    // Entire grid cooperates
```

## Integration Steps

To use the Device API in your application:

1. **Build UCX with CUDA support**
   ```bash
   ./configure --with-cuda [--with-doca-gpunetio]
   make install
   ```

2. **Create UCT interface with device capability**
   ```c
   uct_iface_params_t params = {
       .field_mask = UCT_IFACE_PARAM_FIELD_OPEN_MODE,
       .open_mode = UCT_IFACE_OPEN_MODE_DEVICE
   };
   uct_iface_open(..., &params, ...);
   ```

3. **Get device endpoint handle**
   ```c
   uct_device_ep_h device_ep;
   uct_ep_get_device_ep(host_ep, &device_ep);
   ```

4. **Pass device handle to kernel**
   ```cuda
   my_kernel<<<blocks, threads>>>(device_ep, ...);
   ```

5. **Use Device API in kernel**
   ```cuda
   __global__ void my_kernel(uct_device_ep_h device_ep, ...) {
       uct_device_ep_put_single<UCS_DEVICE_LEVEL_THREAD>(...);
   }
   ```

## Performance Considerations

- **Cooperation level**: Higher levels (warp/block) can be more efficient for bulk operations
- **Batching**: Use `put_multi` for better throughput with multiple operations
- **Completion checking**: Balance between polling frequency and GPU occupancy
- **Memory alignment**: Aligned transfers typically perform better

## Related Examples

For more complete examples, see:
- `test/gtest/uct/cuda/test_cuda_ipc_device.cc` - CUDA IPC device tests
- `test/gtest/ucp/test_ucp_device.cc` - UCP device-level tests
- `src/tools/perf/cuda/ucp_cuda_kernel.cu` - Performance testing kernels

## Troubleshooting

### "Device API not supported" error
- Ensure UCX is built with `--with-cuda`
- Check that CUDA IPC or rc_gda transport is available
- Verify `UCT_IFACE_FLAG_DEVICE_EP` capability is set

### Build errors
- Check CUDA toolkit version (>=10.0 required)
- Verify UCX pkg-config files are in `PKG_CONFIG_PATH`
- Ensure compute capability matches your GPU

### Runtime issues
- Set `CUDA_VISIBLE_DEVICES` if multiple GPUs present
- Check UCX configuration with `ucx_info -d`
- Enable debug output: `UCX_LOG_LEVEL=debug`

## References

- [UCT Device API Documentation](../../docs/device_api.md)
- [UCX Documentation](https://openucx.readthedocs.io)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/)

