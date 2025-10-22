# GDAKI Stubs Headers

This directory contains stub implementations of the GDAKI (GPU Direct Async Kernel Interface) headers.

## Purpose

GDAKI is provided by the DOCA GPUNetIO package from NVIDIA. These stub headers allow UCX device API to be built and installed without requiring the full DOCA GPUNetIO package.

**Installation Condition:** Stub headers are **always installed** since UCX Device API headers (`uct_device_impl.h`) are always installed and include `<uct/ib/mlx5/gdaki/gdaki.cuh>`. This ensures the Device API headers can be used even when UCX is built without CUDA/NVCC.

## Installation Behavior

### When UCX is Built WITHOUT DOCA GPUNetIO

1. **Build time**: UCX build uses stub headers from `${top_srcdir}/stubs`
2. **Install time**: Stub headers are installed to `${prefix}/include/stubs/uct/ib/mlx5/gdaki/`
3. **User's application**:
   ```bash
   # pkg-config ucx --cflags returns:
   -I/usr/include -I/usr/include/stubs

   # Application finds: /usr/include/stubs/uct/ib/mlx5/gdaki/gdaki.cuh (stub)
   # Note: /usr/include/stubs is lower priority
   ```

### When GDAKI Package is Installed

1. **GDAKI installs to**: `${prefix}/include/uct/ib/mlx5/gdaki/` (separate package)
2. **UCX stub is at**: `${prefix}/include/stubs/uct/ib/mlx5/gdaki/` (UCX package)
3. **User's application**:
   ```bash
   # pkg-config ucx --cflags returns:
   -I/usr/include -I/usr/include/stubs

   # Application finds: /usr/include/uct/ib/mlx5/gdaki/gdaki.cuh (real GDAKI)
   # Stub at /usr/include/stubs/uct/ib/mlx5/gdaki/gdaki.cuh exists but not used
   ```

## Header Priority

The include path priority ensures the correct headers are used:

```
-I${includedir}        <-- Higher priority (real GDAKI when installed)
-I${includedir}/stubs  <-- Lower priority (stub headers, always present)
```

When searching for `#include <uct/ib/mlx5/gdaki/gdaki.cuh>`:
1. First searches: `${includedir}/uct/ib/mlx5/gdaki/gdaki.cuh` (real GDAKI)
2. If not found: `${includedir}/stubs/uct/ib/mlx5/gdaki/gdaki.cuh` (stub)

This is controlled in `ucx.pc.in`:
```
Cflags: -I${includedir} @STUB_CFLAGS@
```

## Implementation

### Stub Functions

The stub implementations return `UCS_ERR_UNSUPPORTED` for all operations:

```cuda
template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_rc_mlx5_gda_ep_put_single(...) {
    return UCS_ERR_UNSUPPORTED;
}
```

This allows:
- ✅ Code compiles successfully
- ✅ Graceful runtime failure if GDA operations are attempted
- ✅ Applications can use other UCX device API transports (e.g., CUDA IPC)

### Type Definitions

`gdaki_dev.h` contains type definitions that must match the real GDAKI:
- `uct_rc_gdaki_dev_ep_t`
- `uct_rc_gdaki_device_mem_element_t`
- `uct_rc_gda_completion_t`

These types are used by UCX device API layer regardless of GDAKI availability.

## Build System Integration

### Autotools Files Modified

1. **`stubs/Makefile.am`**: Installs stub headers to `${includedir}/stubs/` (always)
2. **`Makefile.am`**: Adds `stubs` to SUBDIRS
3. **`configure.ac`**: Adds `stubs/Makefile` to AC_CONFIG_FILES
4. **`config/m4/cuda.m4`**: Sets STUB_CFLAGS for pkg-config (always)
5. **`ucx.pc.in`**: Exports `-I${includedir} @STUB_CFLAGS@`

### Install Paths

| Package | Install Path | Priority |
|---------|-------------|----------|
| UCX (stubs) | `${includedir}/stubs/uct/ib/mlx5/gdaki/gdaki.cuh` | Lower (always installed) |
| GDAKI (real) | `${includedir}/uct/ib/mlx5/gdaki/gdaki.cuh` | Higher (when installed) |

### Include Search Order

When compiling user applications:
```
-I${includedir}        # Real GDAKI checked first
-I${includedir}/stubs  # Stub GDAKI checked if real not found
```

## Usage Example

### Application Without GDAKI

```cpp
#include <uct/api/device/uct_device_impl.h>

__global__ void my_kernel(uct_device_ep_h ep, ...) {
    // Will use stub, returns UCS_ERR_UNSUPPORTED
    auto status = uct_device_ep_put_single<UCS_DEVICE_LEVEL_THREAD>(...);
    if (status == UCS_ERR_UNSUPPORTED) {
        // Handle gracefully - use alternative transport
    }
}
```

### Application With GDAKI

```cpp
#include <uct/api/device/uct_device_impl.h>

__global__ void my_kernel(uct_device_ep_h ep, ...) {
    // Uses real GDAKI implementation for rc_gda transport
    auto status = uct_device_ep_put_single<UCS_DEVICE_LEVEL_THREAD>(...);
    if (status == UCS_INPROGRESS) {
        // Operation posted successfully
    }
}
```

## Benefits

✅ **No external dependency**: UCX can be built and installed without DOCA GPUNetIO
✅ **Seamless upgrade**: Installing DOCA GPUNetIO automatically enables full functionality
✅ **ABI compatibility**: Stub types match real GDAKI types
✅ **Graceful degradation**: Runtime errors instead of compile errors
✅ **Transport flexibility**: Other device transports (CUDA IPC) work regardless
✅ **Always available**: Stubs match Device API header availability (always installed)

## Files

- `gdaki.cuh`: Stub device-side function templates
- `gdaki_dev.h`: Type definitions (matches real GDAKI)
- `README.md`: This documentation file

## Related Documentation

- [UCX Device API Documentation](../../../../docs/device_api.md)
- [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/)
- [UCX GitHub](https://github.com/openucx/ucx)
