# GDAKI Stub Headers Design

## Overview

UCX Device API requires GDAKI (GPU Direct Async Kernel Interface) headers to compile device code. However, GDAKI is a separate package (DOCA GPUNetIO) that may not be available on all systems. This design allows UCX to be built and installed without GDAKI, using stub headers as a fallback.

## Installation Paths

### UCX Package (Always Installed)
```
${prefix}/include/stub/uct/ib/mlx5/gdaki/
├── gdaki.cuh      # Stub implementations
└── gdaki_dev.h    # Type definitions
```

### GDAKI Package (Optional, Separate Package)
```
${prefix}/include/uct/ib/mlx5/gdaki/
├── gdaki.cuh      # Real implementations
└── gdaki_dev.h    # Type definitions
```

## Header Priority Mechanism

### pkg-config Output
```bash
$ pkg-config ucx --cflags
-I${prefix}/include -I${prefix}/include/stub
```

### Include Search Order

When compiling: `#include <uct/ib/mlx5/gdaki/gdaki.cuh>`

1. **First search**: `${prefix}/include/uct/ib/mlx5/gdaki/gdaki.cuh` (real GDAKI)
2. **If not found**: `${prefix}/include/stub/uct/ib/mlx5/gdaki/gdaki.cuh` (stub)

## Scenarios

### Scenario 1: UCX built with CUDA/NVCC, without GDAKI

```
System State:
- UCX built with --with-cuda
- NVCC available
- GDAKI NOT installed

Install Paths:
${prefix}/include/stub/uct/ib/mlx5/gdaki/gdaki.cuh  ← stub (from UCX)
${prefix}/include/uct/ib/mlx5/gdaki/gdaki.cuh       ← not present

pkg-config ucx --cflags:
-I${prefix}/include -I${prefix}/include/stub

Compiler Behavior:
-I${prefix}/include        ← searches here first, not found
-I${prefix}/include/stub   ← searches here, finds stub
Result: Uses stub headers
```

### Scenario 2: UCX + GDAKI both installed

```
System State:
- UCX built with --with-cuda
- NVCC available
- GDAKI installed (separate package)

Install Paths:
${prefix}/include/uct/ib/mlx5/gdaki/gdaki.cuh       ← real (from GDAKI)
${prefix}/include/stub/uct/ib/mlx5/gdaki/gdaki.cuh  ← stub (from UCX)

pkg-config ucx --cflags:
-I${prefix}/include -I${prefix}/include/stub

Compiler Behavior:
-I${prefix}/include        ← searches here first, finds real GDAKI
-I${prefix}/include/stub   ← not searched (already found)
Result: Uses real GDAKI headers
```

### Scenario 3: UCX built without CUDA/NVCC

```
System State:
- UCX built without --with-cuda or NVCC not available
- Device code cannot be compiled (no NVCC)
- But Device API headers still installed (uct_device_impl.h)

Install Paths:
${prefix}/include/uct/api/device/uct_device_impl.h  ← Device API headers (always installed)
${prefix}/include/stub/uct/ib/mlx5/gdaki/gdaki.cuh  ← stub headers (always installed)

pkg-config ucx --cflags:
-I${prefix}/include -I${prefix}/include/stubs

Result: Device API headers available, stub provides UCS_ERR_UNSUPPORTED
        Applications can compile but GDAKI operations fail at runtime
```

## Implementation Details

### Stub Functions

```cuda
// All stub functions return UCS_ERR_UNSUPPORTED
template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_rc_mlx5_gda_ep_put_single(...) {
    return UCS_ERR_UNSUPPORTED;
}
```

### Build System

#### stub/Makefile.am
```makefile
# Always install stubs since Device API headers are always installed
# and they include <uct/ib/mlx5/gdaki/gdaki.cuh>
stubincludedir = $(includedir)/stubs
nobase_stubinclude_HEADERS = \
    uct/ib/mlx5/gdaki/gdaki.cuh \
    uct/ib/mlx5/gdaki/gdaki_dev.h
```

#### ucx.pc.in
```
# @STUB_CFLAGS@ is set by configure based on NVCC availability
Cflags: -I${includedir} @STUB_CFLAGS@
```

#### config/m4/cuda.m4
```m4
# Always set stub include path since Device API headers are always installed
STUB_CFLAGS="-I\${includedir}/stubs"
AC_SUBST([STUB_CFLAGS])
```

## Key Design Principles

1. **Separation of Concerns**: GDAKI is a separate package, not modified by UCX
2. **Priority-based Resolution**: Real GDAKI always takes precedence when present
3. **Graceful Degradation**: Stub returns UCS_ERR_UNSUPPORTED instead of compile error
4. **No Ambiguity**: Include path order ensures deterministic header selection
5. **Zero Configuration**: Works automatically when GDAKI is installed
6. **Always Available**: Stubs always installed since Device API headers are always installed

## Benefits

✅ UCX builds without GDAKI dependency  
✅ GDAKI can be installed/upgraded independently  
✅ No conflicts between real and stub headers  
✅ pkg-config automatically provides correct paths  
✅ Applications compile regardless of GDAKI availability  
✅ Runtime detection of GDAKI support  

## Usage Example

```cuda
#include <uct/api/device/uct_device_impl.h>

__global__ void my_kernel(uct_device_ep_h ep, ...) {
    auto status = uct_device_ep_put_single<UCS_DEVICE_LEVEL_THREAD>(...);
    
    switch (status) {
        case UCS_INPROGRESS:
            // Real GDAKI: operation posted successfully
            break;
        case UCS_ERR_UNSUPPORTED:
            // Stub: GDAKI not available, use alternative transport
            break;
        default:
            // Handle other errors
            break;
    }
}
```

## Testing

```bash
# Test 1: UCX with CUDA, without GDAKI
./configure --with-cuda --prefix=/tmp/test1
make install
ls /tmp/test1/include/stubs/uct/ib/mlx5/gdaki/  # Should have stub
ls /tmp/test1/include/uct/ib/mlx5/gdaki/        # Should NOT exist
pkg-config --cflags ucx                          # Should show both paths

# Test 2: UCX without CUDA
./configure --without-cuda --prefix=/tmp/test2
make install
ls /tmp/test2/include/stubs/uct/ib/mlx5/gdaki/  # Should still have stub
ls /tmp/test2/include/uct/api/device/           # Device API headers present
pkg-config --cflags ucx                          # Should show both paths

# Test 3: After installing GDAKI
# (Install GDAKI package to /tmp/test1)
ls /tmp/test1/include/uct/ib/mlx5/gdaki/        # Should have real GDAKI
ls /tmp/test1/include/stubs/uct/ib/mlx5/gdaki/  # Stub still present
# Compiler will find real GDAKI first
```

## Files Modified

1. **`stub/Makefile.am`** - Always install stub to ${includedir}/stubs (no condition)
2. **`Makefile.am`** - Add stub to SUBDIRS
3. **`configure.ac`** - Add stub/Makefile to AC_CONFIG_FILES  
4. **`config/m4/cuda.m4`** - Always set STUB_CFLAGS (stubs always installed)
5. **`ucx.pc.in`** - Export @STUB_CFLAGS@ (always includes stub path)
6. **`stub/uct/ib/mlx5/gdaki/`** - Stub header files and documentation

**NOT Modified:**
- `src/uct/ib/mlx5/gdaki/configure.m4` - GDAKI is separate package, no changes needed
- `src/uct/Makefile.am` - Device API headers always installed (no change needed)
