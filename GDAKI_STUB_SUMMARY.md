# GDAKI Stub Implementation Summary

## Problem
UCX Device API includes `<uct/ib/mlx5/gdaki/gdaki.cuh>` which requires DOCA GPUNetIO package. Without it, UCX cannot be built with device code support.

## Solution
Install stub headers that provide fallback implementations, allowing UCX to build without GDAKI while enabling seamless upgrade when GDAKI is installed.

## Key Design Decisions

### 1. Unconditional Installation
- **Always install stubs** since Device API headers are always installed
- UCX Device API headers (`uct_device_impl.h`) include `<uct/ib/mlx5/gdaki/gdaki.cuh>`
- Stubs ensure these headers can be used even when built without CUDA/NVCC

### 2. Separate Installation Path
- Real GDAKI: `${prefix}/include/uct/ib/mlx5/gdaki/`
- Stub GDAKI: `${prefix}/include/stub/uct/ib/mlx5/gdaki/`
- Stub has lower priority in include search order

### 3. Unconditional pkg-config Export
- Always: `pkg-config ucx --cflags` → `-I${prefix}/include -I${prefix}/include/stubs`
- Ensures Device API headers can be used in all configurations
- Controlled by `@STUB_CFLAGS@` variable (always set)

### 4. No Modifications to GDAKI Package
- GDAKI remains a completely separate package
- No changes to `src/uct/ib/mlx5/gdaki/configure.m4`
- Clean separation of concerns

## Modified Files

1. **`stub/Makefile.am`**
   - Always installs stubs (no condition)
   - Installs to `stubincludedir = $(includedir)/stubs`

2. **`Makefile.am`**
   - Added `stubs` to SUBDIRS

3. **`configure.ac`**
   - Added `stubs/Makefile` to AC_CONFIG_FILES

4. **`config/m4/cuda.m4`**
   - Added at end of `UCX_CHECK_CUDA`:
     ```m4
     STUB_CFLAGS="-I\${includedir}/stubs"
     AC_SUBST([STUB_CFLAGS])
     ```

5. **`ucx.pc.in`**
   - Changed: `Cflags: -I${includedir}`
   - To: `Cflags: -I${includedir} @STUB_CFLAGS@`

6. **`stubs/uct/ib/mlx5/gdaki/`**
   - `gdaki.cuh` - Stub implementations
   - `gdaki_dev.h` - Type definitions
   - `README.md` - Documentation

## Build Scenarios

### Scenario A: UCX with CUDA, no GDAKI
```bash
./configure --with-cuda
make install

# Result:
# - ${prefix}/include/stub/uct/ib/mlx5/gdaki/ ✓ (stub installed)
# - pkg-config ucx --cflags: -I/usr/include -I/usr/include/stub
# - Device code compiles using stub (returns UCS_ERR_UNSUPPORTED)
```

### Scenario B: UCX with CUDA + GDAKI installed
```bash
./configure --with-cuda
make install
# (Then install GDAKI package separately)

# Result:
# - ${prefix}/include/uct/ib/mlx5/gdaki/ ✓ (real GDAKI)
# - ${prefix}/include/stub/uct/ib/mlx5/gdaki/ ✓ (stub, lower priority)
# - pkg-config ucx --cflags: -I/usr/include -I/usr/include/stub
# - Device code uses real GDAKI (found first in search order)
```

### Scenario C: UCX without CUDA/NVCC
```bash
./configure --without-cuda
make install

# Result:
# - ${prefix}/include/stubs/uct/ib/mlx5/gdaki/ ✓ (stub installed)
# - ${prefix}/include/uct/api/device/uct_device_impl.h ✓ (Device API headers)
# - pkg-config ucx --cflags: -I/usr/include -I/usr/include/stubs
# - Device API headers available, stub provides UCS_ERR_UNSUPPORTED
```

## Benefits

✅ **Minimal changes** - Only 5 build system files modified, no GDAKI changes
✅ **Clean separation** - GDAKI package completely independent  
✅ **Always available** - Stubs always installed, matching Device API headers
✅ **Automatic priority** - Real GDAKI always used when present
✅ **No broken builds** - Headers always available, even without CUDA/NVCC
✅ **Graceful degradation** - Stub returns UCS_ERR_UNSUPPORTED at runtime

## Testing

```bash
# Test 1: With CUDA, without GDAKI
./configure --with-cuda --prefix=/tmp/test1
make install
ls /tmp/test1/include/stubs/uct/ib/mlx5/gdaki/  # Should exist
pkg-config --cflags ucx  # Should include stubs path

# Test 2: Without CUDA
./configure --without-cuda --prefix=/tmp/test2
make install
ls /tmp/test2/include/stubs/uct/ib/mlx5/gdaki/  # Should still exist
ls /tmp/test2/include/uct/api/device/            # Device API headers present
pkg-config --cflags ucx  # Should include stubs path

# Test 3: With CUDA + GDAKI
./configure --with-cuda --prefix=/tmp/test3
make install
# Install GDAKI package to /tmp/test3
ls /tmp/test3/include/uct/ib/mlx5/gdaki/        # Real GDAKI
ls /tmp/test3/include/stubs/uct/ib/mlx5/gdaki/  # Stub (lower priority)
# Compilation finds real GDAKI first
```

## Summary

This implementation provides a clean, minimal solution for GDAKI stub headers:
- Always installed (matches Device API headers availability)
- Proper priority mechanism (real > stub)
- No modifications to GDAKI package
- No broken builds when headers are used
- Seamless GDAKI upgrade path
- Works with or without CUDA/NVCC
