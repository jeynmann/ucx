#!/bin/bash
#
# Simple build script for UCT Device Hello World example
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/builddir"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prereqs() {
    print_info "Checking prerequisites..."
    
    # Check meson
    if ! command -v meson &> /dev/null; then
        print_error "meson not found. Install it with: pip3 install meson"
        exit 1
    fi
    
    # Check ninja
    if ! command -v ninja &> /dev/null; then
        print_error "ninja not found. Install it with: pip3 install ninja"
        exit 1
    fi
    
    # Check nvcc
    if ! command -v nvcc &> /dev/null; then
        print_error "nvcc not found. Ensure CUDA toolkit is installed and in PATH"
        exit 1
    fi
    
    # Check UCX
    if ! pkg-config --exists ucx 2>/dev/null; then
        print_warn "UCX not found via pkg-config. You may need to set PKG_CONFIG_PATH"
        print_warn "Example: export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig"
    fi
    
    print_info "All prerequisites found!"
}

# Parse arguments
CLEAN=false
DEBUG=false
CUDA_ARCH=""
UCX_PREFIX=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --cuda-arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --ucx-prefix)
            UCX_PREFIX="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --clean           Clean build directory before building"
            echo "  --debug           Build with debug symbols"
            echo "  --cuda-arch ARCH  Specify CUDA architecture (e.g., sm_80)"
            echo "  --ucx-prefix DIR  UCX installation prefix"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Clean if requested
if [ "$CLEAN" = true ]; then
    print_info "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Check prerequisites
check_prereqs

# Setup meson options
MESON_OPTS=""
if [ "$DEBUG" = true ]; then
    MESON_OPTS="$MESON_OPTS --buildtype=debug"
    print_info "Debug build enabled"
fi

if [ -n "$CUDA_ARCH" ]; then
    MESON_OPTS="$MESON_OPTS -Dcuda_arch=$CUDA_ARCH"
    print_info "CUDA architecture: $CUDA_ARCH"
fi

if [ -n "$UCX_PREFIX" ]; then
    export PKG_CONFIG_PATH="${UCX_PREFIX}/lib/pkgconfig:${UCX_PREFIX}/lib64/pkgconfig:$PKG_CONFIG_PATH"
    print_info "Using UCX prefix: $UCX_PREFIX"
fi

# Configure
if [ ! -d "$BUILD_DIR" ]; then
    print_info "Configuring build with meson..."
    cd "$SCRIPT_DIR"
    meson setup "$BUILD_DIR" $MESON_OPTS
else
    print_info "Build directory already configured"
fi

# Build
print_info "Compiling..."
cd "$SCRIPT_DIR"
meson compile -C "$BUILD_DIR"

# Success
print_info "Build complete!"
print_info "Executable: ${BUILD_DIR}/uct_device_hello"
echo ""
print_info "To run: ${BUILD_DIR}/uct_device_hello"

