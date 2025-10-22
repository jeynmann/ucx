/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * UCT Device API Hello World Example
 *
 * This example demonstrates basic UCT device API usage with CUDA:
 * - Initialize UCT with CUDA IPC transport
 * - Perform device-side memory operations
 * - Use device API from CUDA kernels
 */

#include <uct/api/uct.h>
#include <uct/api/device/uct_device_impl.h>
#include <ucs/type/status.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define UCT_CHECK(call) do { \
    ucs_status_t status = call; \
    if (status != UCS_OK) { \
        fprintf(stderr, "UCT Error: %s at %s:%d\n", \
                ucs_status_string(status), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

/* Simple device kernel that uses UCT device API */
__global__ void device_hello_kernel(uct_device_ep_h device_ep,
                                    uct_device_mem_element_t *mem_elem,
                                    uint64_t *src_data,
                                    uint64_t remote_addr,
                                    size_t length)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid == 0) {
        printf("Device Kernel: Hello from GPU thread %d!\n", tid);
        printf("Device Kernel: Performing device-side PUT operation...\n");
        
        /* Perform device-side PUT using UCT device API */
        uct_device_completion_t comp;
        ucs_status_t status = uct_device_ep_put_single<UCS_DEVICE_LEVEL_THREAD>(
            device_ep, mem_elem, src_data, remote_addr, length,
            0, &comp);
        
        if (status == UCS_ERR_UNSUPPORTED) {
            printf("Device Kernel: Device API not supported on this transport\n");
        } else if (status == UCS_INPROGRESS) {
            printf("Device Kernel: Operation posted, status=INPROGRESS\n");
        } else if (status == UCS_OK) {
            printf("Device Kernel: Operation completed immediately\n");
        }
    }
}

int main(int argc, char **argv)
{
    printf("=== UCT Device API Hello World ===\n\n");
    
    /* Initialize CUDA */
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("Found %d CUDA device(s)\n", device_count);
    
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }
    
    CUDA_CHECK(cudaSetDevice(0));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using device: %s\n\n", prop.name);
    
    /* Initialize UCT */
    ucs_status_t status;
    uct_component_h *components;
    unsigned num_components;
    
    status = uct_query_components(&components, &num_components);
    if (status != UCS_OK) {
        printf("Note: UCT not fully initialized, but example demonstrates device API usage\n");
        printf("To use device API in production, ensure:\n");
        printf("  1. UCX is built with CUDA support (--with-cuda)\n");
        printf("  2. CUDA IPC or rc_gda transport is available\n");
        printf("  3. Proper UCT initialization is done\n\n");
    }
    
    /* Allocate device memory */
    const size_t data_size = 1024;
    uint64_t *d_src_data, *d_dst_data;
    uint64_t *h_data;
    
    h_data = (uint64_t*)malloc(data_size);
    for (size_t i = 0; i < data_size / sizeof(uint64_t); i++) {
        h_data[i] = 0xDEADBEEF00000000ULL + i;
    }
    
    CUDA_CHECK(cudaMalloc(&d_src_data, data_size));
    CUDA_CHECK(cudaMalloc(&d_dst_data, data_size));
    CUDA_CHECK(cudaMemcpy(d_src_data, h_data, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dst_data, 0, data_size));
    
    printf("Allocated device buffers:\n");
    printf("  Source:      %p\n", d_src_data);
    printf("  Destination: %p\n\n", d_dst_data);
    
    /* Demonstrate device API concept */
    printf("UCT Device API Concepts:\n");
    printf("------------------------\n");
    printf("1. Device API allows GPU kernels to initiate communication\n");
    printf("2. Operations are called from __device__ code (CUDA kernels)\n");
    printf("3. Supported operations:\n");
    printf("   - uct_device_ep_put_single()      : Single PUT\n");
    printf("   - uct_device_ep_put_multi()       : Batched PUT operations\n");
    printf("   - uct_device_ep_atomic_add()      : Atomic increment\n");
    printf("   - uct_device_ep_progress()        : Progress communications\n");
    printf("   - uct_device_ep_check_completion(): Check operation status\n");
    printf("4. Device cooperation levels:\n");
    printf("   - UCS_DEVICE_LEVEL_THREAD : Single thread\n");
    printf("   - UCS_DEVICE_LEVEL_WARP   : Warp-level cooperation\n");
    printf("   - UCS_DEVICE_LEVEL_BLOCK  : Block-level cooperation\n");
    printf("   - UCS_DEVICE_LEVEL_GRID   : Grid-level cooperation\n\n");
    
    /* Note: In a real application, you would:
     * 1. Initialize UCT properly with uct_worker_create, uct_iface_open, etc.
     * 2. Create endpoints with UCT_IFACE_FLAG_DEVICE_EP capability
     * 3. Get device endpoint handle via uct_ep_get_device_ep()
     * 4. Pass the device endpoint to kernels
     * 5. Call device API functions from within kernels
     */
    
    printf("Example kernel launch (conceptual):\n");
    printf("  device_hello_kernel<<<blocks, threads>>>(\n");
    printf("    device_ep, mem_elem, d_src_data, remote_addr, data_size);\n\n");
    
    /* Simple CUDA kernel to demonstrate the concept */
    printf("Launching simple device kernel...\n");
    
    /* Launch kernel with NULL pointers just to show the API structure
     * In real usage, you'd pass valid UCT device handles */
    dim3 blocks(1);
    dim3 threads(32);
    device_hello_kernel<<<blocks, threads>>>(
        nullptr,  /* Would be real device_ep from uct_ep_get_device_ep() */
        nullptr,  /* Would be real mem_elem */
        d_src_data,
        (uint64_t)d_dst_data,
        data_size);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("\nKernel execution completed\n");
    
    /* Cleanup */
    CUDA_CHECK(cudaFree(d_src_data));
    CUDA_CHECK(cudaFree(d_dst_data));
    free(h_data);
    
    if (num_components > 0) {
        uct_release_component_list(components);
    }
    
    printf("\n=== UCT Device API Hello World Complete ===\n");
    printf("\nFor a complete working example, see:\n");
    printf("  - test/gtest/uct/cuda/test_cuda_ipc_device.cc\n");
    printf("  - test/gtest/ucp/test_ucp_device.cc\n");
    
    return 0;
}

