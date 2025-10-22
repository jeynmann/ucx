project('uct_device_hello_world', ['c', 'cpp', 'cuda'],
    version: '1.0.0',
    default_options: [
        'warning_level=2',
        'cpp_std=c++11',
    ]
)

# Find CUDA toolkit
cuda = import('unstable-cuda')

# Find UCX via pkg-config
ucx_dep = dependency('ucx', required: true)
ucs_dep = dependency('ucs', required: false)
uct_dep = dependency('uct', required: false)

# If individual UCX components not found, try to use ucx as a whole
deps = []
if ucs_dep.found() and uct_dep.found()
    deps = [ucs_dep, uct_dep]
else
    deps = [ucx_dep]
endif

# Add CUDA runtime
cuda_dep = dependency('cuda', version: '>=10.0', required: true)
deps += [cuda_dep]

# Optional (try to find, don't fail if missing)
gdrcopy_dep = dependency('gdrcopy', required: false)
gpunetio_dep = dependency('doca-gpunetio', required: false)
deps += [gdrcopy_dep, gpunetio_dep]

# NVCC compiler args
nvcc_args = [
    '-gencode', 'arch=compute_70,code=sm_70',  # Volta
    '-gencode', 'arch=compute_75,code=sm_75',  # Turing
    '-gencode', 'arch=compute_80,code=sm_80',  # Ampere
    '-gencode', 'arch=compute_86,code=sm_86',  # Ampere (RTX 30xx)
    '-gencode', 'arch=compute_89,code=sm_89',  # Ada (RTX 40xx)
    '-gencode', 'arch=compute_90,code=sm_90',  # Hopper
    '--expt-relaxed-constexpr',
    '--extended-lambda',
]

# Include directories
inc_dirs = include_directories(
    # '../..',  UCX root
    # '../../src',  UCX source directory
)

# Build executable
executable('uct_device_hello',
    'uct_device_hello.cu',
    dependencies: deps,
    include_directories: inc_dirs,
    cuda_args: nvcc_args,
    install: false,
)

# Print configuration summary
message('UCT Device Hello World Configuration:')
message('  CUDA version: ' + cuda_dep.version())
message('  UCX found: ' + ucx_dep.found().to_string())

