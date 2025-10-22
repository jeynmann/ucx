#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

# Install stub headers whenever UCX device API headers are installed
# These provide fallback implementations when DOCA GPUNetIO package is not installed
# Real GDAKI headers go to ${includedir}/uct/ib/mlx5/gdaki/ (from separate GDAKI package)
# Stub headers go to ${includedir}/stubs/uct/ib/mlx5/gdaki/ (from UCX)
#
# Note: UCX Device API headers (uct_device_impl.h) are always installed and
# include <uct/ib/mlx5/gdaki/gdaki.cuh>, so stubs must always be installed
# to ensure the headers can be used even when built without CUDA/NVCC.

stubincludedir = $(includedir)/stubs

nobase_stubinclude_HEADERS = \
	uct/ib/mlx5/gdaki/gdaki.cuh \
	uct/ib/mlx5/gdaki/gdaki_dev.h

EXTRA_DIST = \
	uct/ib/mlx5/gdaki/README.md

