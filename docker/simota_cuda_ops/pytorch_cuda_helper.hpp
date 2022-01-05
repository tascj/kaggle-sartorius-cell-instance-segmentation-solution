// https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/ops/csrc/common/pytorch_cuda_helper.hpp
// https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/ops/csrc/common/cuda/common_cuda_helper.hpp

#ifndef PYTORCH_CUDA_HELPER
#define PYTORCH_CUDA_HELPER

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>

#include <cuda.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 4096;
  return min(optimal_block_num, max_block_num);
}

using at::Half;
using at::Tensor;
using phalf = at::Half;

#define __PHALF(x) (x)
#endif  // PYTORCH_CUDA_HELPER
