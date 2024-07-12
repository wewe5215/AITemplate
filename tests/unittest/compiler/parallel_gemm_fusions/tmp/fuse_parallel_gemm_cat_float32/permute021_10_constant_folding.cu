

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/util/host_tensor.h"


#include <limits>

#define TILE_SIZE 32
#define CH_K 4

namespace {

//  Copyright (c) Meta Platforms, Inc. and affiliates.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
#ifndef AIT_TENSOR_ACCESSOR_CUH
#define AIT_TENSOR_ACCESSOR_CUH

// Returns a strided address based on a base pointer, an index and strided
// information.
// DATA_T: tensor data type.
// READ_T: actual data type used when reading data. e.g. for a "half"
// tensor, READ_T could be uint4 when all data is aligned.
// data: A base pointer in READ_T type.
// idx: read index in terms of READ_T.
// offset, original_total_elements_from_stride_dim and
// actual_total_elements_from_stride_dim are the corresponding data member
// values of TensorAccessor.
template <typename DATA_T, typename READ_T, bool is_contiguous>
__device__ __forceinline__ READ_T* get_strided_address(
    READ_T* data,
    int64_t idx,
    int64_t offset,
    int64_t original_total_elements_from_stride_dim,
    int64_t actual_total_elements_from_stride_dim) {
  (void)original_total_elements_from_stride_dim; // Suppress incorrect declared
                                                 // but never referenced warning
                                                 // from nvcc.
  (void)actual_total_elements_from_stride_dim; // Ditto.
  if constexpr (is_contiguous) {
    return reinterpret_cast<READ_T*>(reinterpret_cast<DATA_T*>(data) + offset) +
        idx;
  } else {
    constexpr int N_ELEMENTS_PER_READ = sizeof(READ_T) / sizeof(DATA_T);
    int64_t data_idx = idx * N_ELEMENTS_PER_READ;
    int64_t num_rows = data_idx / original_total_elements_from_stride_dim;
    int64_t row_offset = data_idx % original_total_elements_from_stride_dim;
    data_idx =
        num_rows * actual_total_elements_from_stride_dim + row_offset + offset;
    return reinterpret_cast<READ_T*>(
        reinterpret_cast<DATA_T*>(data) + data_idx);
  }
  return nullptr; // Suppress incorrect warning about missing return statement
                  // from nvcc.
}

static inline uint64_t max_power2_divisor(uint64_t n) {
  // max power of 2 which divides n
  return n & (~(n - 1));
}

// A TensorAccessor which handles strided tensor access underneath.
struct TensorAccessor {
  int64_t offset{0};
  bool is_contiguous{true};

  int stride_dim{-1};
  int64_t original_total_elements_from_stride_dim{-1};
  int64_t actual_total_elements_from_stride_dim{-1};

  // Returns an address based on a base pointer and an index.

  // DATA_T: tensor data type.
  // READ_T: actual data type used when reading data. e.g. for a "half"
  // tensor, READ_T could be uint4 when all data is aligned.
  // data: A base pointer in READ_T type.
  // idx: read index in terms of READ_T.
  template <typename DATA_T, typename READ_T>
  __device__ inline READ_T* get(READ_T* data, int64_t idx) const {
    return is_contiguous ? get_strided_address<DATA_T, READ_T, true>(
                               data,
                               idx,
                               offset,
                               original_total_elements_from_stride_dim,
                               actual_total_elements_from_stride_dim)
                         : get_strided_address<DATA_T, READ_T, false>(
                               data,
                               idx,
                               offset,
                               original_total_elements_from_stride_dim,
                               actual_total_elements_from_stride_dim);
  }

  uint64_t max_alignment() const {
    // gcd of max alignments
    auto alignment = max_power2_divisor(offset);
    if (!is_contiguous) {
      alignment |= max_power2_divisor(original_total_elements_from_stride_dim);
      alignment |= max_power2_divisor(actual_total_elements_from_stride_dim);
    }
    return max_power2_divisor(alignment);
  }

  bool is_valid_alignment(uint64_t n) const {
    // n is a power of 2; return whether tensor accessor alignment is divisible
    // by n.
    return !(max_alignment() & (n - 1));
  }
};

#endif


// blockIdx.x -> ni
// blockIdx.y -> hwi
// blockIdx.z -> ci
__device__ __forceinline__ void block_fn_nhc(int32_t& ni, int32_t& hwi, int32_t& ci) {
  ni = blockIdx.x;
  hwi = blockIdx.y;
  ci = blockIdx.z;
}

// blockIdx.x -> ni
// blockIdx.y -> ci
// blockIdx.z -> hwi
__device__ __forceinline__ void block_fn_nch(int32_t& ni, int32_t& hwi, int32_t& ci) {
  ni = blockIdx.x;
  ci = blockIdx.y;
  hwi = blockIdx.z;
}

// blockIdx.x -> ci
// blockIdx.y -> hwi
// blockIdx.z -> ni
__device__ __forceinline__ void block_fn_chn(int32_t& ni, int32_t& hwi, int32_t& ci) {
  ci = blockIdx.x;
  hwi = blockIdx.y;
  ni = blockIdx.z;
}

using BlockFunc = void (*)(int32_t&, int32_t&, int32_t&);

template <typename T, BlockFunc BLOCK_FN>
__global__ void permute021_kernel(T *output,
                                  const T *input,
                                  const int64_t n,
                                  const int32_t h,
                                  const int32_t w,
                                  const int32_t c,
                                  TensorAccessor input_accessor) {

  const int32_t hw = h * w;
  const int32_t hwc = hw * c;

  __shared__ T shbuf[TILE_SIZE * (TILE_SIZE + 1)];

  const int32_t tid  = threadIdx.y * blockDim.x + threadIdx.x;
  const int32_t wid  = tid / TILE_SIZE;
  const int32_t lid  = tid % TILE_SIZE;
  int32_t ni_tmp, hwi_tmp, ci_tmp;
  BLOCK_FN(ni_tmp, hwi_tmp, ci_tmp);
  const int32_t ni = ni_tmp;
  const int32_t hwi0 = hwi_tmp * TILE_SIZE;
  const int32_t ci0  = ci_tmp * TILE_SIZE;

  size_t input_idx = ni * hwc + (hwi0 + wid) * c + ci0;

  const T *A = input_accessor.get<const T, const T>(input, input_idx);

  if (ci0 + lid < c) {
    const int lid_x_33 = lid * (TILE_SIZE + 1);
    if ((hwi0 + TILE_SIZE) <= hw) {
      int hwi = wid;  // between 0 and 7
      #pragma unroll
      for (int cLoopIdx = 0; cLoopIdx < CH_K; cLoopIdx++) {
        shbuf[lid_x_33 + hwi] = *input_accessor.get<const T, const T>(input, input_idx + lid);
        input_idx += TILE_SIZE / CH_K * c;
        hwi += TILE_SIZE / CH_K;
      }
    } else {
      for (int hwi = wid; hwi < TILE_SIZE; hwi += TILE_SIZE / CH_K) {
        if (hwi + hwi0 < hw) {
          shbuf[lid_x_33 + hwi] = *input_accessor.get<const T, const T>(input, input_idx + lid);
        }
        input_idx += TILE_SIZE / CH_K * c;
      }
    }
  }
  __syncthreads();

  const int32_t hwiOut = hwi0 + lid;
  output = &output[ni * hwc + hwiOut];
  if (hwiOut < hw) {
    if (ci0 + TILE_SIZE < c) {
      int cI = wid;
      #pragma unroll
      for (int hwLoopIdx = 0; hwLoopIdx < CH_K; ++hwLoopIdx) {
        output[(ci0 + cI) * hw] = shbuf[cI * (TILE_SIZE + 1) + lid];
        cI += TILE_SIZE / CH_K;
      }
    } else {
      for (int cI = wid; cI < TILE_SIZE; cI += TILE_SIZE / CH_K) {
        if (ci0 + cI < c) {
          output[(ci0 + cI) * hw] = shbuf[cI * (TILE_SIZE + 1) + lid];
        }
      }
    }
  }
}

void permute021_launcher(const void* in_ptr,
                         void* out_ptr,
                         int64_t rank,
                         const int64_t* x_dims,
                         TensorAccessor input_accessor,
                         cudaStream_t stream) {
  int64_t x_dim0 = 1;
  for (int i = 0; i < rank - 2; i++) {
    x_dim0 *= x_dims[i];
  }

  if (x_dims[rank-2] > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error("The second last dim does not fit into int32_t.");
  }
  if (x_dims[rank-1] > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error("The last dim does not fit into int32_t.");
  }

  // given the above checks, we know it's safe
  const int32_t x_dim1 = x_dims[rank-2];
  const int32_t x_dim2 = x_dims[rank-1];

#define THROW_INVALID_LAUNCH_CONFIG                       throw std::runtime_error(                                 std::string("invalid cuda launch config: ") +         std::to_string(grid_c) + ", " +                       std::to_string(grid_hw) + ", " +                      std::to_string(grid_n));

  const int32_t n = static_cast<int32_t>(x_dim0);
  const int32_t h = 1;
  const int32_t w = x_dim1;
  const int32_t c = x_dim2;
  const int32_t grid_c = (c + TILE_SIZE - 1) / TILE_SIZE;
  const int32_t grid_hw = (h * w + TILE_SIZE - 1) / TILE_SIZE;
  const int32_t grid_n = n;
  constexpr int32_t max_grid_z = 65535;
  constexpr int32_t max_grid_x = 2147483647;
  if (grid_c > max_grid_x || grid_hw > max_grid_x || grid_n > max_grid_x) {
    THROW_INVALID_LAUNCH_CONFIG
  }
  if ((grid_c <= max_grid_z && grid_hw <= max_grid_z && grid_n <= max_grid_z) ||
      (grid_c > max_grid_z && grid_hw <= max_grid_z && grid_n <= max_grid_z)) {
    dim3 grid(grid_c, grid_hw, grid_n);
    dim3 block(TILE_SIZE, TILE_SIZE / CH_K);
    permute021_kernel<float, block_fn_chn><<<grid, block, 0, stream>>>(
        static_cast<float*>(out_ptr),
        static_cast<const float*>(in_ptr),
        n, h, w, c, input_accessor
    );
  } else if (grid_n > max_grid_z && grid_hw <= max_grid_z && grid_c <= max_grid_z) {
    dim3 grid(grid_n, grid_c, grid_hw);
    dim3 block(TILE_SIZE, TILE_SIZE / CH_K);
    permute021_kernel<float, block_fn_nch><<<grid, block, 0, stream>>>(
        static_cast<float*>(out_ptr),
        static_cast<const float*>(in_ptr),
        n, h, w, c, input_accessor
    );
  } else if (grid_n > max_grid_z && grid_hw <= max_grid_z && grid_c <= max_grid_z) {
    dim3 grid(grid_n, grid_hw, grid_c);
    dim3 block(TILE_SIZE, TILE_SIZE / CH_K);
    permute021_kernel<float, block_fn_nhc><<<grid, block, 0, stream>>>(
        static_cast<float*>(out_ptr),
        static_cast<const float*>(in_ptr),
        n, h, w, c, input_accessor
    );
  } else {
    THROW_INVALID_LAUNCH_CONFIG
  }
  
}
} // namespace

void permute021_10_constant_folding (
    const void* in_ptr,
    void* out_ptr,
    int64_t rank,
    const int64_t* x_dims,
    cudaStream_t stream
) {
  for (int i = 0; i < rank; i++) {
      if (x_dims[i] == 0) {
          // empty input: nothing to do
          return;
      }
  }
  if (!in_ptr) {
    throw std::runtime_error("in_ptr is NULL!");
  }
  if (!out_ptr) {
    throw std::runtime_error("out_ptr is NULL!");
  }
  

    TensorAccessor input_accessor = {
      0,
      
      true
      
      
    };
permute021_launcher(
    in_ptr,
    out_ptr,
    rank,
    x_dims,
    input_accessor,
    stream
);
return;
}