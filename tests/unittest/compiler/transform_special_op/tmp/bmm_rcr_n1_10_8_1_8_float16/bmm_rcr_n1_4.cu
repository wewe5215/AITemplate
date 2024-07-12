
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"

using bfloat16 = __nv_bfloat16;
using bfloat16_2 =  __nv_bfloat162;

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


template<typename ElemT, typename ReadVecT, int64_t K>
__forceinline__ __device__ bool load_vec_data(
    ReadVecT* a_ptr,
    ReadVecT* b_ptr,
    const int64_t M,
    float alpha,
    TensorAccessor input_a_accessor,
    TensorAccessor input_b_accessor,
    TensorAccessor output_accessor,
    ReadVecT *a_vec,
    ReadVecT *b_vec) {

  int64_t batch_idx = blockIdx.y;
  int64_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;

  constexpr int64_t N_READ_ELEMS_IN_V = sizeof(ReadVecT) / sizeof(ElemT);
  constexpr int64_t N_NUM_ELEMS_IN_V = K / N_READ_ELEMS_IN_V;

  int64_t b_idx_base = (batch_idx * K) / N_READ_ELEMS_IN_V;

  if (blockDim.x >= N_NUM_ELEMS_IN_V) {
    // We have enough threads in a thread block where each thread takes care
    // of loading one vector.
    if (threadIdx.x < N_NUM_ELEMS_IN_V) {
      b_vec[threadIdx.x] = *input_b_accessor.get<ElemT, ReadVecT>(b_ptr, b_idx_base + threadIdx.x);
    }
  } else {
    // We have more vectors than the available threads of a thread block, so each
    // thread may read multiple vectors.
    for (int64_t i = 0; i < N_NUM_ELEMS_IN_V / blockDim.x + 1; i++) {
      int64_t idx = i * blockDim.x + threadIdx.x;
      if (idx < N_NUM_ELEMS_IN_V) {
        b_vec[idx] = *input_b_accessor.get<ElemT, ReadVecT>(b_ptr, b_idx_base + idx);
      }
    }
  }

  __syncthreads();
  if (row_idx >= M) {
    return false;
  }

  int64_t a_batch_stride = M * K;
  int64_t a_idx_base = (batch_idx * a_batch_stride + row_idx * K) / N_READ_ELEMS_IN_V;

  CUTLASS_PRAGMA_UNROLL
  for (int64_t k = 0, i = 0; k < K; k += N_READ_ELEMS_IN_V, i++) {
    a_vec[i] = *input_a_accessor.get<ElemT, ReadVecT>(a_ptr, a_idx_base++);
  }

  return true;
}

namespace detail {
  template<typename TInput>
  struct InputHelper;

  template<>
  struct InputHelper<float>{
    typedef float scalar_type;
    typedef float2 vec2_type;

    static
    __inline__ __device__ vec2_type fma2(vec2_type a, vec2_type b, vec2_type c) {
      return make_float2(__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y));
    }

    static
    __inline__ __device__ scalar_type fma(scalar_type a, scalar_type b, scalar_type c) {
      return __fmaf_rn(a, b, c);
    }

    static
    __inline__ __device__ vec2_type mul2(vec2_type a, vec2_type b) {
      return make_float2(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));
    }

    static
    __inline__ __device__ scalar_type mul(scalar_type a, scalar_type b) {
      return __fmul_rn(a, b);
    }

    static
    __inline__ __device__ vec2_type add2(vec2_type a, vec2_type b) {
      return make_float2(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y));
    }

    static
    __inline__ __device__ scalar_type add(scalar_type a, scalar_type b) {
      return __fadd_rn(a, b);
    }

    static
    __inline__ __device__ scalar_type low(vec2_type a) {
      return a.x;
    }

    static
    __inline__ __device__ scalar_type high(vec2_type a) {
      return a.y;
    }

    static
    __inline__ __device__ float lowf(vec2_type a) {
      return a.x;
    }

    static
    __inline__ __device__ float highf(vec2_type a) {
      return a.y;
    }
  };

  template<>
  struct InputHelper<half>{
    typedef half scalar_type;
    typedef half2 vec2_type;

    static
    __inline__ __device__ vec2_type fma2(vec2_type a, vec2_type b, vec2_type c) {
      return __hfma2(a, b, c);
    }

    static
    __inline__ __device__ scalar_type fma(scalar_type a, scalar_type b, scalar_type c) {
      return __hfma(a, b, c);
    }

    static
    __inline__ __device__ vec2_type mul2(vec2_type a, vec2_type b) {
      return __hmul2(a, b);
    }

    static
    __inline__ __device__ scalar_type mul(scalar_type a, scalar_type b) {
      return __hmul(a, b);
    }

    static
    __inline__ __device__ vec2_type add2(vec2_type a, vec2_type b) {
      return __hadd2(a, b);
    }

    static
    __inline__ __device__ scalar_type add(scalar_type a, scalar_type b) {
      return __hadd(a, b);
    }

    static
    __inline__ __device__ scalar_type low(vec2_type a) {
      return __low2half(a);
    }

    static
    __inline__ __device__ scalar_type high(vec2_type a) {
      return __high2half(a);
    }

    static
    __inline__ __device__ float lowf(vec2_type a) {
      return __low2float(a);
    }

    static
    __inline__ __device__ float highf(vec2_type a) {
      return __high2float(a);
    }
  };

#if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 800)
  template<>
  struct InputHelper<bfloat16> {
    typedef bfloat16 scalar_type;
    typedef bfloat16_2 vec2_type;

    static
    __inline__ __device__ vec2_type fma2(vec2_type a, vec2_type b, vec2_type c) {
      return __hfma2(a, b, c);
    }

    static
    __inline__ __device__ scalar_type fma(scalar_type a, scalar_type b, scalar_type c) {
      return __hfma(a, b, c);
    }

    static
    __inline__ __device__ vec2_type mul2(vec2_type a, vec2_type b) {
      return __hmul2(a, b);
    }

    static
    __inline__ __device__ scalar_type mul(scalar_type a, scalar_type b) {
      return __hmul(a, b);
    }

    static
    __inline__ __device__ vec2_type add2(vec2_type a, vec2_type b) {
      return __hadd2(a, b);
    }

    static
    __inline__ __device__ scalar_type add(scalar_type a, scalar_type b) {
      return __hadd(a, b);
    }

    static
    __inline__ __device__ scalar_type low(vec2_type a) {
      return __low2bfloat16(a);
    }

    static
    __inline__ __device__ scalar_type high(vec2_type a) {
      return __high2bfloat16(a);
    }

    static
    __inline__ __device__ float lowf(vec2_type a) {
      return __low2float(a);
    }

    static
    __inline__ __device__ float highf(vec2_type a) {
      return __high2float(a);
    }
  }; // struct InputHelper<bfloat16>
#endif
} // namespace detail

// Each thread reads one row from "a" and one column from "b",
// computes dot_product(a_row, b_col), and writes the result to "c".
// This kernel assumes loading "a" and "b" can be fully vectorized,
// so it reads both "a" and "b" in ReadVecT.
template<typename ElemT, typename ReadVecT, int64_t K>
__global__ void bmm_rcr_n1_kernel_fp32_acc_vec(
    ReadVecT* a_ptr,
    ReadVecT* b_ptr,
    ElemT* c_ptr,
    const int64_t M,
    float alpha,
    TensorAccessor input_a_accessor,
    TensorAccessor input_b_accessor,
    TensorAccessor output_accessor) {

  static_assert(sizeof(ReadVecT) % sizeof(ElemT) == 0, "invalid vector type");
  constexpr int64_t N_READ_ELEMS_IN_V = sizeof(ReadVecT) / sizeof(ElemT);
  static_assert(N_READ_ELEMS_IN_V % 2 == 0, "invalid vector type for read");
  static_assert(K % N_READ_ELEMS_IN_V == 0, "cannot vectorize input");
  constexpr int64_t N_NUM_ELEMS_IN_V = K / N_READ_ELEMS_IN_V;

  __shared__ ReadVecT b_vec[N_NUM_ELEMS_IN_V];
  ReadVecT a_vec[N_NUM_ELEMS_IN_V];

  if (!load_vec_data<ElemT, ReadVecT, K>(
        a_ptr, b_ptr, M, alpha, input_a_accessor, input_b_accessor,
        output_accessor, a_vec, b_vec)) {
    return;
  }

  float result = 0.0;

  using dispatch = typename detail::InputHelper<ElemT>;
  using vec2_type = typename dispatch::vec2_type;
  CUTLASS_PRAGMA_UNROLL
  for (int64_t i = 0; i < N_NUM_ELEMS_IN_V; i++) {
    auto* a_vec_h2 = reinterpret_cast<const vec2_type*>(&a_vec[i]);
    auto* b_vec_h2 = reinterpret_cast<const vec2_type*>(&b_vec[i]);
    CUTLASS_PRAGMA_UNROLL
    for (int64_t j = 0; j < N_READ_ELEMS_IN_V / 2; ++j) {
      auto c_h2 = dispatch::mul2(a_vec_h2[j], b_vec_h2[j]);
      result += dispatch::lowf(c_h2) + dispatch::highf(c_h2);
    }
  }

  int64_t batch_idx = blockIdx.y;
  int64_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  *output_accessor.get<ElemT, ElemT>(c_ptr, batch_idx * M + row_idx) = alpha * result;
}

template<typename ElemT, int64_t K>
__forceinline__ __device__ bool load_data(
    ElemT* a_ptr,
    ElemT* b_ptr,
    const int64_t M,
    float alpha,
    TensorAccessor input_a_accessor,
    TensorAccessor input_b_accessor,
    TensorAccessor output_accessor,
    ElemT *a_data,
    ElemT *b_data) {

  int64_t batch_idx = blockIdx.y;
  int64_t b_idx_base = batch_idx * K;
  int64_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (blockDim.x >= K) {
    // We have enough threads in a thread block where each thread takes care
    // of loading one element.
    if (threadIdx.x < K) {
      b_data[threadIdx.x] = *input_b_accessor.get<ElemT, ElemT>(b_ptr, b_idx_base + threadIdx.x);
    }
  } else {
    // We have more elements than the available threads of a thread block, so each
    // thread may load multiple elements.
    for (int64_t i = 0; i < K / blockDim.x + 1; i++) {
      int64_t idx = i * blockDim.x + threadIdx.x;
      if (idx < K) {
        b_data[idx] = *input_b_accessor.get<ElemT, ElemT>(b_ptr, b_idx_base + idx);
      }
    }
  }

  __syncthreads();

  if (row_idx >= M) {
    return false;
  }

  int64_t a_batch_stride = M * K;
  int64_t a_idx_base = batch_idx * a_batch_stride + row_idx * K;

  CUTLASS_PRAGMA_UNROLL
  for (int64_t i = 0; i < K; i++) {
    a_data[i] = *input_a_accessor.get<ElemT, ElemT>(a_ptr, a_idx_base++);
  }

  return true;
}

// Each thread reads one row from "a" and one column from "b",
// computes dot_product(a_row, b_col), and writes the result to "c".
// It reads both "a" and "b" one by one in ElemT.
template<typename ElemT, typename ReadVecT, int64_t K>
__global__ void bmm_rcr_n1_kernel_fp32_acc(
    ElemT* a_ptr,
    ElemT* b_ptr,
    ElemT* c_ptr,
    const int64_t M,
    float alpha,
    TensorAccessor input_a_accessor,
    TensorAccessor input_b_accessor,
    TensorAccessor output_accessor) {

  __shared__ ElemT b_data[K];
  ElemT a_data[K];

  if (!load_data<ElemT, K>(
        a_ptr, b_ptr, M, alpha, input_a_accessor, input_b_accessor,
        output_accessor, a_data, b_data)) {
    return;
  }

  float result = 0.0;

  using dispatch = typename detail::InputHelper<ElemT>;
  using vec2_type = typename dispatch::vec2_type;

  auto* a_data_h2 = reinterpret_cast<const vec2_type*>(&a_data[0]);
  auto* b_data_h2 = reinterpret_cast<const vec2_type*>(&b_data[0]);
  CUTLASS_PRAGMA_UNROLL
  for (int64_t i = 0; i < K / 2; ++i) {
    auto c_h2 = dispatch::mul2(a_data_h2[i], b_data_h2[i]);
    result += dispatch::lowf(c_h2) + dispatch::highf(c_h2);
  }
  if (K % 2) {
    result += float(dispatch::mul(a_data[K-1], b_data[K-1]));
  }

  int64_t batch_idx = blockIdx.y;
  int64_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;

  *output_accessor.get<ElemT, ElemT>(c_ptr, batch_idx * M + row_idx) = alpha * result;
}

template<typename ElemT, typename ReadVecT, int64_t K>
__global__ void bmm_rcr_n1_kernel_fp16_acc_vec(
    ReadVecT* a_ptr,
    ReadVecT* b_ptr,
    ElemT* c_ptr,
    const int64_t M,
    float alpha,
    TensorAccessor input_a_accessor,
    TensorAccessor input_b_accessor,
    TensorAccessor output_accessor) {

  static_assert(sizeof(ReadVecT) % sizeof(ElemT) == 0, "invalid vector type");
  constexpr int64_t N_READ_ELEMS_IN_V = sizeof(ReadVecT) / sizeof(ElemT);
  static_assert(N_READ_ELEMS_IN_V % 2 == 0, "invalid vector type for read");
  static_assert(K % N_READ_ELEMS_IN_V == 0, "cannot vectorize input");
  constexpr int64_t N_NUM_ELEMS_IN_V = K / N_READ_ELEMS_IN_V;

  __shared__ ReadVecT b_vec[N_NUM_ELEMS_IN_V];
  ReadVecT a_vec[N_NUM_ELEMS_IN_V];

  if (!load_vec_data<ElemT, ReadVecT, K>(
        a_ptr, b_ptr, M, alpha, input_a_accessor, input_b_accessor,
        output_accessor, a_vec, b_vec)) {
    return;
  }

  using dispatch = typename detail::InputHelper<ElemT>;
  using vec2_type = typename dispatch::vec2_type;
  vec2_type result_h2 = {0.0, 0.0};

  CUTLASS_PRAGMA_UNROLL
  for (int64_t i = 0; i < N_NUM_ELEMS_IN_V; i++) {
    auto* a_vec_h2 = reinterpret_cast<const vec2_type*>(&a_vec[i]);
    auto* b_vec_h2 = reinterpret_cast<const vec2_type*>(&b_vec[i]);
    CUTLASS_PRAGMA_UNROLL
    for (int64_t j = 0; j < N_READ_ELEMS_IN_V / 2; ++j) {
      result_h2 = dispatch::fma2(a_vec_h2[j], b_vec_h2[j], result_h2);
    }
  }

  float result = float(dispatch::add(dispatch::low(result_h2), dispatch::high(result_h2)));

  int64_t batch_idx = blockIdx.y;
  int64_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  *output_accessor.get<ElemT, ElemT>(c_ptr, batch_idx * M + row_idx) = alpha * result;
}

template<typename ElemT, typename ReadVecT, int64_t K>
__global__ void bmm_rcr_n1_kernel_fp16_acc(
    ElemT* a_ptr,
    ElemT* b_ptr,
    ElemT* c_ptr,
    const int64_t M,
    float alpha,
    TensorAccessor input_a_accessor,
    TensorAccessor input_b_accessor,
    TensorAccessor output_accessor) {

  __shared__ ElemT b_data[K];
  ElemT a_data[K];

  if (!load_data<ElemT, K>(
        a_ptr, b_ptr, M, alpha, input_a_accessor, input_b_accessor,
        output_accessor, a_data, b_data)) {
    return;
  }

  using dispatch = typename detail::InputHelper<ElemT>;
  using vec2_type = typename dispatch::vec2_type;

  vec2_type result_h2 = {0.0, 0.0};

  const auto* a_data_h2 = reinterpret_cast<const vec2_type*>(&a_data[0]);
  const auto* b_data_h2 = reinterpret_cast<const vec2_type*>(&b_data[0]);
  CUTLASS_PRAGMA_UNROLL
  for (int64_t i = 0; i < K / 2; ++i) {
    result_h2 = dispatch::fma2(a_data_h2[i], b_data_h2[i], result_h2);
  }

  auto result = dispatch::add(dispatch::low(result_h2), dispatch::high(result_h2));
  if (K % 2) {
    result = dispatch::fma(a_data[K-1], b_data[K-1], result);
  }

  int64_t batch_idx = blockIdx.y;
  int64_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  *output_accessor.get<ElemT, ElemT>(c_ptr, batch_idx * M + row_idx) =
      alpha * (float)result;
}

// N = 1, K is small
template<typename ElemT, typename ReadVecT, int64_t K>
void bmm_rcr_n1_launcher(ElemT* a_ptr,
                         ElemT* b_ptr,
                         ElemT* c_ptr,
                         int64_t B,
                         int64_t M,
                         float alpha,
                         bool use_fp16_acc,
                         cudaStream_t stream,
                         const TensorAccessor& input_a_accessor,
                         const TensorAccessor& input_b_accessor,
                         const TensorAccessor& output_accessor) {
  const int nthread = 256;
  dim3 thread_block(nthread);
  dim3 grid((M + nthread - 1) / nthread, B);

  if(use_fp16_acc) {
    bmm_rcr_n1_kernel_fp16_acc_vec<ElemT, ReadVecT, K>
    <<<grid, thread_block, 0, stream>>>(
      (ReadVecT*)a_ptr,
      (ReadVecT*)b_ptr,
      c_ptr,
      M,
      alpha,
      input_a_accessor,
      input_b_accessor,
      output_accessor
    );
  } else {
    bmm_rcr_n1_kernel_fp32_acc_vec<ElemT, ReadVecT, K>
    <<<grid, thread_block, 0, stream>>>(
      (ReadVecT*)a_ptr,
      (ReadVecT*)b_ptr,
      c_ptr,
      M,
      alpha,
      input_a_accessor,
      input_b_accessor,
      output_accessor
    );
  }
}

} // namespace

void bmm_rcr_n1_4 (
    void* a_ptr,
    void* b_ptr,
    void* c_ptr,
    
    int64_t *a_dim0,
    
    int64_t *a_dim1,
    
    int64_t *a_dim2,
    
    
    int64_t *b_dim0,
    
    int64_t *b_dim1,
    
    int64_t *b_dim2,
    
    
    int64_t *c_dim0,
    
    int64_t *c_dim1,
    
    int64_t *c_dim2,
    
    float alpha,
    bool use_fp16_acc,
    cudaStream_t stream
) {
  
 int64_t B = (*a_dim0);

 int64_t M = (*a_dim1);

 int64_t N = (*b_dim1);

 int64_t K = (*a_dim2);
  
  int64_t a_size = 1;

    a_size *= *a_dim0;

    a_size *= *a_dim1;

    a_size *= *a_dim2;

  if (a_size != 0 && !a_ptr) {
    throw std::runtime_error("input a is null!");
  }

  int64_t b_size = 1;

    b_size *= *b_dim0;

    b_size *= *b_dim1;

    b_size *= *b_dim2;

  if (b_size != 0 && !b_ptr) {
    throw std::runtime_error("input b is null!");
  }

  int64_t c_size = 1;

    c_size *= *c_dim0;

    c_size *= *c_dim1;

    c_size *= *c_dim2;

  if (c_size != 0) {
    if (!c_ptr) {
      throw std::runtime_error("input c is null!");
    }
  } else {
    // output is empty and safe to return
    return;
  }

  // One of the input tensor are empty
  if (a_size == 0 || b_size == 0) {
    return;
  }
  
    TensorAccessor input_a_accessor = {
      0,
      
      true
      
      
    };
    TensorAccessor input_b_accessor = {
      0,
      
      true
      
      
    };
  
    TensorAccessor output_accessor = {
      0,
      
      true
      
      
    };
  
  bmm_rcr_n1_launcher<half, uint4, 8>(
      (half*)a_ptr,
      (half*)b_ptr,
      (half*)c_ptr,
      B,
      M,
      alpha,
      use_fp16_acc,
      stream,
    input_a_accessor,
    input_b_accessor,
    output_accessor
  );
  return;
}
