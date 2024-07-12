
#include <iostream>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"

namespace {

using bfloat16 = __nv_bfloat16;

__device__ float fma(float a, float b, float c) {
  return __fmaf_rn(a, b, c);
}

__device__ half fma(half a, half b, half c) {
  return __hfma(a, b, c);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ bfloat16 fma(bfloat16 a, bfloat16 b, bfloat16 c) {
  return __hfma(a, b, c);
}
#endif

// For each thread, read
// A tile: 8 x K
// B matrix: K x N
// C tile: 8 x N
template<typename TElem, int num_thread, int N, int K, bool USE_FP16_ACC>
__global__ void gemm_rrr_small_nk_kernel(
    const float4* a_ptr, const float4* b_ptr, float4* c_ptr, int M) {
  int idx = blockIdx.x * num_thread + threadIdx.x;
  constexpr int num_elems_in_float4 = sizeof(float4) / sizeof(TElem);

  if (idx >= (M + num_elems_in_float4 - 1) / num_elems_in_float4) {
    return;
  }

  int a_idx_base = idx * K;
  a_ptr += a_idx_base;

  // load b matrix
  TElem b[K][N];
  auto* b_e = reinterpret_cast<const TElem*>(b_ptr);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      b[i][j] = b_e[i * N + j];
    }
  }

  int c_idx_base = idx * N;
  c_ptr += c_idx_base;

  TElem c_tile[num_elems_in_float4][N];

  if (idx <= M / num_elems_in_float4 - 1) {
    // fast kernel
    // load a
    float4 a_tile_vec[K];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < K; i++) {
      a_tile_vec[i] = __ldg(a_ptr++);
    }
    auto* a_tile = reinterpret_cast<const TElem*>(&a_tile_vec);

    // compute
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < num_elems_in_float4; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < N; ++j) {
        if constexpr (USE_FP16_ACC) {
          TElem sum = 0;
          CUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < K; ++k) {
            sum = fma(a_tile[i * K + k], b[k][j], sum);
          }
          c_tile[i][j] = sum;
        } else {
          float sum = 0;
          if constexpr (std::is_same_v<TElem, half>) {
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < K; ++k) {
              sum += __half2float(__hmul(a_tile[i * K + k], b[k][j]));
            }
            c_tile[i][j] = __float2half_rn(sum);
          } else {
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < K; ++k) {
              sum += __fmul_rn(a_tile[i * K + k], b[k][j]);
            }
            c_tile[i][j] = sum;
          }
        }
      }
    }

    // write c
    float4* c_tile_vec = reinterpret_cast<float4*>(&c_tile);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; i++) {
      c_ptr[i] = c_tile_vec[i];
    }
  } else {
    // process tail
    // load a
    auto* a_e = reinterpret_cast<const TElem*>(a_ptr);
    int m = M - M / num_elems_in_float4 * num_elems_in_float4;
    TElem a_tile[num_elems_in_float4][K];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < m; i++) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < K; j++) {
        a_tile[i][j] = a_e[i * K + j];
      }
    }

    // compute
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < m; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < N; ++j) {
        if constexpr (USE_FP16_ACC) {
          TElem sum = 0;
          CUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < K; ++k) {
            sum = fma(a_tile[i][k], b[k][j], sum);
          }
          c_tile[i][j] = sum;
        } else {
          float sum = 0;
          if constexpr (std::is_same_v<TElem, half>) {
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < K; ++k) {
              sum += __half2float(__hmul(a_tile[i][k], b[k][j]));
            }
            c_tile[i][j] = __float2half_rn(sum);
          }
          else {
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < K; ++k) {
              sum += __fmul_rn(a_tile[i][k], b[k][j]);
            }
            c_tile[i][j] = sum;
          }
        }
      }
    }

    // write c
    auto* c_h = reinterpret_cast<TElem*>(c_ptr);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < m; i++) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < N; j++) {
        c_h[i * N + j] = c_tile[i][j];
      }
    }
  }
}

// N <= 8, K <= 8
template<typename ElemT, int N, int K,
         typename = std::enable_if_t<std::is_same_v<ElemT, float> || std::is_same_v<ElemT, half> || std::is_same_v<ElemT, bfloat16>, void>>
void gemm_rrr_small_nk_launcher(ElemT* a_ptr,
                         ElemT* b_ptr,
                         ElemT* c_ptr,
                         int M,
                         bool use_fp16_acc,
                         cudaStream_t stream) {
  constexpr int num_elems_in_float4 = sizeof(float4) / sizeof(ElemT);
  const int nthread = 256;
  dim3 thread_block(nthread);
  constexpr int n_element_per_t = nthread * num_elems_in_float4;
  dim3 grid((M + n_element_per_t - 1) / n_element_per_t);
  if (use_fp16_acc && (std::is_same_v<ElemT, half> || std::is_same_v<ElemT, bfloat16>)) {
    gemm_rrr_small_nk_kernel<ElemT, nthread, N, K, true><<<grid, thread_block, 0, stream>>>(
      reinterpret_cast<const float4*>(a_ptr),
      reinterpret_cast<const float4*>(b_ptr),
      reinterpret_cast<float4*>(c_ptr),
      M
    );
  } else {
    gemm_rrr_small_nk_kernel<ElemT, nthread, N, K, false><<<grid, thread_block, 0, stream>>>(
      reinterpret_cast<const float4*>(a_ptr),
      reinterpret_cast<const float4*>(b_ptr),
      reinterpret_cast<float4*>(c_ptr),
      M
    );
  }
}

} // namespace

void gemm_rrr_small_nk_3 (
    void* a_ptr,
    void* b_ptr,
    void* c_ptr,
    
    int64_t *a_dim0,
    
    int64_t *a_dim1,
    
    
    int64_t *b_dim0,
    
    int64_t *b_dim1,
    
    
    int64_t *c_dim0,
    
    int64_t *c_dim1,
    
    bool use_fp16_acc,
    cudaStream_t stream
) {
  
 int64_t M = (*a_dim0);

 int64_t N = (*b_dim1);

 int64_t K = (*a_dim1);
  
  int64_t a_size = 1;

    a_size *= *a_dim0;

    a_size *= *a_dim1;

  if (a_size != 0 && !a_ptr) {
    throw std::runtime_error("input a is null!");
  }

  int64_t b_size = 1;

    b_size *= *b_dim0;

    b_size *= *b_dim1;

  if (b_size != 0 && !b_ptr) {
    throw std::runtime_error("input b is null!");
  }

  int64_t c_size = 1;

    c_size *= *c_dim0;

    c_size *= *c_dim1;

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
  
  gemm_rrr_small_nk_launcher<half, 6, 3>(
      (half*)a_ptr,
      (half*)b_ptr,
      (half*)c_ptr,
      M,
      use_fp16_acc,
      stream
  );
  return;
}
