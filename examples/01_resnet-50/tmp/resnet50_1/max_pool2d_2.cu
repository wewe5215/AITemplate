
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"

namespace {
extern __shared__ char* shared_mem[];

template <int kernel_size,
          int stride,
          int padding,
          int block_ch,
          int block_h,
          int block_w>
__global__ void max_pool_nhwc_kernel(const half* input_raw,
                                     half* output_raw,
                                     const int N,
                                     const int H,
                                     const int W,
                                     const int C,
                                     const int HO,
                                     const int WO) {

  const half2* input = (const half2*)input_raw;
  half2* output = (half2*)output_raw;
  half2* shm = (half2*)shared_mem;

  const int ldg_h = (block_h - 1) * stride + kernel_size;
  const int ldg_w = (block_w - 1) * stride + kernel_size;
  const int ldg_hw_num = ldg_h * ldg_w;

  const int n_idx = blockIdx.x;
  const int out_h_start_idx = blockIdx.y * block_h;
  const int out_w_start_idx = blockIdx.z * block_w;

  int ldg_h_start_idx = out_h_start_idx * stride - padding;

  int ldg_w_start_idx = out_w_start_idx * stride - padding;

  input += n_idx * H * W * C;

  const int hw_start_idx_of_thread = threadIdx.y;
  const int ch_thread_idx = threadIdx.x;


  const half2 min = {static_cast<half>(-65503.0f),
                     static_cast<half>(-65503.0f)};


  for (int i = hw_start_idx_of_thread; i < ldg_hw_num; i += block_ch) {
    const int shm_h_idx = i / ldg_w;
    const int shm_w_idx = i % ldg_w;
    const int input_h_idx = ldg_h_start_idx + shm_h_idx;
    const int input_w_idx = ldg_w_start_idx + shm_w_idx;
    const int input_idx = (input_h_idx * W + input_w_idx) * C + ch_thread_idx;
    const int shm_idx = i * C + ch_thread_idx;
    if (input_h_idx >= 0 && input_h_idx < H && input_w_idx >= 0 &&
        input_w_idx < W) {
      shm[shm_idx] = __ldg(input + input_idx);
    } else {
      shm[shm_idx] = min;
    }
  }

  __syncthreads();

  for (int i = hw_start_idx_of_thread; i < block_h * block_w; i += block_ch) {
    const int out_h_offset = i / block_w;
    const int out_w_offset = i % block_w;
    const int out_h_idx = out_h_start_idx + out_h_offset;
    const int out_w_idx = out_w_start_idx + out_w_offset;
    if (out_h_idx >= 0 && out_h_idx < HO && out_w_idx >= 0 &&
        out_w_idx < WO) {
      auto max = min;

      const int shm_h_start_idx = out_h_offset * stride;
      const int shm_h_end_idx = shm_h_start_idx + kernel_size;
      const int shm_w_start_idx = out_w_offset * stride;
      const int shm_w_end_idx = shm_w_start_idx + kernel_size;

      for (int shm_h_idx = shm_h_start_idx; shm_h_idx < shm_h_end_idx;
           shm_h_idx++) {
        #pragma unroll
        for (int shm_w_idx = shm_w_start_idx; shm_w_idx < shm_w_end_idx;
             shm_w_idx++) {
          const int shm_idx =
              (shm_h_idx * ldg_w + shm_w_idx) * C + ch_thread_idx;
          const auto tmp = shm[shm_idx];
          max.x = (tmp.x > max.x) ? tmp.x : max.x;
          max.y = (tmp.y > max.y) ? tmp.y : max.y;
        }
      }
      output[((n_idx * HO + out_h_idx) * WO + out_w_idx) * C +
             ch_thread_idx] = max;
    }
  }
}

template <typename ElemT, int kernel_size, int stride, int pad>
void max_pooling_launcher(const ElemT* input,
                          ElemT* output,
                          int NI,
                          int HI,
                          int WI,
                          int CI,
                          int HO,
                          int WO,
                          cudaStream_t stream)
{
  const int block_ch = 4;
  const int block_w = 4;
  const int block_h = 4;
  const size_t shm_size = ((block_h - 1) * stride + kernel_size) *
                          ((block_w - 1) * stride + kernel_size) * CI *
                          sizeof(ElemT);
  dim3 grid(NI, (HO + block_h - 1) / block_h,
            (WO + block_w - 1) / block_w);
  dim3 block(CI / 2, block_ch);
  auto kernel_func = max_pool_nhwc_kernel<kernel_size, stride, pad, 4, 4, 4>;
  cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
  max_pool_nhwc_kernel<kernel_size, stride, pad, 4, 4, 4>
      <<<grid, block, shm_size, stream>>>(input, output, NI, HI,
                                          WI, CI / 2, HO, WO);
}
} // namespace

void max_pool2d_2 (
    const void* in_ptr,
    void* out_ptr,
    int64_t* batch,
    int64_t* in_h,
    int64_t* in_w,
    int64_t* in_ch,
    int64_t* out_batch,
    int64_t* out_h,
    int64_t* out_w,
    cudaStream_t stream
) {
  
  int64_t NI = *batch;
  int64_t HI = *in_h;
  int64_t WI = *in_w;
  int64_t CI = *in_ch;
  int64_t CO = *in_ch;
  int64_t KH = 3;
  int64_t KW = 3;
  int64_t SH = 2;
  int64_t SW = 2;
  int64_t PH = 1;
  int64_t PW = 1;
  int64_t NO = NI;
  int64_t HO = (HI + PH + PH - KH) / SH + 1;
  int64_t WO = (WI + PW + PW - KW) / SW + 1;
  *out_batch = NO;
  *out_h = HO;
  *out_w = WO;
  
  if (true) {
    
    max_pooling_launcher<half, 3, 2, 1>(
        static_cast<const half*>(in_ptr),
        static_cast<half*>(out_ptr),
        NI,
        HI,
        WI,
        CI,
        HO,
        WO,
        stream
    );
    return;
  }
  throw std::runtime_error(
      "Unsupported workload for this max pool2d specialization."
  );
}