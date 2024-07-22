
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"

namespace {

template <int kernel_size, int stride, int padding>
__global__ void avg_pool_nhwc_kernel(const half* input_raw,
                                     half* output_raw,
                                     const int N,
                                     const int H,
                                     const int W,
                                     const int C,
                                     const int HO,
                                     const int WO) {

  const half2* input = (const half2*)input_raw;
  half2* output = (half2*)output_raw;

  const int tid = threadIdx.x;
  const int n_idx = blockIdx.x;
  const int out_h_idx = blockIdx.y;
  const int out_w_idx = blockIdx.z;

  int h_start_idx = out_h_idx * stride - padding;
  int h_end_idx = h_start_idx + kernel_size;
  h_start_idx = (h_start_idx < 0) ? 0 : h_start_idx;
  h_end_idx = h_end_idx > H ? H : h_end_idx;

  int w_start_idx = out_w_idx * stride - padding;
  int w_end_idx = w_start_idx + kernel_size;
  w_start_idx = (w_start_idx < 0) ? 0 : w_start_idx;
  w_end_idx = w_end_idx > W ? W : w_end_idx;

  input += n_idx * H * W * C;
  output += ((n_idx * HO + out_h_idx) * WO + out_w_idx) * C;
  const float norm_factor =
      static_cast<float>(1.0f / (kernel_size * kernel_size));
  for (int c_idx = tid; c_idx < C; c_idx += blockDim.x) {
    float2 avg = {0.f, 0.f};
    for (int h = h_start_idx; h < h_end_idx; h++) {
      #pragma unroll
      for (int w = w_start_idx; w < w_end_idx; w++) {
        const int idx = (h * W + w) * C;
        const half2 tmp = __ldg(input + (idx + c_idx));

        avg.x += __half2float(tmp.x);
        avg.y += __half2float(tmp.y);

      }
    }

    avg.x *= norm_factor;
    avg.y *= norm_factor;

    output[c_idx] = __float22half2_rn(avg);

  }
}

template <typename ElemT, int kernel_size, int stride, int padding>
void avg_pool_launcher(const ElemT* input,
                       ElemT* output,
                       const int N,
                       const int H,
                       const int W,
                       const int C,
                       const int HO,
                       const int WO,
                       cudaStream_t stream)
{
  int num_thread = C / 2;
  if (num_thread > 256) {
      num_thread = 256;
  } else if (num_thread == 0) {
      num_thread = 1;
  }
  dim3 grid(N, HO, WO);
  dim3 block(num_thread);
  avg_pool_nhwc_kernel<kernel_size, stride, padding>
      <<<grid, block, 0, stream>>>(input, output, N, H,
                                   W, C / 2, HO, WO);
}
} // namespace

void avg_pool2d_55 (
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
  int64_t KH = 7;
  int64_t KW = 7;
  int64_t SH = 1;
  int64_t SW = 1;
  int64_t PH = 0;
  int64_t PW = 0;
  int64_t NO = NI;
  int64_t HO = (HI + PH + PH - KH) / SH + 1;
  int64_t WO = (WI + PW + PW - KW) / SW + 1;
  *out_batch = NO;
  *out_h = HO;
  *out_w = WO;
  
  if (true) {
    
    avg_pool_launcher<half, 7, 1, 0>(
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
      "Unsupported workload for this avg pool2d specialization."
  );
}