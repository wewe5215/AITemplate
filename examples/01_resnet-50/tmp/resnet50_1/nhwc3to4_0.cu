
#include "logging.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"

// fast kernel for c_in = 3 & c_out = 4
template <typename Tio, typename Telement, int element_in_Tio>
__global__ void nhwc_padding_channel_3To4_kernel(const int32_t n,
                                                 const int32_t h,
                                                 const int32_t w,
                                                 const Tio *input,
                                                 Tio *output,
                                                 const int32_t max_output_element,
                                                 const int32_t max_input_element,
                                                 const Tio zero_io,
                                                 const Telement zero_element){
  __shared__ Tio shm[192];
  const int tidx = blockIdx.x * 192 + threadIdx.x;
  const int threadidx = threadIdx.x;

  shm[threadIdx.x] = tidx >= max_input_element ? zero_io : input[tidx];
  __syncthreads();

  const int ouput_offset = blockIdx.x * 256;
  const int lower_bound = max_output_element < ouput_offset + 256 ? max_output_element : ouput_offset + 256;
  for (int i = ouput_offset + threadidx, j = threadidx ; i < lower_bound ; i+=192, j+=192)
  {
    const Telement* shm_element = (const Telement*)shm + j*3*element_in_Tio/4;
    Telement array[element_in_Tio];
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0 ; k < element_in_Tio ; k++)
      array[k] = ((k+1)%4 == 0) ? zero_element : shm_element[(k > 3) ? (k - 1) : k];
    output[i] = *((const Tio *)array);
  }
}

template <typename ElemT>
void nhwc3to4_launcher(const ElemT* in_ptr,
                       ElemT* out_ptr,
                       int NI,
                       int HI,
                       int WI,
                       cudaStream_t stream) {
  dim3 block(192);
  const int nhw = NI * HI * WI;
  const int nhwc = nhw * 3;
  CHECK_EQ(nhw % 8, 0);
  const int element_in_Tio = sizeof(int4) / sizeof(ElemT);
  const int max_input_element = nhwc / element_in_Tio;
  const int max_output_element = nhw * 4 / element_in_Tio;
  const int4 zero_io = {0, 0, 0, 0};
  const ElemT zero_element = static_cast<ElemT>(0.0f);
  dim3 grid((nhwc + 192 * element_in_Tio - 1)/(192 * element_in_Tio));
  nhwc_padding_channel_3To4_kernel<int4, ElemT, element_in_Tio><<<grid, block, 0, stream>>>
          (NI, HI, WI,
          (const int4 *)in_ptr,
          (int4 *)out_ptr,
          max_output_element,
          max_input_element,
          zero_io,
          zero_element);
}

void nhwc3to4_0 (
    void* in_ptr,
    void* out_ptr,
    int64_t* batch,
    int64_t* in_h,
    int64_t* in_w,
    int64_t* out_batch,
    int64_t* out_h,
    int64_t* out_w,
    cudaStream_t stream
) {
  
  int64_t NI = *batch;
  int64_t HI = *in_h;
  int64_t WI = *in_w;
  int64_t NO = NI;
  int64_t HO = HI;
  int64_t WO = WI;
  int64_t CO = 4;
  *out_batch = NO;
  *out_h = HO;
  *out_w = WO;
  
nhwc3to4_launcher<half>(
    static_cast<const half*>(in_ptr),
    static_cast<half*>(out_ptr),
    NI,
    HI,
    WI,
    stream
);
return;
}
