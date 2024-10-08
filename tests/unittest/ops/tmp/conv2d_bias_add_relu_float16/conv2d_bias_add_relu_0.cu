
#include <cstdio>
#include <stdexcept>

#include "cutlass/cutlass.h"

#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/default_conv2d_group_fprop.h"

#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"


#include <cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h>
#include <cutlass/epilogue/thread/linear_combination_residual_block.h>


#define CUTLASS_CHECK(status)                                                         \
  {                                                                                   \
    cutlass::Status error = status;                                                   \
    if (error != cutlass::Status::kSuccess) {                                         \
      static char msg[2048];                                                          \
      snprintf(msg, sizeof(msg), "[%s] Got cutlass error: %s at: %s",                 \
        __FILE__, cutlassGetStatusString(error), __LINE__);                           \
      fprintf(stderr, msg);                                                           \
      throw std::runtime_error(msg);                                                  \
    }                                                                                 \
  }



    
  using cutlass_tensorop_f16_s16816fprop_optimized_f16_64x64_32x10_nhwc_single_group_align8_base =
  typename cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64,
                             64,
                             32>,
    cutlass::gemm::GemmShape<32,
                             32,
                             32 >,
    cutlass::gemm::GemmShape<16,
                             8,
                             16>,
    cutlass::epilogue::thread::LinearCombinationResidualBlock<
      cutlass::half_t,
      float,
      float,
      cutlass::half_t,
      8,
      cutlass::epilogue::thread::Identity,
      cutlass::plus,
      cutlass::epilogue::thread::ReLu
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    10,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided,
    8,
    8
  >::Kernel;
  
using f7de63d7392b0aee95c9eaf9b5770eaa922e5396c = cutlass::conv::device::ImplicitGemmConvolution<cutlass_tensorop_f16_s16816fprop_optimized_f16_64x64_32x10_nhwc_single_group_align8_base>;


void conv2d_bias_add_relu_0 (
    void* in_ptr,
    void* weight_ptr,
    void* out_ptr,

    void* bias_ptr,
    void* res_ptr,

    uint8_t* workspace,
    int64_t* batch,
    int64_t* out_ch,
    int64_t* in_ch,
    int64_t* kernel_h,
    int64_t* kernel_w,
    int64_t* in_h,
    int64_t* in_w,
    int64_t* out_batch,
    int64_t* out_h,
    int64_t* out_w,
    int strideh,
    int dilationh,
    int padh,
    int stridew,
    int dilationw,
    int padw,
    cudaStream_t stream
  ) {

  
  int64_t NI = *batch;
  int64_t HI = *in_h;
  int64_t WI = *in_w;
  int64_t CI = *in_ch;
  int64_t CO = *out_ch;
  int64_t KH = *kernel_h;
  int64_t KW = *kernel_w;
  int64_t SH = strideh;
  int64_t SW = stridew;
  int64_t DH = dilationh;
  int64_t DW = dilationw;
  int64_t PH = padh;
  int64_t PW = padw;
  int64_t KHEff = (KH - 1) * DH + 1;
  int64_t KWEff = (KW - 1) * DW + 1;
  int64_t NO = NI;
  int64_t HO = (HI + PH + PH - KHEff) / SH + 1;
  int64_t WO = (WI + PW + PW - KWEff) / SW + 1;
  *out_batch = NO;
  *out_h = HO;
  *out_w = WO;
  *out_ch = CO;

  int i32_batch = *batch;
  int i32_in_h = *in_h;
  int i32_in_w = *in_w;
  int i32_in_ch = *in_ch;
  int i32_out_ch = *out_ch;
  int i32_kernel_h = *kernel_h;
  int i32_kernel_w = *kernel_w;
  int i32_out_batch = *out_batch;
  int i32_out_h = *out_h;
  int i32_out_w = *out_w;

  using cutlass::layout::TensorNHWC;
  TensorNHWC layout_A(TensorNHWC::packed(cutlass::make_Coord(i32_batch, i32_in_h, i32_in_w, i32_in_ch)));

  TensorNHWC layout_B(TensorNHWC::packed(cutlass::make_Coord(i32_out_ch, i32_kernel_h, i32_kernel_w, i32_in_ch)));

  TensorNHWC layout_C(TensorNHWC::packed(cutlass::make_Coord(i32_out_batch, i32_out_h, i32_out_w, i32_out_ch)));

  cutlass::conv::Conv2dProblemSize problem_size(

    {i32_batch, i32_in_h, i32_in_w, i32_in_ch},           // cutlass::Tensor4DCoord input_size


    {i32_out_ch, i32_kernel_h, i32_kernel_w, i32_in_ch},  // cutlass::Tensor4DCoord filter_size

    {padh, padh, padw, padw},                                 // cutlass::Tensor4DCoord padding
    {strideh, stridew},                                     // cutlass::MatrixCoord stride
    {dilationh, dilationw},                                 // cutlass::MatrixCoord dilation

    {i32_out_batch, i32_out_h, i32_out_w, i32_out_ch},    // cutlass::Tensor4DCoord output_size

    cutlass::conv::Mode::kCrossCorrelation,               // cutlass::conv::Mode mode
    1                                                     // int split_k_slices
  );

  
  if (NI == 4 && HI == 28 && WI == 28 && CI == 128) {
    
    using ElementComputeEpilogue = typename f7de63d7392b0aee95c9eaf9b5770eaa922e5396c::ElementCompute;
    //  TODO: cast to right dtype
    typename f7de63d7392b0aee95c9eaf9b5770eaa922e5396c::Arguments arguments{
        problem_size,                                                                 // ConvProblemSize const & problem_size
        {static_cast<cutlass::half_t*>(in_ptr), layout_A},                                  // TensorRefA const & ref_A
        {static_cast<cutlass::half_t*>(weight_ptr), layout_B},                              // TensorRefA const & ref_B

        {static_cast<cutlass::half_t*>(res_ptr), layout_C},                                 // TensorRefC const & ref_C

        {static_cast<cutlass::half_t*>(out_ptr), layout_C},                                 // TensorRefC const & ref_D

        {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},                       // typename EpilogueOutputOp::Params const & output_op
        cutlass::conv::SplitKMode::kSerial,                                           // SplitKMode const & split_k_mode
        static_cast<cutlass::half_t*>(bias_ptr),                                            // void * ptr_Vector
        nullptr,                                                                      // void * ptr_Tensor
        0,                                                                            // typename LayoutC::Stride::Index ldr
        *out_ch,                                                                      // typename LayoutC::Stride::Index ldt

    };
    f7de63d7392b0aee95c9eaf9b5770eaa922e5396c conv_op;

    auto status = conv_op.can_implement(arguments);
    CUTLASS_CHECK(status);
    status = conv_op.initialize(arguments, workspace);
    CUTLASS_CHECK(status);
    status = conv_op(stream);
    CUTLASS_CHECK(status);
    return;
  }

  throw std::runtime_error(
    "Unsupported workload for this conv2d specialization."
  );
}