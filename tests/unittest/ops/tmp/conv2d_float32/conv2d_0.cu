
#include <cstdio>
#include <stdexcept>

#include "cutlass/cutlass.h"

#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/default_conv2d_group_fprop.h"

#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"



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



  // Conv2dFprop Optimized kernel instance "cutlass_tensorop_s1688tf32fprop_optimized_64x64_16x10_nhwc_single_group_align4"
  using cutlass_tensorop_s1688tf32fprop_optimized_64x64_16x10_nhwc_single_group_align4_base =
  typename cutlass::conv::kernel::DefaultConv2dGroupFprop<
    float,
    cutlass::layout::TensorNHWC,
    float,
    cutlass::layout::TensorNHWC,
    float,
    cutlass::layout::TensorNHWC,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16 >,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    10,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::GroupMode::kSingleGroup,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided,
    4,
    4
  >::Kernel;

using f7de63d7392b0aee95c9eaf9b5770eaa922e5396c = cutlass::conv::device::ImplicitGemmConvolution<cutlass_tensorop_s1688tf32fprop_optimized_64x64_16x10_nhwc_single_group_align4_base>;


void conv2d_0 (
    void* in_ptr,
    void* weight_ptr,
    void* out_ptr,

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
        {static_cast<float*>(in_ptr), layout_A},                                  // TensorRefA const & ref_A
        {static_cast<float*>(weight_ptr), layout_B},                              // TensorRefA const & ref_B

        {static_cast<float*>(out_ptr), layout_C},                                 // TensorRefC const & ref_C

        {static_cast<float*>(out_ptr), layout_C},                                 // TensorRefC const & ref_D

        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},                       // typename EpilogueOutputOp::Params const & output_op

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