
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_residual_block.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/device_memory.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/collective/epilogue_tensor_broadcast.hpp"
#include "cutlass/epilogue/thread/linear_combination_tensor_broadcast.hpp"

using bfloat16 = nv_bfloat16;

#define CUTLASS_CHECK(status)                                                         \
  {                                                                                   \
    cutlass::Status error = status;                                                   \
    if (error != cutlass::Status::kSuccess) {                                         \
      auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +              \
          cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);         \
      std::cerr << msg << std::endl;                                                  \
      throw std::runtime_error(msg);                                                  \
    }                                                                                 \
  }



  // Gemm operator cutlass_tensorop_h16816gemm_64x64_32x10_tn_align8
  using Operation_cutlass_tensorop_h16816gemm_64x64_32x10_tn_align8 = 
    cutlass::gemm::device::GemmUniversalWithBroadcast<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<64, 64, 32>,
            cutlass::gemm::GemmShape<32, 32, 32>,
            cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombinationResidualBlock<
            cutlass::half_t, cutlass::half_t, cutlass::half_t,
            cutlass::half_t,       8,
            cutlass::epilogue::thread::Identity, cutlass::plus, cutlass::epilogue::thread::Identity

        >,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
            10,
            8,
            8
    >;

using f2c7cabe06db5f8ef8025d111d3591271c96fb081 = Operation_cutlass_tensorop_h16816gemm_64x64_32x10_tn_align8;

void gemm_rcr_bias_add_4 (
    void* a_ptr,
    void* b_ptr,
    void* bias_ptr,
    void* d0_ptr,
    void* c_ptr,
    uint8_t* workspace,
    int split_k,
    int64_t* a_dim0,
    int64_t* a_dim1,
    int64_t* b_dim0,
    int64_t* b_dim1,
    int64_t* c_dim0,
    int64_t* c_dim1,
    cudaStream_t stream
  ) {
  
 int64_t M = (*a_dim0);

 int64_t N = (*b_dim0);

 int64_t K = (*a_dim1);
  
  int64_t input_a_batch_stride = M * K;
  int64_t input_a_stride = K;
  int64_t input_a_offset = 0; // default to 0
  int64_t input_b_batch_stride = N * K;
  int64_t input_b_stride = K;
  int64_t input_b_offset = 0; // default to 0
    
  
  
  int64_t output_stride = N;
  int64_t output_offset = 0;
  
    
  
  
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

  if (!bias_ptr) {
    throw std::runtime_error("bias is null!");
  }
  if (!d0_ptr) {
    throw std::runtime_error("d0_ptr is null!");
  }

  
  if (M == 64 && N == 768 && K == 768) {
    
//  TODO: cast to right dtype
    using ElementComputeEpilogue = typename f2c7cabe06db5f8ef8025d111d3591271c96fb081::ElementAccumulator;

    using coord_t = cutlass::gemm::GemmCoord::Index;
    typename f2c7cabe06db5f8ef8025d111d3591271c96fb081::Arguments arguments;

    if constexpr (cutlass::gemm::detail::IsCutlass3GemmKernel<typename f2c7cabe06db5f8ef8025d111d3591271c96fb081::GemmKernel>::value) {
    arguments = {

    };
    } else {
    arguments = {

    cutlass::gemm::GemmUniversalMode::kGemm,                 // GemmUniversalMode mode
    {
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K)
    },                                                       // GemmCoord problem_size

    split_k,                                                 // int batch_count

    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},  // typename EpilogueOutputOp::Params epilogue
    (cutlass::half_t*)(a_ptr) + input_a_offset,          // void const * ptr_A
    (cutlass::half_t*)(b_ptr) + input_b_offset,          // void const * ptr_B
    (cutlass::half_t*)(d0_ptr),                         // void const * ptr_C1

    (cutlass::half_t*) (c_ptr) + output_offset,         // void * ptr_D
    (cutlass::half_t*) (bias_ptr),                       // void * ptr_Vector
    nullptr,                                                 // void * ptr_Tensor
    input_a_batch_stride,                                    // int64_t batch_stride_A
    input_b_batch_stride,                                    // int64_t batch_stride_B
    0,                                                       // int64_t batch_stride_C1

    0,                                                       // int64_t batch_stride_D
    0,                                                       // int64_t batch_stride_Vector
    0,                                                       // int64_t batch_stride_Tensor
    input_a_stride,                                          // typename LayoutA::Stride::Index lda
    input_b_stride,                                          // typename LayoutB::Stride::Index ldb
    N,                                     // typename LayoutC::Stride::Index ldc1

    output_stride,                                           // typename LayoutC::Stride::Index ldd
    0,                                                       // typename LayoutC::Stride::Index ldr
    0,                                                       // typename LayoutC::Stride::Index ldt
    };
    }


    f2c7cabe06db5f8ef8025d111d3591271c96fb081 gemm_op;

    auto status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);
    status = gemm_op.initialize(arguments, workspace, stream);
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
    return;
  }
  throw std::runtime_error(
      "Unsupported workload for this gemm_rcr_bias_add_4 specialization."
  );
}