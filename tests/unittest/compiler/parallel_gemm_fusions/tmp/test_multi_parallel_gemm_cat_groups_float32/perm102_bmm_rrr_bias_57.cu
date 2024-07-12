
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <iostream>
#include <cuda_bf16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/device_memory.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"

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



  // Gemm operator cutlass_tensorop_s1688tf32gemm_64x64_32x5_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_64x64_32x5_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    5,
    4,
    4,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using fdc627c249f16833f2e9ae7e01b9fd07d08be1a87 = Operation_cutlass_tensorop_s1688tf32gemm_64x64_32x5_tt_align4;

void perm102_bmm_rrr_bias_57 (
    void* a_ptr,
    void* b_ptr,
    void* bias_ptr,
    void* c_ptr,
    uint8_t* workspace,
    int64_t* a_dim0,
    int64_t* a_dim1,
    int64_t* a_dim2,
    int64_t* b_dim0,
    int64_t* b_dim1,
    int64_t* b_dim2,
    int64_t* c_dim0,
    int64_t* c_dim1,
    int64_t* c_dim2,
  cudaStream_t stream
  ) {
  
 int64_t B = (*c_dim1);

 int64_t M = (*a_dim0);

 int64_t N = (*b_dim2);

 int64_t K = (*a_dim2);
  
  
  int64_t output_batch_stride = N;
  int64_t output_stride = 1280;
  int64_t output_offset = 256; // default to 0
    
  
  
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

  if (!bias_ptr) {
    throw std::runtime_error("bias_ptr is null!");
  }

  
  if (B == 4 && M == 256 && N == 128 && K == 120) {
    
//  TODO: cast to right dtype
    using ElementComputeEpilogue = typename fdc627c249f16833f2e9ae7e01b9fd07d08be1a87::ElementAccumulator;

    using coord_t = cutlass::gemm::GemmCoord::Index;
    typename fdc627c249f16833f2e9ae7e01b9fd07d08be1a87::Arguments arguments;

    if constexpr (cutlass::gemm::detail::IsCutlass3GemmKernel<typename fdc627c249f16833f2e9ae7e01b9fd07d08be1a87::GemmKernel>::value) {
    arguments = {

    };
    } else {
    arguments = {

    cutlass::gemm::GemmUniversalMode::kBatched,                                                         // GemmUniversalMode mode
    {
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K)
    },                                                                                                  // GemmCoord problem_size
    B,                                                                             // int batch_count
    {ElementComputeEpilogue(1.0), ElementComputeEpilogue(1)},  // typename EpilogueOutputOp::Params epilogue
    (a_ptr),                                                                                  // void const * ptr_A
    (b_ptr),                                                                                  // void const * ptr_B
    (bias_ptr),                                                                               // void const * ptr_C
    (float*)(c_ptr) + output_offset,                                                                                  // void * ptr_D
    K,                                                                         // int64_t batch_stride_A
    N * K,                                                                         // int64_t batch_stride_B
    N,                                                                      // int64_t batch_stride_C
    output_batch_stride,                                                                         // int64_t batch_stride_D
    K * B,                                                                                    // typename LayoutA::Stride::LongIndex lda
    N,                                                                                    // typename LayoutB::Stride::LongIndex ldb
    0,                                                                                 // typename LayoutC::Stride::LongIndex ldc
    output_stride,                                                                                    // typename LayoutC::Stride::LongIndex ldd
    };
    }


    fdc627c249f16833f2e9ae7e01b9fd07d08be1a87 gemm_op;

    auto status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);
    status = gemm_op.initialize(arguments, workspace, stream);
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
    return;
  }
      std::cout << "input_ndims0: " << *a_dim0 << std::endl;
      std::cout << "input_ndims1: " << *a_dim1 << std::endl;
      std::cout << "input_ndims2: " << *a_dim2 << std::endl;
      std::cout << "weight_ndims0: " << *b_dim0 << std::endl;
      std::cout << "weight_ndims1: " << *b_dim1 << std::endl;
      std::cout << "weight_ndims2: " << *b_dim2 << std::endl;
      std::cout << "output_ndims0: " << *c_dim0 << std::endl;
      std::cout << "output_ndims1: " << *c_dim1 << std::endl;
      std::cout << "output_ndims2: " << *c_dim2 << std::endl;
  throw std::runtime_error(
      "Unsupported workload for this perm102_bmm_rrr_bias_57 specialization."
  );
}