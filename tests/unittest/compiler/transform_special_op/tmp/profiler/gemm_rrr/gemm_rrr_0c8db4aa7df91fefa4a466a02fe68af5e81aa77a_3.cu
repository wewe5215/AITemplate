
size_t GLOBAL_WORKSPACE_SIZE = 0;

#include <sstream>


#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
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



  // Gemm operator cutlass_tensorop_s1688tf32gemm_256x128_16x3_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_256x128_16x3_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_0 = Operation_cutlass_tensorop_s1688tf32gemm_256x128_16x3_tt_align4;


  // Gemm operator cutlass_tensorop_s1688tf32gemm_128x256_16x3_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_128x256_16x3_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_1 = Operation_cutlass_tensorop_s1688tf32gemm_128x256_16x3_tt_align4;


  // Gemm operator cutlass_tensorop_s1688tf32gemm_256x64_16x4_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_256x64_16x4_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    4,
    4,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_2 = Operation_cutlass_tensorop_s1688tf32gemm_256x64_16x4_tt_align4;


  // Gemm operator cutlass_tensorop_s1688tf32gemm_64x256_16x4_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_64x256_16x4_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    4,
    4,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_3 = Operation_cutlass_tensorop_s1688tf32gemm_64x256_16x4_tt_align4;


  // Gemm operator cutlass_tensorop_s1688tf32gemm_128x128_16x5_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_128x128_16x5_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
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

using GemmInstance_4 = Operation_cutlass_tensorop_s1688tf32gemm_128x128_16x5_tt_align4;


  // Gemm operator cutlass_tensorop_s1688tf32gemm_128x128_16x4_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_128x128_16x4_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    4,
    4,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_5 = Operation_cutlass_tensorop_s1688tf32gemm_128x128_16x4_tt_align4;


  // Gemm operator cutlass_tensorop_s1688tf32gemm_128x128_16x3_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_128x128_16x3_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_6 = Operation_cutlass_tensorop_s1688tf32gemm_128x128_16x3_tt_align4;


  // Gemm operator cutlass_tensorop_s1688tf32gemm_128x64_16x6_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_128x64_16x6_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 16>,
    cutlass::gemm::GemmShape<64, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    6,
    4,
    4,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_7 = Operation_cutlass_tensorop_s1688tf32gemm_128x64_16x6_tt_align4;


  // Gemm operator cutlass_tensorop_s1688tf32gemm_64x128_16x6_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_64x128_16x6_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    6,
    4,
    4,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_8 = Operation_cutlass_tensorop_s1688tf32gemm_64x128_16x6_tt_align4;


  // Gemm operator cutlass_tensorop_s1688tf32gemm_64x64_16x10_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_64x64_16x10_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    10,
    4,
    4,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_9 = Operation_cutlass_tensorop_s1688tf32gemm_64x64_16x10_tt_align4;


  // Gemm operator cutlass_tensorop_s1688tf32gemm_256x128_32x3_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_256x128_32x3_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_10 = Operation_cutlass_tensorop_s1688tf32gemm_256x128_32x3_tt_align4;


  // Gemm operator cutlass_tensorop_s1688tf32gemm_128x256_32x3_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_128x256_32x3_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_11 = Operation_cutlass_tensorop_s1688tf32gemm_128x256_32x3_tt_align4;


  // Gemm operator cutlass_tensorop_s1688tf32gemm_256x64_32x4_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_256x64_32x4_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    4,
    4,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_12 = Operation_cutlass_tensorop_s1688tf32gemm_256x64_32x4_tt_align4;


  // Gemm operator cutlass_tensorop_s1688tf32gemm_64x256_32x4_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_64x256_32x4_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    4,
    4,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_13 = Operation_cutlass_tensorop_s1688tf32gemm_64x256_32x4_tt_align4;


  // Gemm operator cutlass_tensorop_s1688tf32gemm_128x128_32x4_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_128x128_32x4_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    4,
    4,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_14 = Operation_cutlass_tensorop_s1688tf32gemm_128x128_32x4_tt_align4;


  // Gemm operator cutlass_tensorop_s1688tf32gemm_128x128_32x3_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_128x128_32x3_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_15 = Operation_cutlass_tensorop_s1688tf32gemm_128x128_32x3_tt_align4;


  // Gemm operator cutlass_tensorop_s1688tf32gemm_128x64_32x3_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_128x64_32x3_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_16 = Operation_cutlass_tensorop_s1688tf32gemm_128x64_32x3_tt_align4;


  // Gemm operator cutlass_tensorop_s1688tf32gemm_64x128_32x3_tt_align4
  using Operation_cutlass_tensorop_s1688tf32gemm_64x128_32x3_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_17 = Operation_cutlass_tensorop_s1688tf32gemm_64x128_32x3_tt_align4;


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

using GemmInstance_18 = Operation_cutlass_tensorop_s1688tf32gemm_64x64_32x5_tt_align4;


  // Gemm operator cutlass_tensorop_s1688gemm_128x128_16x4_tt_align4
  using Operation_cutlass_tensorop_s1688gemm_128x128_16x4_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    4,
    4,
    
    cutlass::arch::OpMultiplyAddFastF32
    
  >;

using GemmInstance_19 = Operation_cutlass_tensorop_s1688gemm_128x128_16x4_tt_align4;


  // Gemm operator cutlass_tensorop_s1688gemm_128x128_16x3_tt_align4
  using Operation_cutlass_tensorop_s1688gemm_128x128_16x3_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    
    cutlass::arch::OpMultiplyAddFastF32
    
  >;

using GemmInstance_20 = Operation_cutlass_tensorop_s1688gemm_128x128_16x3_tt_align4;


  // Gemm operator cutlass_tensorop_s1688gemm_256x64_16x3_tt_align4
  using Operation_cutlass_tensorop_s1688gemm_256x64_16x3_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 16>,
    cutlass::gemm::GemmShape<64, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    
    cutlass::arch::OpMultiplyAddFastF32
    
  >;

using GemmInstance_21 = Operation_cutlass_tensorop_s1688gemm_256x64_16x3_tt_align4;


  // Gemm operator cutlass_tensorop_s1688gemm_64x256_16x3_tt_align4
  using Operation_cutlass_tensorop_s1688gemm_64x256_16x3_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    
    cutlass::arch::OpMultiplyAddFastF32
    
  >;

using GemmInstance_22 = Operation_cutlass_tensorop_s1688gemm_64x256_16x3_tt_align4;


  // Gemm operator cutlass_tensorop_s1688gemm_128x64_16x4_tt_align4
  using Operation_cutlass_tensorop_s1688gemm_128x64_16x4_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 16>,
    cutlass::gemm::GemmShape<64, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    4,
    4,
    
    cutlass::arch::OpMultiplyAddFastF32
    
  >;

using GemmInstance_23 = Operation_cutlass_tensorop_s1688gemm_128x64_16x4_tt_align4;


  // Gemm operator cutlass_tensorop_s1688gemm_64x128_16x4_tt_align4
  using Operation_cutlass_tensorop_s1688gemm_64x128_16x4_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    4,
    4,
    
    cutlass::arch::OpMultiplyAddFastF32
    
  >;

using GemmInstance_24 = Operation_cutlass_tensorop_s1688gemm_64x128_16x4_tt_align4;


  // Gemm operator cutlass_tensorop_s1688gemm_64x64_16x3_tt_align4
  using Operation_cutlass_tensorop_s1688gemm_64x64_16x3_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    
    cutlass::arch::OpMultiplyAddFastF32
    
  >;

using GemmInstance_25 = Operation_cutlass_tensorop_s1688gemm_64x64_16x3_tt_align4;


  // Gemm operator cutlass_tensorop_s1688gemm_128x128_32x3_tt_align4
  using Operation_cutlass_tensorop_s1688gemm_128x128_32x3_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    
    cutlass::arch::OpMultiplyAddFastF32
    
  >;

using GemmInstance_26 = Operation_cutlass_tensorop_s1688gemm_128x128_32x3_tt_align4;


  // Gemm operator cutlass_tensorop_s1688gemm_256x64_32x3_tt_align4
  using Operation_cutlass_tensorop_s1688gemm_256x64_32x3_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    
    cutlass::arch::OpMultiplyAddFastF32
    
  >;

using GemmInstance_27 = Operation_cutlass_tensorop_s1688gemm_256x64_32x3_tt_align4;


  // Gemm operator cutlass_tensorop_s1688gemm_64x256_32x3_tt_align4
  using Operation_cutlass_tensorop_s1688gemm_64x256_32x3_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    
    cutlass::arch::OpMultiplyAddFastF32
    
  >;

using GemmInstance_28 = Operation_cutlass_tensorop_s1688gemm_64x256_32x3_tt_align4;


  // Gemm operator cutlass_tensorop_s1688gemm_128x64_32x3_tt_align4
  using Operation_cutlass_tensorop_s1688gemm_128x64_32x3_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    
    cutlass::arch::OpMultiplyAddFastF32
    
  >;

using GemmInstance_29 = Operation_cutlass_tensorop_s1688gemm_128x64_32x3_tt_align4;


  // Gemm operator cutlass_tensorop_s1688gemm_64x128_32x3_tt_align4
  using Operation_cutlass_tensorop_s1688gemm_64x128_32x3_tt_align4 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    
    cutlass::arch::OpMultiplyAddFastF32
    
  >;

using GemmInstance_30 = Operation_cutlass_tensorop_s1688gemm_64x128_32x3_tt_align4;


  // Gemm operator cutlass_tensorop_s1688gemm_64x64_32x3_tt_align4
  using Operation_cutlass_tensorop_s1688gemm_64x64_32x3_tt_align4 = cutlass::gemm::device::GemmUniversal<
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
    3,
    4,
    4,
    
    cutlass::arch::OpMultiplyAddFastF32
    
  >;

using GemmInstance_31 = Operation_cutlass_tensorop_s1688gemm_64x64_32x3_tt_align4;

template <typename GemmInstance>
void gemm (
    GemmInstance& gemm_op,
    void* a_ptr,
    void* b_ptr,
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

  int64_t N = (*b_dim1);

  int64_t K = (*a_dim1);
  
  
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

  
//  TODO: cast to right dtype
  using ElementComputeEpilogue = typename GemmInstance::ElementAccumulator;

  using coord_t = cutlass::gemm::GemmCoord::Index;
  typename GemmInstance::Arguments arguments;

  if constexpr (cutlass::gemm::detail::IsCutlass3GemmKernel<typename GemmInstance::GemmKernel>::value) {
  arguments = {

    cutlass::gemm::GemmUniversalMode::kGemm,                     // GemmUniversalMode mode
    {
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K),
        static_cast<coord_t>(1)
    },                                                           // ProblemShape problem_shape
    (float*)(a_ptr),                               // ElementA const* ptr_A
    {K, cute::Int<1>{}, cute::Int<0>{}},                         // StrideA dA
    (float*)(b_ptr),                               // ElementB const* ptr_B
    {cute::Int<1>{}, N, cute::Int<0>{}},                         // StrideB dB
    {
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},  // typename ThreadEpilogueOp::Params thread
        nullptr,                                                 // ElementC const* ptr_C
        {N, cute::Int<1>{}, cute::Int<0>{}},                     // StrideC dC
        (float*)(c_ptr) + output_offset,          // ElementD const* ptr_D
        {output_stride, cute::Int<1>{}, cute::Int<0>{}},         // StrideD dD
    },                                                           // EpilogueArguments epilogue
  };
  } else {
  arguments = {

    cutlass::gemm::GemmUniversalMode::kGemm,                 // GemmUniversalMode mode
    cutlass::gemm::GemmCoord{
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K)
    },                                                       // GemmCoord problem_size
    split_k,                                                 // int batch_count
    {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},  // typename EpilogueOutputOp::Params epilogue
    (float*)(a_ptr),                           // void const * ptr_A
    (float*)(b_ptr),                           // void const * ptr_B
    (float*)(c_ptr),                          // void const * ptr_C
    (float*)(c_ptr) + output_offset,          // void * ptr_D
    M * K,                                                   // int64_t batch_stride_A
    N * K,                                                   // int64_t batch_stride_B
    M * N,                                                   // int64_t batch_stride_C
    M * N,                                                   // int64_t batch_stride_D
    K,                                                       // typename LayoutA::Stride::LongIndex lda
    N,                                                       // typename LayoutB::Stride::LongIndex ldb
    N,                                                       // typename LayoutC::Stride::LongIndex ldc
    output_stride,                                           // typename LayoutC::Stride::LongIndex ldd
  };
  }


  size_t workspace_size = gemm_op.get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> local_workspace(workspace_size);
  workspace = local_workspace.get();
  GLOBAL_WORKSPACE_SIZE = workspace_size;

  auto status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace, stream);
  CUTLASS_CHECK(status);
  status = gemm_op(stream);
  CUTLASS_CHECK(status);
  return;
      std::cout << "input_ndims0: " << *a_dim0 << std::endl;
      std::cout << "input_ndims1: " << *a_dim1 << std::endl;
      std::cout << "weight_ndims0: " << *b_dim0 << std::endl;
      std::cout << "weight_ndims1: " << *b_dim1 << std::endl;
      std::cout << "output_ndims0: " << *c_dim0 << std::endl;
      std::cout << "output_ndims1: " << *c_dim1 << std::endl;
  throw std::runtime_error(
      "Unsupported workload for this gemm specialization."
  );
}

template <typename DType>
struct ProfilerMemoryPool;

template <typename GemmInstance>
int benchmark_gemm (


    GemmInstance &gemm_op,
    const char *gemm_op_name,
    ProfilerMemoryPool<float>* memory_pool,
    uint8_t* global_workspace_,

    int split_k,


    int64_t* a_dim0,

    int64_t* a_dim1,


    int64_t* b_dim0,

    int64_t* b_dim1,


    int64_t* c_dim0,

    int64_t* c_dim1,

    cudaStream_t stream

  ) {
  // warmup
  for (int i = 0; i < 5; ++i) {
    
{

gemm(

    gemm_op,

    memory_pool->RequestTensorByIdx(0),
    memory_pool->RequestTensorByIdx(1),

    memory_pool->RequestTensorByIdx(2),
    global_workspace_,
    split_k,

    a_dim0,

    a_dim1,


    b_dim0,

    b_dim1,


    c_dim0,

    c_dim1,

    stream
);
}
  }
  cudaEvent_t events[2];
  for (auto & event : events) {
    cudaEventCreate(&event);
  }
  cudaEventRecord(events[0], stream);
  for (int i = 0; i < 10; ++i) {
    
{

gemm(

    gemm_op,

    memory_pool->RequestTensorByIdx(0),
    memory_pool->RequestTensorByIdx(1),

    memory_pool->RequestTensorByIdx(2),
    global_workspace_,
    split_k,

    a_dim0,

    a_dim1,


    b_dim0,

    b_dim1,


    c_dim0,

    c_dim1,

    stream
);
}
  }
  cudaEventRecord(events[1], stream);
  cudaEventSynchronize(events[1]);
  float runtime_ms = 0;
  cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }
  // TODO: output workspace
  if (runtime_ms < 0.00001) {
      throw std::runtime_error(
      "OOB in cutlass."
    );
  }
  std::cout << "OP:" << gemm_op_name << ",";
  std::cout << "TIME:" << runtime_ms << ",";
  std::cout << "WS:" << GLOBAL_WORKSPACE_SIZE << std::endl;
  return 0;
}

template <typename DType>
struct ProfilerMemoryPool {
  ProfilerMemoryPool() : shared_input_tensor(false) {
    std::random_device rd;
    gen = std::mt19937(rd());
    uniform_dist = std::uniform_int_distribution<int64_t>(1, 48964896);
    offsets.reserve(512);
    strides.reserve(512);
    copies.reserve(512);
    ptrs.reserve(512);
    blobs.reserve(512);
  }
  ~ProfilerMemoryPool() {}

  int64_t ComputeMemPoolSize(size_t one_copy_sz, size_t ptr_max_sz, size_t l2_cache_bytes) {
    int times_covers_l2_cache = (int)std::ceil(l2_cache_bytes / sizeof(DType) / ptr_max_sz);
    int64_t mem_pool_sz = std::max(2, std::min(512, times_covers_l2_cache));
    size_t free_global_mem = 0;
    size_t total_global_mem = 0;
    cudaError_t cuda_error = cudaMemGetInfo(&free_global_mem, &total_global_mem);
    if (cuda_error != cudaSuccess) {
      auto error_msg = std::string("Failed to invoke cudaMemGetInfo: ") +
          cudaGetErrorName(cuda_error) + ", at " + __FILE__;
      throw std::runtime_error(error_msg);
    }
    size_t single_copy_nbytes = one_copy_sz * sizeof(DType);
    while (mem_pool_sz > 0) {
      size_t nbytes = single_copy_nbytes * mem_pool_sz;
      if (nbytes < free_global_mem) {
        break;
      }
      mem_pool_sz--;
    }

    if (mem_pool_sz <= 1) {
      size_t minimal_required_nbytes = ptr_max_sz * sizeof(DType);
      if (minimal_required_nbytes > free_global_mem) {
        // We absolutely run out of memory
        auto error_msg = std::string("no enough GPU memory: requested ") +
            std::to_string(minimal_required_nbytes) + ", available: " +
            std::to_string(free_global_mem) + ", ptr_max_sz: " +
            std::to_string(ptr_max_sz) + ", at " + __FILE__;
        throw std::runtime_error(error_msg);
      } else {
        // Let's try to allocate a single blob that is large enough to hold
        // all input tensors. Note that this is still an approximation, because
        // we may still hit cudaErrorMemoryAllocation error while allocating
        // memory for the output. We will rely on cudaMalloc to throw out
        // an exception in such a case.
        shared_input_tensor = true;
        AllocateGaussianTensor(ptr_max_sz);
      }
      return 1;
    }
    return mem_pool_sz;
  }

  DType* AllocateGaussianTensor(int64_t size) {
    size_t length = size * sizeof(DType);
    blobs.emplace_back(length);
    DType* ptr = reinterpret_cast<DType*>(blobs.back().get());

    uint64_t seed = uniform_dist(gen);
    double mean = 0.f;
    double std = 1.f;

    cutlass::reference::device::BlockFillRandomGaussian(ptr, size, seed, mean,
                                                        std);

    return ptr;
  }

  int AllocateTensor(int64_t size, int64_t copy, bool is_output = false) {
    offsets.push_back(0);
    strides.push_back(size);
    copies.push_back(copy);
    DType *ptr;
    if (!is_output && shared_input_tensor) {
      ptr = reinterpret_cast<DType*>(blobs.back().get());
    } else {
      ptr = AllocateGaussianTensor(size * copy);
    }
    ptrs.push_back(reinterpret_cast<void*>(ptr));
    return ptrs.size() - 1;
  }

  DType* RequestTensorByIdx(int idx) {
    auto copy = copies.at(idx);
    auto offset = offsets.at(idx);
    auto stride = strides.at(idx);
    DType* ptr = reinterpret_cast<DType*>(ptrs.at(idx));
    ptr += offset;
    offset += stride;
    if (offset == copy * stride) {
        offset = 0;
    }
    offsets[idx] = offset;
    return ptr;
  }

  std::vector<int64_t> offsets;
  std::vector<int64_t> strides;
  std::vector<int64_t> copies;
  std::vector<void*> ptrs;
  std::vector<cutlass::DeviceAllocation<uint8_t> > blobs;
  std::mt19937 gen;
  std::uniform_int_distribution<int64_t> uniform_dist;
  // make a shared blob to hold all inputs in cases we do not have
  // enough GPU memory
  bool shared_input_tensor;
};


int main(int argc, char** argv) {
  int device_idx;
  cudaDeviceProp device_properties;
  cudaError_t result = cudaGetDevice(&device_idx);
  auto memory_pool = std::make_unique<ProfilerMemoryPool<float>>();
  if (result != cudaSuccess) {
    std::ostringstream errorStream;
    errorStream << "cudaGetDevice() call failed! "
                << "Error code: " << cudaGetErrorName(result)
                << " Error message: " << cudaGetErrorString(result);
    throw std::runtime_error(errorStream.str());
  }

  result = cudaGetDeviceProperties(&device_properties, device_idx);

  if (result != cudaSuccess) {
    std::ostringstream errorStream;
    errorStream << "cudaGetDeviceProperties() call failed! "
                << "Error code: " << cudaGetErrorName(result)
                << " Error message: " << cudaGetErrorString(result);
    throw std::runtime_error(errorStream.str());
  }

  
  int64_t M = std::atoi(argv[1]);
  int64_t N = std::atoi(argv[2]);
  int64_t K = std::atoi(argv[3]);
  int64_t split_k = std::atoi(argv[4]);

  int64_t a_dim0 = M;
  int64_t a_dim1 = K;
  int64_t b_dim0 = K;
  int64_t b_dim1 = N;
  int64_t c_dim0 = M;
  int64_t c_dim1 = N;

  uint8_t* global_workspace_ = nullptr;
  cudaStream_t stream = nullptr;

  
  int64_t a_ptr_sz = a_dim0 * a_dim1;
  int64_t b_ptr_sz = b_dim0 * b_dim1;
  int64_t c_ptr_sz = c_dim0 * c_dim1;

  // The value 1 is used to force ptr_max_sz to be non-zero
  int64_t ptr_max_sz = std::max<int64_t>({1, a_ptr_sz, b_ptr_sz, c_ptr_sz});

  size_t one_copy_sz = a_ptr_sz + b_ptr_sz + c_ptr_sz;

  int64_t mem_pool_sz = memory_pool->ComputeMemPoolSize(one_copy_sz, ptr_max_sz, device_properties.l2CacheSize);

  memory_pool->AllocateTensor(a_ptr_sz, mem_pool_sz);  // a_ptr: index 0
  memory_pool->AllocateTensor(b_ptr_sz, mem_pool_sz);  // b_ptr: index 1
  memory_pool->AllocateTensor(c_ptr_sz, mem_pool_sz, /*is_output*/true);  // c_ptr: index 2



  
  {
  
  GemmInstance_0 gemm_op_0;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_256x128_16x3_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_0,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_1 gemm_op_1;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_128x256_16x3_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_1,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_2 gemm_op_2;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_256x64_16x4_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_2,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_3 gemm_op_3;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_64x256_16x4_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_3,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_4 gemm_op_4;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_128x128_16x5_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_4,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_5 gemm_op_5;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_128x128_16x4_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_5,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_6 gemm_op_6;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_128x128_16x3_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_6,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_7 gemm_op_7;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_128x64_16x6_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_7,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_8 gemm_op_8;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_64x128_16x6_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_8,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_9 gemm_op_9;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_64x64_16x10_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_9,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_10 gemm_op_10;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_256x128_32x3_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_10,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_11 gemm_op_11;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_128x256_32x3_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_11,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_12 gemm_op_12;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_256x64_32x4_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_12,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_13 gemm_op_13;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_64x256_32x4_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_13,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_14 gemm_op_14;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_128x128_32x4_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_14,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_15 gemm_op_15;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_128x128_32x3_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_15,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_16 gemm_op_16;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_128x64_32x3_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_16,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_17 gemm_op_17;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_64x128_32x3_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_17,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_18 gemm_op_18;
  const char *gemm_op_name = "cutlass_tensorop_s1688tf32gemm_64x64_32x5_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_18,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_19 gemm_op_19;
  const char *gemm_op_name = "cutlass_tensorop_s1688gemm_128x128_16x4_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_19,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_20 gemm_op_20;
  const char *gemm_op_name = "cutlass_tensorop_s1688gemm_128x128_16x3_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_20,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_21 gemm_op_21;
  const char *gemm_op_name = "cutlass_tensorop_s1688gemm_256x64_16x3_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_21,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_22 gemm_op_22;
  const char *gemm_op_name = "cutlass_tensorop_s1688gemm_64x256_16x3_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_22,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_23 gemm_op_23;
  const char *gemm_op_name = "cutlass_tensorop_s1688gemm_128x64_16x4_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_23,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_24 gemm_op_24;
  const char *gemm_op_name = "cutlass_tensorop_s1688gemm_64x128_16x4_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_24,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_25 gemm_op_25;
  const char *gemm_op_name = "cutlass_tensorop_s1688gemm_64x64_16x3_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_25,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_26 gemm_op_26;
  const char *gemm_op_name = "cutlass_tensorop_s1688gemm_128x128_32x3_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_26,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_27 gemm_op_27;
  const char *gemm_op_name = "cutlass_tensorop_s1688gemm_256x64_32x3_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_27,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_28 gemm_op_28;
  const char *gemm_op_name = "cutlass_tensorop_s1688gemm_64x256_32x3_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_28,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_29 gemm_op_29;
  const char *gemm_op_name = "cutlass_tensorop_s1688gemm_128x64_32x3_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_29,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_30 gemm_op_30;
  const char *gemm_op_name = "cutlass_tensorop_s1688gemm_64x128_32x3_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_30,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_31 gemm_op_31;
  const char *gemm_op_name = "cutlass_tensorop_s1688gemm_64x64_32x3_tt_align_4_4";
  int ret = 0;
  try {
  ret = benchmark_gemm(
      gemm_op_31,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,

      split_k,


      &a_dim0,

      &a_dim1,


      &b_dim0,

      &b_dim1,


      &c_dim0,

      &c_dim1,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }
  return 0;
}