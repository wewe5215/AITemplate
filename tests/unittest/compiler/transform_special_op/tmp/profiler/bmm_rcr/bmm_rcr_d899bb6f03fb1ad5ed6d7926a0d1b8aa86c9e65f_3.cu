
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



  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_0 = Operation_cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_128x256_32x3_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_128x256_32x3_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_1 = Operation_cutlass_tensorop_f16_s16816gemm_f16_128x256_32x3_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_256x64_32x3_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_256x64_32x3_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_2 = Operation_cutlass_tensorop_f16_s16816gemm_f16_256x64_32x3_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_256x64_32x4_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_256x64_32x4_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_3 = Operation_cutlass_tensorop_f16_s16816gemm_f16_256x64_32x4_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_64x256_32x4_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_64x256_32x4_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_4 = Operation_cutlass_tensorop_f16_s16816gemm_f16_64x256_32x4_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_5 = Operation_cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_6 = Operation_cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    5,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_7 = Operation_cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_128x64_32x6_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_128x64_32x6_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    6,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_8 = Operation_cutlass_tensorop_f16_s16816gemm_f16_128x64_32x6_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_64x128_32x6_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_64x128_32x6_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    6,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_9 = Operation_cutlass_tensorop_f16_s16816gemm_f16_64x128_32x6_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_64x64_32x10_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_64x64_32x10_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    10,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_10 = Operation_cutlass_tensorop_f16_s16816gemm_f16_64x64_32x10_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_256x128_64x3_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_256x128_64x3_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_11 = Operation_cutlass_tensorop_f16_s16816gemm_f16_256x128_64x3_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_128x256_64x3_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_128x256_64x3_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_12 = Operation_cutlass_tensorop_f16_s16816gemm_f16_128x256_64x3_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_256x64_64x4_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_256x64_64x4_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_13 = Operation_cutlass_tensorop_f16_s16816gemm_f16_256x64_64x4_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_64x256_64x4_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_64x256_64x4_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_14 = Operation_cutlass_tensorop_f16_s16816gemm_f16_64x256_64x4_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_128x128_64x4_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_128x128_64x4_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_15 = Operation_cutlass_tensorop_f16_s16816gemm_f16_128x128_64x4_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_256x64_64x3_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_256x64_64x3_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_16 = Operation_cutlass_tensorop_f16_s16816gemm_f16_256x64_64x3_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_64x256_64x3_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_64x256_64x3_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_17 = Operation_cutlass_tensorop_f16_s16816gemm_f16_64x256_64x3_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_128x128_64x3_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_128x128_64x3_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_18 = Operation_cutlass_tensorop_f16_s16816gemm_f16_128x128_64x3_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_128x64_64x3_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_128x64_64x3_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<64, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_19 = Operation_cutlass_tensorop_f16_s16816gemm_f16_128x64_64x3_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_64x128_64x3_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_64x128_64x3_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_20 = Operation_cutlass_tensorop_f16_s16816gemm_f16_64x128_64x3_tn_align8;


  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tn_align8
  using Operation_cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tn_align8 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    5,
    8,
    8,
    
    cutlass::arch::OpMultiplyAdd
    
  >;

using GemmInstance_21 = Operation_cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tn_align8;

template <typename GemmInstance>
void bmm (
    GemmInstance& gemm_op,
    void* a_ptr,
    void* b_ptr,
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

  
//  TODO: cast to right dtype
  using ElementComputeEpilogue = typename GemmInstance::ElementAccumulator;

  using coord_t = cutlass::gemm::GemmCoord::Index;
  typename GemmInstance::Arguments arguments;

  if constexpr (cutlass::gemm::detail::IsCutlass3GemmKernel<typename GemmInstance::GemmKernel>::value) {
  arguments = {

    cutlass::gemm::GemmUniversalMode::kBatched,                                 // GemmUniversalMode mode
    {
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K),
        static_cast<coord_t>(B)
    },                                                                          // ProblemShape problem_shape
    (cutlass::half_t*)(a_ptr),                                                          // ElementA const* ptr_A

    { K, cute::Int<1>{}, M * K },            // StrideA dA

    (cutlass::half_t*)(b_ptr),                                                          // ElementB const* ptr_B

    { K, cute::Int<1>{}, N * K },            // StrideB dB

    {
        {
            ElementComputeEpilogue(1.0),
            ElementComputeEpilogue(0)
        },                                                                      // typename ThreadEpilogueOp::Params thread
        (cutlass::half_t*)(c_ptr),                                                   // ElementC const* ptr_C

        { N, cute::Int<1>{}, M * N },  // StrideC dC

        (cutlass::half_t*)(c_ptr),                                                      // ElementD const* ptr_D

        { N, cute::Int<1>{}, M * N },        // StrideD dD

    },                                                                          // EpilogueArguments epilogue
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
    {ElementComputeEpilogue(1.0), ElementComputeEpilogue(0)},  // typename EpilogueOutputOp::Params epilogue
    a_ptr,                                                                                  // void const * ptr_A
    b_ptr,                                                                                  // void const * ptr_B
    c_ptr,                                                                               // void const * ptr_C
    c_ptr,                                                                                  // void * ptr_D
    M * K,                                                                         // int64_t batch_stride_A
    N * K,                                                                         // int64_t batch_stride_B
    M * N,                                                                      // int64_t batch_stride_C
    M * N,                                                                         // int64_t batch_stride_D
    K,                                                                                    // typename LayoutA::Stride::LongIndex lda
    K,                                                                                    // typename LayoutB::Stride::LongIndex ldb
    N,                                                                                 // typename LayoutC::Stride::LongIndex ldc
    N,                                                                                    // typename LayoutC::Stride::LongIndex ldd
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
      std::cout << "input_ndims2: " << *a_dim2 << std::endl;
      std::cout << "weight_ndims0: " << *b_dim0 << std::endl;
      std::cout << "weight_ndims1: " << *b_dim1 << std::endl;
      std::cout << "weight_ndims2: " << *b_dim2 << std::endl;
      std::cout << "output_ndims0: " << *c_dim0 << std::endl;
      std::cout << "output_ndims1: " << *c_dim1 << std::endl;
      std::cout << "output_ndims2: " << *c_dim2 << std::endl;
  throw std::runtime_error(
      "Unsupported workload for this bmm specialization."
  );
}

template <typename DType>
struct ProfilerMemoryPool;

template <typename GemmInstance>
int benchmark_bmm (


    GemmInstance &gemm_op,
    const char *gemm_op_name,
    ProfilerMemoryPool<half>* memory_pool,
    uint8_t* global_workspace_,


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
  // warmup
  for (int i = 0; i < 5; ++i) {
    
{

bmm(

    gemm_op,

    memory_pool->RequestTensorByIdx(0),
    memory_pool->RequestTensorByIdx(1),


    memory_pool->RequestTensorByIdx(2),
    global_workspace_,

    a_dim0,

    a_dim1,

    a_dim2,


    b_dim0,

    b_dim1,

    b_dim2,


    c_dim0,

    c_dim1,

    c_dim2,

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

bmm(

    gemm_op,

    memory_pool->RequestTensorByIdx(0),
    memory_pool->RequestTensorByIdx(1),


    memory_pool->RequestTensorByIdx(2),
    global_workspace_,

    a_dim0,

    a_dim1,

    a_dim2,


    b_dim0,

    b_dim1,

    b_dim2,


    c_dim0,

    c_dim1,

    c_dim2,

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
  auto memory_pool = std::make_unique<ProfilerMemoryPool<half>>();
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

  
  int64_t B = std::atoi(argv[1]);
  int64_t M = std::atoi(argv[2]);
  int64_t N = std::atoi(argv[3]);
  int64_t K = std::atoi(argv[4]);


  int64_t a_dim0 = B;

  int64_t a_dim1 = M;

  int64_t a_dim2 = K;


  int64_t b_dim0 = B;

  int64_t b_dim1 = N;

  int64_t b_dim2 = K;


  int64_t c_dim0 = B;

  int64_t c_dim1 = M;

  int64_t c_dim2 = N;


  uint8_t* global_workspace_ = nullptr;
  cudaStream_t stream = nullptr;

  
  // cast to int64_t to avoid overflow
  int64_t a_ptr_sz = 1;
  
    a_ptr_sz *= static_cast<int64_t>(a_dim0);
  
    a_ptr_sz *= static_cast<int64_t>(a_dim1);
  
    a_ptr_sz *= static_cast<int64_t>(a_dim2);
  

  int64_t b_ptr_sz = 1;
  
    b_ptr_sz *= static_cast<int64_t>(b_dim0);
  
    b_ptr_sz *= static_cast<int64_t>(b_dim1);
  
    b_ptr_sz *= static_cast<int64_t>(b_dim2);
  

  int64_t c_ptr_sz = 1;
  
    c_ptr_sz *= static_cast<int64_t>(c_dim0);
  
    c_ptr_sz *= static_cast<int64_t>(c_dim1);
  
    c_ptr_sz *= static_cast<int64_t>(c_dim2);
  

  // The value 1 is used to force ptr_max_sz to be non-zero
  int64_t ptr_max_sz = std::max<int64_t>({1, a_ptr_sz, b_ptr_sz, c_ptr_sz});
  size_t one_copy_sz = a_ptr_sz + b_ptr_sz + c_ptr_sz;


  int64_t mem_pool_sz = memory_pool->ComputeMemPoolSize(one_copy_sz, ptr_max_sz, device_properties.l2CacheSize);

  memory_pool->AllocateTensor(a_ptr_sz, mem_pool_sz);  // a_ptr: index 0
  memory_pool->AllocateTensor(b_ptr_sz, mem_pool_sz);  // b_ptr: index 1
  memory_pool->AllocateTensor(c_ptr_sz, mem_pool_sz, /*is_output*/true);  // c_ptr: index 2



  
  {
  
  GemmInstance_0 gemm_op_0;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_0,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_1 gemm_op_1;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_128x256_32x3_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_1,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_2 gemm_op_2;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_256x64_32x3_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_2,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_3 gemm_op_3;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_256x64_32x4_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_3,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_4 gemm_op_4;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_64x256_32x4_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_4,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_5 gemm_op_5;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_5,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_6 gemm_op_6;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_6,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_7 gemm_op_7;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_7,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_8 gemm_op_8;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_128x64_32x6_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_8,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_9 gemm_op_9;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_64x128_32x6_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_9,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_10 gemm_op_10;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_64x64_32x10_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_10,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_11 gemm_op_11;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_256x128_64x3_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_11,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_12 gemm_op_12;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_128x256_64x3_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_12,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_13 gemm_op_13;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_256x64_64x4_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_13,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_14 gemm_op_14;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_64x256_64x4_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_14,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_15 gemm_op_15;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_128x128_64x4_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_15,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_16 gemm_op_16;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_256x64_64x3_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_16,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_17 gemm_op_17;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_64x256_64x3_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_17,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_18 gemm_op_18;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_128x128_64x3_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_18,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_19 gemm_op_19;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_128x64_64x3_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_19,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_20 gemm_op_20;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_64x128_64x3_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_20,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }

  {
  
  GemmInstance_21 gemm_op_21;
  const char *gemm_op_name = "cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_tn_align_8_8";
  int ret = 0;
  try {
  ret = benchmark_bmm(
      gemm_op_21,
      gemm_op_name,
      memory_pool.get(),
      global_workspace_,


      &a_dim0,

      &a_dim1,

      &a_dim2,


      &b_dim0,

      &b_dim1,

      &b_dim2,


      &c_dim0,

      &c_dim1,

      &c_dim2,

      stream
  );
  } catch (...) {}
  if (ret != 0)
    return ret;
  
  }
  return 0;
}