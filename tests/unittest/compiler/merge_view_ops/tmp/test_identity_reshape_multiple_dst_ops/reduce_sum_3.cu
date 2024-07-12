

// Modified from cutlass/examples/35_gemm_softmax/gemm_with_softmax.h

/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**

*/

#include <cmath>
#include <iostream>
#include <vector>
#include <limits>
#include <numeric>

#include "cutlass/cutlass.h"

#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/array.h"
#include "cutlass/device_kernel.h"
#include "cutlass/functional.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/fast_math.h"

#ifndef CHECK_ERROR_REDUCE
#define CHECK_ERROR_REDUCE(expr)                             \
  do {                                                       \
    cudaError_t status = (expr);                             \
    if (status != cudaSuccess) {                             \
      auto msg = std::string("Got error: ") +                \
        cudaGetErrorString(status) +                         \
        " at " + __FILE__ + ": " + std::to_string(__LINE__); \
      std::cerr << msg << std::endl;                         \
      throw std::runtime_error(msg);                         \
    }                                                        \
  } while (0)
#endif // CHECK_ERROR_REDUCE

#ifndef LAUNCH_CHECK_REDUCE
#define LAUNCH_CHECK_REDUCE() CHECK_ERROR_REDUCE(cudaGetLastError())
#endif // LAUNCH_CHECK_REDUCE



//  Copyright (c) Meta Platforms, Inc. and affiliates.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
#ifndef AIT_TENSOR_ACCESSOR_CUH
#define AIT_TENSOR_ACCESSOR_CUH

// Returns a strided address based on a base pointer, an index and strided
// information.
// DATA_T: tensor data type.
// READ_T: actual data type used when reading data. e.g. for a "half"
// tensor, READ_T could be uint4 when all data is aligned.
// data: A base pointer in READ_T type.
// idx: read index in terms of READ_T.
// offset, original_total_elements_from_stride_dim and
// actual_total_elements_from_stride_dim are the corresponding data member
// values of TensorAccessor.
template <typename DATA_T, typename READ_T, bool is_contiguous>
__device__ __forceinline__ READ_T* get_strided_address(
    READ_T* data,
    int64_t idx,
    int64_t offset,
    int64_t original_total_elements_from_stride_dim,
    int64_t actual_total_elements_from_stride_dim) {
  (void)original_total_elements_from_stride_dim; // Suppress incorrect declared
                                                 // but never referenced warning
                                                 // from nvcc.
  (void)actual_total_elements_from_stride_dim; // Ditto.
  if constexpr (is_contiguous) {
    return reinterpret_cast<READ_T*>(reinterpret_cast<DATA_T*>(data) + offset) +
        idx;
  } else {
    constexpr int N_ELEMENTS_PER_READ = sizeof(READ_T) / sizeof(DATA_T);
    int64_t data_idx = idx * N_ELEMENTS_PER_READ;
    int64_t num_rows = data_idx / original_total_elements_from_stride_dim;
    int64_t row_offset = data_idx % original_total_elements_from_stride_dim;
    data_idx =
        num_rows * actual_total_elements_from_stride_dim + row_offset + offset;
    return reinterpret_cast<READ_T*>(
        reinterpret_cast<DATA_T*>(data) + data_idx);
  }
  return nullptr; // Suppress incorrect warning about missing return statement
                  // from nvcc.
}

static inline uint64_t max_power2_divisor(uint64_t n) {
  // max power of 2 which divides n
  return n & (~(n - 1));
}

// A TensorAccessor which handles strided tensor access underneath.
struct TensorAccessor {
  int64_t offset{0};
  bool is_contiguous{true};

  int stride_dim{-1};
  int64_t original_total_elements_from_stride_dim{-1};
  int64_t actual_total_elements_from_stride_dim{-1};

  // Returns an address based on a base pointer and an index.

  // DATA_T: tensor data type.
  // READ_T: actual data type used when reading data. e.g. for a "half"
  // tensor, READ_T could be uint4 when all data is aligned.
  // data: A base pointer in READ_T type.
  // idx: read index in terms of READ_T.
  template <typename DATA_T, typename READ_T>
  __device__ inline READ_T* get(READ_T* data, int64_t idx) const {
    return is_contiguous ? get_strided_address<DATA_T, READ_T, true>(
                               data,
                               idx,
                               offset,
                               original_total_elements_from_stride_dim,
                               actual_total_elements_from_stride_dim)
                         : get_strided_address<DATA_T, READ_T, false>(
                               data,
                               idx,
                               offset,
                               original_total_elements_from_stride_dim,
                               actual_total_elements_from_stride_dim);
  }

  uint64_t max_alignment() const {
    // gcd of max alignments
    auto alignment = max_power2_divisor(offset);
    if (!is_contiguous) {
      alignment |= max_power2_divisor(original_total_elements_from_stride_dim);
      alignment |= max_power2_divisor(actual_total_elements_from_stride_dim);
    }
    return max_power2_divisor(alignment);
  }

  bool is_valid_alignment(uint64_t n) const {
    // n is a power of 2; return whether tensor accessor alignment is divisible
    // by n.
    return !(max_alignment() & (n - 1));
  }
};

#endif


namespace {

template <typename DATA_T, typename READ_T>
__device__ __forceinline__ READ_T* get_strided_address_at_idx(
    READ_T *data, int64_t data_idx) {

  return get_strided_address<DATA_T, READ_T, true>(
      data, data_idx, 0, 0, 0);

}
}



namespace {

template <
  typename ElementOutput,
  typename ElementInput,
  typename ElementCompute,
  int Alignment,
  typename Layout_ = cutlass::layout::RowMajor,
  typename Shape_ = cutlass::MatrixShape<4, 16>
>
struct ReductionKernel3D {

  static int const kAlignment = Alignment;

  using Layout = Layout_;
  using Shape = Shape_;

  using TensorOutput = cutlass::TensorRef<ElementOutput, Layout>;
  using TensorInput = cutlass::TensorRef<ElementInput, Layout>;
  using TensorCompute = cutlass::TensorRef<ElementCompute, Layout>;

  struct Arguments {

    TensorOutput ref_output;     ///< Output tensor
    TensorInput ref_input;       ///< Input tensor
    cutlass::MatrixCoord extent; ///< Extent of input and output tensors
    int64_t input_row_stride;    ///< stride for accessing next element in
                                 ///< the same row. It's 1 for RowMajor and
                                 ///< extent.row() for ColMajor
    int64_t batch_count;         ///< Batch count
    int64_t batch_stride_output; ///< Batch stride for Output tensor
    int64_t batch_stride_input;  ///< Batch stride for Input tensor

    Arguments(
      TensorOutput    ref_output_,        ///< Output tensor
      TensorInput     ref_input_,         ///< Input tensor
      cutlass::MatrixCoord extent_,       ///< Extent of input and output tensors
      int64_t         input_row_stride_,  ///< stride for accessing input rows
      int64_t         batch_count_,       ///< Batch count
      int64_t         batch_stride_output_ = 0,
      int64_t         batch_stride_input_ = 0
    ):
      ref_output(ref_output_),
      ref_input(ref_input_),
      extent(extent_),
      input_row_stride(input_row_stride_),
      batch_count(batch_count_),
      batch_stride_output(batch_stride_output_),
      batch_stride_input(batch_stride_input_)
    { }
  };

  struct Params {
    Arguments args;
    Params() { }
    Params(Arguments const &args_): args(args_) { }
  };

  struct SharedStorage {
    cutlass::AlignedArray<ElementCompute, Shape::kCount, Shape::kCount * alignof(ElementCompute)> exchange;
  };

  CUTLASS_DEVICE
  ReductionKernel3D() { }

  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    reduce_partial(params.args, shared_storage);

    __syncthreads();

    reduce_final(params.args, shared_storage);

    __syncthreads();
  }

  /// Partial reduction
  CUTLASS_DEVICE
  void reduce_partial(Arguments const &args, SharedStorage &shared_storage) {

    using AccessTypeInput = cutlass::AlignedArray<ElementInput, kAlignment>;

    int block_batch = blockIdx.z;
    int block_m = blockIdx.x * Shape::kRow;
    int block_n = 0;

    int thread_m = threadIdx.y;
    int thread_n = threadIdx.x * kAlignment;

    int idx_m = block_m + thread_m;
    int idx_n = block_n + thread_n;

    AccessTypeInput *access_input = reinterpret_cast<AccessTypeInput *>(
      args.ref_input.data() +
      args.batch_stride_input * block_batch +
      args.ref_input.layout()({idx_m, idx_n}));

    using ConvertS = cutlass::NumericArrayConverter<ElementCompute, ElementInput, kAlignment>;
    ConvertS convert_s;

    using FragmentCompute = cutlass::Array<ElementCompute, kAlignment>;
    using ReduceVectorOp = cutlass::plus<FragmentCompute>;
    using ReduceScalarOp = cutlass::plus<ElementCompute>;
    ReduceVectorOp reduce_v_op;
    ReduceScalarOp reduce_s_op;

    FragmentCompute frag_compute;


    // initialize the frag_compute with default values
    frag_compute.clear();


    if (idx_m < args.extent.row()) {

      CUTLASS_PRAGMA_UNROLL
      for (
        int idx = 0;
        idx < args.extent.column();
        idx += Shape::kColumn * kAlignment) {

        if (idx_n < args.extent.column()) {

          AccessTypeInput fetch;
          cutlass::arch::global_load<AccessTypeInput, sizeof(AccessTypeInput)>(
              fetch, access_input, true);
          auto prologue_fn = [&] (FragmentCompute fragment) {


        return fragment;

          };
          FragmentCompute tmp = prologue_fn(convert_s(fetch));
          frag_compute = reduce_v_op(frag_compute, tmp);
        }

        access_input += Shape::kColumn * args.input_row_stride;
        idx_n += Shape::kColumn * kAlignment;
      }

      // Reduce the elements owned by one thread
      ElementCompute result = frag_compute[0];

      CUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < kAlignment; ++i) {
        result = reduce_s_op(result, frag_compute[i]);
      }

      shared_storage.exchange.data()[threadIdx.x + threadIdx.y * Shape::kColumn] = result;
    }
  }

  /// Compute the final summation from data in SMEM
  CUTLASS_DEVICE
  void reduce_final(Arguments const &args, SharedStorage &shared_storage) {

    //
    // SMEM has shape `Shape::Row`-by-`Shape::Column`
    //
    // This computes a reduction across the `Column` dimension yielding a `Row-by-1` vector.
    //

    //
    // Tuning parameters tradeoff ILP with latency
    //
    // During each step of the reduction, each thread performs `kAccesses` of
    // vector size `kReduceVector`

    // Tune the number of accesses per reduction
    int const kAccesses = 2;

    // Tune the memory access size
    int const kReduceVector = 4;

    //
    // Static asserts to ensure integrity
    //

    static_assert(kAccesses * kReduceVector,
      "Zero-size steps would infinitely loop.");

    static_assert(
      cutlass::is_pow2<Shape::kColumn>::value &&
      cutlass::is_pow2<kAccesses>::value &&
      cutlass::is_pow2<kReduceVector>::value,
      "Powers of two only.");

    static_assert(!(Shape::kColumn % (kAccesses * kReduceVector)),
      "Divisibility not satisfied");

    //
    // Reduction operators
    //

    using FragmentCompute = cutlass::Array<ElementCompute, kReduceVector>;
    using ReduceVectorOp = cutlass::plus<FragmentCompute>;
    using ReduceScalarOp = cutlass::plus<ElementCompute>;
    ReduceVectorOp reduce_v_op;
    ReduceScalarOp reduce_s_op;

    // Tree reduction
    ElementCompute *smem_ptr = shared_storage.exchange.data() + threadIdx.y * Shape::kColumn;

    ElementCompute result = ElementCompute();

    CUTLASS_PRAGMA_UNROLL
    for (
      int tidx_limit = Shape::kColumn / (kAccesses * kReduceVector);
      tidx_limit > 0;
      tidx_limit /= (kAccesses * kReduceVector)) {

      if (threadIdx.x < tidx_limit) {
        FragmentCompute fetch;

        cutlass::arch::shared_load<sizeof(FragmentCompute)>(
            &fetch,
            cutlass::arch::cutlass_get_smem_pointer(smem_ptr + threadIdx.x * kReduceVector));

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < kAccesses; ++i) {
          FragmentCompute extra;

          cutlass::arch::shared_load<sizeof(FragmentCompute)>(
              &extra,
              cutlass::arch::cutlass_get_smem_pointer(
                  smem_ptr + threadIdx.x * kReduceVector + tidx_limit * kReduceVector * i));

          fetch = reduce_v_op(fetch, extra);
        }

        // Reduce to one element
        result = fetch[0];

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < kReduceVector; ++i) {
          result = reduce_s_op(result, fetch[i]);
        }
      }
      __syncthreads();

      if (threadIdx.x < tidx_limit) {
        smem_ptr[threadIdx.x] = result;
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {

      int const kLgResidual =
        (cutlass::log2_down<Shape::kColumn>::value %
         cutlass::log2_down<kAccesses * kReduceVector>::value);

      // Certain shape combinations require an additional reduction step
      if (kLgResidual) {
        result = ElementCompute();

        int const kResidualVector = (1 << kLgResidual);
        cutlass::Array<ElementCompute, kResidualVector> fetch;

        cutlass::arch::shared_load<sizeof(FragmentCompute)>(
            &fetch,
            cutlass::arch::cutlass_get_smem_pointer(smem_ptr));

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kResidualVector; ++i) {
          result = reduce_s_op(result, fetch[i]);
        }
      }

      int block_batch = blockIdx.z;
      int block_m = blockIdx.x * Shape::kRow;
      int thread_m = threadIdx.y;
      int idx_m = block_m + thread_m;
      if (idx_m >= args.extent.row()) {
        return;
      }

      int64_t output_idx = args.batch_stride_output * block_batch +
                           args.ref_output.layout()({idx_m, 0});
      ElementOutput *access_output =
          get_strided_address_at_idx<ElementOutput, ElementOutput>(
              reinterpret_cast<ElementOutput*>(args.ref_output.data()), output_idx);

      cutlass::NumericConverter<ElementOutput, ElementCompute> convert_output;

      auto epilogue_scalar_fn = [&] (ElementCompute reduced_result,
                                     int num_reduced_elems) {


            return reduced_result;

      };
      ElementCompute tmp = epilogue_scalar_fn(result, args.extent.column());
      *access_output = convert_output(tmp);
    }
  }
};


using ReductionKernelRowMajor_8 = ReductionKernel3D<
    float, /* ElementOutput */
    float, /* ElementInput */
    float, /*ElementCompute */
    8,
    cutlass::layout::RowMajor, /*Layout*/
    cutlass::MatrixShape<1, 8>
>;

using ReductionKernelRowMajor_4 = ReductionKernel3D<
    float, /* ElementOutput */
    float, /* ElementInput */
    float, /*ElementCompute */
    4,
    cutlass::layout::RowMajor, /*Layout*/
    cutlass::MatrixShape<1, 8>
>;

using ReductionKernelRowMajor_2 = ReductionKernel3D<
    float, /* ElementOutput */
    float, /* ElementInput */
    float, /*ElementCompute */
    2,
    cutlass::layout::RowMajor, /*Layout*/
    cutlass::MatrixShape<1, 8>
>;

using ReductionKernelRowMajor_1 = ReductionKernel3D<
    float, /* ElementOutput */
    float, /* ElementInput */
    float, /*ElementCompute */
    1,
    cutlass::layout::RowMajor, /*Layout*/
    cutlass::MatrixShape<1, 8>
>;

using ReductionKernelColumnMajor_1 = ReductionKernel3D<
    float, /* ElementOutput */
    float, /* ElementInput */
    float, /*ElementCompute */
    1,
    cutlass::layout::ColumnMajor, /*Layout*/
    cutlass::MatrixShape<1, 8>
>;


template<typename ElementOutput, typename ElementInput>
void reduce_mean_launcher_RowMajor_8(
  ElementOutput *output,
  ElementInput *input,
  int64_t batch_count,
  int64_t rows,
  int64_t columns,
  int64_t batch_stride_input,
  int64_t batch_stride_output,
  cudaStream_t stream
) {

  using ReductionKernel = ReductionKernelRowMajor_8;

  dim3 apply_block(ReductionKernel::Shape::kColumn,
                   ReductionKernel::Shape::kRow);

  int cta_rows = ReductionKernel::Shape::kRow;
  int cta_columns = ReductionKernel::Shape::kColumn * ReductionKernel::kAlignment;

  dim3 apply_grid(static_cast<int>((rows + cta_rows - 1) / cta_rows),
                  static_cast<int>((columns + cta_columns - 1) / cta_columns),
                  static_cast<int>(batch_count));

  // row major
  int64_t lda_output = 1;
  int64_t lda_input = columns;
  ReductionKernel::Layout output_layout(lda_output);
  ReductionKernel::Layout input_layout(lda_input);

  ReductionKernel::TensorOutput output_tensor(output, output_layout);
  ReductionKernel::TensorInput input_tensor(input, input_layout);
  ReductionKernel::Arguments kernel_args(
      output_tensor,
      input_tensor,
      cutlass::MatrixCoord(static_cast<int>(rows), static_cast<int>(columns)),
      1 /*input_row_stride*/,
      static_cast<int>(batch_count),
      batch_stride_output,
      batch_stride_input
  );

  cutlass::Kernel<ReductionKernel><<<
      apply_grid,
      apply_block,
      sizeof(typename ReductionKernel::SharedStorage),
      stream
  >>>(kernel_args);

  LAUNCH_CHECK_REDUCE();
}

template<typename ElementOutput, typename ElementInput>
void reduce_mean_launcher_RowMajor_4(
  ElementOutput *output,
  ElementInput *input,
  int64_t batch_count,
  int64_t rows,
  int64_t columns,
  int64_t batch_stride_input,
  int64_t batch_stride_output,
  cudaStream_t stream
) {

  using ReductionKernel = ReductionKernelRowMajor_4;

  dim3 apply_block(ReductionKernel::Shape::kColumn,
                   ReductionKernel::Shape::kRow);

  int cta_rows = ReductionKernel::Shape::kRow;
  int cta_columns = ReductionKernel::Shape::kColumn * ReductionKernel::kAlignment;

  dim3 apply_grid(static_cast<int>((rows + cta_rows - 1) / cta_rows),
                  static_cast<int>((columns + cta_columns - 1) / cta_columns),
                  static_cast<int>(batch_count));

  // row major
  int64_t lda_output = 1;
  int64_t lda_input = columns;
  ReductionKernel::Layout output_layout(lda_output);
  ReductionKernel::Layout input_layout(lda_input);

  ReductionKernel::TensorOutput output_tensor(output, output_layout);
  ReductionKernel::TensorInput input_tensor(input, input_layout);
  ReductionKernel::Arguments kernel_args(
      output_tensor,
      input_tensor,
      cutlass::MatrixCoord(static_cast<int>(rows), static_cast<int>(columns)),
      1 /*input_row_stride*/,
      static_cast<int>(batch_count),
      batch_stride_output,
      batch_stride_input
  );

  cutlass::Kernel<ReductionKernel><<<
      apply_grid,
      apply_block,
      sizeof(typename ReductionKernel::SharedStorage),
      stream
  >>>(kernel_args);

  LAUNCH_CHECK_REDUCE();
}

template<typename ElementOutput, typename ElementInput>
void reduce_mean_launcher_RowMajor_2(
  ElementOutput *output,
  ElementInput *input,
  int64_t batch_count,
  int64_t rows,
  int64_t columns,
  int64_t batch_stride_input,
  int64_t batch_stride_output,
  cudaStream_t stream
) {

  using ReductionKernel = ReductionKernelRowMajor_2;

  dim3 apply_block(ReductionKernel::Shape::kColumn,
                   ReductionKernel::Shape::kRow);

  int cta_rows = ReductionKernel::Shape::kRow;
  int cta_columns = ReductionKernel::Shape::kColumn * ReductionKernel::kAlignment;

  dim3 apply_grid(static_cast<int>((rows + cta_rows - 1) / cta_rows),
                  static_cast<int>((columns + cta_columns - 1) / cta_columns),
                  static_cast<int>(batch_count));

  // row major
  int64_t lda_output = 1;
  int64_t lda_input = columns;
  ReductionKernel::Layout output_layout(lda_output);
  ReductionKernel::Layout input_layout(lda_input);

  ReductionKernel::TensorOutput output_tensor(output, output_layout);
  ReductionKernel::TensorInput input_tensor(input, input_layout);
  ReductionKernel::Arguments kernel_args(
      output_tensor,
      input_tensor,
      cutlass::MatrixCoord(static_cast<int>(rows), static_cast<int>(columns)),
      1 /*input_row_stride*/,
      static_cast<int>(batch_count),
      batch_stride_output,
      batch_stride_input
  );

  cutlass::Kernel<ReductionKernel><<<
      apply_grid,
      apply_block,
      sizeof(typename ReductionKernel::SharedStorage),
      stream
  >>>(kernel_args);

  LAUNCH_CHECK_REDUCE();
}

template<typename ElementOutput, typename ElementInput>
void reduce_mean_launcher_RowMajor_1(
  ElementOutput *output,
  ElementInput *input,
  int64_t batch_count,
  int64_t rows,
  int64_t columns,
  int64_t batch_stride_input,
  int64_t batch_stride_output,
  cudaStream_t stream
) {

  using ReductionKernel = ReductionKernelRowMajor_1;

  dim3 apply_block(ReductionKernel::Shape::kColumn,
                   ReductionKernel::Shape::kRow);

  int cta_rows = ReductionKernel::Shape::kRow;
  int cta_columns = ReductionKernel::Shape::kColumn * ReductionKernel::kAlignment;

  dim3 apply_grid(static_cast<int>((rows + cta_rows - 1) / cta_rows),
                  static_cast<int>((columns + cta_columns - 1) / cta_columns),
                  static_cast<int>(batch_count));

  // row major
  int64_t lda_output = 1;
  int64_t lda_input = columns;
  ReductionKernel::Layout output_layout(lda_output);
  ReductionKernel::Layout input_layout(lda_input);

  ReductionKernel::TensorOutput output_tensor(output, output_layout);
  ReductionKernel::TensorInput input_tensor(input, input_layout);
  ReductionKernel::Arguments kernel_args(
      output_tensor,
      input_tensor,
      cutlass::MatrixCoord(static_cast<int>(rows), static_cast<int>(columns)),
      1 /*input_row_stride*/,
      static_cast<int>(batch_count),
      batch_stride_output,
      batch_stride_input
  );

  cutlass::Kernel<ReductionKernel><<<
      apply_grid,
      apply_block,
      sizeof(typename ReductionKernel::SharedStorage),
      stream
  >>>(kernel_args);

  LAUNCH_CHECK_REDUCE();
}


template<typename ElementOutput, typename ElementInput>
void reduce_mean_launcher_ColumnMajor_1(
  ElementOutput *output,
  ElementInput *input,
  int64_t batch_count,
  int64_t rows,
  int64_t columns,
  int64_t batch_stride_input,
  int64_t batch_stride_output,
  cudaStream_t stream
) {

  using ReductionKernel = ReductionKernelColumnMajor_1;

  dim3 apply_block(ReductionKernel::Shape::kColumn,
                   ReductionKernel::Shape::kRow);

  int cta_rows = ReductionKernel::Shape::kRow;
  int cta_columns = ReductionKernel::Shape::kColumn * ReductionKernel::kAlignment;

  dim3 apply_grid(static_cast<int>((rows + cta_rows - 1) / cta_rows),
                  static_cast<int>((columns + cta_columns - 1) / cta_columns),
                  static_cast<int>(batch_count));

  // column major
  int64_t lda_output = 1;
  int64_t lda_input = rows;
  ReductionKernel::Layout output_layout(lda_output);
  ReductionKernel::Layout input_layout(lda_input);

  ReductionKernel::TensorOutput output_tensor(output, output_layout);
  ReductionKernel::TensorInput input_tensor(input, input_layout);
  ReductionKernel::Arguments kernel_args(
      output_tensor,
      input_tensor,
      cutlass::MatrixCoord(static_cast<int>(rows), static_cast<int>(columns)),
      rows /*input_row_stride*/,
      static_cast<int>(batch_count),
      batch_stride_output,
      batch_stride_input
  );

  cutlass::Kernel<ReductionKernel><<<
      apply_grid,
      apply_block,
      sizeof(typename ReductionKernel::SharedStorage),
      stream
  >>>(kernel_args);

  LAUNCH_CHECK_REDUCE();
}


constexpr const int ThreadsPerBlock = 128;

template <typename ElementInput,
          typename ElementOutput,
          typename ElementCompute,
          typename ReadVecT,
          typename WriteVecT,
          int64_t num_rows_per_thread,
          int64_t num_cols>
__global__ void reduce_small_in_v_out_v(
    ElementOutput *output,
    const ElementInput *input,
    int64_t num_rows,
    int64_t batch_stride_input,
    int64_t batch_stride_output,
    ElementCompute reduction_identity) {
  int block_batch = blockIdx.y;
  // index within the batch
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t idx = tid * num_rows_per_thread;
  if (idx >= num_rows)
    return;
  // input within the batch
  int64_t input_offset = idx * num_cols;
  const ElementInput *this_input =
      input + block_batch * batch_stride_input + input_offset;
  size_t output_idx = block_batch * batch_stride_output + idx;
  ElementOutput *this_output = get_strided_address_at_idx<ElementOutput, ElementOutput>(output, output_idx);

  static_assert(sizeof(ReadVecT) % sizeof(ElementInput) == 0);
  constexpr int n_read_elems_in_v = sizeof(ReadVecT) / sizeof(ElementInput);
  // number of original elements
  constexpr int64_t num_elems_per_thread = num_rows_per_thread * num_cols;
  // number of vector elements
  static_assert(num_elems_per_thread % n_read_elems_in_v == 0);
  constexpr int64_t num_elems_per_thread_v =
      num_elems_per_thread / n_read_elems_in_v;

  ReadVecT read_elems_v[num_elems_per_thread_v];
  const ReadVecT *this_input_v = reinterpret_cast<const ReadVecT*>(this_input);
  // read
  CUTLASS_PRAGMA_UNROLL
  for (int64_t i = 0; i < num_elems_per_thread_v; i++) {
    cutlass::arch::global_load<ReadVecT, sizeof(ReadVecT)>(
        read_elems_v[i], this_input_v + i, true
    );
  }

  // compute
  using FragmentCompute = ElementCompute;
  ElementInput *read_elems = reinterpret_cast<ElementInput *>(read_elems_v);
  using ReduceScalarOp = cutlass::plus<ElementCompute>;
  ReduceScalarOp reduce_s_op;
  constexpr int num_reduced_elems = num_cols;

  auto prologue_fn = [&] (FragmentCompute fragment) {
    
        return fragment;
  };
  auto epilogue_scalar_fn = [&] (ElementCompute reduced_result) {
    
            return reduced_result;
  };

  ElementOutput reduced_elems[num_rows_per_thread];
  static_assert(num_elems_per_thread % num_cols == 0);
  cutlass::NumericConverter<ElementCompute, ElementInput> convert_input;
  CUTLASS_PRAGMA_UNROLL
  for (int64_t i = 0; i < num_elems_per_thread / num_cols; i++) {
    static_assert(num_elems_per_thread % num_rows_per_thread == 0);
    FragmentCompute frag_compute = FragmentCompute(reduction_identity);
    CUTLASS_PRAGMA_UNROLL
    for (int64_t j = 0; j < num_cols; j++) {
      int64_t read_idx = i * num_cols + j;
      FragmentCompute tmp = prologue_fn(convert_input(read_elems[read_idx]));
      frag_compute = reduce_s_op(frag_compute, tmp);
    }
    cutlass::NumericConverter<ElementOutput, ElementCompute> convert_output;
    ElementCompute tmp = epilogue_scalar_fn(frag_compute);
    reduced_elems[i] = convert_output(tmp);
  }

  WriteVecT *this_output_v = reinterpret_cast<WriteVecT*>(this_output);
  WriteVecT *reduced_elems_v = reinterpret_cast<WriteVecT*>(&reduced_elems[0]);
  constexpr int n_write_elems_in_v = sizeof(WriteVecT) / sizeof(ElementOutput);
  CUTLASS_PRAGMA_UNROLL

  for (int64_t i = 0; i < num_rows_per_thread / n_write_elems_in_v; i++) {
    WriteVecT tmp = reduced_elems_v[i];
    this_output_v[i] = tmp;
  }

}

template <typename ElementOutput,
          typename ElementInput,
          typename ElementCompute,
          int64_t num_cols>
void reduce_mean_launcher_small_axis(
  ElementOutput *output,
  ElementInput *input,
  int64_t num_batches,
  int64_t num_rows,
  int64_t batch_stride_input,
  int64_t batch_stride_output,
  cudaStream_t stream
) {
  constexpr int64_t num_read_v =
      sizeof(uint4) / sizeof(ElementInput);
  constexpr int64_t row_gcd = std::gcd(num_cols, num_read_v);
  constexpr int64_t num_rows_per_thread = num_read_v / row_gcd;

  constexpr int64_t num_write_bytes_v =
      num_rows_per_thread * sizeof(ElementOutput);


  assert(num_rows % num_rows_per_thread == 0);
  int64_t real_rows = num_rows / num_rows_per_thread;
  dim3 grid(static_cast<int>(real_rows + ThreadsPerBlock -1 ) / ThreadsPerBlock,
            static_cast<int>(num_batches));

  if (num_rows % num_rows_per_thread == 0) {

#define HANDLE_ONE_WRITE_VEC(write_bytes, write_vec_type) \
    if (write_bytes == num_write_bytes_v) {               \
      reduce_small_in_v_out_v<ElementInput,               \
                              ElementOutput,              \
                              ElementCompute,             \
                              uint4,          \
                              write_vec_type,             \
                              num_rows_per_thread,        \
                              num_cols>                   \
      <<<grid, ThreadsPerBlock, 0, stream>>>(             \
          output,                                         \
          input,                                          \
          num_rows,                                       \
          batch_stride_input,                             \
          batch_stride_output,                            \
          ElementCompute());                        \
      LAUNCH_CHECK_REDUCE();                              \
      return;                                             \
    }
    HANDLE_ONE_WRITE_VEC(16, uint4)
    HANDLE_ONE_WRITE_VEC(8, uint2)
    HANDLE_ONE_WRITE_VEC(4, unsigned)
    if constexpr (std::is_same_v<ElementOutput, cutlass::half_t>) {
      HANDLE_ONE_WRITE_VEC(2, cutlass::half_t)
    }
    else if constexpr (std::is_same_v<ElementOutput, cutlass::bfloat16_t>) {
      HANDLE_ONE_WRITE_VEC(2, cutlass::bfloat16_t)
    }
    throw std::runtime_error("unsupported vector size for write");
  } else {
    throw std::runtime_error("unsupported num_row_per_threads");
  }
}

template <typename ElementOutput, typename ElementInput>
void reduce_mean_launcher_small_axis_column_major(
  ElementOutput *output,
  ElementInput *input,
  int64_t num_batches,
  int64_t num_rows,
  int64_t num_columns,
  int64_t batch_stride_input,
  int64_t batch_stride_output,
  cudaStream_t stream
) {
}


} // namespace


static int normalize_axis(int axis, int rank) {
  if (axis >= 0) return axis;
  return rank + axis;
}

static int64_t get_size(const int64_t *input_shape, int from, int to) {
  int64_t sz = 1;
  for (int i = from; i < to; i++) {
    sz *= input_shape[i];
  }
  return sz;
}

static void normalize_input_shape(
  int64_t *new_input_shape,
  int *reduction_axis,
  const int64_t *input_shape,
  int *rank
) {
  if (*reduction_axis == 0 && *rank > 1) {
      new_input_shape[0] = 1;
      new_input_shape[1] = input_shape[0];
      new_input_shape[2] = get_size(input_shape, 1, *rank);
      *reduction_axis = 1;
      *rank = 3;
      return;
  }

  if (*rank <= 3) {
    for (int i = 0; i < *rank; i++) {
      new_input_shape[i] = input_shape[i];
    }
    return;
  }

  if (*reduction_axis == *rank - 1) {
    new_input_shape[0] = input_shape[0];
    new_input_shape[1] = get_size(input_shape, 1, *reduction_axis);
    new_input_shape[2] = input_shape[*reduction_axis];
    *reduction_axis = 2;
    *rank = 3;
    return;
  }

  new_input_shape[0] = get_size(input_shape, 0, *reduction_axis);
  new_input_shape[1] = input_shape[*reduction_axis];
  new_input_shape[2] = get_size(input_shape, *reduction_axis + 1, *rank);
  *reduction_axis = 1;
  *rank = 3;
}

void reduce_sum_3(
  void *output,
  void *input,
  int reduction_axis,
  int64_t *output_shape[],
  const int64_t *orig_input_shape,
  int rank,
  bool keep_dim,
  cudaStream_t stream
) {

  reduction_axis = normalize_axis(reduction_axis, rank);
  if (reduction_axis >= rank) {
    throw std::runtime_error("reduction_axis must < rank");
  }
  if (reduction_axis < 0) {
    throw std::runtime_error("reduction_axis must >= 0");
  }
  if (rank == 0) {
    return;
  }


  for (int i = 0, j = 0; i < rank; i++, j++) {
    if (i == reduction_axis) {
      if (keep_dim) {
        *(output_shape[j]) = orig_input_shape[j] == 0 ? 0 : 1;
      } else {
        j--;
      }
    } else {
      if (orig_input_shape[i] != *(output_shape[j])) {
        throw std::runtime_error("input/output dim values do not match");
      }
    }
  }


  int64_t input_shape[3] = {1, 1, 1};
  normalize_input_shape(
      input_shape, &reduction_axis, orig_input_shape, &rank
  );

  for (int i = 0; i < rank; i++) {
    if (input_shape[i] == 0)
      return;
  }
  // make sure input and output are valid
  if (!output) {
    throw std::runtime_error("output is NULL!");
  }
  if (!input) {
    throw std::runtime_error("input is NULL!");
  }

  int64_t b = 1;
  int64_t m = 1;
  int64_t n = 1;
  int64_t batch_stride_input = 0;
  int64_t batch_stride_output = 0;

  

  if (input_shape[reduction_axis] <= 128) {
    if (reduction_axis == rank - 1) {
      constexpr int64_t cst_n = 8;
      if (rank == 3) {
        b = input_shape[0];
        m = input_shape[1];
        if (b > 1) {
          batch_stride_input = m * cst_n;
          batch_stride_output = m;
        }
      } else if (rank == 2) {
        m = input_shape[0];
      } else if (rank == 1) {
        // nothing to do
      } else {
        throw std::runtime_error("reduce_small_axis: invalid rank rank");
      }
      reduce_mean_launcher_small_axis<float,
                                      float,
                                      float,
                                      cst_n>(
            static_cast<float*>(output),
            static_cast<float*>(input),
            b, m, batch_stride_input,
            batch_stride_output, stream);
      return;
    } else {
      // TODO: support more reduction axis
      // fall-through to the general reduction kernel for now
    }
  }

#define SKIP_GENERAL_REDUCTION


// If we can statically determine we always fall into special exec cond above,
// it's safe to skip the general exec path below
#ifndef SKIP_GENERAL_REDUCTION
  if (reduction_axis == rank - 1) {
    if (rank == 3) {
      b = input_shape[0];
      m = input_shape[1];
      n = input_shape[2];
      if (b > 1) {
        batch_stride_input = m * n;
        batch_stride_output = m;
      }
    } else if (rank == 2) {
      m = input_shape[0];
      n = input_shape[1];
    } else if (rank == 1) {
      n = input_shape[0];
    } else {
      throw std::runtime_error("unreachable: invalid rank");
    }

    if (input_shape[reduction_axis] % 8 == 0) {
      reduce_mean_launcher_RowMajor_8(
        static_cast<float*>(output),
        static_cast<float*>(input),
        b, m, n, batch_stride_input, batch_stride_output, stream);
      return;
    }

    if (input_shape[reduction_axis] % 4 == 0) {
      reduce_mean_launcher_RowMajor_4(
        static_cast<float*>(output),
        static_cast<float*>(input),
        b, m, n, batch_stride_input, batch_stride_output, stream);
      return;
    }

    if (input_shape[reduction_axis] % 2 == 0) {
      reduce_mean_launcher_RowMajor_2(
        static_cast<float*>(output),
        static_cast<float*>(input),
        b, m, n, batch_stride_input, batch_stride_output, stream);
      return;
    }

    if (input_shape[reduction_axis] % 1 == 0) {
      reduce_mean_launcher_RowMajor_1(
        static_cast<float*>(output),
        static_cast<float*>(input),
        b, m, n, batch_stride_input, batch_stride_output, stream);
      return;
    }

    throw std::runtime_error("unreachable: invalid alignment");
  } else if (reduction_axis == rank - 2) {
    if (rank == 3) {
      b = input_shape[0];
      m = input_shape[2];
      n = input_shape[1];
      if (b > 1) {
        batch_stride_input = m * n;
        batch_stride_output = m;
      }
    } else if (rank == 2) {
      m = input_shape[1];
      n = input_shape[0];
    } else {
      throw std::runtime_error("unreachable: invalid rank");
    }
    reduce_mean_launcher_ColumnMajor_1(
      static_cast<float*>(output),
      static_cast<float*>(input),
      b, m, n, batch_stride_input, batch_stride_output, stream);
    return;
  }
#else
#undef SKIP_GENERAL_REDUCTION
#endif // !SKIP_GENERAL_REDUCTION

  throw std::runtime_error(
    "unsupported reduction_axis value for reduce_sum_3"
  );
}