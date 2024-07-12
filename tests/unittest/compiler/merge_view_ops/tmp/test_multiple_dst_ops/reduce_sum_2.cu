
#include <cassert>
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/reduction/thread/reduction_operators.h"
#include "cutlass/reduction/device/tensor_reduce.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/util/host_tensor.h"


#include "cutlass/arch/memory.h"

#define SKIP_GENERAL_REDUCTION

template <typename ElemOutputType, typename ElemInputType, typename ElementCompute, typename VecType>
__global__ void reduce_sum_2_kernel(
    ElemOutputType* output,
    ElemInputType* input,
    const int nbatches,
    const int nrows,
    const int ncols,
    ElementCompute reductionIdentity) {
  constexpr int32_t elemsPerThread = sizeof(VecType) / sizeof(ElemInputType);
  int32_t batch = blockIdx.x;
  int32_t batchOffset = batch * nrows * ncols;
  int32_t colOffset = threadIdx.x * elemsPerThread;

  VecType vecIn[1];
  ElementCompute fragReduce[elemsPerThread];
  cutlass::NumericConverter<ElementCompute, ElemInputType> toComputeType;
  using ReductionOp = cutlass::plus<ElementCompute>;
  ReductionOp reduce;
#pragma unroll
  for (int32_t r = 0; r < nrows; r++) {
    // Vectorized load and reduce.
    int32_t rowOffset = r * ncols;
    int readIdx = batchOffset + rowOffset + colOffset;

    cutlass::arch::global_load<VecType, sizeof(VecType)>(*vecIn, input + readIdx, true);
    ElemInputType* in = reinterpret_cast<ElemInputType*>(vecIn);

    #pragma unroll
    for (int32_t i = 0; i < elemsPerThread; i++) {
        if (r == 0) fragReduce[i] = reductionIdentity;
        fragReduce[i] = reduce(fragReduce[i], toComputeType(in[i]));
    }
  }

  // Finished reduction now convert back to output type.
  alignas(sizeof(VecType)) ElemOutputType reduced[elemsPerThread];
  cutlass::NumericConverter<ElemOutputType, ElementCompute> toOutputType;
  for (int32_t i = 0; i < elemsPerThread; i++) {
    reduced[i] = toOutputType(fragReduce[i]);
  }

  // Vectorized stores.
  int writeIdx = (batch * ncols) + colOffset;
  VecType* vecOut = reinterpret_cast<VecType*>(&output[writeIdx]);
  *vecOut = *reinterpret_cast<VecType*>(reduced);  // vectorized store
}

template <typename ElemOutputType, typename ElemInputType, typename ElementCompute, typename VecType>
void reduce_sum_2_launcher(
    void* dst_ptr,
    void* src_ptr,
    int reduction_axis,
    const int64_t* shape,
    const int rank,
    cudaStream_t stream) {
    static_assert(sizeof(ElemOutputType) == sizeof(ElemInputType));
    int nbatches = shape[rank - 3];
    int nrows = shape[rank - 2];
    int ncols = shape[rank - 1];
    int elemsPerThread = sizeof(VecType) / sizeof(ElemInputType);
    int nthreads = ncols / elemsPerThread;

    reduce_sum_2_kernel<ElemOutputType, ElemInputType, ElementCompute, VecType>
        <<<nbatches, nthreads, 0, stream>>>(
            static_cast<ElemOutputType*>(dst_ptr),
            static_cast<ElemInputType*>(src_ptr),
            nbatches,
            nrows,
            ncols,
            ElementCompute());
}

#define CUTLASS_CHECK_REDUCE(status)                                                  \
  {                                                                                   \
    cutlass::Status error = status;                                                   \
    if (error != cutlass::Status::kSuccess) {                                         \
      auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +              \
          cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);         \
      std::cerr << msg << std::endl;                                                  \
      throw std::runtime_error(msg);                                                  \
    }                                                                                 \
  }

#ifndef SKIP_GENERAL_REDUCTION
template <typename ElemOutputType, typename ElemInputType, int VectorLength = 1>
void reduce_sum_2_launcher(
    ElemOutputType *dst_ptr,
    ElemInputType *src_ptr,
    int reduction_axis,
    const int64_t *shape,
    const int rank,
    uint8_t* workspace,
    cudaStream_t stream) {
  // Instead of making our own 4D tensor definition,
  // we simply use TensoeNHWC as a 4D tensor
  using Layout = cutlass::layout::TensorNHWC;
  // Match pytorch's behavior where the accumuation type is the same
  // as the output type
  using ElementCompute = float;
  using ReductionOp = cutlass::plus<ElementCompute>;
  constexpr int NUM_DIMS = 4;
  assert(rank <= NUM_DIMS);
  assert(reduction_axis < rank);
  assert(rank > 0);
  using TensorReduction = cutlass::reduction::device::TensorReduction<
    ElemOutputType,
    ElemInputType,
    Layout,
    ReductionOp,
    VectorLength,
    ElementCompute
  >;
  assert(shape[rank - 1] % VectorLength == 0);
  // adjust reduction_axis
  reduction_axis = NUM_DIMS - rank + reduction_axis;
  // cutlass's tensor_reduce only supports 4D tensors at the moment
  int64_t dst_dims[NUM_DIMS];
  int64_t src_dims[NUM_DIMS];
  for (int i = 0; i < NUM_DIMS; i++) {
    dst_dims[i] = 1;
    src_dims[i] = 1;
  }
  for (int i = 0; i < rank; i++) {
    int idx = NUM_DIMS - rank + i;
    dst_dims[idx] = shape[i];
    src_dims[idx] = shape[i];
  }
  dst_dims[reduction_axis] = 1;
  Layout::TensorCoord dst_extent(
    dst_dims[0], dst_dims[1], dst_dims[2], dst_dims[3]
  );
  Layout dst_layout(Layout::packed(dst_extent));
  Layout::TensorCoord src_extent(
    src_dims[0], src_dims[1], src_dims[2], src_dims[3]
  );
  Layout src_layout(Layout::packed(src_extent));
  ElementCompute reduction_identity = ElementCompute();
  TensorReduction reduction(src_extent, reduction_axis);
  ReductionOp reduction_op = ReductionOp();
  assert(dst_ptr);
  assert(src_ptr);
  cutlass::Status status = reduction.reduce(
      {dst_ptr, dst_layout},
      {src_ptr, src_layout},
      nullptr,
      reduction_identity,
      reduction_op,
      stream
    );
  CUTLASS_CHECK_REDUCE(status);
}
#else
#undef SKIP_GENERAL_REDUCTION
#endif // !SKIP_GENERAL_REDUCTION
#undef CUTLASS_CHECK_REDUCE

void reduce_sum_2(
    void *dst_ptr,
    void *src_ptr,
    int reduction_axis,
    const int64_t *shape,
    const int rank,
    uint8_t *workspace,
    cudaStream_t stream) {
  if (!dst_ptr) {
    throw std::runtime_error("dst_ptr is nullptr!");
  }
  if (!src_ptr) {
    throw std::runtime_error("src_ptr is nullptr!");
  }
  
  if (shape[rank - 1] % 4 == 0) {
    reduce_sum_2_launcher<float, float, float, uint4>
      (dst_ptr, src_ptr, reduction_axis, shape, rank, stream);
    return;
  }
  if (shape[rank - 1] % 2 == 0) {
    reduce_sum_2_launcher<float, float, float, uint2>
      (dst_ptr, src_ptr, reduction_axis, shape, rank, stream);
    return;
  }
  if (shape[rank - 1] % 1 == 0) {
    reduce_sum_2_launcher<float, float, float, unsigned>
      (dst_ptr, src_ptr, reduction_axis, shape, rank, stream);
    return;
  }
  throw std::runtime_error(
    "Unsupported workload for this reduce_sum_2 specialization."
  );
}