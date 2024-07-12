
#include <cassert>
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/reduction/thread/reduction_operators.h"
#include "cutlass/reduction/device/tensor_reduce.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/util/host_tensor.h"



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
void reduce_sum_3_launcher(
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

void reduce_sum_3(
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
  
  if (shape[rank - 1] % 32 == 0) {
    reduce_sum_3_launcher<float, float, 32>(
        static_cast<float*>(dst_ptr),
        static_cast<float*>(src_ptr),
        reduction_axis,
        shape,
        rank,
        workspace,
        stream);
    return;
}
  if (shape[rank - 1] % 16 == 0) {
    reduce_sum_3_launcher<float, float, 16>(
        static_cast<float*>(dst_ptr),
        static_cast<float*>(src_ptr),
        reduction_axis,
        shape,
        rank,
        workspace,
        stream);
    return;
}
  if (shape[rank - 1] % 8 == 0) {
    reduce_sum_3_launcher<float, float, 8>(
        static_cast<float*>(dst_ptr),
        static_cast<float*>(src_ptr),
        reduction_axis,
        shape,
        rank,
        workspace,
        stream);
    return;
}
  if (shape[rank - 1] % 4 == 0) {
    reduce_sum_3_launcher<float, float, 4>(
        static_cast<float*>(dst_ptr),
        static_cast<float*>(src_ptr),
        reduction_axis,
        shape,
        rank,
        workspace,
        stream);
    return;
}
  if (shape[rank - 1] % 1 == 0) {
    reduce_sum_3_launcher<float, float, 1>(
        static_cast<float*>(dst_ptr),
        static_cast<float*>(src_ptr),
        reduction_axis,
        shape,
        rank,
        workspace,
        stream);
    return;
}
  throw std::runtime_error(
    "Unsupported workload for this reduce_sum_3 specialization."
  );
}