

#include <vector>
#include <assert.h>
#include <iostream>
#include <stdexcept>
#include <string>


#include <cuda_fp16.h>
#include <cuda_bf16.h>

using bfloat16 = nv_bfloat16;
using bfloat16_2 = nv_bfloat162;


        

#ifndef CHECK_ERROR_SPLIT
#define CHECK_ERROR_SPLIT(expr)                              \
  do {                                                       \
    cudaError_t status = (expr);                       \
    if (status != cudaSuccess) {                       \
      auto msg = std::string("Got error: ") +                \
        cudaGetErrorString(status) +                   \
        " at " + __FILE__ + ": " + std::to_string(__LINE__); \
      std::cerr << msg << std::endl;                         \
      throw std::runtime_error(msg);                         \
    }                                                        \
  } while (0)
#endif // CHECK_ERROR_SPLIT

#ifndef LAUNCH_CHECK_SPLIT
#define LAUNCH_CHECK_SPLIT() CHECK_ERROR_SPLIT(cudaGetLastError())
#endif // LAUNCH_CHECK_SPLIT

template <typename T, int64_t NumSplits>
struct OutputMetaData {
  T* outputs[NumSplits]; /* pointer to each output */
  int64_t split_dim_offsets[NumSplits]; /* offset of each output along
                                           the split dimension */
  int64_t split_dim_sizes[NumSplits]; /* cat dimension size of each output */
  int64_t num_elems[NumSplits]; /* number of the elements of each output */
};

template <int64_t Rank>
struct InputMetaData {
  int64_t input_shape[Rank];
  int64_t input_strides[Rank];
};

__host__ __device__ __forceinline__
int64_t get_num_elems(const int64_t *shape, int64_t rank) {
  int64_t num = 1;
  for (int64_t i = 0; i < rank; i++) {
    num *= shape[i];
  }
  return num;
}

template <int64_t Rank>
__host__ __device__ int64_t compute_input_elem_offset(
    const int64_t *input_shape,
    int64_t *input_strides,
    int64_t split_dim_size,
    int64_t split_dim,
    int64_t linear_idx) {
  int64_t offset = 0;
  for (int64_t i = Rank - 1; i >= 1; --i) {
    int64_t cur_dim_size = i == split_dim ? split_dim_size : input_shape[i];
    int64_t next_dim_idx = linear_idx / cur_dim_size;
    int64_t cur_dim_idx = linear_idx - cur_dim_size * next_dim_idx;
    int64_t cur_dim_offset = cur_dim_idx * input_strides[i];
    offset += cur_dim_offset;
    linear_idx = next_dim_idx;
  }
  return offset + linear_idx * input_strides[0];
}

template <typename READ_T, typename ELEM_T, int64_t Rank,
          int64_t NumSplits, int64_t ElemsPerThread>
__global__ void
split_kernel(
    const ELEM_T *orig_input,
    InputMetaData<Rank> input_meta,
    OutputMetaData<ELEM_T, NumSplits> output_meta,
    const int64_t split_dim,
    const int64_t input_split_dim_stride) {
  // split is the inverse of concat, so we
  //   (1) use blockIdx.y to specify the blocks for each ouput; and
  //   (2) use tid to access each output;
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const READ_T* input = reinterpret_cast<const READ_T*>(orig_input);

  READ_T* output =
      reinterpret_cast<READ_T*>(output_meta.outputs[blockIdx.y]);
  int64_t output_offset = output_meta.split_dim_offsets[blockIdx.y];
  int64_t num_output_elems = output_meta.num_elems[blockIdx.y];
  int64_t split_dim_size = output_meta.split_dim_sizes[blockIdx.y];
  int64_t input_offset = output_offset * input_split_dim_stride;

  unsigned constexpr read_t_sz = sizeof(READ_T);
  unsigned constexpr elem_t_sz = sizeof(ELEM_T);
  static_assert(read_t_sz >= elem_t_sz && (read_t_sz % elem_t_sz == 0));
  int64_t n_of_elem_t = read_t_sz / elem_t_sz;
  // number of READ_T elements per thread
  int64_t reads_per_thread_in_read_t = ElemsPerThread / n_of_elem_t;
  const int64_t num_elems_in_read_t = num_output_elems / n_of_elem_t;
  int64_t read_idx = tid;

#pragma unroll
  for (int64_t i = 0; i < reads_per_thread_in_read_t;
       i++, read_idx += blockDim.x * gridDim.x) {
    if (read_idx >= num_elems_in_read_t) {
      break;
    }
    /* make sure to adjust read_idx, which refers to location at
       (read_idx * n_of_elem_t) actually */
    int64_t input_elem_offset =
        compute_input_elem_offset<Rank>(input_meta.input_shape,
                                        input_meta.input_strides,
                                        split_dim_size,
                                        split_dim,
                                        read_idx * n_of_elem_t);

    READ_T tmp_v = input[(input_offset + input_elem_offset) / n_of_elem_t];
    output[read_idx] = tmp_v;
  }
}

enum class LoadVecType {
  VT_HALF = 0,
  VT_BFLOAT16,
  VT_FLOAT,
  VT_FLOAT2,
  VT_FLOAT4
};

template <typename ELEM_T>
static inline LoadVecType get_vec_type(
    const int64_t *shape, int64_t rank, int64_t dim) {
  assert(rank > 0);
  assert(dim < rank && dim >= 0);
  int64_t running_stride = shape[rank - 1];
  for (int64_t i = rank - 2; i >= dim; i--) {
    running_stride *= shape[i];
  }
  int64_t size_elem_t = sizeof(ELEM_T);

#define HANDLE_ONE_VEC_TYPE(load_vec_type, vec_type)  \
  if (sizeof(vec_type) % size_elem_t == 0) {          \
    int64_t n_of_elem_t = sizeof(vec_type) / size_elem_t; \
    if (running_stride % n_of_elem_t == 0) {          \
      return load_vec_type;                           \
    }                                                 \
  }

  HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT4, float4)
  HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT2, float2)
  HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT, float)
  if constexpr (std::is_same_v<ELEM_T, half>) {
    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_HALF, half)
  } else if constexpr (std::is_same_v<ELEM_T, bfloat16>) {
    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_BFLOAT16, bfloat16)
  }

#undef HANDLE_ONE_VEC_TYPE
  throw std::runtime_error(
      "Cannot resolve LoadVecType."
  );
}

template <typename ELEM_T, int64_t Rank, int64_t NumSplits,
          int64_t ElemsPerThread, int64_t ThreadsPerBlock>
void split_kernel_launcher(
    void *outputs[],
    int64_t *output_shapes[],
    const bool output_masks[],
    const void *input,
    const int64_t *input_shape,
    const int64_t split_dim,
    const int64_t split_sizes[],
    cudaStream_t stream
) {

  InputMetaData<Rank> input_meta;
  input_meta.input_strides[Rank - 1] = 1;
  input_meta.input_shape[Rank - 1] = input_shape[Rank - 1];
  for (int64_t i = Rank - 2; i >= 0; i--) {
    input_meta.input_strides[i] =
        input_meta.input_strides[i+1] * input_shape[i+1];
    input_meta.input_shape[i] = input_shape[i];
  }

  OutputMetaData<ELEM_T, NumSplits> output_meta;
  int64_t offset = 0;
  int64_t split_sizes_idx = 0;
  LoadVecType min_vec_type = LoadVecType::VT_FLOAT4;
  for (int64_t i = 0; i < NumSplits; i++) {
    while (!output_masks[split_sizes_idx]) {
      offset += split_sizes[split_sizes_idx];
      split_sizes_idx++;
    }
    output_meta.outputs[i] = static_cast<ELEM_T*>(outputs[i]);
    output_meta.split_dim_offsets[i] = offset;
    output_meta.split_dim_sizes[i] = output_shapes[i][split_dim];
    output_meta.num_elems[i] = get_num_elems(output_shapes[i], Rank);
    offset += output_meta.split_dim_sizes[i];
    split_sizes_idx++;
    LoadVecType vec_type =
        get_vec_type<ELEM_T>(output_shapes[i], Rank, split_dim);
    min_vec_type = vec_type < min_vec_type ? vec_type : min_vec_type;
  }

  int64_t max_num_output_elems = 0;
  for (int64_t i = 0; i < NumSplits; i++) {
    int64_t num_outputs = get_num_elems(output_shapes[i], Rank);
    max_num_output_elems = num_outputs > max_num_output_elems ?
                           num_outputs : max_num_output_elems;
  }
  int64_t m = (max_num_output_elems % (ThreadsPerBlock * ElemsPerThread) != 0);
  int64_t num_blocks_x =
      (max_num_output_elems / (ThreadsPerBlock * ElemsPerThread)) + m;
  dim3 grid_config = dim3(num_blocks_x, NumSplits);

#define HANDLE_ONE_VEC_TYPE(load_vec_type, vec_type)                   \
    if (min_vec_type == load_vec_type) {                               \
      if (ElemsPerThread * sizeof(ELEM_T) < sizeof(vec_type)) {        \
         throw std::runtime_error(                                     \
           std::string("No valid kernel available for ") + #vec_type); \
      }                                                                \
      split_kernel<vec_type, ELEM_T, Rank, NumSplits, ElemsPerThread>  \
        <<<grid_config, ThreadsPerBlock, 0, stream>>>(                 \
            static_cast<const ELEM_T*>(input),                         \
            input_meta,                                                \
            output_meta,                                               \
            split_dim,                                                 \
            input_meta.input_strides[split_dim]);                      \
      LAUNCH_CHECK_SPLIT();                                            \
      return;                                                          \
    }

    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT4, float4)
    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT2, float2)
    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT, float)
    if constexpr (std::is_same_v<ELEM_T, half>) {
      HANDLE_ONE_VEC_TYPE(LoadVecType::VT_HALF, half)
    } else if constexpr (std::is_same_v<ELEM_T, bfloat16>) {
      HANDLE_ONE_VEC_TYPE(LoadVecType::VT_BFLOAT16, bfloat16)
    }

  throw std::runtime_error("Invalid LoadVecType\n");
#undef HANDLE_ONE_VEC_TYPE
}

#undef CHECK_ERROR_SPLIT
#undef LAUNCH_CHECK_SPLIT

void split_1(
    void* outputs[],
    int64_t **output_shapes[],
    const bool output_masks[],
    const void* input,
    const int64_t *input_shape,
    int64_t real_num_splits,
    int64_t all_num_splits,
    int64_t split_sizes[],
    int64_t split_dim,
    int64_t rank,
    cudaStream_t stream
    ) {

  if (rank <= 0) {
    throw std::runtime_error("rank must be larger than 0!");
  }
  if (split_dim >= rank) {
    throw std::runtime_error("cat_dim must be smaller than rank!");
  }
  if (real_num_splits < 1) {
    throw std::runtime_error("the number of splits must be larger than 0!");
  }

  // now we update the shape for each output
  int64_t real_idx = 0;
  for (int64_t i = 0; i < all_num_splits; i++) {
    if (!output_masks[i]) {
      continue;
    }
    int64_t **shape_ptr = output_shapes[real_idx];
    for (int64_t dim_idx = 0; dim_idx < rank; dim_idx++) {
      *(shape_ptr[dim_idx]) = input_shape[dim_idx];
    }
    // update dim size for the split axis
    *(shape_ptr[split_dim]) = split_sizes[i];
    real_idx++;
  }

  int64_t split_dim_size = input_shape[split_dim];
  int64_t sum_of_split_sizes = 0;
  for (int64_t i = 0; i < all_num_splits; i++) {
    sum_of_split_sizes += split_sizes[i];
  }
  if (split_dim_size != sum_of_split_sizes) {
      throw std::runtime_error("unmatched split dim size!");
  }

  // If split dim is zero, we are done
  if (split_dim_size == 0) {
    return;
  }
  // If the input tensor is empty, we are done
  if (get_num_elems(input_shape, rank) == 0) {
    return;
  }
  // make sure input and outputs are valid
  if (!input) {
    throw std::runtime_error("input is NULL!");
  }
  for (int i = 0; i < real_num_splits; i++) {
    if (split_sizes[i] && !outputs[i]) {
      throw std::runtime_error("NULL output found at: " + std::to_string(i));
    }
  }


  if (rank == 2 && real_num_splits == 4) {


    int64_t local_shape0[2];

    local_shape0[0] = input_shape[0];

    local_shape0[1] = input_shape[1];

    local_shape0[split_dim] = split_sizes[0];



    int64_t local_shape1[2];

    local_shape1[0] = input_shape[0];

    local_shape1[1] = input_shape[1];

    local_shape1[split_dim] = split_sizes[1];



    int64_t local_shape2[2];

    local_shape2[0] = input_shape[0];

    local_shape2[1] = input_shape[1];

    local_shape2[split_dim] = split_sizes[2];



    int64_t local_shape3[2];

    local_shape3[0] = input_shape[0];

    local_shape3[1] = input_shape[1];

    local_shape3[split_dim] = split_sizes[3];



    int64_t* local_output_shapes[4] = {

      local_shape0,

      local_shape1,

      local_shape2,

      local_shape3
    };
    /* TODO: more profiling on ElemsPerThread and ThreadsPerBlock */
    split_kernel_launcher<float,
                          2/*Rank*/,
                          4/*NumSplits*/,
                          128/*ElemsPerThread*/,
                          128/*THREADS_PER_BLOCK*/>(
        outputs, local_output_shapes, output_masks, input, input_shape, split_dim, split_sizes, stream);
    return;
  }

  throw std::runtime_error(
      "Unsupported split kernel specialization!"
  );
}