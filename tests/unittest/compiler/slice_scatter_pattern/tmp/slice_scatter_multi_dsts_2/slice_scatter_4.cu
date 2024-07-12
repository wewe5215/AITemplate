


#include <cuda_fp16.h>
#include <cuda_bf16.h>

using bfloat16 = nv_bfloat16;
using bfloat16_2 = nv_bfloat162;


        

#include <assert.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>



namespace {
#ifndef CHECK_ERROR_SLICE
#define CHECK_ERROR_SLICE(expr)                              \
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
#endif // CHECK_ERROR_SLICE

#ifndef LAUNCH_CHECK_SLICE
#define LAUNCH_CHECK_SLICE() CHECK_ERROR_SLICE(cudaGetLastError())
#endif // LAUNCH_CHECK_SLICE



template <typename T, int64_t  Rank, int64_t  NumInputs>
struct SliceMetaData {
  const T *inputs[NumInputs];
  int64_t slice_start_indices[NumInputs][Rank];
  int64_t slice_end_indices[NumInputs][Rank];
  int64_t  dim; // scatter dimension
  int64_t input_strides[NumInputs][Rank];
  int64_t num_elems[NumInputs];
  int64_t offsets[NumInputs];  // value of (dim_offset * output_dim_stride) at
                               // the dim axis in the output, where dim_offset
                               // is the offset of the scattered input at the
                               // dimension axis in the output
  int64_t dim_sizes[NumInputs];  // dimension size of the input to be scattered
                                 // at the dim axis
};

template <int64_t  Rank, int64_t  NumInputs>
struct ScatterMetaData {
  int64_t output_shape[Rank];
  int64_t output_strides[Rank];
};

__host__ __device__ __forceinline__
int64_t get_num_elems(const int64_t *shape, int64_t  rank) {
  int64_t  num = 1;
  for (int64_t  i = 0; i < rank; i++) {
    num *= shape[i];
  }
  return num;
}

template <int64_t  Rank>
__host__ __device__ int64_t compute_input_linear_index(
    const int64_t *input_strides,
    const int64_t *slice_start_indices,
    const int64_t *slice_end_indices,
    int64_t linear_idx) {
  int64_t input_offset = slice_start_indices[0] * input_strides[0];
  for (int64_t  i = Rank - 1; i > 0; i--) {
    int64_t  curr_output_dim_size = slice_end_indices[i] - slice_start_indices[i];
    int64_t curr_output_idx = linear_idx % curr_output_dim_size;
    int64_t curr_input_idx = curr_output_idx + slice_start_indices[i];
    input_offset += curr_input_idx * input_strides[i];
    linear_idx /= curr_output_dim_size;
  }
  return input_offset + linear_idx * input_strides[0];
}

template <int64_t  Rank>
__host__ __device__ int64_t compute_output_elem_offset(
    const int64_t *output_shape,
    const int64_t *output_strides,
    int64_t scatter_dim_size,
    const int64_t  scatter_dim,
    int64_t linear_idx) {
  int64_t offset = 0;
  for (int64_t  i = Rank - 1; i >= 1; --i) {
    int64_t cur_dim_size = i == scatter_dim ?  scatter_dim_size : output_shape[i];
    int64_t next_dim_idx = linear_idx / cur_dim_size;
    int64_t cur_dim_idx = linear_idx - cur_dim_size * next_dim_idx;
    int64_t cur_dim_offset = cur_dim_idx * output_strides[i];
    offset += cur_dim_offset;
    linear_idx = next_dim_idx;
  }
  return offset + linear_idx * output_strides[0];
}

template <typename READ_T, typename ELEM_T, int64_t  Rank,
          int64_t  NumInputs, int64_t  ElemsPerThread>
__global__ void
slice_scatter_kernel(
    ELEM_T *orig_output,
    SliceMetaData<ELEM_T, Rank, NumInputs> slice_meta_data,
    ScatterMetaData<Rank, NumInputs> scatter_meta_data) {
  const int64_t  tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t  block_y = blockIdx.y % NumInputs;

  READ_T* output = reinterpret_cast<READ_T*>(orig_output);
  const READ_T* input =
      reinterpret_cast<const READ_T*>(slice_meta_data.inputs[block_y]);
  int64_t num_elems = slice_meta_data.num_elems[block_y];
  const int64_t *input_strides = slice_meta_data.input_strides[block_y];
  const int64_t *slice_start_indices =
      slice_meta_data.slice_start_indices[block_y];
  const int64_t *slice_end_indices =
      slice_meta_data.slice_end_indices[block_y];

  int64_t  scatter_dim = slice_meta_data.dim;
  int64_t scatter_dim_size = slice_meta_data.dim_sizes[block_y];
  int64_t scatter_offset = slice_meta_data.offsets[block_y];

  constexpr unsigned read_t_sz = sizeof(READ_T);
  constexpr unsigned elem_t_sz = sizeof(ELEM_T);
  static_assert(read_t_sz >= elem_t_sz && (read_t_sz % elem_t_sz == 0));
  int64_t  n_of_elem_t = read_t_sz / elem_t_sz;
  // number of READ_T elements per thread
  int64_t  reads_per_thread_in_read_t = ElemsPerThread / n_of_elem_t;
  const int64_t num_elems_in_read_t = num_elems / n_of_elem_t;
  int64_t  read_idx = tid;

#pragma unroll
  for (int64_t  i = 0; i < reads_per_thread_in_read_t;
       i++, read_idx += blockDim.x * gridDim.x) {
    if (read_idx >= num_elems_in_read_t) {
      break;
    }
    /* make sure to adjust read_idx, which refers to location at
       (read_idx * n_of_elem_t) actually */
    int64_t input_idx = compute_input_linear_index<Rank>(
        input_strides,
        slice_start_indices,
        slice_end_indices,
        read_idx * n_of_elem_t);
    int64_t output_elem_offset = compute_output_elem_offset<Rank>(
        scatter_meta_data.output_shape,
        scatter_meta_data.output_strides,
        scatter_dim_size,
        scatter_dim,
        read_idx * n_of_elem_t);

    READ_T tmp_v = input[input_idx / n_of_elem_t];
    int64_t output_idx = (scatter_offset + output_elem_offset) / n_of_elem_t;
    
    output[output_idx] = tmp_v;
    
  }
}

enum class LoadVecType {
  VT_HALF = 0,
  VT_BFLOAT16 = 0,
  VT_FLOAT,
  VT_FLOAT2,
  VT_FLOAT4
};


template <typename ELEM_T>
static inline LoadVecType get_vec_type(int64_t dim_size) {
  int64_t  size_elem_t = sizeof(ELEM_T);

#define HANDLE_ONE_VEC_TYPE(load_vec_type, vec_type)  \
  if (sizeof(vec_type) % size_elem_t == 0) {          \
    int64_t  n_of_elem_t = sizeof(vec_type) / size_elem_t; \
    if (dim_size % n_of_elem_t == 0) {                \
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

template <typename ELEM_T, int64_t  Rank>
static LoadVecType get_input_vec_type(
    const int64_t *output_strides,
    const int64_t *input_shape,
    const int64_t *input_strides,
    const int64_t *slice_start_indices,
    const int64_t *slice_end_indices,
    int64_t  scatter_dim,
    int64_t  scatter_offset,
    int64_t  dim_size) {
  // get the outermost index where we continuous element accesses
  int64_t  flatten_index = Rank - 1;
  for (; flatten_index >= 0; flatten_index--) {
    if (slice_end_indices[flatten_index] - slice_start_indices[flatten_index] !=
        input_shape[flatten_index]) {
      break;
    }
  }
  // We have a full slice for the entire input
  if (flatten_index == -1) {
    flatten_index = 0;
  }

  int64_t input_start_offset =
      compute_input_linear_index<Rank>(input_strides,
                                       slice_start_indices,
                                       slice_end_indices,
                                       /*linear_idx*/0);
  LoadVecType slice_vec_type1 =
      get_vec_type<ELEM_T>(input_start_offset);
  LoadVecType slice_vec_type2;
  if (Rank == 1) {
    int64_t continuous_read_size = slice_end_indices[0] - slice_start_indices[0];
    slice_vec_type2 = get_vec_type<ELEM_T>(continuous_read_size);
  } else {
    int64_t continuous_read_size =
      (slice_end_indices[flatten_index] - slice_start_indices[flatten_index]) *
      input_strides[flatten_index];
    LoadVecType vec_type1 = get_vec_type<ELEM_T>(continuous_read_size);
    continuous_read_size =
      (input_shape[flatten_index] - slice_end_indices[flatten_index]) *
      input_strides[flatten_index];
    LoadVecType vec_type2 = get_vec_type<ELEM_T>(continuous_read_size);
    // find the smaller alignment reqirement between the sliced piece
    // and the rest along the flattened dimensions
    slice_vec_type2 = vec_type1 < vec_type2 ?  vec_type1 : vec_type2;
  }
  LoadVecType slice_min_vec_type = slice_vec_type1 < slice_vec_type2 ?
                                   slice_vec_type1 : slice_vec_type2;

  LoadVecType scatter_vec_type1 = get_vec_type<ELEM_T>(dim_size);
  LoadVecType scatter_vec_type2 = get_vec_type<ELEM_T>(scatter_offset);
  LoadVecType scatter_min_vec_type = scatter_vec_type1 < scatter_vec_type2 ?
                                     scatter_vec_type1 : scatter_vec_type2;

  LoadVecType min_vec_type = slice_min_vec_type < scatter_min_vec_type ?
                             slice_min_vec_type : scatter_min_vec_type;
  return min_vec_type;
}

template <typename ELEM_T, int64_t  Rank, int64_t  NumInputs>
void prepare_one_meta_data(
    int64_t  input_idx,
    SliceMetaData<ELEM_T, Rank, NumInputs> &slice_meta_data,
    ScatterMetaData<Rank, NumInputs> &scatter_meta_data,
    const ELEM_T *input,
    const int64_t *input_shape,
    const int64_t *slice_start_indices,
    const int64_t *slice_end_indices,
    int64_t  scatter_dim,
    int64_t  scatter_dim_offset) {
  slice_meta_data.inputs[input_idx] = input;
  slice_meta_data.input_strides[input_idx][Rank-1] = 1;
  for (int64_t  i = Rank - 2; i >= 0; i--) {
    slice_meta_data.input_strides[input_idx][i] =
        slice_meta_data.input_strides[input_idx][i+1] * input_shape[i+1];
  }

  slice_meta_data.num_elems[input_idx] = 1;
  for (int64_t  i = 0; i < Rank; i++) {
    int64_t slice_start_idx = slice_start_indices[i];
    int64_t slice_end_idx = slice_end_indices[i];
    int64_t input_dim = input_shape[i];

    if (!(slice_start_idx >= 0 && slice_start_idx <= input_dim)) {
        throw std::runtime_error("invalid slice_start_idx: " +
            std::to_string(slice_start_idx) +
            ", input_dim: " +
            std::to_string(input_dim) +
            ", i: " + std::to_string(i));
    }
    if (!(slice_end_idx >= 0 && slice_end_idx <= input_dim)) {
        throw std::runtime_error("invalid slice_end_idx: " +
            std::to_string(slice_end_idx) +
            ", input_dim: " +
            std::to_string(input_dim) +
            ", i: " + std::to_string(i));
    }
    if (slice_start_idx > slice_end_idx) {
        throw std::runtime_error(
            "expected slice_start_idx <= slice_end_idx but got slice_start_idx: " +
            std::to_string(slice_start_idx) +
            " and slice_end_idx: " +
            std::to_string(slice_end_idx) +
            ", i: " + std::to_string(i));
    }

    slice_meta_data.num_elems[input_idx] *= slice_end_idx - slice_start_idx;
    slice_meta_data.slice_start_indices[input_idx][i] = slice_start_idx;
    slice_meta_data.slice_end_indices[input_idx][i] = slice_end_idx;
  }

  slice_meta_data.dim_sizes[input_idx] =
      slice_end_indices[scatter_dim] - slice_start_indices[scatter_dim];
  slice_meta_data.offsets[input_idx] =
      scatter_dim_offset * scatter_meta_data.output_strides[scatter_dim];
}

template <typename ELEM_T, int64_t  Rank, int64_t  NumInputs,
          int64_t  ElemsPerThread, int64_t  ThreadsPerBlock>
void slice_scatter_kernel_launcher(
    ELEM_T *output,
    int64_t output_offset,
    const int64_t *output_shape,
    const ELEM_T *inputs[],
    const int64_t *input_shapes[],
    const std::vector<std::vector<int64_t>> &slice_start_indices,
    const std::vector<std::vector<int64_t>> &slice_end_indices,
    int64_t  scatter_dim,
    cudaStream_t stream
) {
  SliceMetaData<ELEM_T, Rank, NumInputs> slice_meta_data;
  ScatterMetaData<Rank, NumInputs> scatter_meta_data;

  // meta data for placing sliced output
  scatter_meta_data.output_strides[Rank-1] = 1;
  if (output_shape[Rank-1] < 0) {
    throw std::runtime_error("invalid output_shape[Rank-1]: " +
        std::to_string(output_shape[Rank-1]) +
        ", Rank: " + std::to_string(Rank));
  }
  scatter_meta_data.output_shape[Rank-1] = output_shape[Rank-1];
  for (int64_t  i = Rank - 2; i >= 0; i--) {
    scatter_meta_data.output_strides[i] =
        scatter_meta_data.output_strides[i+1] * output_shape[i+1];
    if (output_shape[i] < 0) {
      throw std::runtime_error("invalid output_shape[i]: " +
          std::to_string(output_shape[i]) +
          ", i: " + std::to_string(i));
    }
    scatter_meta_data.output_shape[i] = output_shape[i];
  }

  int64_t  scatter_dim_offset = 0;
  slice_meta_data.dim = scatter_dim;
  for (int64_t  i = 0; i < NumInputs; i++) {
    prepare_one_meta_data(i, slice_meta_data, scatter_meta_data,
                          inputs[i], input_shapes[i],
                          slice_start_indices[i].data(),
                          slice_end_indices[i].data(),
                          scatter_dim, scatter_dim_offset);
    scatter_dim_offset += slice_meta_data.dim_sizes[i];
  }

  LoadVecType min_vec_type = get_vec_type<ELEM_T>(output_offset);
  for (int64_t  i = 0; i < NumInputs; i++) {
    LoadVecType vec_type = get_input_vec_type<ELEM_T, Rank>(
        scatter_meta_data.output_strides,
        input_shapes[i],
        slice_meta_data.input_strides[i],
        slice_start_indices[i].data(),
        slice_end_indices[i].data(),
        scatter_dim,
        slice_meta_data.offsets[i],
        slice_meta_data.dim_sizes[i]);
    min_vec_type = vec_type < min_vec_type ? vec_type : min_vec_type;
  }

  // setup kernel configs
  int64_t max_num_elems = 0;
  for (int64_t  i = 0; i < NumInputs; i++) {
    if (slice_meta_data.num_elems[i] > max_num_elems) {
      max_num_elems =  slice_meta_data.num_elems[i];
    }
  }

  if (max_num_elems <= 0) {
    throw std::runtime_error("invalid max_num_elems: " +
        std::to_string(max_num_elems));
  }

  int64_t  m = max_num_elems % (ThreadsPerBlock * ElemsPerThread) != 0;
  int64_t  num_blocks_x =
      (max_num_elems / (ThreadsPerBlock * ElemsPerThread)) + m;
  dim3 grid_config = dim3(num_blocks_x, NumInputs);

#define HANDLE_ONE_VEC_TYPE(load_vec_type, vec_type)                          \
    if (min_vec_type == load_vec_type) {                                      \
      if (ElemsPerThread * sizeof(ELEM_T) < sizeof(vec_type)) {               \
         throw std::runtime_error(                                            \
           std::string("No valid kernel available for ") + #vec_type);        \
      }                                                                       \
      slice_scatter_kernel<vec_type, ELEM_T, Rank, NumInputs, ElemsPerThread> \
        <<<grid_config, ThreadsPerBlock, 0, stream>>>(                        \
            output + output_offset,                                           \
            slice_meta_data,                                                  \
            scatter_meta_data);                                               \
      LAUNCH_CHECK_SLICE();                                                   \
      return;                                                                 \
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

std::tuple<std::vector<int64_t>, std::vector<int64_t>>
normalize_slice_indices(
    const int64_t *input_shape,
    const int64_t *orig_slice_start_indices,
    const int64_t *orig_slice_end_indices,
    int64_t  rank) {
  std::vector<int64_t> slice_start_indices(rank);
  std::vector<int64_t> slice_end_indices(rank);
  for (int64_t  i = 0; i < rank; i++) {
    if (input_shape[i] < 0) {
        throw std::runtime_error("invalid input_shape: " +
            std::to_string(input_shape[i]) +
            ", i: " +
            std::to_string(i));
    }
    slice_start_indices[i] = orig_slice_start_indices[i] < 0 ?
                             input_shape[i] + orig_slice_start_indices[i]:
                             orig_slice_start_indices[i];
    // make it compatible with PyTorch
    slice_start_indices[i] = slice_start_indices[i] < 0 ?
                             0 : slice_start_indices[i];
    if (slice_start_indices[i] < 0) {
      slice_start_indices[i] = 0;
    }
    if (slice_start_indices[i] > input_shape[i]) {
      slice_start_indices[i] = input_shape[i];
    }

    slice_end_indices[i] =  orig_slice_end_indices[i] < 0 ?
                            input_shape[i] + orig_slice_end_indices[i]:
                            orig_slice_end_indices[i];
    // make it compatible with PyTorch
    slice_end_indices[i] = slice_end_indices[i] < 0 ?
                           0 : slice_end_indices[i];
    if (slice_end_indices[i] < 0) {
      slice_end_indices[i] = 0;
    }
    if (slice_end_indices[i] > input_shape[i]) {
      slice_end_indices[i] = input_shape[i];
    }

    // make it compatible with PyTorch
    if (slice_start_indices[i] > slice_end_indices[i]) {
      slice_start_indices[i] = slice_end_indices[i];
    }
  }

  return {slice_start_indices, slice_end_indices};
}
} // namespace


void slice_scatter_4(
    void *output,
    int64_t *output_shape[],
    const void *inputs[],
    const int64_t *input_shapes[],
    const int64_t *orig_slice_start_indices[],
    const int64_t *orig_slice_end_indices[],
    int64_t  scatter_dim,
    int64_t  rank,
    int64_t  num_inputs,
    cudaStream_t stream
    ) {

  if (rank <= 0) {
    throw std::runtime_error("rank must > 0!");
  }
  if (scatter_dim >= rank) {
    throw std::runtime_error("scatter_dim must < rank!");
  }
  if (num_inputs < 1) {
    throw std::runtime_error("num_inputs must be larger than 0!");
  }

  // clip slip start and end indices
  std::vector<std::vector<int64_t>> slice_start_indices(num_inputs);
  std::vector<std::vector<int64_t>> slice_end_indices(num_inputs);
  std::vector<int64_t> output_dim_sizes;
  for (int64_t i = 0; i < num_inputs; i++) {
    std::vector<int64_t> start_indices;
    std::vector<int64_t> end_indices;
    std::tie(start_indices, end_indices) =
        normalize_slice_indices(input_shapes[i],
                                orig_slice_start_indices[i],
                                orig_slice_end_indices[i],
                                rank);
    slice_start_indices[i] = start_indices;
    slice_end_indices[i] = end_indices;
  }


  int64_t output_scatter_dim_value = 0;
  for (int64_t i = 0; i < num_inputs; i++) {
    output_scatter_dim_value +=
        slice_end_indices[i][scatter_dim] - slice_start_indices[i][scatter_dim];
  }
  
  for (int64_t  i = 0; i < rank; i++) {
    if (i == scatter_dim) {

      // skip updating output_shape[i]

    } else {
      int64_t dim = slice_end_indices[0][i] - slice_start_indices[0][i];
      for (int64_t  j = 1; j < num_inputs; j++) {
        if (slice_end_indices[j][i] - slice_start_indices[j][i] != dim) {
          throw std::runtime_error("invalid indices");
        }

      // skip updating output_shape[i]

      }
    }
  }

  // If all input tensors are empty, we are done
  bool empty = true;
  for (int64_t i = 0; i < num_inputs; i++) {
    if (get_num_elems(input_shapes[i], rank) != 0) {
      empty = false;
      // make sure input is valid for each non-zero-size tensor
      if (!inputs[i]) {
        throw std::runtime_error("NULL input is found at: " + std::to_string(i));
      }
    }
  }

  if (empty)
    return;

  // if we output has any zero dim size, we are done
  for (int64_t i = 0; i < rank; i++) {
    if (*output_shape[i] == 0)
      return;
  }
  // make sure we have a valid output pointer
  if (!output) {
    throw std::runtime_error("output is NULL!");
  }


  if (rank == 3 && num_inputs == 4) {
    int64_t local_output_shape[3];

    local_output_shape[0] = *output_shape[0];

    local_output_shape[1] = *output_shape[1];

    local_output_shape[2] = *output_shape[2];

    slice_scatter_kernel_launcher<half,
                                  3/*Rank*/,
                                  4/*NumInputs*/,
                                  256/*ElemsPerThread*/,
                                  128/*ThreadsPerBlock*/>(
        static_cast<half*>(output), 0, local_output_shape,
        reinterpret_cast<const half**>(inputs), input_shapes,
        slice_start_indices, slice_end_indices, scatter_dim, stream);
    return;
  }

  throw std::runtime_error(
      "Unsupported cat kernel specialization!"
  );
}