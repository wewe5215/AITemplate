
#pragma once

#include "logging.h"
#include "device_functions-generated.h"
#include "model_interface.h"
#include "raii_wrapper.h"
#include "model.h"
#include "macros.h"
#include "jagged.h"
#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <math.h>
#include <iomanip>


void gemm_rcr_bias_0(
  void*,
  void*,
  void*,
  void*,
  uint8_t*,

    int,


  int64_t*,

  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,

  int64_t*,

  cudaStream_t
);

    void flash_attention_2(void* output,
                   const void* qkv,
                   const int* cu_seqlens,
                   float* softmax_lse,
                   float* o_tmp,
                   int batch_size,
                   int seq_len,
                   int num_heads,
                   int head_size,
                   float p_dropout,
                   float softmax_scale,
                   bool is_causal,
                   bool loop,
                   cudaStream_t stream);
    

void gemm_rcr_bias_add_4(
  void*,
  void*,
  void*,
  void*,

  void*,
  uint8_t*,

    int,


  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,

  cudaStream_t
);

    cudaError_t layernorm_6(void* output,
                   void* input,
                   const void* gamma,
                   const void* beta,
                   int m,
                   int n,
                   const float eps,
                   cudaStream_t stream);
    

void gemm_rcr_bias_fast_gelu_7(
  void*,
  void*,
  void*,
  void*,
  uint8_t*,

    int,


  int64_t*,

  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,

  int64_t*,

  cudaStream_t
);

void gemm_rcr_bias_add_8(
  void*,
  void*,
  void*,
  void*,

  void*,
  uint8_t*,

    int,


  int64_t*,

  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,

  int64_t*,

  cudaStream_t
);

namespace ait {

// Model is the class that actually performs inference. It owns memory for
// intermediate tensors and dynamic dimensions. Constants are owned by
// the model's owning container object, and input/output memory is owned
// by the user.
// Once an inference run has started, it is not safe to re-use the Model
// until the run has finished!
class Model : public ModelBase<Model> {
  
  

  public:
    Model(
        size_t blob_size,
        size_t workspace_size,
        size_t unique_workspace_size,
        size_t num_inputs,
        size_t num_outputs,
        size_t num_unbound_constants,
        uint8_t* constants,
        AITemplateAllocator& allocator)
        : ModelBase(
            blob_size,
            workspace_size,
            unique_workspace_size,
            num_inputs,
            num_outputs,
            num_unbound_constants,
            constants,
            allocator) {
         constant_name_to_ptr_["bert_encoder_layer_0_attention_self_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_0_attention_self_qkv_weight));
     constant_name_to_ptr_["bert_encoder_layer_0_attention_self_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_0_attention_self_qkv_bias));
     constant_name_to_ptr_["bert_encoder_layer_0_attention_self_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_0_attention_self_cu_length));
     constant_name_to_ptr_["bert_encoder_layer_0_attention_self_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_0_attention_self_proj_weight));
     constant_name_to_ptr_["bert_encoder_layer_0_attention_self_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_0_attention_self_proj_bias));
     constant_name_to_ptr_["bert_encoder_layer_0_attention_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_0_attention_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_0_attention_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_0_attention_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_0_intermediate_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_0_intermediate_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_0_intermediate_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_0_intermediate_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_0_output_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_0_output_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_0_output_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_0_output_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_0_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_0_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_0_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_0_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_1_attention_self_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_1_attention_self_qkv_weight));
     constant_name_to_ptr_["bert_encoder_layer_1_attention_self_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_1_attention_self_qkv_bias));
     constant_name_to_ptr_["bert_encoder_layer_1_attention_self_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_1_attention_self_cu_length));
     constant_name_to_ptr_["bert_encoder_layer_1_attention_self_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_1_attention_self_proj_weight));
     constant_name_to_ptr_["bert_encoder_layer_1_attention_self_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_1_attention_self_proj_bias));
     constant_name_to_ptr_["bert_encoder_layer_1_attention_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_1_attention_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_1_attention_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_1_attention_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_1_intermediate_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_1_intermediate_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_1_intermediate_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_1_intermediate_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_1_output_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_1_output_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_1_output_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_1_output_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_1_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_1_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_1_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_1_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_2_attention_self_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_2_attention_self_qkv_weight));
     constant_name_to_ptr_["bert_encoder_layer_2_attention_self_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_2_attention_self_qkv_bias));
     constant_name_to_ptr_["bert_encoder_layer_2_attention_self_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_2_attention_self_cu_length));
     constant_name_to_ptr_["bert_encoder_layer_2_attention_self_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_2_attention_self_proj_weight));
     constant_name_to_ptr_["bert_encoder_layer_2_attention_self_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_2_attention_self_proj_bias));
     constant_name_to_ptr_["bert_encoder_layer_2_attention_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_2_attention_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_2_attention_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_2_attention_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_2_intermediate_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_2_intermediate_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_2_intermediate_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_2_intermediate_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_2_output_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_2_output_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_2_output_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_2_output_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_2_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_2_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_2_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_2_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_3_attention_self_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_3_attention_self_qkv_weight));
     constant_name_to_ptr_["bert_encoder_layer_3_attention_self_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_3_attention_self_qkv_bias));
     constant_name_to_ptr_["bert_encoder_layer_3_attention_self_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_3_attention_self_cu_length));
     constant_name_to_ptr_["bert_encoder_layer_3_attention_self_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_3_attention_self_proj_weight));
     constant_name_to_ptr_["bert_encoder_layer_3_attention_self_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_3_attention_self_proj_bias));
     constant_name_to_ptr_["bert_encoder_layer_3_attention_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_3_attention_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_3_attention_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_3_attention_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_3_intermediate_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_3_intermediate_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_3_intermediate_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_3_intermediate_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_3_output_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_3_output_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_3_output_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_3_output_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_3_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_3_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_3_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_3_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_4_attention_self_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_4_attention_self_qkv_weight));
     constant_name_to_ptr_["bert_encoder_layer_4_attention_self_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_4_attention_self_qkv_bias));
     constant_name_to_ptr_["bert_encoder_layer_4_attention_self_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_4_attention_self_cu_length));
     constant_name_to_ptr_["bert_encoder_layer_4_attention_self_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_4_attention_self_proj_weight));
     constant_name_to_ptr_["bert_encoder_layer_4_attention_self_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_4_attention_self_proj_bias));
     constant_name_to_ptr_["bert_encoder_layer_4_attention_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_4_attention_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_4_attention_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_4_attention_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_4_intermediate_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_4_intermediate_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_4_intermediate_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_4_intermediate_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_4_output_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_4_output_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_4_output_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_4_output_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_4_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_4_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_4_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_4_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_5_attention_self_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_5_attention_self_qkv_weight));
     constant_name_to_ptr_["bert_encoder_layer_5_attention_self_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_5_attention_self_qkv_bias));
     constant_name_to_ptr_["bert_encoder_layer_5_attention_self_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_5_attention_self_cu_length));
     constant_name_to_ptr_["bert_encoder_layer_5_attention_self_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_5_attention_self_proj_weight));
     constant_name_to_ptr_["bert_encoder_layer_5_attention_self_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_5_attention_self_proj_bias));
     constant_name_to_ptr_["bert_encoder_layer_5_attention_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_5_attention_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_5_attention_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_5_attention_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_5_intermediate_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_5_intermediate_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_5_intermediate_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_5_intermediate_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_5_output_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_5_output_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_5_output_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_5_output_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_5_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_5_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_5_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_5_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_6_attention_self_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_6_attention_self_qkv_weight));
     constant_name_to_ptr_["bert_encoder_layer_6_attention_self_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_6_attention_self_qkv_bias));
     constant_name_to_ptr_["bert_encoder_layer_6_attention_self_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_6_attention_self_cu_length));
     constant_name_to_ptr_["bert_encoder_layer_6_attention_self_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_6_attention_self_proj_weight));
     constant_name_to_ptr_["bert_encoder_layer_6_attention_self_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_6_attention_self_proj_bias));
     constant_name_to_ptr_["bert_encoder_layer_6_attention_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_6_attention_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_6_attention_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_6_attention_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_6_intermediate_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_6_intermediate_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_6_intermediate_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_6_intermediate_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_6_output_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_6_output_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_6_output_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_6_output_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_6_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_6_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_6_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_6_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_7_attention_self_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_7_attention_self_qkv_weight));
     constant_name_to_ptr_["bert_encoder_layer_7_attention_self_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_7_attention_self_qkv_bias));
     constant_name_to_ptr_["bert_encoder_layer_7_attention_self_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_7_attention_self_cu_length));
     constant_name_to_ptr_["bert_encoder_layer_7_attention_self_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_7_attention_self_proj_weight));
     constant_name_to_ptr_["bert_encoder_layer_7_attention_self_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_7_attention_self_proj_bias));
     constant_name_to_ptr_["bert_encoder_layer_7_attention_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_7_attention_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_7_attention_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_7_attention_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_7_intermediate_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_7_intermediate_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_7_intermediate_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_7_intermediate_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_7_output_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_7_output_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_7_output_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_7_output_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_7_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_7_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_7_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_7_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_8_attention_self_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_8_attention_self_qkv_weight));
     constant_name_to_ptr_["bert_encoder_layer_8_attention_self_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_8_attention_self_qkv_bias));
     constant_name_to_ptr_["bert_encoder_layer_8_attention_self_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_8_attention_self_cu_length));
     constant_name_to_ptr_["bert_encoder_layer_8_attention_self_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_8_attention_self_proj_weight));
     constant_name_to_ptr_["bert_encoder_layer_8_attention_self_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_8_attention_self_proj_bias));
     constant_name_to_ptr_["bert_encoder_layer_8_attention_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_8_attention_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_8_attention_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_8_attention_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_8_intermediate_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_8_intermediate_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_8_intermediate_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_8_intermediate_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_8_output_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_8_output_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_8_output_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_8_output_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_8_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_8_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_8_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_8_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_9_attention_self_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_9_attention_self_qkv_weight));
     constant_name_to_ptr_["bert_encoder_layer_9_attention_self_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_9_attention_self_qkv_bias));
     constant_name_to_ptr_["bert_encoder_layer_9_attention_self_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_9_attention_self_cu_length));
     constant_name_to_ptr_["bert_encoder_layer_9_attention_self_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_9_attention_self_proj_weight));
     constant_name_to_ptr_["bert_encoder_layer_9_attention_self_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_9_attention_self_proj_bias));
     constant_name_to_ptr_["bert_encoder_layer_9_attention_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_9_attention_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_9_attention_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_9_attention_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_9_intermediate_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_9_intermediate_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_9_intermediate_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_9_intermediate_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_9_output_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_9_output_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_9_output_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_9_output_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_9_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_9_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_9_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_9_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_10_attention_self_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_10_attention_self_qkv_weight));
     constant_name_to_ptr_["bert_encoder_layer_10_attention_self_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_10_attention_self_qkv_bias));
     constant_name_to_ptr_["bert_encoder_layer_10_attention_self_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_10_attention_self_cu_length));
     constant_name_to_ptr_["bert_encoder_layer_10_attention_self_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_10_attention_self_proj_weight));
     constant_name_to_ptr_["bert_encoder_layer_10_attention_self_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_10_attention_self_proj_bias));
     constant_name_to_ptr_["bert_encoder_layer_10_attention_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_10_attention_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_10_attention_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_10_attention_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_10_intermediate_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_10_intermediate_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_10_intermediate_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_10_intermediate_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_10_output_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_10_output_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_10_output_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_10_output_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_10_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_10_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_10_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_10_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_11_attention_self_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_11_attention_self_qkv_weight));
     constant_name_to_ptr_["bert_encoder_layer_11_attention_self_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_11_attention_self_qkv_bias));
     constant_name_to_ptr_["bert_encoder_layer_11_attention_self_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_11_attention_self_cu_length));
     constant_name_to_ptr_["bert_encoder_layer_11_attention_self_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_11_attention_self_proj_weight));
     constant_name_to_ptr_["bert_encoder_layer_11_attention_self_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_11_attention_self_proj_bias));
     constant_name_to_ptr_["bert_encoder_layer_11_attention_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_11_attention_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_11_attention_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_11_attention_output_LayerNorm_bias));
     constant_name_to_ptr_["bert_encoder_layer_11_intermediate_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_11_intermediate_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_11_intermediate_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_11_intermediate_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_11_output_dense_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_11_output_dense_weight));
     constant_name_to_ptr_["bert_encoder_layer_11_output_dense_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_11_output_dense_bias));
     constant_name_to_ptr_["bert_encoder_layer_11_output_LayerNorm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_11_output_LayerNorm_weight));
     constant_name_to_ptr_["bert_encoder_layer_11_output_LayerNorm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&bert_encoder_layer_11_output_LayerNorm_bias));
    auto* blob_ptr = static_cast<uint8_t*>(blob_.get());
        reshape_1_0 = reinterpret_cast<decltype(reshape_1_0)>(blob_ptr + 0);
    flash_attention_2_0 = reinterpret_cast<decltype(flash_attention_2_0)>(blob_ptr + 393216);
    reshape_5_0 = reinterpret_cast<decltype(reshape_5_0)>(blob_ptr + 0);
    layernorm_6_0 = reinterpret_cast<decltype(layernorm_6_0)>(blob_ptr + 393216);
    gemm_rcr_bias_fast_gelu_7_0 = reinterpret_cast<decltype(gemm_rcr_bias_fast_gelu_7_0)>(blob_ptr + 0);
    gemm_rcr_bias_add_8_0 = reinterpret_cast<decltype(gemm_rcr_bias_add_8_0)>(blob_ptr + 491520);
    layernorm_9_0 = reinterpret_cast<decltype(layernorm_9_0)>(blob_ptr + 294912);
    reshape_11_0 = reinterpret_cast<decltype(reshape_11_0)>(blob_ptr + 0);
    flash_attention_12_0 = reinterpret_cast<decltype(flash_attention_12_0)>(blob_ptr + 393216);
    reshape_15_0 = reinterpret_cast<decltype(reshape_15_0)>(blob_ptr + 0);
    layernorm_16_0 = reinterpret_cast<decltype(layernorm_16_0)>(blob_ptr + 393216);
    gemm_rcr_bias_fast_gelu_17_0 = reinterpret_cast<decltype(gemm_rcr_bias_fast_gelu_17_0)>(blob_ptr + 0);
    gemm_rcr_bias_add_18_0 = reinterpret_cast<decltype(gemm_rcr_bias_add_18_0)>(blob_ptr + 491520);
    layernorm_19_0 = reinterpret_cast<decltype(layernorm_19_0)>(blob_ptr + 294912);
    reshape_21_0 = reinterpret_cast<decltype(reshape_21_0)>(blob_ptr + 0);
    flash_attention_22_0 = reinterpret_cast<decltype(flash_attention_22_0)>(blob_ptr + 393216);
    reshape_25_0 = reinterpret_cast<decltype(reshape_25_0)>(blob_ptr + 0);
    layernorm_26_0 = reinterpret_cast<decltype(layernorm_26_0)>(blob_ptr + 393216);
    gemm_rcr_bias_fast_gelu_27_0 = reinterpret_cast<decltype(gemm_rcr_bias_fast_gelu_27_0)>(blob_ptr + 0);
    gemm_rcr_bias_add_28_0 = reinterpret_cast<decltype(gemm_rcr_bias_add_28_0)>(blob_ptr + 491520);
    layernorm_29_0 = reinterpret_cast<decltype(layernorm_29_0)>(blob_ptr + 294912);
    reshape_31_0 = reinterpret_cast<decltype(reshape_31_0)>(blob_ptr + 0);
    flash_attention_32_0 = reinterpret_cast<decltype(flash_attention_32_0)>(blob_ptr + 393216);
    reshape_35_0 = reinterpret_cast<decltype(reshape_35_0)>(blob_ptr + 0);
    layernorm_36_0 = reinterpret_cast<decltype(layernorm_36_0)>(blob_ptr + 393216);
    gemm_rcr_bias_fast_gelu_37_0 = reinterpret_cast<decltype(gemm_rcr_bias_fast_gelu_37_0)>(blob_ptr + 0);
    gemm_rcr_bias_add_38_0 = reinterpret_cast<decltype(gemm_rcr_bias_add_38_0)>(blob_ptr + 491520);
    layernorm_39_0 = reinterpret_cast<decltype(layernorm_39_0)>(blob_ptr + 294912);
    reshape_41_0 = reinterpret_cast<decltype(reshape_41_0)>(blob_ptr + 0);
    flash_attention_42_0 = reinterpret_cast<decltype(flash_attention_42_0)>(blob_ptr + 393216);
    reshape_45_0 = reinterpret_cast<decltype(reshape_45_0)>(blob_ptr + 0);
    layernorm_46_0 = reinterpret_cast<decltype(layernorm_46_0)>(blob_ptr + 393216);
    gemm_rcr_bias_fast_gelu_47_0 = reinterpret_cast<decltype(gemm_rcr_bias_fast_gelu_47_0)>(blob_ptr + 0);
    gemm_rcr_bias_add_48_0 = reinterpret_cast<decltype(gemm_rcr_bias_add_48_0)>(blob_ptr + 491520);
    layernorm_49_0 = reinterpret_cast<decltype(layernorm_49_0)>(blob_ptr + 294912);
    reshape_51_0 = reinterpret_cast<decltype(reshape_51_0)>(blob_ptr + 0);
    flash_attention_52_0 = reinterpret_cast<decltype(flash_attention_52_0)>(blob_ptr + 393216);
    reshape_55_0 = reinterpret_cast<decltype(reshape_55_0)>(blob_ptr + 0);
    layernorm_56_0 = reinterpret_cast<decltype(layernorm_56_0)>(blob_ptr + 393216);
    gemm_rcr_bias_fast_gelu_57_0 = reinterpret_cast<decltype(gemm_rcr_bias_fast_gelu_57_0)>(blob_ptr + 0);
    gemm_rcr_bias_add_58_0 = reinterpret_cast<decltype(gemm_rcr_bias_add_58_0)>(blob_ptr + 491520);
    layernorm_59_0 = reinterpret_cast<decltype(layernorm_59_0)>(blob_ptr + 294912);
    reshape_61_0 = reinterpret_cast<decltype(reshape_61_0)>(blob_ptr + 0);
    flash_attention_62_0 = reinterpret_cast<decltype(flash_attention_62_0)>(blob_ptr + 393216);
    reshape_65_0 = reinterpret_cast<decltype(reshape_65_0)>(blob_ptr + 0);
    layernorm_66_0 = reinterpret_cast<decltype(layernorm_66_0)>(blob_ptr + 393216);
    gemm_rcr_bias_fast_gelu_67_0 = reinterpret_cast<decltype(gemm_rcr_bias_fast_gelu_67_0)>(blob_ptr + 0);
    gemm_rcr_bias_add_68_0 = reinterpret_cast<decltype(gemm_rcr_bias_add_68_0)>(blob_ptr + 491520);
    layernorm_69_0 = reinterpret_cast<decltype(layernorm_69_0)>(blob_ptr + 294912);
    reshape_71_0 = reinterpret_cast<decltype(reshape_71_0)>(blob_ptr + 0);
    flash_attention_72_0 = reinterpret_cast<decltype(flash_attention_72_0)>(blob_ptr + 393216);
    reshape_75_0 = reinterpret_cast<decltype(reshape_75_0)>(blob_ptr + 0);
    layernorm_76_0 = reinterpret_cast<decltype(layernorm_76_0)>(blob_ptr + 393216);
    gemm_rcr_bias_fast_gelu_77_0 = reinterpret_cast<decltype(gemm_rcr_bias_fast_gelu_77_0)>(blob_ptr + 0);
    gemm_rcr_bias_add_78_0 = reinterpret_cast<decltype(gemm_rcr_bias_add_78_0)>(blob_ptr + 491520);
    layernorm_79_0 = reinterpret_cast<decltype(layernorm_79_0)>(blob_ptr + 294912);
    reshape_81_0 = reinterpret_cast<decltype(reshape_81_0)>(blob_ptr + 0);
    flash_attention_82_0 = reinterpret_cast<decltype(flash_attention_82_0)>(blob_ptr + 393216);
    reshape_85_0 = reinterpret_cast<decltype(reshape_85_0)>(blob_ptr + 0);
    layernorm_86_0 = reinterpret_cast<decltype(layernorm_86_0)>(blob_ptr + 393216);
    gemm_rcr_bias_fast_gelu_87_0 = reinterpret_cast<decltype(gemm_rcr_bias_fast_gelu_87_0)>(blob_ptr + 0);
    gemm_rcr_bias_add_88_0 = reinterpret_cast<decltype(gemm_rcr_bias_add_88_0)>(blob_ptr + 491520);
    layernorm_89_0 = reinterpret_cast<decltype(layernorm_89_0)>(blob_ptr + 294912);
    reshape_91_0 = reinterpret_cast<decltype(reshape_91_0)>(blob_ptr + 0);
    flash_attention_92_0 = reinterpret_cast<decltype(flash_attention_92_0)>(blob_ptr + 393216);
    reshape_95_0 = reinterpret_cast<decltype(reshape_95_0)>(blob_ptr + 0);
    layernorm_96_0 = reinterpret_cast<decltype(layernorm_96_0)>(blob_ptr + 393216);
    gemm_rcr_bias_fast_gelu_97_0 = reinterpret_cast<decltype(gemm_rcr_bias_fast_gelu_97_0)>(blob_ptr + 0);
    gemm_rcr_bias_add_98_0 = reinterpret_cast<decltype(gemm_rcr_bias_add_98_0)>(blob_ptr + 491520);
    layernorm_99_0 = reinterpret_cast<decltype(layernorm_99_0)>(blob_ptr + 294912);
    reshape_101_0 = reinterpret_cast<decltype(reshape_101_0)>(blob_ptr + 0);
    flash_attention_102_0 = reinterpret_cast<decltype(flash_attention_102_0)>(blob_ptr + 393216);
    reshape_105_0 = reinterpret_cast<decltype(reshape_105_0)>(blob_ptr + 0);
    layernorm_106_0 = reinterpret_cast<decltype(layernorm_106_0)>(blob_ptr + 393216);
    gemm_rcr_bias_fast_gelu_107_0 = reinterpret_cast<decltype(gemm_rcr_bias_fast_gelu_107_0)>(blob_ptr + 0);
    gemm_rcr_bias_add_108_0 = reinterpret_cast<decltype(gemm_rcr_bias_add_108_0)>(blob_ptr + 491520);
    layernorm_109_0 = reinterpret_cast<decltype(layernorm_109_0)>(blob_ptr + 294912);
    reshape_111_0 = reinterpret_cast<decltype(reshape_111_0)>(blob_ptr + 0);
    flash_attention_112_0 = reinterpret_cast<decltype(flash_attention_112_0)>(blob_ptr + 393216);
    reshape_115_0 = reinterpret_cast<decltype(reshape_115_0)>(blob_ptr + 0);
    layernorm_116_0 = reinterpret_cast<decltype(layernorm_116_0)>(blob_ptr + 393216);
    gemm_rcr_bias_fast_gelu_117_0 = reinterpret_cast<decltype(gemm_rcr_bias_fast_gelu_117_0)>(blob_ptr + 0);
    gemm_rcr_bias_add_118_0 = reinterpret_cast<decltype(gemm_rcr_bias_add_118_0)>(blob_ptr + 491520);
    
         params_[0].shape_ptrs = {ParamDim(1, 1, &input_dim_0), ParamDim(64, 64, &input_dim_1), ParamDim(768, 768, &input_dim_2)};
     params_[2].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_0_attention_self_qkv_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_0_attention_self_qkv_weight_dim_1)};
     params_[3].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_0_attention_self_qkv_bias_dim_0)};
     params_[4].shape_ptrs = {ParamDim(2, 2, &bert_encoder_layer_0_attention_self_cu_length_dim_0)};
     params_[5].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_0_attention_self_proj_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_0_attention_self_proj_weight_dim_1)};
     params_[6].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_0_attention_self_proj_bias_dim_0)};
     params_[7].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_0_attention_output_LayerNorm_weight_dim_0)};
     params_[8].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_0_attention_output_LayerNorm_bias_dim_0)};
     params_[9].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_0_intermediate_dense_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_0_intermediate_dense_weight_dim_1)};
     params_[10].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_0_intermediate_dense_bias_dim_0)};
     params_[11].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_0_output_dense_weight_dim_0), ParamDim(3072, 3072, &bert_encoder_layer_0_output_dense_weight_dim_1)};
     params_[12].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_0_output_dense_bias_dim_0)};
     params_[13].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_0_output_LayerNorm_weight_dim_0)};
     params_[14].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_0_output_LayerNorm_bias_dim_0)};
     params_[15].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_1_attention_self_qkv_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_1_attention_self_qkv_weight_dim_1)};
     params_[16].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_1_attention_self_qkv_bias_dim_0)};
     params_[17].shape_ptrs = {ParamDim(2, 2, &bert_encoder_layer_1_attention_self_cu_length_dim_0)};
     params_[18].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_1_attention_self_proj_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_1_attention_self_proj_weight_dim_1)};
     params_[19].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_1_attention_self_proj_bias_dim_0)};
     params_[20].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_1_attention_output_LayerNorm_weight_dim_0)};
     params_[21].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_1_attention_output_LayerNorm_bias_dim_0)};
     params_[22].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_1_intermediate_dense_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_1_intermediate_dense_weight_dim_1)};
     params_[23].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_1_intermediate_dense_bias_dim_0)};
     params_[24].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_1_output_dense_weight_dim_0), ParamDim(3072, 3072, &bert_encoder_layer_1_output_dense_weight_dim_1)};
     params_[25].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_1_output_dense_bias_dim_0)};
     params_[26].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_1_output_LayerNorm_weight_dim_0)};
     params_[27].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_1_output_LayerNorm_bias_dim_0)};
     params_[28].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_2_attention_self_qkv_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_2_attention_self_qkv_weight_dim_1)};
     params_[29].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_2_attention_self_qkv_bias_dim_0)};
     params_[30].shape_ptrs = {ParamDim(2, 2, &bert_encoder_layer_2_attention_self_cu_length_dim_0)};
     params_[31].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_2_attention_self_proj_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_2_attention_self_proj_weight_dim_1)};
     params_[32].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_2_attention_self_proj_bias_dim_0)};
     params_[33].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_2_attention_output_LayerNorm_weight_dim_0)};
     params_[34].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_2_attention_output_LayerNorm_bias_dim_0)};
     params_[35].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_2_intermediate_dense_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_2_intermediate_dense_weight_dim_1)};
     params_[36].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_2_intermediate_dense_bias_dim_0)};
     params_[37].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_2_output_dense_weight_dim_0), ParamDim(3072, 3072, &bert_encoder_layer_2_output_dense_weight_dim_1)};
     params_[38].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_2_output_dense_bias_dim_0)};
     params_[39].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_2_output_LayerNorm_weight_dim_0)};
     params_[40].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_2_output_LayerNorm_bias_dim_0)};
     params_[41].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_3_attention_self_qkv_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_3_attention_self_qkv_weight_dim_1)};
     params_[42].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_3_attention_self_qkv_bias_dim_0)};
     params_[43].shape_ptrs = {ParamDim(2, 2, &bert_encoder_layer_3_attention_self_cu_length_dim_0)};
     params_[44].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_3_attention_self_proj_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_3_attention_self_proj_weight_dim_1)};
     params_[45].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_3_attention_self_proj_bias_dim_0)};
     params_[46].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_3_attention_output_LayerNorm_weight_dim_0)};
     params_[47].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_3_attention_output_LayerNorm_bias_dim_0)};
     params_[48].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_3_intermediate_dense_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_3_intermediate_dense_weight_dim_1)};
     params_[49].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_3_intermediate_dense_bias_dim_0)};
     params_[50].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_3_output_dense_weight_dim_0), ParamDim(3072, 3072, &bert_encoder_layer_3_output_dense_weight_dim_1)};
     params_[51].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_3_output_dense_bias_dim_0)};
     params_[52].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_3_output_LayerNorm_weight_dim_0)};
     params_[53].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_3_output_LayerNorm_bias_dim_0)};
     params_[54].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_4_attention_self_qkv_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_4_attention_self_qkv_weight_dim_1)};
     params_[55].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_4_attention_self_qkv_bias_dim_0)};
     params_[56].shape_ptrs = {ParamDim(2, 2, &bert_encoder_layer_4_attention_self_cu_length_dim_0)};
     params_[57].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_4_attention_self_proj_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_4_attention_self_proj_weight_dim_1)};
     params_[58].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_4_attention_self_proj_bias_dim_0)};
     params_[59].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_4_attention_output_LayerNorm_weight_dim_0)};
     params_[60].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_4_attention_output_LayerNorm_bias_dim_0)};
     params_[61].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_4_intermediate_dense_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_4_intermediate_dense_weight_dim_1)};
     params_[62].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_4_intermediate_dense_bias_dim_0)};
     params_[63].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_4_output_dense_weight_dim_0), ParamDim(3072, 3072, &bert_encoder_layer_4_output_dense_weight_dim_1)};
     params_[64].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_4_output_dense_bias_dim_0)};
     params_[65].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_4_output_LayerNorm_weight_dim_0)};
     params_[66].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_4_output_LayerNorm_bias_dim_0)};
     params_[67].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_5_attention_self_qkv_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_5_attention_self_qkv_weight_dim_1)};
     params_[68].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_5_attention_self_qkv_bias_dim_0)};
     params_[69].shape_ptrs = {ParamDim(2, 2, &bert_encoder_layer_5_attention_self_cu_length_dim_0)};
     params_[70].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_5_attention_self_proj_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_5_attention_self_proj_weight_dim_1)};
     params_[71].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_5_attention_self_proj_bias_dim_0)};
     params_[72].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_5_attention_output_LayerNorm_weight_dim_0)};
     params_[73].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_5_attention_output_LayerNorm_bias_dim_0)};
     params_[74].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_5_intermediate_dense_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_5_intermediate_dense_weight_dim_1)};
     params_[75].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_5_intermediate_dense_bias_dim_0)};
     params_[76].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_5_output_dense_weight_dim_0), ParamDim(3072, 3072, &bert_encoder_layer_5_output_dense_weight_dim_1)};
     params_[77].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_5_output_dense_bias_dim_0)};
     params_[78].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_5_output_LayerNorm_weight_dim_0)};
     params_[79].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_5_output_LayerNorm_bias_dim_0)};
     params_[80].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_6_attention_self_qkv_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_6_attention_self_qkv_weight_dim_1)};
     params_[81].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_6_attention_self_qkv_bias_dim_0)};
     params_[82].shape_ptrs = {ParamDim(2, 2, &bert_encoder_layer_6_attention_self_cu_length_dim_0)};
     params_[83].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_6_attention_self_proj_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_6_attention_self_proj_weight_dim_1)};
     params_[84].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_6_attention_self_proj_bias_dim_0)};
     params_[85].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_6_attention_output_LayerNorm_weight_dim_0)};
     params_[86].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_6_attention_output_LayerNorm_bias_dim_0)};
     params_[87].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_6_intermediate_dense_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_6_intermediate_dense_weight_dim_1)};
     params_[88].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_6_intermediate_dense_bias_dim_0)};
     params_[89].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_6_output_dense_weight_dim_0), ParamDim(3072, 3072, &bert_encoder_layer_6_output_dense_weight_dim_1)};
     params_[90].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_6_output_dense_bias_dim_0)};
     params_[91].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_6_output_LayerNorm_weight_dim_0)};
     params_[92].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_6_output_LayerNorm_bias_dim_0)};
     params_[93].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_7_attention_self_qkv_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_7_attention_self_qkv_weight_dim_1)};
     params_[94].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_7_attention_self_qkv_bias_dim_0)};
     params_[95].shape_ptrs = {ParamDim(2, 2, &bert_encoder_layer_7_attention_self_cu_length_dim_0)};
     params_[96].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_7_attention_self_proj_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_7_attention_self_proj_weight_dim_1)};
     params_[97].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_7_attention_self_proj_bias_dim_0)};
     params_[98].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_7_attention_output_LayerNorm_weight_dim_0)};
     params_[99].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_7_attention_output_LayerNorm_bias_dim_0)};
     params_[100].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_7_intermediate_dense_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_7_intermediate_dense_weight_dim_1)};
     params_[101].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_7_intermediate_dense_bias_dim_0)};
     params_[102].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_7_output_dense_weight_dim_0), ParamDim(3072, 3072, &bert_encoder_layer_7_output_dense_weight_dim_1)};
     params_[103].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_7_output_dense_bias_dim_0)};
     params_[104].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_7_output_LayerNorm_weight_dim_0)};
     params_[105].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_7_output_LayerNorm_bias_dim_0)};
     params_[106].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_8_attention_self_qkv_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_8_attention_self_qkv_weight_dim_1)};
     params_[107].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_8_attention_self_qkv_bias_dim_0)};
     params_[108].shape_ptrs = {ParamDim(2, 2, &bert_encoder_layer_8_attention_self_cu_length_dim_0)};
     params_[109].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_8_attention_self_proj_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_8_attention_self_proj_weight_dim_1)};
     params_[110].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_8_attention_self_proj_bias_dim_0)};
     params_[111].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_8_attention_output_LayerNorm_weight_dim_0)};
     params_[112].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_8_attention_output_LayerNorm_bias_dim_0)};
     params_[113].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_8_intermediate_dense_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_8_intermediate_dense_weight_dim_1)};
     params_[114].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_8_intermediate_dense_bias_dim_0)};
     params_[115].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_8_output_dense_weight_dim_0), ParamDim(3072, 3072, &bert_encoder_layer_8_output_dense_weight_dim_1)};
     params_[116].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_8_output_dense_bias_dim_0)};
     params_[117].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_8_output_LayerNorm_weight_dim_0)};
     params_[118].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_8_output_LayerNorm_bias_dim_0)};
     params_[119].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_9_attention_self_qkv_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_9_attention_self_qkv_weight_dim_1)};
     params_[120].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_9_attention_self_qkv_bias_dim_0)};
     params_[121].shape_ptrs = {ParamDim(2, 2, &bert_encoder_layer_9_attention_self_cu_length_dim_0)};
     params_[122].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_9_attention_self_proj_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_9_attention_self_proj_weight_dim_1)};
     params_[123].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_9_attention_self_proj_bias_dim_0)};
     params_[124].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_9_attention_output_LayerNorm_weight_dim_0)};
     params_[125].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_9_attention_output_LayerNorm_bias_dim_0)};
     params_[126].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_9_intermediate_dense_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_9_intermediate_dense_weight_dim_1)};
     params_[127].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_9_intermediate_dense_bias_dim_0)};
     params_[128].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_9_output_dense_weight_dim_0), ParamDim(3072, 3072, &bert_encoder_layer_9_output_dense_weight_dim_1)};
     params_[129].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_9_output_dense_bias_dim_0)};
     params_[130].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_9_output_LayerNorm_weight_dim_0)};
     params_[131].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_9_output_LayerNorm_bias_dim_0)};
     params_[132].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_10_attention_self_qkv_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_10_attention_self_qkv_weight_dim_1)};
     params_[133].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_10_attention_self_qkv_bias_dim_0)};
     params_[134].shape_ptrs = {ParamDim(2, 2, &bert_encoder_layer_10_attention_self_cu_length_dim_0)};
     params_[135].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_10_attention_self_proj_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_10_attention_self_proj_weight_dim_1)};
     params_[136].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_10_attention_self_proj_bias_dim_0)};
     params_[137].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_10_attention_output_LayerNorm_weight_dim_0)};
     params_[138].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_10_attention_output_LayerNorm_bias_dim_0)};
     params_[139].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_10_intermediate_dense_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_10_intermediate_dense_weight_dim_1)};
     params_[140].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_10_intermediate_dense_bias_dim_0)};
     params_[141].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_10_output_dense_weight_dim_0), ParamDim(3072, 3072, &bert_encoder_layer_10_output_dense_weight_dim_1)};
     params_[142].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_10_output_dense_bias_dim_0)};
     params_[143].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_10_output_LayerNorm_weight_dim_0)};
     params_[144].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_10_output_LayerNorm_bias_dim_0)};
     params_[145].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_11_attention_self_qkv_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_11_attention_self_qkv_weight_dim_1)};
     params_[146].shape_ptrs = {ParamDim(2304, 2304, &bert_encoder_layer_11_attention_self_qkv_bias_dim_0)};
     params_[147].shape_ptrs = {ParamDim(2, 2, &bert_encoder_layer_11_attention_self_cu_length_dim_0)};
     params_[148].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_11_attention_self_proj_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_11_attention_self_proj_weight_dim_1)};
     params_[149].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_11_attention_self_proj_bias_dim_0)};
     params_[150].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_11_attention_output_LayerNorm_weight_dim_0)};
     params_[151].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_11_attention_output_LayerNorm_bias_dim_0)};
     params_[152].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_11_intermediate_dense_weight_dim_0), ParamDim(768, 768, &bert_encoder_layer_11_intermediate_dense_weight_dim_1)};
     params_[153].shape_ptrs = {ParamDim(3072, 3072, &bert_encoder_layer_11_intermediate_dense_bias_dim_0)};
     params_[154].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_11_output_dense_weight_dim_0), ParamDim(3072, 3072, &bert_encoder_layer_11_output_dense_weight_dim_1)};
     params_[155].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_11_output_dense_bias_dim_0)};
     params_[156].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_11_output_LayerNorm_weight_dim_0)};
     params_[157].shape_ptrs = {ParamDim(768, 768, &bert_encoder_layer_11_output_LayerNorm_bias_dim_0)};
     params_[1].shape_ptrs = {ParamDim(1, 1, &reshape_115_0_dim_0), ParamDim(64, 64, &reshape_115_0_dim_1), ParamDim(768, 768, &bert_encoder_layer_11_output_dense_weight_dim_0)};

      
      
    }

    ~Model() {
      
      
    }

    void SetUpInputsOutputs() {
             input = static_cast<decltype(input)>(params_[0].ptr);

if (input == nullptr) {
    throw std::runtime_error("Constant input was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_0_attention_self_qkv_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_0_attention_self_qkv_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_0_attention_self_qkv_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_0_attention_self_qkv_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_0_attention_self_cu_length == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_0_attention_self_cu_length was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_0_attention_self_proj_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_0_attention_self_proj_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_0_attention_self_proj_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_0_attention_self_proj_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_0_attention_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_0_attention_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_0_attention_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_0_attention_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_0_intermediate_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_0_intermediate_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_0_intermediate_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_0_intermediate_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_0_output_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_0_output_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_0_output_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_0_output_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_0_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_0_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_0_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_0_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_1_attention_self_qkv_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_1_attention_self_qkv_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_1_attention_self_qkv_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_1_attention_self_qkv_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_1_attention_self_cu_length == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_1_attention_self_cu_length was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_1_attention_self_proj_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_1_attention_self_proj_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_1_attention_self_proj_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_1_attention_self_proj_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_1_attention_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_1_attention_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_1_attention_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_1_attention_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_1_intermediate_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_1_intermediate_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_1_intermediate_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_1_intermediate_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_1_output_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_1_output_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_1_output_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_1_output_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_1_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_1_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_1_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_1_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_2_attention_self_qkv_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_2_attention_self_qkv_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_2_attention_self_qkv_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_2_attention_self_qkv_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_2_attention_self_cu_length == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_2_attention_self_cu_length was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_2_attention_self_proj_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_2_attention_self_proj_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_2_attention_self_proj_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_2_attention_self_proj_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_2_attention_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_2_attention_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_2_attention_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_2_attention_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_2_intermediate_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_2_intermediate_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_2_intermediate_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_2_intermediate_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_2_output_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_2_output_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_2_output_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_2_output_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_2_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_2_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_2_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_2_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_3_attention_self_qkv_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_3_attention_self_qkv_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_3_attention_self_qkv_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_3_attention_self_qkv_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_3_attention_self_cu_length == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_3_attention_self_cu_length was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_3_attention_self_proj_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_3_attention_self_proj_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_3_attention_self_proj_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_3_attention_self_proj_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_3_attention_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_3_attention_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_3_attention_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_3_attention_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_3_intermediate_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_3_intermediate_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_3_intermediate_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_3_intermediate_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_3_output_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_3_output_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_3_output_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_3_output_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_3_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_3_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_3_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_3_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_4_attention_self_qkv_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_4_attention_self_qkv_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_4_attention_self_qkv_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_4_attention_self_qkv_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_4_attention_self_cu_length == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_4_attention_self_cu_length was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_4_attention_self_proj_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_4_attention_self_proj_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_4_attention_self_proj_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_4_attention_self_proj_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_4_attention_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_4_attention_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_4_attention_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_4_attention_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_4_intermediate_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_4_intermediate_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_4_intermediate_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_4_intermediate_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_4_output_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_4_output_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_4_output_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_4_output_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_4_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_4_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_4_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_4_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_5_attention_self_qkv_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_5_attention_self_qkv_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_5_attention_self_qkv_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_5_attention_self_qkv_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_5_attention_self_cu_length == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_5_attention_self_cu_length was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_5_attention_self_proj_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_5_attention_self_proj_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_5_attention_self_proj_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_5_attention_self_proj_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_5_attention_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_5_attention_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_5_attention_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_5_attention_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_5_intermediate_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_5_intermediate_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_5_intermediate_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_5_intermediate_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_5_output_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_5_output_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_5_output_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_5_output_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_5_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_5_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_5_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_5_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_6_attention_self_qkv_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_6_attention_self_qkv_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_6_attention_self_qkv_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_6_attention_self_qkv_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_6_attention_self_cu_length == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_6_attention_self_cu_length was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_6_attention_self_proj_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_6_attention_self_proj_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_6_attention_self_proj_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_6_attention_self_proj_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_6_attention_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_6_attention_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_6_attention_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_6_attention_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_6_intermediate_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_6_intermediate_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_6_intermediate_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_6_intermediate_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_6_output_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_6_output_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_6_output_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_6_output_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_6_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_6_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_6_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_6_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_7_attention_self_qkv_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_7_attention_self_qkv_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_7_attention_self_qkv_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_7_attention_self_qkv_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_7_attention_self_cu_length == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_7_attention_self_cu_length was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_7_attention_self_proj_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_7_attention_self_proj_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_7_attention_self_proj_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_7_attention_self_proj_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_7_attention_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_7_attention_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_7_attention_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_7_attention_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_7_intermediate_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_7_intermediate_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_7_intermediate_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_7_intermediate_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_7_output_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_7_output_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_7_output_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_7_output_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_7_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_7_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_7_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_7_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_8_attention_self_qkv_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_8_attention_self_qkv_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_8_attention_self_qkv_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_8_attention_self_qkv_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_8_attention_self_cu_length == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_8_attention_self_cu_length was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_8_attention_self_proj_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_8_attention_self_proj_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_8_attention_self_proj_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_8_attention_self_proj_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_8_attention_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_8_attention_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_8_attention_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_8_attention_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_8_intermediate_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_8_intermediate_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_8_intermediate_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_8_intermediate_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_8_output_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_8_output_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_8_output_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_8_output_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_8_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_8_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_8_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_8_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_9_attention_self_qkv_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_9_attention_self_qkv_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_9_attention_self_qkv_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_9_attention_self_qkv_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_9_attention_self_cu_length == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_9_attention_self_cu_length was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_9_attention_self_proj_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_9_attention_self_proj_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_9_attention_self_proj_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_9_attention_self_proj_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_9_attention_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_9_attention_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_9_attention_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_9_attention_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_9_intermediate_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_9_intermediate_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_9_intermediate_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_9_intermediate_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_9_output_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_9_output_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_9_output_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_9_output_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_9_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_9_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_9_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_9_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_10_attention_self_qkv_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_10_attention_self_qkv_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_10_attention_self_qkv_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_10_attention_self_qkv_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_10_attention_self_cu_length == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_10_attention_self_cu_length was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_10_attention_self_proj_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_10_attention_self_proj_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_10_attention_self_proj_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_10_attention_self_proj_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_10_attention_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_10_attention_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_10_attention_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_10_attention_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_10_intermediate_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_10_intermediate_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_10_intermediate_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_10_intermediate_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_10_output_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_10_output_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_10_output_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_10_output_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_10_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_10_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_10_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_10_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_11_attention_self_qkv_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_11_attention_self_qkv_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_11_attention_self_qkv_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_11_attention_self_qkv_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_11_attention_self_cu_length == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_11_attention_self_cu_length was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_11_attention_self_proj_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_11_attention_self_proj_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_11_attention_self_proj_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_11_attention_self_proj_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_11_attention_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_11_attention_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_11_attention_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_11_attention_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_11_intermediate_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_11_intermediate_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_11_intermediate_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_11_intermediate_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_11_output_dense_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_11_output_dense_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_11_output_dense_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_11_output_dense_bias was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_11_output_LayerNorm_weight == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_11_output_LayerNorm_weight was not set! Set the value with set_constant.");
}
    

if (bert_encoder_layer_11_output_LayerNorm_bias == nullptr) {
    throw std::runtime_error("Constant bert_encoder_layer_11_output_LayerNorm_bias was not set! Set the value with set_constant.");
}
    
     output_0 = static_cast<decltype(output_0)>(params_[1].ptr);

if (output_0 == nullptr) {
    throw std::runtime_error("Constant output_0 was not set! Set the value with set_constant.");
}
    
    }

    void ResetConstants(uint8_t* constants) {
        /*
         * This can be called if we want to use a different piece of memory
         * for the constants to be consumed.
         */
        
    }

    void DeviceToDeviceCopies(StreamType stream) {
  
    }


    ///////////////////////////////////////////////////////////////////////////
    // default RunImpl implemenation
    void RunImpl(StreamType stream) {
        
  
  
    {
    

    gemm_rcr_bias_0(

        input,
        bert_encoder_layer_0_attention_self_qkv_weight,

        bert_encoder_layer_0_attention_self_qkv_bias,

        reshape_1_0,
        global_workspace_,
        1,

        &input_dim_0,

        &input_dim_1,

        &input_dim_2,


        &bert_encoder_layer_0_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_0_attention_self_qkv_weight_dim_1,


        &input_dim_0,

        &input_dim_1,

        &bert_encoder_layer_0_attention_self_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_2(
       flash_attention_2_0, reshape_1_0, reinterpret_cast<int*>(bert_encoder_layer_0_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_2_0,
        bert_encoder_layer_0_attention_self_proj_weight,
        bert_encoder_layer_0_attention_self_proj_bias,
        input,

        reshape_5_0,
        global_workspace_,

     1,


        &reshape_3_0_dim_0,

        &reshape_3_0_dim_1,


        &bert_encoder_layer_0_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_0_attention_self_proj_weight_dim_1,


        &reshape_3_0_dim_0,

        &bert_encoder_layer_0_attention_self_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_5_0_dim_0;

        M *= reshape_5_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_5_0_dim_2;

    
        layernorm_6(
           layernorm_6_0, reshape_5_0, bert_encoder_layer_0_attention_output_LayerNorm_weight, bert_encoder_layer_0_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_6_0,
        bert_encoder_layer_0_intermediate_dense_weight,

        bert_encoder_layer_0_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_7_0,
        global_workspace_,
        1,

        &reshape_5_0_dim_0,

        &reshape_5_0_dim_1,

        &reshape_5_0_dim_2,


        &bert_encoder_layer_0_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_0_intermediate_dense_weight_dim_1,


        &reshape_5_0_dim_0,

        &reshape_5_0_dim_1,

        &bert_encoder_layer_0_intermediate_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_7_0,
        bert_encoder_layer_0_output_dense_weight,
        bert_encoder_layer_0_output_dense_bias,
        layernorm_6_0,

        gemm_rcr_bias_add_8_0,
        global_workspace_,

     1,


        &reshape_5_0_dim_0,

        &reshape_5_0_dim_1,

        &bert_encoder_layer_0_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_0_output_dense_weight_dim_0,

        &bert_encoder_layer_0_output_dense_weight_dim_1,


        &reshape_5_0_dim_0,

        &reshape_5_0_dim_1,

        &bert_encoder_layer_0_output_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_5_0_dim_0;

        M *= reshape_5_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_0_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_9_0, gemm_rcr_bias_add_8_0, bert_encoder_layer_0_output_LayerNorm_weight, bert_encoder_layer_0_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_0(

        layernorm_9_0,
        bert_encoder_layer_1_attention_self_qkv_weight,

        bert_encoder_layer_1_attention_self_qkv_bias,

        reshape_11_0,
        global_workspace_,
        1,

        &reshape_5_0_dim_0,

        &reshape_5_0_dim_1,

        &bert_encoder_layer_0_output_dense_weight_dim_0,


        &bert_encoder_layer_1_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_1_attention_self_qkv_weight_dim_1,


        &reshape_5_0_dim_0,

        &reshape_5_0_dim_1,

        &bert_encoder_layer_1_attention_self_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_2(
       flash_attention_12_0, reshape_11_0, reinterpret_cast<int*>(bert_encoder_layer_1_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_12_0,
        bert_encoder_layer_1_attention_self_proj_weight,
        bert_encoder_layer_1_attention_self_proj_bias,
        layernorm_9_0,

        reshape_15_0,
        global_workspace_,

     1,


        &reshape_13_0_dim_0,

        &reshape_13_0_dim_1,


        &bert_encoder_layer_1_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_1_attention_self_proj_weight_dim_1,


        &reshape_13_0_dim_0,

        &bert_encoder_layer_1_attention_self_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_15_0_dim_0;

        M *= reshape_15_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_15_0_dim_2;

    
        layernorm_6(
           layernorm_16_0, reshape_15_0, bert_encoder_layer_1_attention_output_LayerNorm_weight, bert_encoder_layer_1_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_16_0,
        bert_encoder_layer_1_intermediate_dense_weight,

        bert_encoder_layer_1_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_17_0,
        global_workspace_,
        1,

        &reshape_15_0_dim_0,

        &reshape_15_0_dim_1,

        &reshape_15_0_dim_2,


        &bert_encoder_layer_1_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_1_intermediate_dense_weight_dim_1,


        &reshape_15_0_dim_0,

        &reshape_15_0_dim_1,

        &bert_encoder_layer_1_intermediate_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_17_0,
        bert_encoder_layer_1_output_dense_weight,
        bert_encoder_layer_1_output_dense_bias,
        layernorm_16_0,

        gemm_rcr_bias_add_18_0,
        global_workspace_,

     1,


        &reshape_15_0_dim_0,

        &reshape_15_0_dim_1,

        &bert_encoder_layer_1_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_1_output_dense_weight_dim_0,

        &bert_encoder_layer_1_output_dense_weight_dim_1,


        &reshape_15_0_dim_0,

        &reshape_15_0_dim_1,

        &bert_encoder_layer_1_output_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_15_0_dim_0;

        M *= reshape_15_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_1_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_19_0, gemm_rcr_bias_add_18_0, bert_encoder_layer_1_output_LayerNorm_weight, bert_encoder_layer_1_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_0(

        layernorm_19_0,
        bert_encoder_layer_2_attention_self_qkv_weight,

        bert_encoder_layer_2_attention_self_qkv_bias,

        reshape_21_0,
        global_workspace_,
        1,

        &reshape_15_0_dim_0,

        &reshape_15_0_dim_1,

        &bert_encoder_layer_1_output_dense_weight_dim_0,


        &bert_encoder_layer_2_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_2_attention_self_qkv_weight_dim_1,


        &reshape_15_0_dim_0,

        &reshape_15_0_dim_1,

        &bert_encoder_layer_2_attention_self_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_2(
       flash_attention_22_0, reshape_21_0, reinterpret_cast<int*>(bert_encoder_layer_2_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_22_0,
        bert_encoder_layer_2_attention_self_proj_weight,
        bert_encoder_layer_2_attention_self_proj_bias,
        layernorm_19_0,

        reshape_25_0,
        global_workspace_,

     1,


        &reshape_23_0_dim_0,

        &reshape_23_0_dim_1,


        &bert_encoder_layer_2_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_2_attention_self_proj_weight_dim_1,


        &reshape_23_0_dim_0,

        &bert_encoder_layer_2_attention_self_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_25_0_dim_0;

        M *= reshape_25_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_25_0_dim_2;

    
        layernorm_6(
           layernorm_26_0, reshape_25_0, bert_encoder_layer_2_attention_output_LayerNorm_weight, bert_encoder_layer_2_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_26_0,
        bert_encoder_layer_2_intermediate_dense_weight,

        bert_encoder_layer_2_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_27_0,
        global_workspace_,
        1,

        &reshape_25_0_dim_0,

        &reshape_25_0_dim_1,

        &reshape_25_0_dim_2,


        &bert_encoder_layer_2_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_2_intermediate_dense_weight_dim_1,


        &reshape_25_0_dim_0,

        &reshape_25_0_dim_1,

        &bert_encoder_layer_2_intermediate_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_27_0,
        bert_encoder_layer_2_output_dense_weight,
        bert_encoder_layer_2_output_dense_bias,
        layernorm_26_0,

        gemm_rcr_bias_add_28_0,
        global_workspace_,

     1,


        &reshape_25_0_dim_0,

        &reshape_25_0_dim_1,

        &bert_encoder_layer_2_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_2_output_dense_weight_dim_0,

        &bert_encoder_layer_2_output_dense_weight_dim_1,


        &reshape_25_0_dim_0,

        &reshape_25_0_dim_1,

        &bert_encoder_layer_2_output_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_25_0_dim_0;

        M *= reshape_25_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_2_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_29_0, gemm_rcr_bias_add_28_0, bert_encoder_layer_2_output_LayerNorm_weight, bert_encoder_layer_2_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_0(

        layernorm_29_0,
        bert_encoder_layer_3_attention_self_qkv_weight,

        bert_encoder_layer_3_attention_self_qkv_bias,

        reshape_31_0,
        global_workspace_,
        1,

        &reshape_25_0_dim_0,

        &reshape_25_0_dim_1,

        &bert_encoder_layer_2_output_dense_weight_dim_0,


        &bert_encoder_layer_3_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_3_attention_self_qkv_weight_dim_1,


        &reshape_25_0_dim_0,

        &reshape_25_0_dim_1,

        &bert_encoder_layer_3_attention_self_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_2(
       flash_attention_32_0, reshape_31_0, reinterpret_cast<int*>(bert_encoder_layer_3_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_32_0,
        bert_encoder_layer_3_attention_self_proj_weight,
        bert_encoder_layer_3_attention_self_proj_bias,
        layernorm_29_0,

        reshape_35_0,
        global_workspace_,

     1,


        &reshape_33_0_dim_0,

        &reshape_33_0_dim_1,


        &bert_encoder_layer_3_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_3_attention_self_proj_weight_dim_1,


        &reshape_33_0_dim_0,

        &bert_encoder_layer_3_attention_self_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_35_0_dim_0;

        M *= reshape_35_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_35_0_dim_2;

    
        layernorm_6(
           layernorm_36_0, reshape_35_0, bert_encoder_layer_3_attention_output_LayerNorm_weight, bert_encoder_layer_3_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_36_0,
        bert_encoder_layer_3_intermediate_dense_weight,

        bert_encoder_layer_3_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_37_0,
        global_workspace_,
        1,

        &reshape_35_0_dim_0,

        &reshape_35_0_dim_1,

        &reshape_35_0_dim_2,


        &bert_encoder_layer_3_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_3_intermediate_dense_weight_dim_1,


        &reshape_35_0_dim_0,

        &reshape_35_0_dim_1,

        &bert_encoder_layer_3_intermediate_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_37_0,
        bert_encoder_layer_3_output_dense_weight,
        bert_encoder_layer_3_output_dense_bias,
        layernorm_36_0,

        gemm_rcr_bias_add_38_0,
        global_workspace_,

     1,


        &reshape_35_0_dim_0,

        &reshape_35_0_dim_1,

        &bert_encoder_layer_3_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_3_output_dense_weight_dim_0,

        &bert_encoder_layer_3_output_dense_weight_dim_1,


        &reshape_35_0_dim_0,

        &reshape_35_0_dim_1,

        &bert_encoder_layer_3_output_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_35_0_dim_0;

        M *= reshape_35_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_3_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_39_0, gemm_rcr_bias_add_38_0, bert_encoder_layer_3_output_LayerNorm_weight, bert_encoder_layer_3_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_0(

        layernorm_39_0,
        bert_encoder_layer_4_attention_self_qkv_weight,

        bert_encoder_layer_4_attention_self_qkv_bias,

        reshape_41_0,
        global_workspace_,
        1,

        &reshape_35_0_dim_0,

        &reshape_35_0_dim_1,

        &bert_encoder_layer_3_output_dense_weight_dim_0,


        &bert_encoder_layer_4_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_4_attention_self_qkv_weight_dim_1,


        &reshape_35_0_dim_0,

        &reshape_35_0_dim_1,

        &bert_encoder_layer_4_attention_self_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_2(
       flash_attention_42_0, reshape_41_0, reinterpret_cast<int*>(bert_encoder_layer_4_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_42_0,
        bert_encoder_layer_4_attention_self_proj_weight,
        bert_encoder_layer_4_attention_self_proj_bias,
        layernorm_39_0,

        reshape_45_0,
        global_workspace_,

     1,


        &reshape_43_0_dim_0,

        &reshape_43_0_dim_1,


        &bert_encoder_layer_4_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_4_attention_self_proj_weight_dim_1,


        &reshape_43_0_dim_0,

        &bert_encoder_layer_4_attention_self_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_45_0_dim_0;

        M *= reshape_45_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_45_0_dim_2;

    
        layernorm_6(
           layernorm_46_0, reshape_45_0, bert_encoder_layer_4_attention_output_LayerNorm_weight, bert_encoder_layer_4_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_46_0,
        bert_encoder_layer_4_intermediate_dense_weight,

        bert_encoder_layer_4_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_47_0,
        global_workspace_,
        1,

        &reshape_45_0_dim_0,

        &reshape_45_0_dim_1,

        &reshape_45_0_dim_2,


        &bert_encoder_layer_4_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_4_intermediate_dense_weight_dim_1,


        &reshape_45_0_dim_0,

        &reshape_45_0_dim_1,

        &bert_encoder_layer_4_intermediate_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_47_0,
        bert_encoder_layer_4_output_dense_weight,
        bert_encoder_layer_4_output_dense_bias,
        layernorm_46_0,

        gemm_rcr_bias_add_48_0,
        global_workspace_,

     1,


        &reshape_45_0_dim_0,

        &reshape_45_0_dim_1,

        &bert_encoder_layer_4_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_4_output_dense_weight_dim_0,

        &bert_encoder_layer_4_output_dense_weight_dim_1,


        &reshape_45_0_dim_0,

        &reshape_45_0_dim_1,

        &bert_encoder_layer_4_output_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_45_0_dim_0;

        M *= reshape_45_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_4_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_49_0, gemm_rcr_bias_add_48_0, bert_encoder_layer_4_output_LayerNorm_weight, bert_encoder_layer_4_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_0(

        layernorm_49_0,
        bert_encoder_layer_5_attention_self_qkv_weight,

        bert_encoder_layer_5_attention_self_qkv_bias,

        reshape_51_0,
        global_workspace_,
        1,

        &reshape_45_0_dim_0,

        &reshape_45_0_dim_1,

        &bert_encoder_layer_4_output_dense_weight_dim_0,


        &bert_encoder_layer_5_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_5_attention_self_qkv_weight_dim_1,


        &reshape_45_0_dim_0,

        &reshape_45_0_dim_1,

        &bert_encoder_layer_5_attention_self_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_2(
       flash_attention_52_0, reshape_51_0, reinterpret_cast<int*>(bert_encoder_layer_5_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_52_0,
        bert_encoder_layer_5_attention_self_proj_weight,
        bert_encoder_layer_5_attention_self_proj_bias,
        layernorm_49_0,

        reshape_55_0,
        global_workspace_,

     1,


        &reshape_53_0_dim_0,

        &reshape_53_0_dim_1,


        &bert_encoder_layer_5_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_5_attention_self_proj_weight_dim_1,


        &reshape_53_0_dim_0,

        &bert_encoder_layer_5_attention_self_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_55_0_dim_0;

        M *= reshape_55_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_55_0_dim_2;

    
        layernorm_6(
           layernorm_56_0, reshape_55_0, bert_encoder_layer_5_attention_output_LayerNorm_weight, bert_encoder_layer_5_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_56_0,
        bert_encoder_layer_5_intermediate_dense_weight,

        bert_encoder_layer_5_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_57_0,
        global_workspace_,
        1,

        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &reshape_55_0_dim_2,


        &bert_encoder_layer_5_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_5_intermediate_dense_weight_dim_1,


        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &bert_encoder_layer_5_intermediate_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_57_0,
        bert_encoder_layer_5_output_dense_weight,
        bert_encoder_layer_5_output_dense_bias,
        layernorm_56_0,

        gemm_rcr_bias_add_58_0,
        global_workspace_,

     1,


        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &bert_encoder_layer_5_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_5_output_dense_weight_dim_0,

        &bert_encoder_layer_5_output_dense_weight_dim_1,


        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &bert_encoder_layer_5_output_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_55_0_dim_0;

        M *= reshape_55_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_5_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_59_0, gemm_rcr_bias_add_58_0, bert_encoder_layer_5_output_LayerNorm_weight, bert_encoder_layer_5_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_0(

        layernorm_59_0,
        bert_encoder_layer_6_attention_self_qkv_weight,

        bert_encoder_layer_6_attention_self_qkv_bias,

        reshape_61_0,
        global_workspace_,
        1,

        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &bert_encoder_layer_5_output_dense_weight_dim_0,


        &bert_encoder_layer_6_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_6_attention_self_qkv_weight_dim_1,


        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &bert_encoder_layer_6_attention_self_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_2(
       flash_attention_62_0, reshape_61_0, reinterpret_cast<int*>(bert_encoder_layer_6_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_62_0,
        bert_encoder_layer_6_attention_self_proj_weight,
        bert_encoder_layer_6_attention_self_proj_bias,
        layernorm_59_0,

        reshape_65_0,
        global_workspace_,

     1,


        &reshape_63_0_dim_0,

        &reshape_63_0_dim_1,


        &bert_encoder_layer_6_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_6_attention_self_proj_weight_dim_1,


        &reshape_63_0_dim_0,

        &bert_encoder_layer_6_attention_self_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_65_0_dim_0;

        M *= reshape_65_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_65_0_dim_2;

    
        layernorm_6(
           layernorm_66_0, reshape_65_0, bert_encoder_layer_6_attention_output_LayerNorm_weight, bert_encoder_layer_6_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_66_0,
        bert_encoder_layer_6_intermediate_dense_weight,

        bert_encoder_layer_6_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_67_0,
        global_workspace_,
        1,

        &reshape_65_0_dim_0,

        &reshape_65_0_dim_1,

        &reshape_65_0_dim_2,


        &bert_encoder_layer_6_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_6_intermediate_dense_weight_dim_1,


        &reshape_65_0_dim_0,

        &reshape_65_0_dim_1,

        &bert_encoder_layer_6_intermediate_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_67_0,
        bert_encoder_layer_6_output_dense_weight,
        bert_encoder_layer_6_output_dense_bias,
        layernorm_66_0,

        gemm_rcr_bias_add_68_0,
        global_workspace_,

     1,


        &reshape_65_0_dim_0,

        &reshape_65_0_dim_1,

        &bert_encoder_layer_6_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_6_output_dense_weight_dim_0,

        &bert_encoder_layer_6_output_dense_weight_dim_1,


        &reshape_65_0_dim_0,

        &reshape_65_0_dim_1,

        &bert_encoder_layer_6_output_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_65_0_dim_0;

        M *= reshape_65_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_6_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_69_0, gemm_rcr_bias_add_68_0, bert_encoder_layer_6_output_LayerNorm_weight, bert_encoder_layer_6_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_0(

        layernorm_69_0,
        bert_encoder_layer_7_attention_self_qkv_weight,

        bert_encoder_layer_7_attention_self_qkv_bias,

        reshape_71_0,
        global_workspace_,
        1,

        &reshape_65_0_dim_0,

        &reshape_65_0_dim_1,

        &bert_encoder_layer_6_output_dense_weight_dim_0,


        &bert_encoder_layer_7_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_7_attention_self_qkv_weight_dim_1,


        &reshape_65_0_dim_0,

        &reshape_65_0_dim_1,

        &bert_encoder_layer_7_attention_self_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_2(
       flash_attention_72_0, reshape_71_0, reinterpret_cast<int*>(bert_encoder_layer_7_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_72_0,
        bert_encoder_layer_7_attention_self_proj_weight,
        bert_encoder_layer_7_attention_self_proj_bias,
        layernorm_69_0,

        reshape_75_0,
        global_workspace_,

     1,


        &reshape_73_0_dim_0,

        &reshape_73_0_dim_1,


        &bert_encoder_layer_7_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_7_attention_self_proj_weight_dim_1,


        &reshape_73_0_dim_0,

        &bert_encoder_layer_7_attention_self_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_75_0_dim_0;

        M *= reshape_75_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_75_0_dim_2;

    
        layernorm_6(
           layernorm_76_0, reshape_75_0, bert_encoder_layer_7_attention_output_LayerNorm_weight, bert_encoder_layer_7_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_76_0,
        bert_encoder_layer_7_intermediate_dense_weight,

        bert_encoder_layer_7_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_77_0,
        global_workspace_,
        1,

        &reshape_75_0_dim_0,

        &reshape_75_0_dim_1,

        &reshape_75_0_dim_2,


        &bert_encoder_layer_7_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_7_intermediate_dense_weight_dim_1,


        &reshape_75_0_dim_0,

        &reshape_75_0_dim_1,

        &bert_encoder_layer_7_intermediate_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_77_0,
        bert_encoder_layer_7_output_dense_weight,
        bert_encoder_layer_7_output_dense_bias,
        layernorm_76_0,

        gemm_rcr_bias_add_78_0,
        global_workspace_,

     1,


        &reshape_75_0_dim_0,

        &reshape_75_0_dim_1,

        &bert_encoder_layer_7_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_7_output_dense_weight_dim_0,

        &bert_encoder_layer_7_output_dense_weight_dim_1,


        &reshape_75_0_dim_0,

        &reshape_75_0_dim_1,

        &bert_encoder_layer_7_output_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_75_0_dim_0;

        M *= reshape_75_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_7_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_79_0, gemm_rcr_bias_add_78_0, bert_encoder_layer_7_output_LayerNorm_weight, bert_encoder_layer_7_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_0(

        layernorm_79_0,
        bert_encoder_layer_8_attention_self_qkv_weight,

        bert_encoder_layer_8_attention_self_qkv_bias,

        reshape_81_0,
        global_workspace_,
        1,

        &reshape_75_0_dim_0,

        &reshape_75_0_dim_1,

        &bert_encoder_layer_7_output_dense_weight_dim_0,


        &bert_encoder_layer_8_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_8_attention_self_qkv_weight_dim_1,


        &reshape_75_0_dim_0,

        &reshape_75_0_dim_1,

        &bert_encoder_layer_8_attention_self_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_2(
       flash_attention_82_0, reshape_81_0, reinterpret_cast<int*>(bert_encoder_layer_8_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_82_0,
        bert_encoder_layer_8_attention_self_proj_weight,
        bert_encoder_layer_8_attention_self_proj_bias,
        layernorm_79_0,

        reshape_85_0,
        global_workspace_,

     1,


        &reshape_83_0_dim_0,

        &reshape_83_0_dim_1,


        &bert_encoder_layer_8_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_8_attention_self_proj_weight_dim_1,


        &reshape_83_0_dim_0,

        &bert_encoder_layer_8_attention_self_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_85_0_dim_0;

        M *= reshape_85_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_85_0_dim_2;

    
        layernorm_6(
           layernorm_86_0, reshape_85_0, bert_encoder_layer_8_attention_output_LayerNorm_weight, bert_encoder_layer_8_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_86_0,
        bert_encoder_layer_8_intermediate_dense_weight,

        bert_encoder_layer_8_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_87_0,
        global_workspace_,
        1,

        &reshape_85_0_dim_0,

        &reshape_85_0_dim_1,

        &reshape_85_0_dim_2,


        &bert_encoder_layer_8_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_8_intermediate_dense_weight_dim_1,


        &reshape_85_0_dim_0,

        &reshape_85_0_dim_1,

        &bert_encoder_layer_8_intermediate_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_87_0,
        bert_encoder_layer_8_output_dense_weight,
        bert_encoder_layer_8_output_dense_bias,
        layernorm_86_0,

        gemm_rcr_bias_add_88_0,
        global_workspace_,

     1,


        &reshape_85_0_dim_0,

        &reshape_85_0_dim_1,

        &bert_encoder_layer_8_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_8_output_dense_weight_dim_0,

        &bert_encoder_layer_8_output_dense_weight_dim_1,


        &reshape_85_0_dim_0,

        &reshape_85_0_dim_1,

        &bert_encoder_layer_8_output_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_85_0_dim_0;

        M *= reshape_85_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_8_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_89_0, gemm_rcr_bias_add_88_0, bert_encoder_layer_8_output_LayerNorm_weight, bert_encoder_layer_8_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_0(

        layernorm_89_0,
        bert_encoder_layer_9_attention_self_qkv_weight,

        bert_encoder_layer_9_attention_self_qkv_bias,

        reshape_91_0,
        global_workspace_,
        1,

        &reshape_85_0_dim_0,

        &reshape_85_0_dim_1,

        &bert_encoder_layer_8_output_dense_weight_dim_0,


        &bert_encoder_layer_9_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_9_attention_self_qkv_weight_dim_1,


        &reshape_85_0_dim_0,

        &reshape_85_0_dim_1,

        &bert_encoder_layer_9_attention_self_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_2(
       flash_attention_92_0, reshape_91_0, reinterpret_cast<int*>(bert_encoder_layer_9_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_92_0,
        bert_encoder_layer_9_attention_self_proj_weight,
        bert_encoder_layer_9_attention_self_proj_bias,
        layernorm_89_0,

        reshape_95_0,
        global_workspace_,

     1,


        &reshape_93_0_dim_0,

        &reshape_93_0_dim_1,


        &bert_encoder_layer_9_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_9_attention_self_proj_weight_dim_1,


        &reshape_93_0_dim_0,

        &bert_encoder_layer_9_attention_self_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_95_0_dim_0;

        M *= reshape_95_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_95_0_dim_2;

    
        layernorm_6(
           layernorm_96_0, reshape_95_0, bert_encoder_layer_9_attention_output_LayerNorm_weight, bert_encoder_layer_9_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_96_0,
        bert_encoder_layer_9_intermediate_dense_weight,

        bert_encoder_layer_9_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_97_0,
        global_workspace_,
        1,

        &reshape_95_0_dim_0,

        &reshape_95_0_dim_1,

        &reshape_95_0_dim_2,


        &bert_encoder_layer_9_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_9_intermediate_dense_weight_dim_1,


        &reshape_95_0_dim_0,

        &reshape_95_0_dim_1,

        &bert_encoder_layer_9_intermediate_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_97_0,
        bert_encoder_layer_9_output_dense_weight,
        bert_encoder_layer_9_output_dense_bias,
        layernorm_96_0,

        gemm_rcr_bias_add_98_0,
        global_workspace_,

     1,


        &reshape_95_0_dim_0,

        &reshape_95_0_dim_1,

        &bert_encoder_layer_9_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_9_output_dense_weight_dim_0,

        &bert_encoder_layer_9_output_dense_weight_dim_1,


        &reshape_95_0_dim_0,

        &reshape_95_0_dim_1,

        &bert_encoder_layer_9_output_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_95_0_dim_0;

        M *= reshape_95_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_9_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_99_0, gemm_rcr_bias_add_98_0, bert_encoder_layer_9_output_LayerNorm_weight, bert_encoder_layer_9_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_0(

        layernorm_99_0,
        bert_encoder_layer_10_attention_self_qkv_weight,

        bert_encoder_layer_10_attention_self_qkv_bias,

        reshape_101_0,
        global_workspace_,
        1,

        &reshape_95_0_dim_0,

        &reshape_95_0_dim_1,

        &bert_encoder_layer_9_output_dense_weight_dim_0,


        &bert_encoder_layer_10_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_10_attention_self_qkv_weight_dim_1,


        &reshape_95_0_dim_0,

        &reshape_95_0_dim_1,

        &bert_encoder_layer_10_attention_self_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_2(
       flash_attention_102_0, reshape_101_0, reinterpret_cast<int*>(bert_encoder_layer_10_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_102_0,
        bert_encoder_layer_10_attention_self_proj_weight,
        bert_encoder_layer_10_attention_self_proj_bias,
        layernorm_99_0,

        reshape_105_0,
        global_workspace_,

     1,


        &reshape_103_0_dim_0,

        &reshape_103_0_dim_1,


        &bert_encoder_layer_10_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_10_attention_self_proj_weight_dim_1,


        &reshape_103_0_dim_0,

        &bert_encoder_layer_10_attention_self_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_105_0_dim_0;

        M *= reshape_105_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_105_0_dim_2;

    
        layernorm_6(
           layernorm_106_0, reshape_105_0, bert_encoder_layer_10_attention_output_LayerNorm_weight, bert_encoder_layer_10_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_106_0,
        bert_encoder_layer_10_intermediate_dense_weight,

        bert_encoder_layer_10_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_107_0,
        global_workspace_,
        1,

        &reshape_105_0_dim_0,

        &reshape_105_0_dim_1,

        &reshape_105_0_dim_2,


        &bert_encoder_layer_10_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_10_intermediate_dense_weight_dim_1,


        &reshape_105_0_dim_0,

        &reshape_105_0_dim_1,

        &bert_encoder_layer_10_intermediate_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_107_0,
        bert_encoder_layer_10_output_dense_weight,
        bert_encoder_layer_10_output_dense_bias,
        layernorm_106_0,

        gemm_rcr_bias_add_108_0,
        global_workspace_,

     1,


        &reshape_105_0_dim_0,

        &reshape_105_0_dim_1,

        &bert_encoder_layer_10_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_10_output_dense_weight_dim_0,

        &bert_encoder_layer_10_output_dense_weight_dim_1,


        &reshape_105_0_dim_0,

        &reshape_105_0_dim_1,

        &bert_encoder_layer_10_output_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_105_0_dim_0;

        M *= reshape_105_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_10_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_109_0, gemm_rcr_bias_add_108_0, bert_encoder_layer_10_output_LayerNorm_weight, bert_encoder_layer_10_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_0(

        layernorm_109_0,
        bert_encoder_layer_11_attention_self_qkv_weight,

        bert_encoder_layer_11_attention_self_qkv_bias,

        reshape_111_0,
        global_workspace_,
        1,

        &reshape_105_0_dim_0,

        &reshape_105_0_dim_1,

        &bert_encoder_layer_10_output_dense_weight_dim_0,


        &bert_encoder_layer_11_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_11_attention_self_qkv_weight_dim_1,


        &reshape_105_0_dim_0,

        &reshape_105_0_dim_1,

        &bert_encoder_layer_11_attention_self_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_2(
       flash_attention_112_0, reshape_111_0, reinterpret_cast<int*>(bert_encoder_layer_11_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_112_0,
        bert_encoder_layer_11_attention_self_proj_weight,
        bert_encoder_layer_11_attention_self_proj_bias,
        layernorm_109_0,

        reshape_115_0,
        global_workspace_,

     1,


        &reshape_113_0_dim_0,

        &reshape_113_0_dim_1,


        &bert_encoder_layer_11_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_11_attention_self_proj_weight_dim_1,


        &reshape_113_0_dim_0,

        &bert_encoder_layer_11_attention_self_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_115_0_dim_0;

        M *= reshape_115_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_115_0_dim_2;

    
        layernorm_6(
           layernorm_116_0, reshape_115_0, bert_encoder_layer_11_attention_output_LayerNorm_weight, bert_encoder_layer_11_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_116_0,
        bert_encoder_layer_11_intermediate_dense_weight,

        bert_encoder_layer_11_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_117_0,
        global_workspace_,
        1,

        &reshape_115_0_dim_0,

        &reshape_115_0_dim_1,

        &reshape_115_0_dim_2,


        &bert_encoder_layer_11_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_11_intermediate_dense_weight_dim_1,


        &reshape_115_0_dim_0,

        &reshape_115_0_dim_1,

        &bert_encoder_layer_11_intermediate_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_117_0,
        bert_encoder_layer_11_output_dense_weight,
        bert_encoder_layer_11_output_dense_bias,
        layernorm_116_0,

        gemm_rcr_bias_add_118_0,
        global_workspace_,

     1,


        &reshape_115_0_dim_0,

        &reshape_115_0_dim_1,

        &bert_encoder_layer_11_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_11_output_dense_weight_dim_0,

        &bert_encoder_layer_11_output_dense_weight_dim_1,


        &reshape_115_0_dim_0,

        &reshape_115_0_dim_1,

        &bert_encoder_layer_11_output_dense_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_115_0_dim_0;

        M *= reshape_115_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_11_output_dense_weight_dim_0;

    
        layernorm_6(
           output_0, gemm_rcr_bias_add_118_0, bert_encoder_layer_11_output_LayerNorm_weight, bert_encoder_layer_11_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
    }




    void ProfileImpl(StreamType stream, size_t iters, const std::string& filename) {
#ifdef OPTIMIZE_FOR_COMPILATION_TIME
      throw std::runtime_error("Profile is disabled, please recompile without OPTIMIZE_FOR_COMPILE_TIME flag");
#else
      std::ofstream ss(filename);
      if (!ss) {
        throw std::runtime_error(std::string("Could not open file ") + filename);
      }

      int deviceId;
      char* L2CacheSlab = nullptr;
      DevicePropertyType deviceProperties;
      GetDevice(&deviceId);
      GetDeviceProperties(&deviceProperties, deviceId);
      const size_t L2SizeInBytes = deviceProperties.l2CacheSize;
      DeviceMalloc((void**) &L2CacheSlab, L2SizeInBytes);

      ss << "{\n";
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_0" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_0(

        input,
        bert_encoder_layer_0_attention_self_qkv_weight,

        bert_encoder_layer_0_attention_self_qkv_bias,

        reshape_1_0,
        global_workspace_,
        1,

        &input_dim_0,

        &input_dim_1,

        &input_dim_2,


        &bert_encoder_layer_0_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_0_attention_self_qkv_weight_dim_1,


        &input_dim_0,

        &input_dim_1,

        &bert_encoder_layer_0_attention_self_qkv_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_0" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"2304\", \"768\"], [\"2304\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"2304\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "flash_attention_2" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    flash_attention_2(
       flash_attention_2_0, reshape_1_0, reinterpret_cast<int*>(bert_encoder_layer_0_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "flash_attention_2" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"3\", \"12\", \"64\"], [\"2\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"12\", \"64\"]]"
        
          << ", \"batch_size\": " << "\"1\""
        
          << ", \"dropout\": " << "\"0.0\""
        
          << ", \"max_seq_len\": " << "\"64\""
        
          << ", \"causal\": " << "\"False\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_4" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_2_0,
        bert_encoder_layer_0_attention_self_proj_weight,
        bert_encoder_layer_0_attention_self_proj_bias,
        input,

        reshape_5_0,
        global_workspace_,

     1,


        &reshape_3_0_dim_0,

        &reshape_3_0_dim_1,


        &bert_encoder_layer_0_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_0_attention_self_proj_weight_dim_1,


        &reshape_3_0_dim_0,

        &bert_encoder_layer_0_attention_self_proj_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_4" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"768\"], [\"768\", \"768\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_6" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_5_0_dim_0;

        M *= reshape_5_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_5_0_dim_2;

    
        layernorm_6(
           layernorm_6_0, reshape_5_0, bert_encoder_layer_0_attention_output_LayerNorm_weight, bert_encoder_layer_0_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_6" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_fast_gelu_7" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_6_0,
        bert_encoder_layer_0_intermediate_dense_weight,

        bert_encoder_layer_0_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_7_0,
        global_workspace_,
        1,

        &reshape_5_0_dim_0,

        &reshape_5_0_dim_1,

        &reshape_5_0_dim_2,


        &bert_encoder_layer_0_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_0_intermediate_dense_weight_dim_1,


        &reshape_5_0_dim_0,

        &reshape_5_0_dim_1,

        &bert_encoder_layer_0_intermediate_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_fast_gelu_7" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"3072\", \"768\"], [\"3072\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"3072\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_8" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_7_0,
        bert_encoder_layer_0_output_dense_weight,
        bert_encoder_layer_0_output_dense_bias,
        layernorm_6_0,

        gemm_rcr_bias_add_8_0,
        global_workspace_,

     1,


        &reshape_5_0_dim_0,

        &reshape_5_0_dim_1,

        &bert_encoder_layer_0_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_0_output_dense_weight_dim_0,

        &bert_encoder_layer_0_output_dense_weight_dim_1,


        &reshape_5_0_dim_0,

        &reshape_5_0_dim_1,

        &bert_encoder_layer_0_output_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_8" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"3072\"], [\"768\", \"3072\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_9" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_5_0_dim_0;

        M *= reshape_5_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_0_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_9_0, gemm_rcr_bias_add_8_0, bert_encoder_layer_0_output_LayerNorm_weight, bert_encoder_layer_0_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_9" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_10" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_0(

        layernorm_9_0,
        bert_encoder_layer_1_attention_self_qkv_weight,

        bert_encoder_layer_1_attention_self_qkv_bias,

        reshape_11_0,
        global_workspace_,
        1,

        &reshape_5_0_dim_0,

        &reshape_5_0_dim_1,

        &bert_encoder_layer_0_output_dense_weight_dim_0,


        &bert_encoder_layer_1_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_1_attention_self_qkv_weight_dim_1,


        &reshape_5_0_dim_0,

        &reshape_5_0_dim_1,

        &bert_encoder_layer_1_attention_self_qkv_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_10" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"2304\", \"768\"], [\"2304\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"2304\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "flash_attention_12" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    flash_attention_2(
       flash_attention_12_0, reshape_11_0, reinterpret_cast<int*>(bert_encoder_layer_1_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "flash_attention_12" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"3\", \"12\", \"64\"], [\"2\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"12\", \"64\"]]"
        
          << ", \"batch_size\": " << "\"1\""
        
          << ", \"dropout\": " << "\"0.0\""
        
          << ", \"max_seq_len\": " << "\"64\""
        
          << ", \"causal\": " << "\"False\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_14" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_12_0,
        bert_encoder_layer_1_attention_self_proj_weight,
        bert_encoder_layer_1_attention_self_proj_bias,
        layernorm_9_0,

        reshape_15_0,
        global_workspace_,

     1,


        &reshape_13_0_dim_0,

        &reshape_13_0_dim_1,


        &bert_encoder_layer_1_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_1_attention_self_proj_weight_dim_1,


        &reshape_13_0_dim_0,

        &bert_encoder_layer_1_attention_self_proj_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_14" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"768\"], [\"768\", \"768\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_16" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_15_0_dim_0;

        M *= reshape_15_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_15_0_dim_2;

    
        layernorm_6(
           layernorm_16_0, reshape_15_0, bert_encoder_layer_1_attention_output_LayerNorm_weight, bert_encoder_layer_1_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_16" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_fast_gelu_17" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_16_0,
        bert_encoder_layer_1_intermediate_dense_weight,

        bert_encoder_layer_1_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_17_0,
        global_workspace_,
        1,

        &reshape_15_0_dim_0,

        &reshape_15_0_dim_1,

        &reshape_15_0_dim_2,


        &bert_encoder_layer_1_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_1_intermediate_dense_weight_dim_1,


        &reshape_15_0_dim_0,

        &reshape_15_0_dim_1,

        &bert_encoder_layer_1_intermediate_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_fast_gelu_17" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"3072\", \"768\"], [\"3072\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"3072\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_18" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_17_0,
        bert_encoder_layer_1_output_dense_weight,
        bert_encoder_layer_1_output_dense_bias,
        layernorm_16_0,

        gemm_rcr_bias_add_18_0,
        global_workspace_,

     1,


        &reshape_15_0_dim_0,

        &reshape_15_0_dim_1,

        &bert_encoder_layer_1_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_1_output_dense_weight_dim_0,

        &bert_encoder_layer_1_output_dense_weight_dim_1,


        &reshape_15_0_dim_0,

        &reshape_15_0_dim_1,

        &bert_encoder_layer_1_output_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_18" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"3072\"], [\"768\", \"3072\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_19" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_15_0_dim_0;

        M *= reshape_15_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_1_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_19_0, gemm_rcr_bias_add_18_0, bert_encoder_layer_1_output_LayerNorm_weight, bert_encoder_layer_1_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_19" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_20" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_0(

        layernorm_19_0,
        bert_encoder_layer_2_attention_self_qkv_weight,

        bert_encoder_layer_2_attention_self_qkv_bias,

        reshape_21_0,
        global_workspace_,
        1,

        &reshape_15_0_dim_0,

        &reshape_15_0_dim_1,

        &bert_encoder_layer_1_output_dense_weight_dim_0,


        &bert_encoder_layer_2_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_2_attention_self_qkv_weight_dim_1,


        &reshape_15_0_dim_0,

        &reshape_15_0_dim_1,

        &bert_encoder_layer_2_attention_self_qkv_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_20" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"2304\", \"768\"], [\"2304\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"2304\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "flash_attention_22" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    flash_attention_2(
       flash_attention_22_0, reshape_21_0, reinterpret_cast<int*>(bert_encoder_layer_2_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "flash_attention_22" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"3\", \"12\", \"64\"], [\"2\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"12\", \"64\"]]"
        
          << ", \"batch_size\": " << "\"1\""
        
          << ", \"dropout\": " << "\"0.0\""
        
          << ", \"max_seq_len\": " << "\"64\""
        
          << ", \"causal\": " << "\"False\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_24" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_22_0,
        bert_encoder_layer_2_attention_self_proj_weight,
        bert_encoder_layer_2_attention_self_proj_bias,
        layernorm_19_0,

        reshape_25_0,
        global_workspace_,

     1,


        &reshape_23_0_dim_0,

        &reshape_23_0_dim_1,


        &bert_encoder_layer_2_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_2_attention_self_proj_weight_dim_1,


        &reshape_23_0_dim_0,

        &bert_encoder_layer_2_attention_self_proj_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_24" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"768\"], [\"768\", \"768\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_26" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_25_0_dim_0;

        M *= reshape_25_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_25_0_dim_2;

    
        layernorm_6(
           layernorm_26_0, reshape_25_0, bert_encoder_layer_2_attention_output_LayerNorm_weight, bert_encoder_layer_2_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_26" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_fast_gelu_27" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_26_0,
        bert_encoder_layer_2_intermediate_dense_weight,

        bert_encoder_layer_2_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_27_0,
        global_workspace_,
        1,

        &reshape_25_0_dim_0,

        &reshape_25_0_dim_1,

        &reshape_25_0_dim_2,


        &bert_encoder_layer_2_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_2_intermediate_dense_weight_dim_1,


        &reshape_25_0_dim_0,

        &reshape_25_0_dim_1,

        &bert_encoder_layer_2_intermediate_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_fast_gelu_27" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"3072\", \"768\"], [\"3072\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"3072\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_28" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_27_0,
        bert_encoder_layer_2_output_dense_weight,
        bert_encoder_layer_2_output_dense_bias,
        layernorm_26_0,

        gemm_rcr_bias_add_28_0,
        global_workspace_,

     1,


        &reshape_25_0_dim_0,

        &reshape_25_0_dim_1,

        &bert_encoder_layer_2_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_2_output_dense_weight_dim_0,

        &bert_encoder_layer_2_output_dense_weight_dim_1,


        &reshape_25_0_dim_0,

        &reshape_25_0_dim_1,

        &bert_encoder_layer_2_output_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_28" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"3072\"], [\"768\", \"3072\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_29" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_25_0_dim_0;

        M *= reshape_25_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_2_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_29_0, gemm_rcr_bias_add_28_0, bert_encoder_layer_2_output_LayerNorm_weight, bert_encoder_layer_2_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_29" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_30" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_0(

        layernorm_29_0,
        bert_encoder_layer_3_attention_self_qkv_weight,

        bert_encoder_layer_3_attention_self_qkv_bias,

        reshape_31_0,
        global_workspace_,
        1,

        &reshape_25_0_dim_0,

        &reshape_25_0_dim_1,

        &bert_encoder_layer_2_output_dense_weight_dim_0,


        &bert_encoder_layer_3_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_3_attention_self_qkv_weight_dim_1,


        &reshape_25_0_dim_0,

        &reshape_25_0_dim_1,

        &bert_encoder_layer_3_attention_self_qkv_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_30" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"2304\", \"768\"], [\"2304\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"2304\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "flash_attention_32" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    flash_attention_2(
       flash_attention_32_0, reshape_31_0, reinterpret_cast<int*>(bert_encoder_layer_3_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "flash_attention_32" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"3\", \"12\", \"64\"], [\"2\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"12\", \"64\"]]"
        
          << ", \"batch_size\": " << "\"1\""
        
          << ", \"dropout\": " << "\"0.0\""
        
          << ", \"max_seq_len\": " << "\"64\""
        
          << ", \"causal\": " << "\"False\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_34" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_32_0,
        bert_encoder_layer_3_attention_self_proj_weight,
        bert_encoder_layer_3_attention_self_proj_bias,
        layernorm_29_0,

        reshape_35_0,
        global_workspace_,

     1,


        &reshape_33_0_dim_0,

        &reshape_33_0_dim_1,


        &bert_encoder_layer_3_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_3_attention_self_proj_weight_dim_1,


        &reshape_33_0_dim_0,

        &bert_encoder_layer_3_attention_self_proj_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_34" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"768\"], [\"768\", \"768\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_36" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_35_0_dim_0;

        M *= reshape_35_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_35_0_dim_2;

    
        layernorm_6(
           layernorm_36_0, reshape_35_0, bert_encoder_layer_3_attention_output_LayerNorm_weight, bert_encoder_layer_3_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_36" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_fast_gelu_37" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_36_0,
        bert_encoder_layer_3_intermediate_dense_weight,

        bert_encoder_layer_3_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_37_0,
        global_workspace_,
        1,

        &reshape_35_0_dim_0,

        &reshape_35_0_dim_1,

        &reshape_35_0_dim_2,


        &bert_encoder_layer_3_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_3_intermediate_dense_weight_dim_1,


        &reshape_35_0_dim_0,

        &reshape_35_0_dim_1,

        &bert_encoder_layer_3_intermediate_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_fast_gelu_37" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"3072\", \"768\"], [\"3072\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"3072\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_38" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_37_0,
        bert_encoder_layer_3_output_dense_weight,
        bert_encoder_layer_3_output_dense_bias,
        layernorm_36_0,

        gemm_rcr_bias_add_38_0,
        global_workspace_,

     1,


        &reshape_35_0_dim_0,

        &reshape_35_0_dim_1,

        &bert_encoder_layer_3_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_3_output_dense_weight_dim_0,

        &bert_encoder_layer_3_output_dense_weight_dim_1,


        &reshape_35_0_dim_0,

        &reshape_35_0_dim_1,

        &bert_encoder_layer_3_output_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_38" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"3072\"], [\"768\", \"3072\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_39" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_35_0_dim_0;

        M *= reshape_35_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_3_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_39_0, gemm_rcr_bias_add_38_0, bert_encoder_layer_3_output_LayerNorm_weight, bert_encoder_layer_3_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_39" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_40" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_0(

        layernorm_39_0,
        bert_encoder_layer_4_attention_self_qkv_weight,

        bert_encoder_layer_4_attention_self_qkv_bias,

        reshape_41_0,
        global_workspace_,
        1,

        &reshape_35_0_dim_0,

        &reshape_35_0_dim_1,

        &bert_encoder_layer_3_output_dense_weight_dim_0,


        &bert_encoder_layer_4_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_4_attention_self_qkv_weight_dim_1,


        &reshape_35_0_dim_0,

        &reshape_35_0_dim_1,

        &bert_encoder_layer_4_attention_self_qkv_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_40" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"2304\", \"768\"], [\"2304\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"2304\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "flash_attention_42" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    flash_attention_2(
       flash_attention_42_0, reshape_41_0, reinterpret_cast<int*>(bert_encoder_layer_4_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "flash_attention_42" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"3\", \"12\", \"64\"], [\"2\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"12\", \"64\"]]"
        
          << ", \"batch_size\": " << "\"1\""
        
          << ", \"dropout\": " << "\"0.0\""
        
          << ", \"max_seq_len\": " << "\"64\""
        
          << ", \"causal\": " << "\"False\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_44" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_42_0,
        bert_encoder_layer_4_attention_self_proj_weight,
        bert_encoder_layer_4_attention_self_proj_bias,
        layernorm_39_0,

        reshape_45_0,
        global_workspace_,

     1,


        &reshape_43_0_dim_0,

        &reshape_43_0_dim_1,


        &bert_encoder_layer_4_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_4_attention_self_proj_weight_dim_1,


        &reshape_43_0_dim_0,

        &bert_encoder_layer_4_attention_self_proj_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_44" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"768\"], [\"768\", \"768\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_46" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_45_0_dim_0;

        M *= reshape_45_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_45_0_dim_2;

    
        layernorm_6(
           layernorm_46_0, reshape_45_0, bert_encoder_layer_4_attention_output_LayerNorm_weight, bert_encoder_layer_4_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_46" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_fast_gelu_47" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_46_0,
        bert_encoder_layer_4_intermediate_dense_weight,

        bert_encoder_layer_4_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_47_0,
        global_workspace_,
        1,

        &reshape_45_0_dim_0,

        &reshape_45_0_dim_1,

        &reshape_45_0_dim_2,


        &bert_encoder_layer_4_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_4_intermediate_dense_weight_dim_1,


        &reshape_45_0_dim_0,

        &reshape_45_0_dim_1,

        &bert_encoder_layer_4_intermediate_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_fast_gelu_47" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"3072\", \"768\"], [\"3072\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"3072\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_48" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_47_0,
        bert_encoder_layer_4_output_dense_weight,
        bert_encoder_layer_4_output_dense_bias,
        layernorm_46_0,

        gemm_rcr_bias_add_48_0,
        global_workspace_,

     1,


        &reshape_45_0_dim_0,

        &reshape_45_0_dim_1,

        &bert_encoder_layer_4_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_4_output_dense_weight_dim_0,

        &bert_encoder_layer_4_output_dense_weight_dim_1,


        &reshape_45_0_dim_0,

        &reshape_45_0_dim_1,

        &bert_encoder_layer_4_output_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_48" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"3072\"], [\"768\", \"3072\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_49" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_45_0_dim_0;

        M *= reshape_45_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_4_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_49_0, gemm_rcr_bias_add_48_0, bert_encoder_layer_4_output_LayerNorm_weight, bert_encoder_layer_4_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_49" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_50" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_0(

        layernorm_49_0,
        bert_encoder_layer_5_attention_self_qkv_weight,

        bert_encoder_layer_5_attention_self_qkv_bias,

        reshape_51_0,
        global_workspace_,
        1,

        &reshape_45_0_dim_0,

        &reshape_45_0_dim_1,

        &bert_encoder_layer_4_output_dense_weight_dim_0,


        &bert_encoder_layer_5_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_5_attention_self_qkv_weight_dim_1,


        &reshape_45_0_dim_0,

        &reshape_45_0_dim_1,

        &bert_encoder_layer_5_attention_self_qkv_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_50" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"2304\", \"768\"], [\"2304\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"2304\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "flash_attention_52" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    flash_attention_2(
       flash_attention_52_0, reshape_51_0, reinterpret_cast<int*>(bert_encoder_layer_5_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "flash_attention_52" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"3\", \"12\", \"64\"], [\"2\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"12\", \"64\"]]"
        
          << ", \"batch_size\": " << "\"1\""
        
          << ", \"dropout\": " << "\"0.0\""
        
          << ", \"max_seq_len\": " << "\"64\""
        
          << ", \"causal\": " << "\"False\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_54" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_52_0,
        bert_encoder_layer_5_attention_self_proj_weight,
        bert_encoder_layer_5_attention_self_proj_bias,
        layernorm_49_0,

        reshape_55_0,
        global_workspace_,

     1,


        &reshape_53_0_dim_0,

        &reshape_53_0_dim_1,


        &bert_encoder_layer_5_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_5_attention_self_proj_weight_dim_1,


        &reshape_53_0_dim_0,

        &bert_encoder_layer_5_attention_self_proj_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_54" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"768\"], [\"768\", \"768\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_56" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_55_0_dim_0;

        M *= reshape_55_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_55_0_dim_2;

    
        layernorm_6(
           layernorm_56_0, reshape_55_0, bert_encoder_layer_5_attention_output_LayerNorm_weight, bert_encoder_layer_5_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_56" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_fast_gelu_57" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_56_0,
        bert_encoder_layer_5_intermediate_dense_weight,

        bert_encoder_layer_5_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_57_0,
        global_workspace_,
        1,

        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &reshape_55_0_dim_2,


        &bert_encoder_layer_5_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_5_intermediate_dense_weight_dim_1,


        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &bert_encoder_layer_5_intermediate_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_fast_gelu_57" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"3072\", \"768\"], [\"3072\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"3072\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_58" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_57_0,
        bert_encoder_layer_5_output_dense_weight,
        bert_encoder_layer_5_output_dense_bias,
        layernorm_56_0,

        gemm_rcr_bias_add_58_0,
        global_workspace_,

     1,


        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &bert_encoder_layer_5_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_5_output_dense_weight_dim_0,

        &bert_encoder_layer_5_output_dense_weight_dim_1,


        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &bert_encoder_layer_5_output_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_58" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"3072\"], [\"768\", \"3072\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_59" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_55_0_dim_0;

        M *= reshape_55_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_5_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_59_0, gemm_rcr_bias_add_58_0, bert_encoder_layer_5_output_LayerNorm_weight, bert_encoder_layer_5_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_59" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_60" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_0(

        layernorm_59_0,
        bert_encoder_layer_6_attention_self_qkv_weight,

        bert_encoder_layer_6_attention_self_qkv_bias,

        reshape_61_0,
        global_workspace_,
        1,

        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &bert_encoder_layer_5_output_dense_weight_dim_0,


        &bert_encoder_layer_6_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_6_attention_self_qkv_weight_dim_1,


        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &bert_encoder_layer_6_attention_self_qkv_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_60" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"2304\", \"768\"], [\"2304\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"2304\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "flash_attention_62" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    flash_attention_2(
       flash_attention_62_0, reshape_61_0, reinterpret_cast<int*>(bert_encoder_layer_6_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "flash_attention_62" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"3\", \"12\", \"64\"], [\"2\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"12\", \"64\"]]"
        
          << ", \"batch_size\": " << "\"1\""
        
          << ", \"dropout\": " << "\"0.0\""
        
          << ", \"max_seq_len\": " << "\"64\""
        
          << ", \"causal\": " << "\"False\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_64" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_62_0,
        bert_encoder_layer_6_attention_self_proj_weight,
        bert_encoder_layer_6_attention_self_proj_bias,
        layernorm_59_0,

        reshape_65_0,
        global_workspace_,

     1,


        &reshape_63_0_dim_0,

        &reshape_63_0_dim_1,


        &bert_encoder_layer_6_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_6_attention_self_proj_weight_dim_1,


        &reshape_63_0_dim_0,

        &bert_encoder_layer_6_attention_self_proj_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_64" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"768\"], [\"768\", \"768\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_66" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_65_0_dim_0;

        M *= reshape_65_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_65_0_dim_2;

    
        layernorm_6(
           layernorm_66_0, reshape_65_0, bert_encoder_layer_6_attention_output_LayerNorm_weight, bert_encoder_layer_6_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_66" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_fast_gelu_67" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_66_0,
        bert_encoder_layer_6_intermediate_dense_weight,

        bert_encoder_layer_6_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_67_0,
        global_workspace_,
        1,

        &reshape_65_0_dim_0,

        &reshape_65_0_dim_1,

        &reshape_65_0_dim_2,


        &bert_encoder_layer_6_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_6_intermediate_dense_weight_dim_1,


        &reshape_65_0_dim_0,

        &reshape_65_0_dim_1,

        &bert_encoder_layer_6_intermediate_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_fast_gelu_67" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"3072\", \"768\"], [\"3072\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"3072\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_68" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_67_0,
        bert_encoder_layer_6_output_dense_weight,
        bert_encoder_layer_6_output_dense_bias,
        layernorm_66_0,

        gemm_rcr_bias_add_68_0,
        global_workspace_,

     1,


        &reshape_65_0_dim_0,

        &reshape_65_0_dim_1,

        &bert_encoder_layer_6_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_6_output_dense_weight_dim_0,

        &bert_encoder_layer_6_output_dense_weight_dim_1,


        &reshape_65_0_dim_0,

        &reshape_65_0_dim_1,

        &bert_encoder_layer_6_output_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_68" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"3072\"], [\"768\", \"3072\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_69" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_65_0_dim_0;

        M *= reshape_65_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_6_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_69_0, gemm_rcr_bias_add_68_0, bert_encoder_layer_6_output_LayerNorm_weight, bert_encoder_layer_6_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_69" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_70" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_0(

        layernorm_69_0,
        bert_encoder_layer_7_attention_self_qkv_weight,

        bert_encoder_layer_7_attention_self_qkv_bias,

        reshape_71_0,
        global_workspace_,
        1,

        &reshape_65_0_dim_0,

        &reshape_65_0_dim_1,

        &bert_encoder_layer_6_output_dense_weight_dim_0,


        &bert_encoder_layer_7_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_7_attention_self_qkv_weight_dim_1,


        &reshape_65_0_dim_0,

        &reshape_65_0_dim_1,

        &bert_encoder_layer_7_attention_self_qkv_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_70" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"2304\", \"768\"], [\"2304\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"2304\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "flash_attention_72" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    flash_attention_2(
       flash_attention_72_0, reshape_71_0, reinterpret_cast<int*>(bert_encoder_layer_7_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "flash_attention_72" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"3\", \"12\", \"64\"], [\"2\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"12\", \"64\"]]"
        
          << ", \"batch_size\": " << "\"1\""
        
          << ", \"dropout\": " << "\"0.0\""
        
          << ", \"max_seq_len\": " << "\"64\""
        
          << ", \"causal\": " << "\"False\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_74" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_72_0,
        bert_encoder_layer_7_attention_self_proj_weight,
        bert_encoder_layer_7_attention_self_proj_bias,
        layernorm_69_0,

        reshape_75_0,
        global_workspace_,

     1,


        &reshape_73_0_dim_0,

        &reshape_73_0_dim_1,


        &bert_encoder_layer_7_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_7_attention_self_proj_weight_dim_1,


        &reshape_73_0_dim_0,

        &bert_encoder_layer_7_attention_self_proj_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_74" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"768\"], [\"768\", \"768\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_76" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_75_0_dim_0;

        M *= reshape_75_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_75_0_dim_2;

    
        layernorm_6(
           layernorm_76_0, reshape_75_0, bert_encoder_layer_7_attention_output_LayerNorm_weight, bert_encoder_layer_7_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_76" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_fast_gelu_77" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_76_0,
        bert_encoder_layer_7_intermediate_dense_weight,

        bert_encoder_layer_7_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_77_0,
        global_workspace_,
        1,

        &reshape_75_0_dim_0,

        &reshape_75_0_dim_1,

        &reshape_75_0_dim_2,


        &bert_encoder_layer_7_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_7_intermediate_dense_weight_dim_1,


        &reshape_75_0_dim_0,

        &reshape_75_0_dim_1,

        &bert_encoder_layer_7_intermediate_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_fast_gelu_77" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"3072\", \"768\"], [\"3072\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"3072\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_78" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_77_0,
        bert_encoder_layer_7_output_dense_weight,
        bert_encoder_layer_7_output_dense_bias,
        layernorm_76_0,

        gemm_rcr_bias_add_78_0,
        global_workspace_,

     1,


        &reshape_75_0_dim_0,

        &reshape_75_0_dim_1,

        &bert_encoder_layer_7_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_7_output_dense_weight_dim_0,

        &bert_encoder_layer_7_output_dense_weight_dim_1,


        &reshape_75_0_dim_0,

        &reshape_75_0_dim_1,

        &bert_encoder_layer_7_output_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_78" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"3072\"], [\"768\", \"3072\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_79" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_75_0_dim_0;

        M *= reshape_75_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_7_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_79_0, gemm_rcr_bias_add_78_0, bert_encoder_layer_7_output_LayerNorm_weight, bert_encoder_layer_7_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_79" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_80" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_0(

        layernorm_79_0,
        bert_encoder_layer_8_attention_self_qkv_weight,

        bert_encoder_layer_8_attention_self_qkv_bias,

        reshape_81_0,
        global_workspace_,
        1,

        &reshape_75_0_dim_0,

        &reshape_75_0_dim_1,

        &bert_encoder_layer_7_output_dense_weight_dim_0,


        &bert_encoder_layer_8_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_8_attention_self_qkv_weight_dim_1,


        &reshape_75_0_dim_0,

        &reshape_75_0_dim_1,

        &bert_encoder_layer_8_attention_self_qkv_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_80" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"2304\", \"768\"], [\"2304\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"2304\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "flash_attention_82" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    flash_attention_2(
       flash_attention_82_0, reshape_81_0, reinterpret_cast<int*>(bert_encoder_layer_8_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "flash_attention_82" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"3\", \"12\", \"64\"], [\"2\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"12\", \"64\"]]"
        
          << ", \"batch_size\": " << "\"1\""
        
          << ", \"dropout\": " << "\"0.0\""
        
          << ", \"max_seq_len\": " << "\"64\""
        
          << ", \"causal\": " << "\"False\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_84" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_82_0,
        bert_encoder_layer_8_attention_self_proj_weight,
        bert_encoder_layer_8_attention_self_proj_bias,
        layernorm_79_0,

        reshape_85_0,
        global_workspace_,

     1,


        &reshape_83_0_dim_0,

        &reshape_83_0_dim_1,


        &bert_encoder_layer_8_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_8_attention_self_proj_weight_dim_1,


        &reshape_83_0_dim_0,

        &bert_encoder_layer_8_attention_self_proj_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_84" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"768\"], [\"768\", \"768\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_86" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_85_0_dim_0;

        M *= reshape_85_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_85_0_dim_2;

    
        layernorm_6(
           layernorm_86_0, reshape_85_0, bert_encoder_layer_8_attention_output_LayerNorm_weight, bert_encoder_layer_8_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_86" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_fast_gelu_87" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_86_0,
        bert_encoder_layer_8_intermediate_dense_weight,

        bert_encoder_layer_8_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_87_0,
        global_workspace_,
        1,

        &reshape_85_0_dim_0,

        &reshape_85_0_dim_1,

        &reshape_85_0_dim_2,


        &bert_encoder_layer_8_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_8_intermediate_dense_weight_dim_1,


        &reshape_85_0_dim_0,

        &reshape_85_0_dim_1,

        &bert_encoder_layer_8_intermediate_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_fast_gelu_87" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"3072\", \"768\"], [\"3072\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"3072\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_88" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_87_0,
        bert_encoder_layer_8_output_dense_weight,
        bert_encoder_layer_8_output_dense_bias,
        layernorm_86_0,

        gemm_rcr_bias_add_88_0,
        global_workspace_,

     1,


        &reshape_85_0_dim_0,

        &reshape_85_0_dim_1,

        &bert_encoder_layer_8_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_8_output_dense_weight_dim_0,

        &bert_encoder_layer_8_output_dense_weight_dim_1,


        &reshape_85_0_dim_0,

        &reshape_85_0_dim_1,

        &bert_encoder_layer_8_output_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_88" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"3072\"], [\"768\", \"3072\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_89" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_85_0_dim_0;

        M *= reshape_85_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_8_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_89_0, gemm_rcr_bias_add_88_0, bert_encoder_layer_8_output_LayerNorm_weight, bert_encoder_layer_8_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_89" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_90" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_0(

        layernorm_89_0,
        bert_encoder_layer_9_attention_self_qkv_weight,

        bert_encoder_layer_9_attention_self_qkv_bias,

        reshape_91_0,
        global_workspace_,
        1,

        &reshape_85_0_dim_0,

        &reshape_85_0_dim_1,

        &bert_encoder_layer_8_output_dense_weight_dim_0,


        &bert_encoder_layer_9_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_9_attention_self_qkv_weight_dim_1,


        &reshape_85_0_dim_0,

        &reshape_85_0_dim_1,

        &bert_encoder_layer_9_attention_self_qkv_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_90" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"2304\", \"768\"], [\"2304\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"2304\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "flash_attention_92" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    flash_attention_2(
       flash_attention_92_0, reshape_91_0, reinterpret_cast<int*>(bert_encoder_layer_9_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "flash_attention_92" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"3\", \"12\", \"64\"], [\"2\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"12\", \"64\"]]"
        
          << ", \"batch_size\": " << "\"1\""
        
          << ", \"dropout\": " << "\"0.0\""
        
          << ", \"max_seq_len\": " << "\"64\""
        
          << ", \"causal\": " << "\"False\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_94" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_92_0,
        bert_encoder_layer_9_attention_self_proj_weight,
        bert_encoder_layer_9_attention_self_proj_bias,
        layernorm_89_0,

        reshape_95_0,
        global_workspace_,

     1,


        &reshape_93_0_dim_0,

        &reshape_93_0_dim_1,


        &bert_encoder_layer_9_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_9_attention_self_proj_weight_dim_1,


        &reshape_93_0_dim_0,

        &bert_encoder_layer_9_attention_self_proj_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_94" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"768\"], [\"768\", \"768\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_96" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_95_0_dim_0;

        M *= reshape_95_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_95_0_dim_2;

    
        layernorm_6(
           layernorm_96_0, reshape_95_0, bert_encoder_layer_9_attention_output_LayerNorm_weight, bert_encoder_layer_9_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_96" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_fast_gelu_97" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_96_0,
        bert_encoder_layer_9_intermediate_dense_weight,

        bert_encoder_layer_9_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_97_0,
        global_workspace_,
        1,

        &reshape_95_0_dim_0,

        &reshape_95_0_dim_1,

        &reshape_95_0_dim_2,


        &bert_encoder_layer_9_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_9_intermediate_dense_weight_dim_1,


        &reshape_95_0_dim_0,

        &reshape_95_0_dim_1,

        &bert_encoder_layer_9_intermediate_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_fast_gelu_97" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"3072\", \"768\"], [\"3072\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"3072\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_98" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_97_0,
        bert_encoder_layer_9_output_dense_weight,
        bert_encoder_layer_9_output_dense_bias,
        layernorm_96_0,

        gemm_rcr_bias_add_98_0,
        global_workspace_,

     1,


        &reshape_95_0_dim_0,

        &reshape_95_0_dim_1,

        &bert_encoder_layer_9_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_9_output_dense_weight_dim_0,

        &bert_encoder_layer_9_output_dense_weight_dim_1,


        &reshape_95_0_dim_0,

        &reshape_95_0_dim_1,

        &bert_encoder_layer_9_output_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_98" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"3072\"], [\"768\", \"3072\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_99" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_95_0_dim_0;

        M *= reshape_95_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_9_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_99_0, gemm_rcr_bias_add_98_0, bert_encoder_layer_9_output_LayerNorm_weight, bert_encoder_layer_9_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_99" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_100" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_0(

        layernorm_99_0,
        bert_encoder_layer_10_attention_self_qkv_weight,

        bert_encoder_layer_10_attention_self_qkv_bias,

        reshape_101_0,
        global_workspace_,
        1,

        &reshape_95_0_dim_0,

        &reshape_95_0_dim_1,

        &bert_encoder_layer_9_output_dense_weight_dim_0,


        &bert_encoder_layer_10_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_10_attention_self_qkv_weight_dim_1,


        &reshape_95_0_dim_0,

        &reshape_95_0_dim_1,

        &bert_encoder_layer_10_attention_self_qkv_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_100" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"2304\", \"768\"], [\"2304\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"2304\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "flash_attention_102" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    flash_attention_2(
       flash_attention_102_0, reshape_101_0, reinterpret_cast<int*>(bert_encoder_layer_10_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "flash_attention_102" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"3\", \"12\", \"64\"], [\"2\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"12\", \"64\"]]"
        
          << ", \"batch_size\": " << "\"1\""
        
          << ", \"dropout\": " << "\"0.0\""
        
          << ", \"max_seq_len\": " << "\"64\""
        
          << ", \"causal\": " << "\"False\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_104" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_102_0,
        bert_encoder_layer_10_attention_self_proj_weight,
        bert_encoder_layer_10_attention_self_proj_bias,
        layernorm_99_0,

        reshape_105_0,
        global_workspace_,

     1,


        &reshape_103_0_dim_0,

        &reshape_103_0_dim_1,


        &bert_encoder_layer_10_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_10_attention_self_proj_weight_dim_1,


        &reshape_103_0_dim_0,

        &bert_encoder_layer_10_attention_self_proj_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_104" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"768\"], [\"768\", \"768\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_106" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_105_0_dim_0;

        M *= reshape_105_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_105_0_dim_2;

    
        layernorm_6(
           layernorm_106_0, reshape_105_0, bert_encoder_layer_10_attention_output_LayerNorm_weight, bert_encoder_layer_10_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_106" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_fast_gelu_107" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_106_0,
        bert_encoder_layer_10_intermediate_dense_weight,

        bert_encoder_layer_10_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_107_0,
        global_workspace_,
        1,

        &reshape_105_0_dim_0,

        &reshape_105_0_dim_1,

        &reshape_105_0_dim_2,


        &bert_encoder_layer_10_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_10_intermediate_dense_weight_dim_1,


        &reshape_105_0_dim_0,

        &reshape_105_0_dim_1,

        &bert_encoder_layer_10_intermediate_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_fast_gelu_107" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"3072\", \"768\"], [\"3072\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"3072\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_108" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_107_0,
        bert_encoder_layer_10_output_dense_weight,
        bert_encoder_layer_10_output_dense_bias,
        layernorm_106_0,

        gemm_rcr_bias_add_108_0,
        global_workspace_,

     1,


        &reshape_105_0_dim_0,

        &reshape_105_0_dim_1,

        &bert_encoder_layer_10_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_10_output_dense_weight_dim_0,

        &bert_encoder_layer_10_output_dense_weight_dim_1,


        &reshape_105_0_dim_0,

        &reshape_105_0_dim_1,

        &bert_encoder_layer_10_output_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_108" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"3072\"], [\"768\", \"3072\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_109" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_105_0_dim_0;

        M *= reshape_105_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_10_output_dense_weight_dim_0;

    
        layernorm_6(
           layernorm_109_0, gemm_rcr_bias_add_108_0, bert_encoder_layer_10_output_LayerNorm_weight, bert_encoder_layer_10_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_109" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_110" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_0(

        layernorm_109_0,
        bert_encoder_layer_11_attention_self_qkv_weight,

        bert_encoder_layer_11_attention_self_qkv_bias,

        reshape_111_0,
        global_workspace_,
        1,

        &reshape_105_0_dim_0,

        &reshape_105_0_dim_1,

        &bert_encoder_layer_10_output_dense_weight_dim_0,


        &bert_encoder_layer_11_attention_self_qkv_weight_dim_0,

        &bert_encoder_layer_11_attention_self_qkv_weight_dim_1,


        &reshape_105_0_dim_0,

        &reshape_105_0_dim_1,

        &bert_encoder_layer_11_attention_self_qkv_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_110" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"2304\", \"768\"], [\"2304\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"2304\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "flash_attention_112" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    flash_attention_2(
       flash_attention_112_0, reshape_111_0, reinterpret_cast<int*>(bert_encoder_layer_11_attention_self_cu_length),
        reinterpret_cast<float*>(global_workspace_), reinterpret_cast<float*>(global_workspace_ + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        false, false, stream /* default stream */
    );
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "flash_attention_112" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"3\", \"12\", \"64\"], [\"2\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"12\", \"64\"]]"
        
          << ", \"batch_size\": " << "\"1\""
        
          << ", \"dropout\": " << "\"0.0\""
        
          << ", \"max_seq_len\": " << "\"64\""
        
          << ", \"causal\": " << "\"False\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_114" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_4(

        flash_attention_112_0,
        bert_encoder_layer_11_attention_self_proj_weight,
        bert_encoder_layer_11_attention_self_proj_bias,
        layernorm_109_0,

        reshape_115_0,
        global_workspace_,

     1,


        &reshape_113_0_dim_0,

        &reshape_113_0_dim_1,


        &bert_encoder_layer_11_attention_self_proj_weight_dim_0,

        &bert_encoder_layer_11_attention_self_proj_weight_dim_1,


        &reshape_113_0_dim_0,

        &bert_encoder_layer_11_attention_self_proj_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_114" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"768\"], [\"768\", \"768\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_116" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_115_0_dim_0;

        M *= reshape_115_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_115_0_dim_2;

    
        layernorm_6(
           layernorm_116_0, reshape_115_0, bert_encoder_layer_11_attention_output_LayerNorm_weight, bert_encoder_layer_11_attention_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_116" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_fast_gelu_117" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_fast_gelu_7(

        layernorm_116_0,
        bert_encoder_layer_11_intermediate_dense_weight,

        bert_encoder_layer_11_intermediate_dense_bias,

        gemm_rcr_bias_fast_gelu_117_0,
        global_workspace_,
        1,

        &reshape_115_0_dim_0,

        &reshape_115_0_dim_1,

        &reshape_115_0_dim_2,


        &bert_encoder_layer_11_intermediate_dense_weight_dim_0,

        &bert_encoder_layer_11_intermediate_dense_weight_dim_1,


        &reshape_115_0_dim_0,

        &reshape_115_0_dim_1,

        &bert_encoder_layer_11_intermediate_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_fast_gelu_117" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"], [\"3072\", \"768\"], [\"3072\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"3072\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_add_118" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_add_8(

        gemm_rcr_bias_fast_gelu_117_0,
        bert_encoder_layer_11_output_dense_weight,
        bert_encoder_layer_11_output_dense_bias,
        layernorm_116_0,

        gemm_rcr_bias_add_118_0,
        global_workspace_,

     1,


        &reshape_115_0_dim_0,

        &reshape_115_0_dim_1,

        &bert_encoder_layer_11_intermediate_dense_weight_dim_0,


        &bert_encoder_layer_11_output_dense_weight_dim_0,

        &bert_encoder_layer_11_output_dense_weight_dim_1,


        &reshape_115_0_dim_0,

        &reshape_115_0_dim_1,

        &bert_encoder_layer_11_output_dense_weight_dim_0,

        stream
    );
    }
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "gemm_rcr_bias_add_118" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"3072\"], [\"768\", \"3072\"], [\"768\"], [\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "layernorm_119" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
      {
        
        int64_t M = 1;

        M *= reshape_115_0_dim_0;

        M *= reshape_115_0_dim_1;

    

        int64_t N = 1;

        N *= bert_encoder_layer_11_output_dense_weight_dim_0;

    
        layernorm_6(
           output_0, gemm_rcr_bias_add_118_0, bert_encoder_layer_11_output_LayerNorm_weight, bert_encoder_layer_11_output_LayerNorm_bias,
           M, N, 1e-12, stream /* default stream */
        );
      }
    
          EventRecord(call_end, stream);
          DeviceCheckLastError(__FILE__, __LINE__);
        }
        EventSynchronize(std::get<1>(call_events.back()));
        float milliseconds = 0.0;
        for (auto& [call_start, call_end] : call_events) {
          float call_milliseconds = 0.0;
          EventElapsedTime(&call_milliseconds, call_start, call_end);
          DestroyEvent(call_start);
          DestroyEvent(call_end);
          milliseconds += call_milliseconds;
        }
        ss << "\"" << "layernorm_119" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"64\", \"768\"]]"
        
          << ", \"normalized_shape\": " << "\"[768]\""
        
           << " } ";
        
          ss << "\n";
        
      }
      
      ss << "}\n";

      DeviceToDeviceCopies(stream);
      std::cout << "AIT per op profiling finished." << std::endl;
      FreeDeviceMemory(L2CacheSlab);
#endif
    }

    static std::unique_ptr<Model> Create(
      AITemplateAllocator& allocator,
      uint8_t* constants
    ) {
      return std::make_unique<Model>(
          589824,
          202752 * (1 + 0),
          0 * (1 + 0),
          1,
          1,
          156,
          constants,
          allocator
      );
    }

  private:
   void* input {nullptr};
   void* bert_encoder_layer_0_attention_self_qkv_weight {nullptr};
   void* bert_encoder_layer_0_attention_self_qkv_bias {nullptr};
   void* bert_encoder_layer_0_attention_self_cu_length {nullptr};
   void* bert_encoder_layer_0_attention_self_proj_weight {nullptr};
   void* bert_encoder_layer_0_attention_self_proj_bias {nullptr};
   void* bert_encoder_layer_0_attention_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_0_attention_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_0_intermediate_dense_weight {nullptr};
   void* bert_encoder_layer_0_intermediate_dense_bias {nullptr};
   void* bert_encoder_layer_0_output_dense_weight {nullptr};
   void* bert_encoder_layer_0_output_dense_bias {nullptr};
   void* bert_encoder_layer_0_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_0_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_1_attention_self_qkv_weight {nullptr};
   void* bert_encoder_layer_1_attention_self_qkv_bias {nullptr};
   void* bert_encoder_layer_1_attention_self_cu_length {nullptr};
   void* bert_encoder_layer_1_attention_self_proj_weight {nullptr};
   void* bert_encoder_layer_1_attention_self_proj_bias {nullptr};
   void* bert_encoder_layer_1_attention_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_1_attention_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_1_intermediate_dense_weight {nullptr};
   void* bert_encoder_layer_1_intermediate_dense_bias {nullptr};
   void* bert_encoder_layer_1_output_dense_weight {nullptr};
   void* bert_encoder_layer_1_output_dense_bias {nullptr};
   void* bert_encoder_layer_1_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_1_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_2_attention_self_qkv_weight {nullptr};
   void* bert_encoder_layer_2_attention_self_qkv_bias {nullptr};
   void* bert_encoder_layer_2_attention_self_cu_length {nullptr};
   void* bert_encoder_layer_2_attention_self_proj_weight {nullptr};
   void* bert_encoder_layer_2_attention_self_proj_bias {nullptr};
   void* bert_encoder_layer_2_attention_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_2_attention_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_2_intermediate_dense_weight {nullptr};
   void* bert_encoder_layer_2_intermediate_dense_bias {nullptr};
   void* bert_encoder_layer_2_output_dense_weight {nullptr};
   void* bert_encoder_layer_2_output_dense_bias {nullptr};
   void* bert_encoder_layer_2_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_2_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_3_attention_self_qkv_weight {nullptr};
   void* bert_encoder_layer_3_attention_self_qkv_bias {nullptr};
   void* bert_encoder_layer_3_attention_self_cu_length {nullptr};
   void* bert_encoder_layer_3_attention_self_proj_weight {nullptr};
   void* bert_encoder_layer_3_attention_self_proj_bias {nullptr};
   void* bert_encoder_layer_3_attention_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_3_attention_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_3_intermediate_dense_weight {nullptr};
   void* bert_encoder_layer_3_intermediate_dense_bias {nullptr};
   void* bert_encoder_layer_3_output_dense_weight {nullptr};
   void* bert_encoder_layer_3_output_dense_bias {nullptr};
   void* bert_encoder_layer_3_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_3_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_4_attention_self_qkv_weight {nullptr};
   void* bert_encoder_layer_4_attention_self_qkv_bias {nullptr};
   void* bert_encoder_layer_4_attention_self_cu_length {nullptr};
   void* bert_encoder_layer_4_attention_self_proj_weight {nullptr};
   void* bert_encoder_layer_4_attention_self_proj_bias {nullptr};
   void* bert_encoder_layer_4_attention_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_4_attention_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_4_intermediate_dense_weight {nullptr};
   void* bert_encoder_layer_4_intermediate_dense_bias {nullptr};
   void* bert_encoder_layer_4_output_dense_weight {nullptr};
   void* bert_encoder_layer_4_output_dense_bias {nullptr};
   void* bert_encoder_layer_4_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_4_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_5_attention_self_qkv_weight {nullptr};
   void* bert_encoder_layer_5_attention_self_qkv_bias {nullptr};
   void* bert_encoder_layer_5_attention_self_cu_length {nullptr};
   void* bert_encoder_layer_5_attention_self_proj_weight {nullptr};
   void* bert_encoder_layer_5_attention_self_proj_bias {nullptr};
   void* bert_encoder_layer_5_attention_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_5_attention_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_5_intermediate_dense_weight {nullptr};
   void* bert_encoder_layer_5_intermediate_dense_bias {nullptr};
   void* bert_encoder_layer_5_output_dense_weight {nullptr};
   void* bert_encoder_layer_5_output_dense_bias {nullptr};
   void* bert_encoder_layer_5_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_5_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_6_attention_self_qkv_weight {nullptr};
   void* bert_encoder_layer_6_attention_self_qkv_bias {nullptr};
   void* bert_encoder_layer_6_attention_self_cu_length {nullptr};
   void* bert_encoder_layer_6_attention_self_proj_weight {nullptr};
   void* bert_encoder_layer_6_attention_self_proj_bias {nullptr};
   void* bert_encoder_layer_6_attention_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_6_attention_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_6_intermediate_dense_weight {nullptr};
   void* bert_encoder_layer_6_intermediate_dense_bias {nullptr};
   void* bert_encoder_layer_6_output_dense_weight {nullptr};
   void* bert_encoder_layer_6_output_dense_bias {nullptr};
   void* bert_encoder_layer_6_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_6_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_7_attention_self_qkv_weight {nullptr};
   void* bert_encoder_layer_7_attention_self_qkv_bias {nullptr};
   void* bert_encoder_layer_7_attention_self_cu_length {nullptr};
   void* bert_encoder_layer_7_attention_self_proj_weight {nullptr};
   void* bert_encoder_layer_7_attention_self_proj_bias {nullptr};
   void* bert_encoder_layer_7_attention_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_7_attention_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_7_intermediate_dense_weight {nullptr};
   void* bert_encoder_layer_7_intermediate_dense_bias {nullptr};
   void* bert_encoder_layer_7_output_dense_weight {nullptr};
   void* bert_encoder_layer_7_output_dense_bias {nullptr};
   void* bert_encoder_layer_7_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_7_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_8_attention_self_qkv_weight {nullptr};
   void* bert_encoder_layer_8_attention_self_qkv_bias {nullptr};
   void* bert_encoder_layer_8_attention_self_cu_length {nullptr};
   void* bert_encoder_layer_8_attention_self_proj_weight {nullptr};
   void* bert_encoder_layer_8_attention_self_proj_bias {nullptr};
   void* bert_encoder_layer_8_attention_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_8_attention_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_8_intermediate_dense_weight {nullptr};
   void* bert_encoder_layer_8_intermediate_dense_bias {nullptr};
   void* bert_encoder_layer_8_output_dense_weight {nullptr};
   void* bert_encoder_layer_8_output_dense_bias {nullptr};
   void* bert_encoder_layer_8_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_8_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_9_attention_self_qkv_weight {nullptr};
   void* bert_encoder_layer_9_attention_self_qkv_bias {nullptr};
   void* bert_encoder_layer_9_attention_self_cu_length {nullptr};
   void* bert_encoder_layer_9_attention_self_proj_weight {nullptr};
   void* bert_encoder_layer_9_attention_self_proj_bias {nullptr};
   void* bert_encoder_layer_9_attention_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_9_attention_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_9_intermediate_dense_weight {nullptr};
   void* bert_encoder_layer_9_intermediate_dense_bias {nullptr};
   void* bert_encoder_layer_9_output_dense_weight {nullptr};
   void* bert_encoder_layer_9_output_dense_bias {nullptr};
   void* bert_encoder_layer_9_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_9_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_10_attention_self_qkv_weight {nullptr};
   void* bert_encoder_layer_10_attention_self_qkv_bias {nullptr};
   void* bert_encoder_layer_10_attention_self_cu_length {nullptr};
   void* bert_encoder_layer_10_attention_self_proj_weight {nullptr};
   void* bert_encoder_layer_10_attention_self_proj_bias {nullptr};
   void* bert_encoder_layer_10_attention_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_10_attention_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_10_intermediate_dense_weight {nullptr};
   void* bert_encoder_layer_10_intermediate_dense_bias {nullptr};
   void* bert_encoder_layer_10_output_dense_weight {nullptr};
   void* bert_encoder_layer_10_output_dense_bias {nullptr};
   void* bert_encoder_layer_10_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_10_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_11_attention_self_qkv_weight {nullptr};
   void* bert_encoder_layer_11_attention_self_qkv_bias {nullptr};
   void* bert_encoder_layer_11_attention_self_cu_length {nullptr};
   void* bert_encoder_layer_11_attention_self_proj_weight {nullptr};
   void* bert_encoder_layer_11_attention_self_proj_bias {nullptr};
   void* bert_encoder_layer_11_attention_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_11_attention_output_LayerNorm_bias {nullptr};
   void* bert_encoder_layer_11_intermediate_dense_weight {nullptr};
   void* bert_encoder_layer_11_intermediate_dense_bias {nullptr};
   void* bert_encoder_layer_11_output_dense_weight {nullptr};
   void* bert_encoder_layer_11_output_dense_bias {nullptr};
   void* bert_encoder_layer_11_output_LayerNorm_weight {nullptr};
   void* bert_encoder_layer_11_output_LayerNorm_bias {nullptr};
   void* reshape_1_0 {nullptr};
   void* flash_attention_2_0 {nullptr};
   void* reshape_5_0 {nullptr};
   void* layernorm_6_0 {nullptr};
   void* gemm_rcr_bias_fast_gelu_7_0 {nullptr};
   void* gemm_rcr_bias_add_8_0 {nullptr};
   void* layernorm_9_0 {nullptr};
   void* reshape_11_0 {nullptr};
   void* flash_attention_12_0 {nullptr};
   void* reshape_15_0 {nullptr};
   void* layernorm_16_0 {nullptr};
   void* gemm_rcr_bias_fast_gelu_17_0 {nullptr};
   void* gemm_rcr_bias_add_18_0 {nullptr};
   void* layernorm_19_0 {nullptr};
   void* reshape_21_0 {nullptr};
   void* flash_attention_22_0 {nullptr};
   void* reshape_25_0 {nullptr};
   void* layernorm_26_0 {nullptr};
   void* gemm_rcr_bias_fast_gelu_27_0 {nullptr};
   void* gemm_rcr_bias_add_28_0 {nullptr};
   void* layernorm_29_0 {nullptr};
   void* reshape_31_0 {nullptr};
   void* flash_attention_32_0 {nullptr};
   void* reshape_35_0 {nullptr};
   void* layernorm_36_0 {nullptr};
   void* gemm_rcr_bias_fast_gelu_37_0 {nullptr};
   void* gemm_rcr_bias_add_38_0 {nullptr};
   void* layernorm_39_0 {nullptr};
   void* reshape_41_0 {nullptr};
   void* flash_attention_42_0 {nullptr};
   void* reshape_45_0 {nullptr};
   void* layernorm_46_0 {nullptr};
   void* gemm_rcr_bias_fast_gelu_47_0 {nullptr};
   void* gemm_rcr_bias_add_48_0 {nullptr};
   void* layernorm_49_0 {nullptr};
   void* reshape_51_0 {nullptr};
   void* flash_attention_52_0 {nullptr};
   void* reshape_55_0 {nullptr};
   void* layernorm_56_0 {nullptr};
   void* gemm_rcr_bias_fast_gelu_57_0 {nullptr};
   void* gemm_rcr_bias_add_58_0 {nullptr};
   void* layernorm_59_0 {nullptr};
   void* reshape_61_0 {nullptr};
   void* flash_attention_62_0 {nullptr};
   void* reshape_65_0 {nullptr};
   void* layernorm_66_0 {nullptr};
   void* gemm_rcr_bias_fast_gelu_67_0 {nullptr};
   void* gemm_rcr_bias_add_68_0 {nullptr};
   void* layernorm_69_0 {nullptr};
   void* reshape_71_0 {nullptr};
   void* flash_attention_72_0 {nullptr};
   void* reshape_75_0 {nullptr};
   void* layernorm_76_0 {nullptr};
   void* gemm_rcr_bias_fast_gelu_77_0 {nullptr};
   void* gemm_rcr_bias_add_78_0 {nullptr};
   void* layernorm_79_0 {nullptr};
   void* reshape_81_0 {nullptr};
   void* flash_attention_82_0 {nullptr};
   void* reshape_85_0 {nullptr};
   void* layernorm_86_0 {nullptr};
   void* gemm_rcr_bias_fast_gelu_87_0 {nullptr};
   void* gemm_rcr_bias_add_88_0 {nullptr};
   void* layernorm_89_0 {nullptr};
   void* reshape_91_0 {nullptr};
   void* flash_attention_92_0 {nullptr};
   void* reshape_95_0 {nullptr};
   void* layernorm_96_0 {nullptr};
   void* gemm_rcr_bias_fast_gelu_97_0 {nullptr};
   void* gemm_rcr_bias_add_98_0 {nullptr};
   void* layernorm_99_0 {nullptr};
   void* reshape_101_0 {nullptr};
   void* flash_attention_102_0 {nullptr};
   void* reshape_105_0 {nullptr};
   void* layernorm_106_0 {nullptr};
   void* gemm_rcr_bias_fast_gelu_107_0 {nullptr};
   void* gemm_rcr_bias_add_108_0 {nullptr};
   void* layernorm_109_0 {nullptr};
   void* reshape_111_0 {nullptr};
   void* flash_attention_112_0 {nullptr};
   void* reshape_115_0 {nullptr};
   void* layernorm_116_0 {nullptr};
   void* gemm_rcr_bias_fast_gelu_117_0 {nullptr};
   void* gemm_rcr_bias_add_118_0 {nullptr};
   void* output_0 {nullptr};
   int64_t input_dim_0 { 1 };
   int64_t input_dim_1 { 64 };
   int64_t input_dim_2 { 768 };
   int64_t bert_encoder_layer_0_attention_self_qkv_weight_dim_0 { 2304 };
   int64_t bert_encoder_layer_0_attention_self_qkv_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_0_attention_self_qkv_bias_dim_0 { 2304 };
   int64_t bert_encoder_layer_0_attention_self_cu_length_dim_0 { 2 };
   int64_t bert_encoder_layer_0_attention_self_proj_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_0_attention_self_proj_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_0_attention_self_proj_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_0_attention_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_0_attention_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_0_intermediate_dense_weight_dim_0 { 3072 };
   int64_t bert_encoder_layer_0_intermediate_dense_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_0_intermediate_dense_bias_dim_0 { 3072 };
   int64_t bert_encoder_layer_0_output_dense_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_0_output_dense_weight_dim_1 { 3072 };
   int64_t bert_encoder_layer_0_output_dense_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_0_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_0_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_1_attention_self_qkv_weight_dim_0 { 2304 };
   int64_t bert_encoder_layer_1_attention_self_qkv_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_1_attention_self_qkv_bias_dim_0 { 2304 };
   int64_t bert_encoder_layer_1_attention_self_cu_length_dim_0 { 2 };
   int64_t bert_encoder_layer_1_attention_self_proj_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_1_attention_self_proj_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_1_attention_self_proj_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_1_attention_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_1_attention_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_1_intermediate_dense_weight_dim_0 { 3072 };
   int64_t bert_encoder_layer_1_intermediate_dense_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_1_intermediate_dense_bias_dim_0 { 3072 };
   int64_t bert_encoder_layer_1_output_dense_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_1_output_dense_weight_dim_1 { 3072 };
   int64_t bert_encoder_layer_1_output_dense_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_1_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_1_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_2_attention_self_qkv_weight_dim_0 { 2304 };
   int64_t bert_encoder_layer_2_attention_self_qkv_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_2_attention_self_qkv_bias_dim_0 { 2304 };
   int64_t bert_encoder_layer_2_attention_self_cu_length_dim_0 { 2 };
   int64_t bert_encoder_layer_2_attention_self_proj_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_2_attention_self_proj_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_2_attention_self_proj_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_2_attention_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_2_attention_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_2_intermediate_dense_weight_dim_0 { 3072 };
   int64_t bert_encoder_layer_2_intermediate_dense_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_2_intermediate_dense_bias_dim_0 { 3072 };
   int64_t bert_encoder_layer_2_output_dense_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_2_output_dense_weight_dim_1 { 3072 };
   int64_t bert_encoder_layer_2_output_dense_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_2_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_2_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_3_attention_self_qkv_weight_dim_0 { 2304 };
   int64_t bert_encoder_layer_3_attention_self_qkv_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_3_attention_self_qkv_bias_dim_0 { 2304 };
   int64_t bert_encoder_layer_3_attention_self_cu_length_dim_0 { 2 };
   int64_t bert_encoder_layer_3_attention_self_proj_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_3_attention_self_proj_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_3_attention_self_proj_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_3_attention_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_3_attention_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_3_intermediate_dense_weight_dim_0 { 3072 };
   int64_t bert_encoder_layer_3_intermediate_dense_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_3_intermediate_dense_bias_dim_0 { 3072 };
   int64_t bert_encoder_layer_3_output_dense_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_3_output_dense_weight_dim_1 { 3072 };
   int64_t bert_encoder_layer_3_output_dense_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_3_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_3_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_4_attention_self_qkv_weight_dim_0 { 2304 };
   int64_t bert_encoder_layer_4_attention_self_qkv_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_4_attention_self_qkv_bias_dim_0 { 2304 };
   int64_t bert_encoder_layer_4_attention_self_cu_length_dim_0 { 2 };
   int64_t bert_encoder_layer_4_attention_self_proj_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_4_attention_self_proj_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_4_attention_self_proj_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_4_attention_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_4_attention_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_4_intermediate_dense_weight_dim_0 { 3072 };
   int64_t bert_encoder_layer_4_intermediate_dense_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_4_intermediate_dense_bias_dim_0 { 3072 };
   int64_t bert_encoder_layer_4_output_dense_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_4_output_dense_weight_dim_1 { 3072 };
   int64_t bert_encoder_layer_4_output_dense_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_4_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_4_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_5_attention_self_qkv_weight_dim_0 { 2304 };
   int64_t bert_encoder_layer_5_attention_self_qkv_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_5_attention_self_qkv_bias_dim_0 { 2304 };
   int64_t bert_encoder_layer_5_attention_self_cu_length_dim_0 { 2 };
   int64_t bert_encoder_layer_5_attention_self_proj_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_5_attention_self_proj_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_5_attention_self_proj_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_5_attention_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_5_attention_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_5_intermediate_dense_weight_dim_0 { 3072 };
   int64_t bert_encoder_layer_5_intermediate_dense_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_5_intermediate_dense_bias_dim_0 { 3072 };
   int64_t bert_encoder_layer_5_output_dense_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_5_output_dense_weight_dim_1 { 3072 };
   int64_t bert_encoder_layer_5_output_dense_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_5_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_5_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_6_attention_self_qkv_weight_dim_0 { 2304 };
   int64_t bert_encoder_layer_6_attention_self_qkv_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_6_attention_self_qkv_bias_dim_0 { 2304 };
   int64_t bert_encoder_layer_6_attention_self_cu_length_dim_0 { 2 };
   int64_t bert_encoder_layer_6_attention_self_proj_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_6_attention_self_proj_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_6_attention_self_proj_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_6_attention_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_6_attention_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_6_intermediate_dense_weight_dim_0 { 3072 };
   int64_t bert_encoder_layer_6_intermediate_dense_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_6_intermediate_dense_bias_dim_0 { 3072 };
   int64_t bert_encoder_layer_6_output_dense_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_6_output_dense_weight_dim_1 { 3072 };
   int64_t bert_encoder_layer_6_output_dense_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_6_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_6_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_7_attention_self_qkv_weight_dim_0 { 2304 };
   int64_t bert_encoder_layer_7_attention_self_qkv_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_7_attention_self_qkv_bias_dim_0 { 2304 };
   int64_t bert_encoder_layer_7_attention_self_cu_length_dim_0 { 2 };
   int64_t bert_encoder_layer_7_attention_self_proj_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_7_attention_self_proj_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_7_attention_self_proj_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_7_attention_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_7_attention_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_7_intermediate_dense_weight_dim_0 { 3072 };
   int64_t bert_encoder_layer_7_intermediate_dense_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_7_intermediate_dense_bias_dim_0 { 3072 };
   int64_t bert_encoder_layer_7_output_dense_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_7_output_dense_weight_dim_1 { 3072 };
   int64_t bert_encoder_layer_7_output_dense_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_7_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_7_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_8_attention_self_qkv_weight_dim_0 { 2304 };
   int64_t bert_encoder_layer_8_attention_self_qkv_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_8_attention_self_qkv_bias_dim_0 { 2304 };
   int64_t bert_encoder_layer_8_attention_self_cu_length_dim_0 { 2 };
   int64_t bert_encoder_layer_8_attention_self_proj_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_8_attention_self_proj_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_8_attention_self_proj_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_8_attention_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_8_attention_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_8_intermediate_dense_weight_dim_0 { 3072 };
   int64_t bert_encoder_layer_8_intermediate_dense_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_8_intermediate_dense_bias_dim_0 { 3072 };
   int64_t bert_encoder_layer_8_output_dense_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_8_output_dense_weight_dim_1 { 3072 };
   int64_t bert_encoder_layer_8_output_dense_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_8_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_8_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_9_attention_self_qkv_weight_dim_0 { 2304 };
   int64_t bert_encoder_layer_9_attention_self_qkv_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_9_attention_self_qkv_bias_dim_0 { 2304 };
   int64_t bert_encoder_layer_9_attention_self_cu_length_dim_0 { 2 };
   int64_t bert_encoder_layer_9_attention_self_proj_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_9_attention_self_proj_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_9_attention_self_proj_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_9_attention_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_9_attention_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_9_intermediate_dense_weight_dim_0 { 3072 };
   int64_t bert_encoder_layer_9_intermediate_dense_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_9_intermediate_dense_bias_dim_0 { 3072 };
   int64_t bert_encoder_layer_9_output_dense_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_9_output_dense_weight_dim_1 { 3072 };
   int64_t bert_encoder_layer_9_output_dense_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_9_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_9_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_10_attention_self_qkv_weight_dim_0 { 2304 };
   int64_t bert_encoder_layer_10_attention_self_qkv_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_10_attention_self_qkv_bias_dim_0 { 2304 };
   int64_t bert_encoder_layer_10_attention_self_cu_length_dim_0 { 2 };
   int64_t bert_encoder_layer_10_attention_self_proj_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_10_attention_self_proj_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_10_attention_self_proj_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_10_attention_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_10_attention_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_10_intermediate_dense_weight_dim_0 { 3072 };
   int64_t bert_encoder_layer_10_intermediate_dense_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_10_intermediate_dense_bias_dim_0 { 3072 };
   int64_t bert_encoder_layer_10_output_dense_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_10_output_dense_weight_dim_1 { 3072 };
   int64_t bert_encoder_layer_10_output_dense_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_10_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_10_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_11_attention_self_qkv_weight_dim_0 { 2304 };
   int64_t bert_encoder_layer_11_attention_self_qkv_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_11_attention_self_qkv_bias_dim_0 { 2304 };
   int64_t bert_encoder_layer_11_attention_self_cu_length_dim_0 { 2 };
   int64_t bert_encoder_layer_11_attention_self_proj_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_11_attention_self_proj_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_11_attention_self_proj_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_11_attention_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_11_attention_output_LayerNorm_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_11_intermediate_dense_weight_dim_0 { 3072 };
   int64_t bert_encoder_layer_11_intermediate_dense_weight_dim_1 { 768 };
   int64_t bert_encoder_layer_11_intermediate_dense_bias_dim_0 { 3072 };
   int64_t bert_encoder_layer_11_output_dense_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_11_output_dense_weight_dim_1 { 3072 };
   int64_t bert_encoder_layer_11_output_dense_bias_dim_0 { 768 };
   int64_t bert_encoder_layer_11_output_LayerNorm_weight_dim_0 { 768 };
   int64_t bert_encoder_layer_11_output_LayerNorm_bias_dim_0 { 768 };
   int64_t reshape_1_0_dim_0 { 64 };
   int64_t reshape_1_0_dim_1 { 3 };
   int64_t reshape_1_0_dim_2 { 12 };
   int64_t reshape_1_0_dim_3 { 64 };
   int64_t flash_attention_2_0_dim_0 { 64 };
   int64_t flash_attention_2_0_dim_1 { 12 };
   int64_t flash_attention_2_0_dim_2 { 64 };
   int64_t reshape_5_0_dim_0 { 1 };
   int64_t reshape_5_0_dim_1 { 64 };
   int64_t reshape_5_0_dim_2 { 768 };
   int64_t reshape_3_0_dim_0 { 64 };
   int64_t reshape_3_0_dim_1 { 768 };
   int64_t reshape_11_0_dim_0 { 64 };
   int64_t reshape_11_0_dim_1 { 3 };
   int64_t reshape_11_0_dim_2 { 12 };
   int64_t reshape_11_0_dim_3 { 64 };
   int64_t flash_attention_12_0_dim_0 { 64 };
   int64_t flash_attention_12_0_dim_1 { 12 };
   int64_t flash_attention_12_0_dim_2 { 64 };
   int64_t reshape_15_0_dim_0 { 1 };
   int64_t reshape_15_0_dim_1 { 64 };
   int64_t reshape_15_0_dim_2 { 768 };
   int64_t reshape_13_0_dim_0 { 64 };
   int64_t reshape_13_0_dim_1 { 768 };
   int64_t reshape_21_0_dim_0 { 64 };
   int64_t reshape_21_0_dim_1 { 3 };
   int64_t reshape_21_0_dim_2 { 12 };
   int64_t reshape_21_0_dim_3 { 64 };
   int64_t flash_attention_22_0_dim_0 { 64 };
   int64_t flash_attention_22_0_dim_1 { 12 };
   int64_t flash_attention_22_0_dim_2 { 64 };
   int64_t reshape_25_0_dim_0 { 1 };
   int64_t reshape_25_0_dim_1 { 64 };
   int64_t reshape_25_0_dim_2 { 768 };
   int64_t reshape_23_0_dim_0 { 64 };
   int64_t reshape_23_0_dim_1 { 768 };
   int64_t reshape_31_0_dim_0 { 64 };
   int64_t reshape_31_0_dim_1 { 3 };
   int64_t reshape_31_0_dim_2 { 12 };
   int64_t reshape_31_0_dim_3 { 64 };
   int64_t flash_attention_32_0_dim_0 { 64 };
   int64_t flash_attention_32_0_dim_1 { 12 };
   int64_t flash_attention_32_0_dim_2 { 64 };
   int64_t reshape_35_0_dim_0 { 1 };
   int64_t reshape_35_0_dim_1 { 64 };
   int64_t reshape_35_0_dim_2 { 768 };
   int64_t reshape_33_0_dim_0 { 64 };
   int64_t reshape_33_0_dim_1 { 768 };
   int64_t reshape_41_0_dim_0 { 64 };
   int64_t reshape_41_0_dim_1 { 3 };
   int64_t reshape_41_0_dim_2 { 12 };
   int64_t reshape_41_0_dim_3 { 64 };
   int64_t flash_attention_42_0_dim_0 { 64 };
   int64_t flash_attention_42_0_dim_1 { 12 };
   int64_t flash_attention_42_0_dim_2 { 64 };
   int64_t reshape_45_0_dim_0 { 1 };
   int64_t reshape_45_0_dim_1 { 64 };
   int64_t reshape_45_0_dim_2 { 768 };
   int64_t reshape_43_0_dim_0 { 64 };
   int64_t reshape_43_0_dim_1 { 768 };
   int64_t reshape_51_0_dim_0 { 64 };
   int64_t reshape_51_0_dim_1 { 3 };
   int64_t reshape_51_0_dim_2 { 12 };
   int64_t reshape_51_0_dim_3 { 64 };
   int64_t flash_attention_52_0_dim_0 { 64 };
   int64_t flash_attention_52_0_dim_1 { 12 };
   int64_t flash_attention_52_0_dim_2 { 64 };
   int64_t reshape_55_0_dim_0 { 1 };
   int64_t reshape_55_0_dim_1 { 64 };
   int64_t reshape_55_0_dim_2 { 768 };
   int64_t reshape_53_0_dim_0 { 64 };
   int64_t reshape_53_0_dim_1 { 768 };
   int64_t reshape_61_0_dim_0 { 64 };
   int64_t reshape_61_0_dim_1 { 3 };
   int64_t reshape_61_0_dim_2 { 12 };
   int64_t reshape_61_0_dim_3 { 64 };
   int64_t flash_attention_62_0_dim_0 { 64 };
   int64_t flash_attention_62_0_dim_1 { 12 };
   int64_t flash_attention_62_0_dim_2 { 64 };
   int64_t reshape_65_0_dim_0 { 1 };
   int64_t reshape_65_0_dim_1 { 64 };
   int64_t reshape_65_0_dim_2 { 768 };
   int64_t reshape_63_0_dim_0 { 64 };
   int64_t reshape_63_0_dim_1 { 768 };
   int64_t reshape_71_0_dim_0 { 64 };
   int64_t reshape_71_0_dim_1 { 3 };
   int64_t reshape_71_0_dim_2 { 12 };
   int64_t reshape_71_0_dim_3 { 64 };
   int64_t flash_attention_72_0_dim_0 { 64 };
   int64_t flash_attention_72_0_dim_1 { 12 };
   int64_t flash_attention_72_0_dim_2 { 64 };
   int64_t reshape_75_0_dim_0 { 1 };
   int64_t reshape_75_0_dim_1 { 64 };
   int64_t reshape_75_0_dim_2 { 768 };
   int64_t reshape_73_0_dim_0 { 64 };
   int64_t reshape_73_0_dim_1 { 768 };
   int64_t reshape_81_0_dim_0 { 64 };
   int64_t reshape_81_0_dim_1 { 3 };
   int64_t reshape_81_0_dim_2 { 12 };
   int64_t reshape_81_0_dim_3 { 64 };
   int64_t flash_attention_82_0_dim_0 { 64 };
   int64_t flash_attention_82_0_dim_1 { 12 };
   int64_t flash_attention_82_0_dim_2 { 64 };
   int64_t reshape_85_0_dim_0 { 1 };
   int64_t reshape_85_0_dim_1 { 64 };
   int64_t reshape_85_0_dim_2 { 768 };
   int64_t reshape_83_0_dim_0 { 64 };
   int64_t reshape_83_0_dim_1 { 768 };
   int64_t reshape_91_0_dim_0 { 64 };
   int64_t reshape_91_0_dim_1 { 3 };
   int64_t reshape_91_0_dim_2 { 12 };
   int64_t reshape_91_0_dim_3 { 64 };
   int64_t flash_attention_92_0_dim_0 { 64 };
   int64_t flash_attention_92_0_dim_1 { 12 };
   int64_t flash_attention_92_0_dim_2 { 64 };
   int64_t reshape_95_0_dim_0 { 1 };
   int64_t reshape_95_0_dim_1 { 64 };
   int64_t reshape_95_0_dim_2 { 768 };
   int64_t reshape_93_0_dim_0 { 64 };
   int64_t reshape_93_0_dim_1 { 768 };
   int64_t reshape_101_0_dim_0 { 64 };
   int64_t reshape_101_0_dim_1 { 3 };
   int64_t reshape_101_0_dim_2 { 12 };
   int64_t reshape_101_0_dim_3 { 64 };
   int64_t flash_attention_102_0_dim_0 { 64 };
   int64_t flash_attention_102_0_dim_1 { 12 };
   int64_t flash_attention_102_0_dim_2 { 64 };
   int64_t reshape_105_0_dim_0 { 1 };
   int64_t reshape_105_0_dim_1 { 64 };
   int64_t reshape_105_0_dim_2 { 768 };
   int64_t reshape_103_0_dim_0 { 64 };
   int64_t reshape_103_0_dim_1 { 768 };
   int64_t reshape_111_0_dim_0 { 64 };
   int64_t reshape_111_0_dim_1 { 3 };
   int64_t reshape_111_0_dim_2 { 12 };
   int64_t reshape_111_0_dim_3 { 64 };
   int64_t flash_attention_112_0_dim_0 { 64 };
   int64_t flash_attention_112_0_dim_1 { 12 };
   int64_t flash_attention_112_0_dim_2 { 64 };
   int64_t reshape_115_0_dim_0 { 1 };
   int64_t reshape_115_0_dim_1 { 64 };
   int64_t reshape_115_0_dim_2 { 768 };
   int64_t reshape_113_0_dim_0 { 64 };
   int64_t reshape_113_0_dim_1 { 768 };


};
} // namespace ait