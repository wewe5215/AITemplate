
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


void concatenate_39_constant_folding(
    void * /*output*/,
    int64_t *[] /*output_shape*/,
    const void *[] /*inputs*/,
    const int64_t *[], /* real_input_shapes, representing shapes of those inputs
                                 whose masks are False, i.e. inputs that will be
                                 copied to the output tensor by concat.*/
    const int64_t *[], /* all_input_shapes, including both kinds of inputs,
                                 i.e. not matter input_mask being True or False */
    const bool [] /*input_masks*/,
    const int64_t [] /*concat_dim_sizes*/,
    int64_t /*concat_dim*/,
    int64_t /*rank*/,
    int64_t /*num_real_inputs*/,
    int64_t /*num_all_inputs*/,
    cudaStream_t
);

void permute021_41_constant_folding(
  const void* /* input */,
  void* /* output */,
  int64_t /* rank */,
  const int64_t* /* x_dims */,
  cudaStream_t /* stream */
);

void concatenate_42_constant_folding(
    void * /*output*/,
    int64_t *[] /*output_shape*/,
    const void *[] /*inputs*/,
    const int64_t *[], /* real_input_shapes, representing shapes of those inputs
                                 whose masks are False, i.e. inputs that will be
                                 copied to the output tensor by concat.*/
    const int64_t *[], /* all_input_shapes, including both kinds of inputs,
                                 i.e. not matter input_mask being True or False */
    const bool [] /*input_masks*/,
    const int64_t [] /*concat_dim_sizes*/,
    int64_t /*concat_dim*/,
    int64_t /*rank*/,
    int64_t /*num_real_inputs*/,
    int64_t /*num_all_inputs*/,
    cudaStream_t
);

void permute021_44_constant_folding(
  const void* /* input */,
  void* /* output */,
  int64_t /* rank */,
  const int64_t* /* x_dims */,
  cudaStream_t /* stream */
);

void concatenate_45_constant_folding(
    void * /*output*/,
    int64_t *[] /*output_shape*/,
    const void *[] /*inputs*/,
    const int64_t *[], /* real_input_shapes, representing shapes of those inputs
                                 whose masks are False, i.e. inputs that will be
                                 copied to the output tensor by concat.*/
    const int64_t *[], /* all_input_shapes, including both kinds of inputs,
                                 i.e. not matter input_mask being True or False */
    const bool [] /*input_masks*/,
    const int64_t [] /*concat_dim_sizes*/,
    int64_t /*concat_dim*/,
    int64_t /*rank*/,
    int64_t /*num_real_inputs*/,
    int64_t /*num_all_inputs*/,
    cudaStream_t
);

void permute021_47_constant_folding(
  const void* /* input */,
  void* /* output */,
  int64_t /* rank */,
  const int64_t* /* x_dims */,
  cudaStream_t /* stream */
);

void concatenate_51_constant_folding(
    void * /*output*/,
    int64_t *[] /*output_shape*/,
    const void *[] /*inputs*/,
    const int64_t *[], /* real_input_shapes, representing shapes of those inputs
                                 whose masks are False, i.e. inputs that will be
                                 copied to the output tensor by concat.*/
    const int64_t *[], /* all_input_shapes, including both kinds of inputs,
                                 i.e. not matter input_mask being True or False */
    const bool [] /*input_masks*/,
    const int64_t [] /*concat_dim_sizes*/,
    int64_t /*concat_dim*/,
    int64_t /*rank*/,
    int64_t /*num_real_inputs*/,
    int64_t /*num_all_inputs*/,
    cudaStream_t
);

void concatenate_55_constant_folding(
    void * /*output*/,
    int64_t *[] /*output_shape*/,
    const void *[] /*inputs*/,
    const int64_t *[], /* real_input_shapes, representing shapes of those inputs
                                 whose masks are False, i.e. inputs that will be
                                 copied to the output tensor by concat.*/
    const int64_t *[], /* all_input_shapes, including both kinds of inputs,
                                 i.e. not matter input_mask being True or False */
    const bool [] /*input_masks*/,
    const int64_t [] /*concat_dim_sizes*/,
    int64_t /*concat_dim*/,
    int64_t /*rank*/,
    int64_t /*num_real_inputs*/,
    int64_t /*num_all_inputs*/,
    cudaStream_t
);

namespace ait {

// Model is the class that actually performs inference. It owns memory for
// intermediate tensors and dynamic dimensions. Constants are owned by
// the model's owning container object, and input/output memory is owned
// by the user.
// Once an inference run has started, it is not safe to re-use the Model
// until the run has finished!
class ConstantFolder : public ModelBase<ConstantFolder> {
  
  

  public:
    ConstantFolder(
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
        w_0 = reinterpret_cast<decltype(w_0)>(constants + 0);
     constant_name_to_ptr_["w_0"] = const_cast<const void**>(reinterpret_cast<void**>(&w_0));
    b_0 = reinterpret_cast<decltype(b_0)>(constants + 32768);
     constant_name_to_ptr_["b_0"] = const_cast<const void**>(reinterpret_cast<void**>(&b_0));
    w_1 = reinterpret_cast<decltype(w_1)>(constants + 33280);
     constant_name_to_ptr_["w_1"] = const_cast<const void**>(reinterpret_cast<void**>(&w_1));
    b_1 = reinterpret_cast<decltype(b_1)>(constants + 66048);
     constant_name_to_ptr_["b_1"] = const_cast<const void**>(reinterpret_cast<void**>(&b_1));
    w_2 = reinterpret_cast<decltype(w_2)>(constants + 66560);
     constant_name_to_ptr_["w_2"] = const_cast<const void**>(reinterpret_cast<void**>(&w_2));
    b_2 = reinterpret_cast<decltype(b_2)>(constants + 128000);
     constant_name_to_ptr_["b_2"] = const_cast<const void**>(reinterpret_cast<void**>(&b_2));
    w_3 = reinterpret_cast<decltype(w_3)>(constants + 128512);
     constant_name_to_ptr_["w_3"] = const_cast<const void**>(reinterpret_cast<void**>(&w_3));
    b_3 = reinterpret_cast<decltype(b_3)>(constants + 189952);
     constant_name_to_ptr_["b_3"] = const_cast<const void**>(reinterpret_cast<void**>(&b_3));
    w_4 = reinterpret_cast<decltype(w_4)>(constants + 190464);
     constant_name_to_ptr_["w_4"] = const_cast<const void**>(reinterpret_cast<void**>(&w_4));
    b_4 = reinterpret_cast<decltype(b_4)>(constants + 251904);
     constant_name_to_ptr_["b_4"] = const_cast<const void**>(reinterpret_cast<void**>(&b_4));
    w_5 = reinterpret_cast<decltype(w_5)>(constants + 252416);
     constant_name_to_ptr_["w_5"] = const_cast<const void**>(reinterpret_cast<void**>(&w_5));
    b_5 = reinterpret_cast<decltype(b_5)>(constants + 313856);
     constant_name_to_ptr_["b_5"] = const_cast<const void**>(reinterpret_cast<void**>(&b_5));
    w_6 = reinterpret_cast<decltype(w_6)>(constants + 314368);
     constant_name_to_ptr_["w_6"] = const_cast<const void**>(reinterpret_cast<void**>(&w_6));
    b_6 = reinterpret_cast<decltype(b_6)>(constants + 351232);
     constant_name_to_ptr_["b_6"] = const_cast<const void**>(reinterpret_cast<void**>(&b_6));
    w_7 = reinterpret_cast<decltype(w_7)>(constants + 351744);
     constant_name_to_ptr_["w_7"] = const_cast<const void**>(reinterpret_cast<void**>(&w_7));
    b_7 = reinterpret_cast<decltype(b_7)>(constants + 388608);
     constant_name_to_ptr_["b_7"] = const_cast<const void**>(reinterpret_cast<void**>(&b_7));
    w_8 = reinterpret_cast<decltype(w_8)>(constants + 389120);
     constant_name_to_ptr_["w_8"] = const_cast<const void**>(reinterpret_cast<void**>(&w_8));
    b_8 = reinterpret_cast<decltype(b_8)>(constants + 421888);
     constant_name_to_ptr_["b_8"] = const_cast<const void**>(reinterpret_cast<void**>(&b_8));
    w_9 = reinterpret_cast<decltype(w_9)>(constants + 422400);
     constant_name_to_ptr_["w_9"] = const_cast<const void**>(reinterpret_cast<void**>(&w_9));
    b_9 = reinterpret_cast<decltype(b_9)>(constants + 455168);
     constant_name_to_ptr_["b_9"] = const_cast<const void**>(reinterpret_cast<void**>(&b_9));
    auto* blob_ptr = static_cast<uint8_t*>(blob_.get());
        concatenate_39_0 = reinterpret_cast<decltype(concatenate_39_0)>(blob_ptr + 0);
    concatenate_42_0 = reinterpret_cast<decltype(concatenate_42_0)>(blob_ptr + 0);
    concatenate_45_0 = reinterpret_cast<decltype(concatenate_45_0)>(blob_ptr + 0);
    concatenate_48_0 = reinterpret_cast<decltype(concatenate_48_0)>(blob_ptr + 0);
    
         params_[0].shape_ptrs = {ParamDim(4, 4, &reshape_40_0_dim_0), ParamDim(120, 120, &reshape_40_0_dim_2), ParamDim(128, 128, &reshape_40_0_dim_1)};
     params_[1].shape_ptrs = {ParamDim(2, 2, &reshape_43_0_dim_0), ParamDim(72, 72, &reshape_43_0_dim_2), ParamDim(128, 128, &reshape_43_0_dim_1)};
     params_[2].shape_ptrs = {ParamDim(2, 2, &reshape_46_0_dim_0), ParamDim(64, 64, &reshape_46_0_dim_2), ParamDim(128, 128, &reshape_46_0_dim_1)};
     params_[3].shape_ptrs = {ParamDim(2, 2, &reshape_49_0_dim_0), ParamDim(64, 64, &reshape_49_0_dim_2), ParamDim(128, 128, &reshape_49_0_dim_1)};
     params_[4].shape_ptrs = {ParamDim(256, 256, &concatenate_51_0_dim_0)};
     params_[5].shape_ptrs = {ParamDim(512, 512, &concatenate_55_0_dim_0)};
     params_[6].shape_ptrs = {ParamDim(256, 256, &concatenate_59_0_dim_0)};
     params_[7].shape_ptrs = {ParamDim(256, 256, &concatenate_63_0_dim_0)};

      
      
    }

    ~ConstantFolder() {
      
      
    }

    void SetUpInputsOutputs() {
             permute021_41_0 = static_cast<decltype(permute021_41_0)>(params_[0].ptr);

if (permute021_41_0 == nullptr) {
    throw std::runtime_error("Constant permute021_41_0 was not set! Set the value with set_constant.");
}
    
     permute021_44_0 = static_cast<decltype(permute021_44_0)>(params_[1].ptr);

if (permute021_44_0 == nullptr) {
    throw std::runtime_error("Constant permute021_44_0 was not set! Set the value with set_constant.");
}
    
     permute021_47_0 = static_cast<decltype(permute021_47_0)>(params_[2].ptr);

if (permute021_47_0 == nullptr) {
    throw std::runtime_error("Constant permute021_47_0 was not set! Set the value with set_constant.");
}
    
     permute021_50_0 = static_cast<decltype(permute021_50_0)>(params_[3].ptr);

if (permute021_50_0 == nullptr) {
    throw std::runtime_error("Constant permute021_50_0 was not set! Set the value with set_constant.");
}
    
     concatenate_51_0 = static_cast<decltype(concatenate_51_0)>(params_[4].ptr);

if (concatenate_51_0 == nullptr) {
    throw std::runtime_error("Constant concatenate_51_0 was not set! Set the value with set_constant.");
}
    
     concatenate_55_0 = static_cast<decltype(concatenate_55_0)>(params_[5].ptr);

if (concatenate_55_0 == nullptr) {
    throw std::runtime_error("Constant concatenate_55_0 was not set! Set the value with set_constant.");
}
    
     concatenate_59_0 = static_cast<decltype(concatenate_59_0)>(params_[6].ptr);

if (concatenate_59_0 == nullptr) {
    throw std::runtime_error("Constant concatenate_59_0 was not set! Set the value with set_constant.");
}
    
     concatenate_63_0 = static_cast<decltype(concatenate_63_0)>(params_[7].ptr);

if (concatenate_63_0 == nullptr) {
    throw std::runtime_error("Constant concatenate_63_0 was not set! Set the value with set_constant.");
}
    
    }

    void ResetConstants(uint8_t* constants) {
        /*
         * This can be called if we want to use a different piece of memory
         * for the constants to be consumed.
         */
            w_0 = reinterpret_cast<decltype(w_0)>(constants + 0);
    b_0 = reinterpret_cast<decltype(b_0)>(constants + 32768);
    w_1 = reinterpret_cast<decltype(w_1)>(constants + 33280);
    b_1 = reinterpret_cast<decltype(b_1)>(constants + 66048);
    w_2 = reinterpret_cast<decltype(w_2)>(constants + 66560);
    b_2 = reinterpret_cast<decltype(b_2)>(constants + 128000);
    w_3 = reinterpret_cast<decltype(w_3)>(constants + 128512);
    b_3 = reinterpret_cast<decltype(b_3)>(constants + 189952);
    w_4 = reinterpret_cast<decltype(w_4)>(constants + 190464);
    b_4 = reinterpret_cast<decltype(b_4)>(constants + 251904);
    w_5 = reinterpret_cast<decltype(w_5)>(constants + 252416);
    b_5 = reinterpret_cast<decltype(b_5)>(constants + 313856);
    w_6 = reinterpret_cast<decltype(w_6)>(constants + 314368);
    b_6 = reinterpret_cast<decltype(b_6)>(constants + 351232);
    w_7 = reinterpret_cast<decltype(w_7)>(constants + 351744);
    b_7 = reinterpret_cast<decltype(b_7)>(constants + 388608);
    w_8 = reinterpret_cast<decltype(w_8)>(constants + 389120);
    b_8 = reinterpret_cast<decltype(b_8)>(constants + 421888);
    w_9 = reinterpret_cast<decltype(w_9)>(constants + 422400);
    b_9 = reinterpret_cast<decltype(b_9)>(constants + 455168);
    }

    void DeviceToDeviceCopies(StreamType stream) {
  
    }


    ///////////////////////////////////////////////////////////////////////////
    // default RunImpl implemenation
    void RunImpl(StreamType stream) {
        
  
  
  {

    const void *inputs[] = {
      w_2,
      w_3,
      w_4,
      w_5
    };


      int64_t w_2_shape_0[] = {
        w_2_dim_0, w_2_dim_1
      };
      int64_t w_3_shape_1[] = {
        w_3_dim_0, w_3_dim_1
      };
      int64_t w_4_shape_2[] = {
        w_4_dim_0, w_4_dim_1
      };
      int64_t w_5_shape_3[] = {
        w_5_dim_0, w_5_dim_1
      };

    const int64_t *real_input_shapes[] = {
      w_2_shape_0, w_3_shape_1, w_4_shape_2, w_5_shape_3
    };



    const int64_t *all_input_shapes[] = {
      w_2_shape_0, w_3_shape_1, w_4_shape_2, w_5_shape_3
    };

    int64_t *concatenate_39_0_shape[] = {
      &concatenate_39_0_dim_0, &w_2_dim_1
    };

    int64_t concat_dim_sizes[] = {
      w_2_dim_0, w_3_dim_0, w_4_dim_0, w_5_dim_0
    };

    bool input_masks[] = {
      true, true, true, true
    };

    concatenate_39_constant_folding(
        concatenate_39_0,
        concatenate_39_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        2/*rank*/,
        4/*num_real_inputs*/,
        4/*num_all_inputs*/,
        stream
    );
  }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
      const int64_t x_dims[] = {reshape_40_0_dim_0, reshape_40_0_dim_1, reshape_40_0_dim_2};
      permute021_41_constant_folding(
          concatenate_39_0,
          permute021_41_0,
          3,
          x_dims,
          stream
      );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    const void *inputs[] = {
      w_6,
      w_7
    };


      int64_t w_6_shape_0[] = {
        w_6_dim_0, w_6_dim_1
      };
      int64_t w_7_shape_1[] = {
        w_7_dim_0, w_7_dim_1
      };

    const int64_t *real_input_shapes[] = {
      w_6_shape_0, w_7_shape_1
    };



    const int64_t *all_input_shapes[] = {
      w_6_shape_0, w_7_shape_1
    };

    int64_t *concatenate_42_0_shape[] = {
      &concatenate_42_0_dim_0, &w_6_dim_1
    };

    int64_t concat_dim_sizes[] = {
      w_6_dim_0, w_7_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_42_constant_folding(
        concatenate_42_0,
        concatenate_42_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        2/*rank*/,
        2/*num_real_inputs*/,
        2/*num_all_inputs*/,
        stream
    );
  }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
      const int64_t x_dims[] = {reshape_43_0_dim_0, reshape_43_0_dim_1, reshape_43_0_dim_2};
      permute021_44_constant_folding(
          concatenate_42_0,
          permute021_44_0,
          3,
          x_dims,
          stream
      );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    const void *inputs[] = {
      w_0,
      w_1
    };


      int64_t w_0_shape_0[] = {
        w_0_dim_0, w_0_dim_1
      };
      int64_t w_1_shape_1[] = {
        w_1_dim_0, w_1_dim_1
      };

    const int64_t *real_input_shapes[] = {
      w_0_shape_0, w_1_shape_1
    };



    const int64_t *all_input_shapes[] = {
      w_0_shape_0, w_1_shape_1
    };

    int64_t *concatenate_45_0_shape[] = {
      &concatenate_45_0_dim_0, &w_0_dim_1
    };

    int64_t concat_dim_sizes[] = {
      w_0_dim_0, w_1_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_45_constant_folding(
        concatenate_45_0,
        concatenate_45_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        2/*rank*/,
        2/*num_real_inputs*/,
        2/*num_all_inputs*/,
        stream
    );
  }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
      const int64_t x_dims[] = {reshape_46_0_dim_0, reshape_46_0_dim_1, reshape_46_0_dim_2};
      permute021_47_constant_folding(
          concatenate_45_0,
          permute021_47_0,
          3,
          x_dims,
          stream
      );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    const void *inputs[] = {
      w_8,
      w_9
    };


      int64_t w_8_shape_0[] = {
        w_8_dim_0, w_8_dim_1
      };
      int64_t w_9_shape_1[] = {
        w_9_dim_0, w_9_dim_1
      };

    const int64_t *real_input_shapes[] = {
      w_8_shape_0, w_9_shape_1
    };



    const int64_t *all_input_shapes[] = {
      w_8_shape_0, w_9_shape_1
    };

    int64_t *concatenate_48_0_shape[] = {
      &concatenate_48_0_dim_0, &w_8_dim_1
    };

    int64_t concat_dim_sizes[] = {
      w_8_dim_0, w_9_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_45_constant_folding(
        concatenate_48_0,
        concatenate_48_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        2/*rank*/,
        2/*num_real_inputs*/,
        2/*num_all_inputs*/,
        stream
    );
  }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
      const int64_t x_dims[] = {reshape_49_0_dim_0, reshape_49_0_dim_1, reshape_49_0_dim_2};
      permute021_47_constant_folding(
          concatenate_48_0,
          permute021_50_0,
          3,
          x_dims,
          stream
      );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    const void *inputs[] = {
      b_0,
      b_1
    };


      int64_t b_0_shape_0[] = {
        b_0_dim_0
      };
      int64_t b_1_shape_1[] = {
        b_1_dim_0
      };

    const int64_t *real_input_shapes[] = {
      b_0_shape_0, b_1_shape_1
    };



    const int64_t *all_input_shapes[] = {
      b_0_shape_0, b_1_shape_1
    };

    int64_t *concatenate_51_0_shape[] = {
      &concatenate_51_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      b_0_dim_0, b_1_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_51_constant_folding(
        concatenate_51_0,
        concatenate_51_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        1/*rank*/,
        2/*num_real_inputs*/,
        2/*num_all_inputs*/,
        stream
    );
  }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    const void *inputs[] = {
      b_2,
      b_3,
      b_4,
      b_5
    };


      int64_t b_2_shape_0[] = {
        b_2_dim_0
      };
      int64_t b_3_shape_1[] = {
        b_3_dim_0
      };
      int64_t b_4_shape_2[] = {
        b_4_dim_0
      };
      int64_t b_5_shape_3[] = {
        b_5_dim_0
      };

    const int64_t *real_input_shapes[] = {
      b_2_shape_0, b_3_shape_1, b_4_shape_2, b_5_shape_3
    };



    const int64_t *all_input_shapes[] = {
      b_2_shape_0, b_3_shape_1, b_4_shape_2, b_5_shape_3
    };

    int64_t *concatenate_55_0_shape[] = {
      &concatenate_55_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      b_2_dim_0, b_3_dim_0, b_4_dim_0, b_5_dim_0
    };

    bool input_masks[] = {
      true, true, true, true
    };

    concatenate_55_constant_folding(
        concatenate_55_0,
        concatenate_55_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        1/*rank*/,
        4/*num_real_inputs*/,
        4/*num_all_inputs*/,
        stream
    );
  }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    const void *inputs[] = {
      b_6,
      b_7
    };


      int64_t b_6_shape_0[] = {
        b_6_dim_0
      };
      int64_t b_7_shape_1[] = {
        b_7_dim_0
      };

    const int64_t *real_input_shapes[] = {
      b_6_shape_0, b_7_shape_1
    };



    const int64_t *all_input_shapes[] = {
      b_6_shape_0, b_7_shape_1
    };

    int64_t *concatenate_59_0_shape[] = {
      &concatenate_59_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      b_6_dim_0, b_7_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_51_constant_folding(
        concatenate_59_0,
        concatenate_59_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        1/*rank*/,
        2/*num_real_inputs*/,
        2/*num_all_inputs*/,
        stream
    );
  }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    const void *inputs[] = {
      b_8,
      b_9
    };


      int64_t b_8_shape_0[] = {
        b_8_dim_0
      };
      int64_t b_9_shape_1[] = {
        b_9_dim_0
      };

    const int64_t *real_input_shapes[] = {
      b_8_shape_0, b_9_shape_1
    };



    const int64_t *all_input_shapes[] = {
      b_8_shape_0, b_9_shape_1
    };

    int64_t *concatenate_63_0_shape[] = {
      &concatenate_63_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      b_8_dim_0, b_9_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_51_constant_folding(
        concatenate_63_0,
        concatenate_63_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        1/*rank*/,
        2/*num_real_inputs*/,
        2/*num_all_inputs*/,
        stream
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
        std::cout << "Profiling: " << "concatenate_39" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
  {

    const void *inputs[] = {
      w_2,
      w_3,
      w_4,
      w_5
    };


      int64_t w_2_shape_0[] = {
        w_2_dim_0, w_2_dim_1
      };
      int64_t w_3_shape_1[] = {
        w_3_dim_0, w_3_dim_1
      };
      int64_t w_4_shape_2[] = {
        w_4_dim_0, w_4_dim_1
      };
      int64_t w_5_shape_3[] = {
        w_5_dim_0, w_5_dim_1
      };

    const int64_t *real_input_shapes[] = {
      w_2_shape_0, w_3_shape_1, w_4_shape_2, w_5_shape_3
    };



    const int64_t *all_input_shapes[] = {
      w_2_shape_0, w_3_shape_1, w_4_shape_2, w_5_shape_3
    };

    int64_t *concatenate_39_0_shape[] = {
      &concatenate_39_0_dim_0, &w_2_dim_1
    };

    int64_t concat_dim_sizes[] = {
      w_2_dim_0, w_3_dim_0, w_4_dim_0, w_5_dim_0
    };

    bool input_masks[] = {
      true, true, true, true
    };

    concatenate_39_constant_folding(
        concatenate_39_0,
        concatenate_39_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        2/*rank*/,
        4/*num_real_inputs*/,
        4/*num_all_inputs*/,
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
        ss << "\"" << "concatenate_39" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"128\", \"120\"], [\"128\", \"120\"], [\"128\", \"120\"], [\"128\", \"120\"]]"
           << ", \"output_sizes\": " << "[[\"512\", \"120\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "permute021_41" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
      const int64_t x_dims[] = {reshape_40_0_dim_0, reshape_40_0_dim_1, reshape_40_0_dim_2};
      permute021_41_constant_folding(
          concatenate_39_0,
          permute021_41_0,
          3,
          x_dims,
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
        ss << "\"" << "permute021_41" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"4\", \"128\", \"120\"]]"
           << ", \"output_sizes\": " << "[[\"4\", \"120\", \"128\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_42" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
  {

    const void *inputs[] = {
      w_6,
      w_7
    };


      int64_t w_6_shape_0[] = {
        w_6_dim_0, w_6_dim_1
      };
      int64_t w_7_shape_1[] = {
        w_7_dim_0, w_7_dim_1
      };

    const int64_t *real_input_shapes[] = {
      w_6_shape_0, w_7_shape_1
    };



    const int64_t *all_input_shapes[] = {
      w_6_shape_0, w_7_shape_1
    };

    int64_t *concatenate_42_0_shape[] = {
      &concatenate_42_0_dim_0, &w_6_dim_1
    };

    int64_t concat_dim_sizes[] = {
      w_6_dim_0, w_7_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_42_constant_folding(
        concatenate_42_0,
        concatenate_42_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        2/*rank*/,
        2/*num_real_inputs*/,
        2/*num_all_inputs*/,
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
        ss << "\"" << "concatenate_42" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"128\", \"72\"], [\"128\", \"72\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"72\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "permute021_44" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
      const int64_t x_dims[] = {reshape_43_0_dim_0, reshape_43_0_dim_1, reshape_43_0_dim_2};
      permute021_44_constant_folding(
          concatenate_42_0,
          permute021_44_0,
          3,
          x_dims,
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
        ss << "\"" << "permute021_44" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"2\", \"128\", \"72\"]]"
           << ", \"output_sizes\": " << "[[\"2\", \"72\", \"128\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_45" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
  {

    const void *inputs[] = {
      w_0,
      w_1
    };


      int64_t w_0_shape_0[] = {
        w_0_dim_0, w_0_dim_1
      };
      int64_t w_1_shape_1[] = {
        w_1_dim_0, w_1_dim_1
      };

    const int64_t *real_input_shapes[] = {
      w_0_shape_0, w_1_shape_1
    };



    const int64_t *all_input_shapes[] = {
      w_0_shape_0, w_1_shape_1
    };

    int64_t *concatenate_45_0_shape[] = {
      &concatenate_45_0_dim_0, &w_0_dim_1
    };

    int64_t concat_dim_sizes[] = {
      w_0_dim_0, w_1_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_45_constant_folding(
        concatenate_45_0,
        concatenate_45_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        2/*rank*/,
        2/*num_real_inputs*/,
        2/*num_all_inputs*/,
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
        ss << "\"" << "concatenate_45" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"128\", \"64\"], [\"128\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"64\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "permute021_47" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
      const int64_t x_dims[] = {reshape_46_0_dim_0, reshape_46_0_dim_1, reshape_46_0_dim_2};
      permute021_47_constant_folding(
          concatenate_45_0,
          permute021_47_0,
          3,
          x_dims,
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
        ss << "\"" << "permute021_47" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"2\", \"128\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"2\", \"64\", \"128\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_48" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
  {

    const void *inputs[] = {
      w_8,
      w_9
    };


      int64_t w_8_shape_0[] = {
        w_8_dim_0, w_8_dim_1
      };
      int64_t w_9_shape_1[] = {
        w_9_dim_0, w_9_dim_1
      };

    const int64_t *real_input_shapes[] = {
      w_8_shape_0, w_9_shape_1
    };



    const int64_t *all_input_shapes[] = {
      w_8_shape_0, w_9_shape_1
    };

    int64_t *concatenate_48_0_shape[] = {
      &concatenate_48_0_dim_0, &w_8_dim_1
    };

    int64_t concat_dim_sizes[] = {
      w_8_dim_0, w_9_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_45_constant_folding(
        concatenate_48_0,
        concatenate_48_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        2/*rank*/,
        2/*num_real_inputs*/,
        2/*num_all_inputs*/,
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
        ss << "\"" << "concatenate_48" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"128\", \"64\"], [\"128\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"64\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "permute021_50" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
      const int64_t x_dims[] = {reshape_49_0_dim_0, reshape_49_0_dim_1, reshape_49_0_dim_2};
      permute021_47_constant_folding(
          concatenate_48_0,
          permute021_50_0,
          3,
          x_dims,
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
        ss << "\"" << "permute021_50" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"2\", \"128\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"2\", \"64\", \"128\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_51" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
  {

    const void *inputs[] = {
      b_0,
      b_1
    };


      int64_t b_0_shape_0[] = {
        b_0_dim_0
      };
      int64_t b_1_shape_1[] = {
        b_1_dim_0
      };

    const int64_t *real_input_shapes[] = {
      b_0_shape_0, b_1_shape_1
    };



    const int64_t *all_input_shapes[] = {
      b_0_shape_0, b_1_shape_1
    };

    int64_t *concatenate_51_0_shape[] = {
      &concatenate_51_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      b_0_dim_0, b_1_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_51_constant_folding(
        concatenate_51_0,
        concatenate_51_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        1/*rank*/,
        2/*num_real_inputs*/,
        2/*num_all_inputs*/,
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
        ss << "\"" << "concatenate_51" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"128\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"256\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_55" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
  {

    const void *inputs[] = {
      b_2,
      b_3,
      b_4,
      b_5
    };


      int64_t b_2_shape_0[] = {
        b_2_dim_0
      };
      int64_t b_3_shape_1[] = {
        b_3_dim_0
      };
      int64_t b_4_shape_2[] = {
        b_4_dim_0
      };
      int64_t b_5_shape_3[] = {
        b_5_dim_0
      };

    const int64_t *real_input_shapes[] = {
      b_2_shape_0, b_3_shape_1, b_4_shape_2, b_5_shape_3
    };



    const int64_t *all_input_shapes[] = {
      b_2_shape_0, b_3_shape_1, b_4_shape_2, b_5_shape_3
    };

    int64_t *concatenate_55_0_shape[] = {
      &concatenate_55_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      b_2_dim_0, b_3_dim_0, b_4_dim_0, b_5_dim_0
    };

    bool input_masks[] = {
      true, true, true, true
    };

    concatenate_55_constant_folding(
        concatenate_55_0,
        concatenate_55_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        1/*rank*/,
        4/*num_real_inputs*/,
        4/*num_all_inputs*/,
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
        ss << "\"" << "concatenate_55" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"128\"], [\"128\"], [\"128\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"512\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_59" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
  {

    const void *inputs[] = {
      b_6,
      b_7
    };


      int64_t b_6_shape_0[] = {
        b_6_dim_0
      };
      int64_t b_7_shape_1[] = {
        b_7_dim_0
      };

    const int64_t *real_input_shapes[] = {
      b_6_shape_0, b_7_shape_1
    };



    const int64_t *all_input_shapes[] = {
      b_6_shape_0, b_7_shape_1
    };

    int64_t *concatenate_59_0_shape[] = {
      &concatenate_59_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      b_6_dim_0, b_7_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_51_constant_folding(
        concatenate_59_0,
        concatenate_59_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        1/*rank*/,
        2/*num_real_inputs*/,
        2/*num_all_inputs*/,
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
        ss << "\"" << "concatenate_59" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"128\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"256\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_63" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
  {

    const void *inputs[] = {
      b_8,
      b_9
    };


      int64_t b_8_shape_0[] = {
        b_8_dim_0
      };
      int64_t b_9_shape_1[] = {
        b_9_dim_0
      };

    const int64_t *real_input_shapes[] = {
      b_8_shape_0, b_9_shape_1
    };



    const int64_t *all_input_shapes[] = {
      b_8_shape_0, b_9_shape_1
    };

    int64_t *concatenate_63_0_shape[] = {
      &concatenate_63_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      b_8_dim_0, b_9_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_51_constant_folding(
        concatenate_63_0,
        concatenate_63_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        1/*rank*/,
        2/*num_real_inputs*/,
        2/*num_all_inputs*/,
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
        ss << "\"" << "concatenate_63" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"128\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"256\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << "\n";
        
      }
      
      ss << "}\n";

      DeviceToDeviceCopies(stream);
      std::cout << "AIT per op profiling finished." << std::endl;
      FreeDeviceMemory(L2CacheSlab);
#endif
    }

    static std::unique_ptr<ConstantFolder> Create(
      AITemplateAllocator& allocator,
      uint8_t* constants
    ) {
      return std::make_unique<ConstantFolder>(
          557056,
          0 * (1 + 0),
          0 * (1 + 0),
          0,
          8,
          0,
          constants,
          allocator
      );
    }

  private:
   void* w_0 {nullptr};
   void* b_0 {nullptr};
   void* w_1 {nullptr};
   void* b_1 {nullptr};
   void* w_2 {nullptr};
   void* b_2 {nullptr};
   void* w_3 {nullptr};
   void* b_3 {nullptr};
   void* w_4 {nullptr};
   void* b_4 {nullptr};
   void* w_5 {nullptr};
   void* b_5 {nullptr};
   void* w_6 {nullptr};
   void* b_6 {nullptr};
   void* w_7 {nullptr};
   void* b_7 {nullptr};
   void* w_8 {nullptr};
   void* b_8 {nullptr};
   void* w_9 {nullptr};
   void* b_9 {nullptr};
   void* concatenate_39_0 {nullptr};
   void* permute021_41_0 {nullptr};
   void* concatenate_42_0 {nullptr};
   void* permute021_44_0 {nullptr};
   void* concatenate_45_0 {nullptr};
   void* permute021_47_0 {nullptr};
   void* concatenate_48_0 {nullptr};
   void* permute021_50_0 {nullptr};
   void* concatenate_51_0 {nullptr};
   void* concatenate_55_0 {nullptr};
   void* concatenate_59_0 {nullptr};
   void* concatenate_63_0 {nullptr};
   int64_t w_0_dim_0 { 128 };
   int64_t w_0_dim_1 { 64 };
   int64_t b_0_dim_0 { 128 };
   int64_t w_1_dim_0 { 128 };
   int64_t w_1_dim_1 { 64 };
   int64_t b_1_dim_0 { 128 };
   int64_t w_2_dim_0 { 128 };
   int64_t w_2_dim_1 { 120 };
   int64_t b_2_dim_0 { 128 };
   int64_t w_3_dim_0 { 128 };
   int64_t w_3_dim_1 { 120 };
   int64_t b_3_dim_0 { 128 };
   int64_t w_4_dim_0 { 128 };
   int64_t w_4_dim_1 { 120 };
   int64_t b_4_dim_0 { 128 };
   int64_t w_5_dim_0 { 128 };
   int64_t w_5_dim_1 { 120 };
   int64_t b_5_dim_0 { 128 };
   int64_t w_6_dim_0 { 128 };
   int64_t w_6_dim_1 { 72 };
   int64_t b_6_dim_0 { 128 };
   int64_t w_7_dim_0 { 128 };
   int64_t w_7_dim_1 { 72 };
   int64_t b_7_dim_0 { 128 };
   int64_t w_8_dim_0 { 128 };
   int64_t w_8_dim_1 { 64 };
   int64_t b_8_dim_0 { 128 };
   int64_t w_9_dim_0 { 128 };
   int64_t w_9_dim_1 { 64 };
   int64_t b_9_dim_0 { 128 };
   int64_t concatenate_39_0_dim_0 { 512 };
   int64_t reshape_40_0_dim_0 { 4 };
   int64_t reshape_40_0_dim_2 { 120 };
   int64_t reshape_40_0_dim_1 { 128 };
   int64_t concatenate_42_0_dim_0 { 256 };
   int64_t reshape_43_0_dim_0 { 2 };
   int64_t reshape_43_0_dim_2 { 72 };
   int64_t reshape_43_0_dim_1 { 128 };
   int64_t concatenate_45_0_dim_0 { 256 };
   int64_t reshape_46_0_dim_0 { 2 };
   int64_t reshape_46_0_dim_2 { 64 };
   int64_t reshape_46_0_dim_1 { 128 };
   int64_t concatenate_48_0_dim_0 { 256 };
   int64_t reshape_49_0_dim_0 { 2 };
   int64_t reshape_49_0_dim_2 { 64 };
   int64_t reshape_49_0_dim_1 { 128 };
   int64_t concatenate_51_0_dim_0 { 256 };
   int64_t concatenate_55_0_dim_0 { 512 };
   int64_t concatenate_59_0_dim_0 { 256 };
   int64_t concatenate_63_0_dim_0 { 256 };


};
} // namespace ait