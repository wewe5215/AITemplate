
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


void permute021_5_constant_folding(
  const void* /* input */,
  void* /* output */,
  int64_t /* rank */,
  const int64_t* /* x_dims */,
  cudaStream_t /* stream */
);

void concatenate_25_constant_folding(
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

void permute021_27_constant_folding(
  const void* /* input */,
  void* /* output */,
  int64_t /* rank */,
  const int64_t* /* x_dims */,
  cudaStream_t /* stream */
);

void concatenate_31_constant_folding(
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
        W = reinterpret_cast<decltype(W)>(constants + 0);
     constant_name_to_ptr_["W"] = const_cast<const void**>(reinterpret_cast<void**>(&W));
    W0 = reinterpret_cast<decltype(W0)>(constants + 32768);
     constant_name_to_ptr_["W0"] = const_cast<const void**>(reinterpret_cast<void**>(&W0));
    B0 = reinterpret_cast<decltype(B0)>(constants + 40960);
     constant_name_to_ptr_["B0"] = const_cast<const void**>(reinterpret_cast<void**>(&B0));
    W1 = reinterpret_cast<decltype(W1)>(constants + 41088);
     constant_name_to_ptr_["W1"] = const_cast<const void**>(reinterpret_cast<void**>(&W1));
    B1 = reinterpret_cast<decltype(B1)>(constants + 49280);
     constant_name_to_ptr_["B1"] = const_cast<const void**>(reinterpret_cast<void**>(&B1));
    W2 = reinterpret_cast<decltype(W2)>(constants + 49408);
     constant_name_to_ptr_["W2"] = const_cast<const void**>(reinterpret_cast<void**>(&W2));
    B2 = reinterpret_cast<decltype(B2)>(constants + 57600);
     constant_name_to_ptr_["B2"] = const_cast<const void**>(reinterpret_cast<void**>(&B2));
    W3 = reinterpret_cast<decltype(W3)>(constants + 57728);
     constant_name_to_ptr_["W3"] = const_cast<const void**>(reinterpret_cast<void**>(&W3));
    B3 = reinterpret_cast<decltype(B3)>(constants + 65920);
     constant_name_to_ptr_["B3"] = const_cast<const void**>(reinterpret_cast<void**>(&B3));
    W4 = reinterpret_cast<decltype(W4)>(constants + 66048);
     constant_name_to_ptr_["W4"] = const_cast<const void**>(reinterpret_cast<void**>(&W4));
    B4 = reinterpret_cast<decltype(B4)>(constants + 74240);
     constant_name_to_ptr_["B4"] = const_cast<const void**>(reinterpret_cast<void**>(&B4));
    W5 = reinterpret_cast<decltype(W5)>(constants + 74368);
     constant_name_to_ptr_["W5"] = const_cast<const void**>(reinterpret_cast<void**>(&W5));
    B5 = reinterpret_cast<decltype(B5)>(constants + 82560);
     constant_name_to_ptr_["B5"] = const_cast<const void**>(reinterpret_cast<void**>(&B5));
    W6 = reinterpret_cast<decltype(W6)>(constants + 82688);
     constant_name_to_ptr_["W6"] = const_cast<const void**>(reinterpret_cast<void**>(&W6));
    B6 = reinterpret_cast<decltype(B6)>(constants + 90880);
     constant_name_to_ptr_["B6"] = const_cast<const void**>(reinterpret_cast<void**>(&B6));
    W7 = reinterpret_cast<decltype(W7)>(constants + 91008);
     constant_name_to_ptr_["W7"] = const_cast<const void**>(reinterpret_cast<void**>(&W7));
    B7 = reinterpret_cast<decltype(B7)>(constants + 99200);
     constant_name_to_ptr_["B7"] = const_cast<const void**>(reinterpret_cast<void**>(&B7));
    auto* blob_ptr = static_cast<uint8_t*>(blob_.get());
        concatenate_25_0 = reinterpret_cast<decltype(concatenate_25_0)>(blob_ptr + 32768);
    concatenate_28_0 = reinterpret_cast<decltype(concatenate_28_0)>(blob_ptr + 32768);
    
         params_[0].shape_ptrs = {ParamDim(4, 4, &W_dim_0), ParamDim(64, 64, &W_dim_2), ParamDim(32, 32, &W_dim_1)};
     params_[1].shape_ptrs = {ParamDim(4, 4, &reshape_26_0_dim_0), ParamDim(64, 64, &reshape_26_0_dim_2), ParamDim(32, 32, &reshape_26_0_dim_1)};
     params_[2].shape_ptrs = {ParamDim(4, 4, &reshape_29_0_dim_0), ParamDim(64, 64, &reshape_29_0_dim_2), ParamDim(32, 32, &reshape_29_0_dim_1)};
     params_[3].shape_ptrs = {ParamDim(128, 128, &concatenate_31_0_dim_0)};
     params_[4].shape_ptrs = {ParamDim(128, 128, &concatenate_35_0_dim_0)};

      
      
    }

    ~ConstantFolder() {
      
      
    }

    void SetUpInputsOutputs() {
             permute021_5_0 = static_cast<decltype(permute021_5_0)>(params_[0].ptr);

if (permute021_5_0 == nullptr) {
    throw std::runtime_error("Constant permute021_5_0 was not set! Set the value with set_constant.");
}
    
     permute021_27_0 = static_cast<decltype(permute021_27_0)>(params_[1].ptr);

if (permute021_27_0 == nullptr) {
    throw std::runtime_error("Constant permute021_27_0 was not set! Set the value with set_constant.");
}
    
     permute021_30_0 = static_cast<decltype(permute021_30_0)>(params_[2].ptr);

if (permute021_30_0 == nullptr) {
    throw std::runtime_error("Constant permute021_30_0 was not set! Set the value with set_constant.");
}
    
     concatenate_31_0 = static_cast<decltype(concatenate_31_0)>(params_[3].ptr);

if (concatenate_31_0 == nullptr) {
    throw std::runtime_error("Constant concatenate_31_0 was not set! Set the value with set_constant.");
}
    
     concatenate_35_0 = static_cast<decltype(concatenate_35_0)>(params_[4].ptr);

if (concatenate_35_0 == nullptr) {
    throw std::runtime_error("Constant concatenate_35_0 was not set! Set the value with set_constant.");
}
    
    }

    void ResetConstants(uint8_t* constants) {
        /*
         * This can be called if we want to use a different piece of memory
         * for the constants to be consumed.
         */
            W = reinterpret_cast<decltype(W)>(constants + 0);
    W0 = reinterpret_cast<decltype(W0)>(constants + 32768);
    B0 = reinterpret_cast<decltype(B0)>(constants + 40960);
    W1 = reinterpret_cast<decltype(W1)>(constants + 41088);
    B1 = reinterpret_cast<decltype(B1)>(constants + 49280);
    W2 = reinterpret_cast<decltype(W2)>(constants + 49408);
    B2 = reinterpret_cast<decltype(B2)>(constants + 57600);
    W3 = reinterpret_cast<decltype(W3)>(constants + 57728);
    B3 = reinterpret_cast<decltype(B3)>(constants + 65920);
    W4 = reinterpret_cast<decltype(W4)>(constants + 66048);
    B4 = reinterpret_cast<decltype(B4)>(constants + 74240);
    W5 = reinterpret_cast<decltype(W5)>(constants + 74368);
    B5 = reinterpret_cast<decltype(B5)>(constants + 82560);
    W6 = reinterpret_cast<decltype(W6)>(constants + 82688);
    B6 = reinterpret_cast<decltype(B6)>(constants + 90880);
    W7 = reinterpret_cast<decltype(W7)>(constants + 91008);
    B7 = reinterpret_cast<decltype(B7)>(constants + 99200);
    }

    void DeviceToDeviceCopies(StreamType stream) {
  
    }


    ///////////////////////////////////////////////////////////////////////////
    // default RunImpl implemenation
    void RunImpl(StreamType stream) {
        
  
  
    {
      const int64_t x_dims[] = {W_dim_0, W_dim_1, W_dim_2};
      permute021_5_constant_folding(
          W,
          permute021_5_0,
          3,
          x_dims,
          stream
      );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    const void *inputs[] = {
      W0,
      W1,
      W2,
      W3
    };


      int64_t W0_shape_0[] = {
        W0_dim_0, W0_dim_1
      };
      int64_t W1_shape_1[] = {
        W1_dim_0, W1_dim_1
      };
      int64_t W2_shape_2[] = {
        W2_dim_0, W2_dim_1
      };
      int64_t W3_shape_3[] = {
        W3_dim_0, W3_dim_1
      };

    const int64_t *real_input_shapes[] = {
      W0_shape_0, W1_shape_1, W2_shape_2, W3_shape_3
    };



    const int64_t *all_input_shapes[] = {
      W0_shape_0, W1_shape_1, W2_shape_2, W3_shape_3
    };

    int64_t *concatenate_25_0_shape[] = {
      &concatenate_25_0_dim_0, &W0_dim_1
    };

    int64_t concat_dim_sizes[] = {
      W0_dim_0, W1_dim_0, W2_dim_0, W3_dim_0
    };

    bool input_masks[] = {
      true, true, true, true
    };

    concatenate_25_constant_folding(
        concatenate_25_0,
        concatenate_25_0_shape,
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
      const int64_t x_dims[] = {reshape_26_0_dim_0, reshape_26_0_dim_1, reshape_26_0_dim_2};
      permute021_27_constant_folding(
          concatenate_25_0,
          permute021_27_0,
          3,
          x_dims,
          stream
      );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    const void *inputs[] = {
      W4,
      W5,
      W6,
      W7
    };


      int64_t W4_shape_0[] = {
        W4_dim_0, W4_dim_1
      };
      int64_t W5_shape_1[] = {
        W5_dim_0, W5_dim_1
      };
      int64_t W6_shape_2[] = {
        W6_dim_0, W6_dim_1
      };
      int64_t W7_shape_3[] = {
        W7_dim_0, W7_dim_1
      };

    const int64_t *real_input_shapes[] = {
      W4_shape_0, W5_shape_1, W6_shape_2, W7_shape_3
    };



    const int64_t *all_input_shapes[] = {
      W4_shape_0, W5_shape_1, W6_shape_2, W7_shape_3
    };

    int64_t *concatenate_28_0_shape[] = {
      &concatenate_28_0_dim_0, &W4_dim_1
    };

    int64_t concat_dim_sizes[] = {
      W4_dim_0, W5_dim_0, W6_dim_0, W7_dim_0
    };

    bool input_masks[] = {
      true, true, true, true
    };

    concatenate_25_constant_folding(
        concatenate_28_0,
        concatenate_28_0_shape,
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
      const int64_t x_dims[] = {reshape_29_0_dim_0, reshape_29_0_dim_1, reshape_29_0_dim_2};
      permute021_27_constant_folding(
          concatenate_28_0,
          permute021_30_0,
          3,
          x_dims,
          stream
      );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    const void *inputs[] = {
      B0,
      B1,
      B2,
      B3
    };


      int64_t B0_shape_0[] = {
        B0_dim_0
      };
      int64_t B1_shape_1[] = {
        B1_dim_0
      };
      int64_t B2_shape_2[] = {
        B2_dim_0
      };
      int64_t B3_shape_3[] = {
        B3_dim_0
      };

    const int64_t *real_input_shapes[] = {
      B0_shape_0, B1_shape_1, B2_shape_2, B3_shape_3
    };



    const int64_t *all_input_shapes[] = {
      B0_shape_0, B1_shape_1, B2_shape_2, B3_shape_3
    };

    int64_t *concatenate_31_0_shape[] = {
      &concatenate_31_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      B0_dim_0, B1_dim_0, B2_dim_0, B3_dim_0
    };

    bool input_masks[] = {
      true, true, true, true
    };

    concatenate_31_constant_folding(
        concatenate_31_0,
        concatenate_31_0_shape,
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
      B4,
      B5,
      B6,
      B7
    };


      int64_t B4_shape_0[] = {
        B4_dim_0
      };
      int64_t B5_shape_1[] = {
        B5_dim_0
      };
      int64_t B6_shape_2[] = {
        B6_dim_0
      };
      int64_t B7_shape_3[] = {
        B7_dim_0
      };

    const int64_t *real_input_shapes[] = {
      B4_shape_0, B5_shape_1, B6_shape_2, B7_shape_3
    };



    const int64_t *all_input_shapes[] = {
      B4_shape_0, B5_shape_1, B6_shape_2, B7_shape_3
    };

    int64_t *concatenate_35_0_shape[] = {
      &concatenate_35_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      B4_dim_0, B5_dim_0, B6_dim_0, B7_dim_0
    };

    bool input_masks[] = {
      true, true, true, true
    };

    concatenate_31_constant_folding(
        concatenate_35_0,
        concatenate_35_0_shape,
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
        std::cout << "Profiling: " << "permute021_5" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
      const int64_t x_dims[] = {W_dim_0, W_dim_1, W_dim_2};
      permute021_5_constant_folding(
          W,
          permute021_5_0,
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
        ss << "\"" << "permute021_5" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"4\", \"32\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"4\", \"64\", \"32\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_25" << " (" << iters << " iterations)" << std::endl;
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
      W0,
      W1,
      W2,
      W3
    };


      int64_t W0_shape_0[] = {
        W0_dim_0, W0_dim_1
      };
      int64_t W1_shape_1[] = {
        W1_dim_0, W1_dim_1
      };
      int64_t W2_shape_2[] = {
        W2_dim_0, W2_dim_1
      };
      int64_t W3_shape_3[] = {
        W3_dim_0, W3_dim_1
      };

    const int64_t *real_input_shapes[] = {
      W0_shape_0, W1_shape_1, W2_shape_2, W3_shape_3
    };



    const int64_t *all_input_shapes[] = {
      W0_shape_0, W1_shape_1, W2_shape_2, W3_shape_3
    };

    int64_t *concatenate_25_0_shape[] = {
      &concatenate_25_0_dim_0, &W0_dim_1
    };

    int64_t concat_dim_sizes[] = {
      W0_dim_0, W1_dim_0, W2_dim_0, W3_dim_0
    };

    bool input_masks[] = {
      true, true, true, true
    };

    concatenate_25_constant_folding(
        concatenate_25_0,
        concatenate_25_0_shape,
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
        ss << "\"" << "concatenate_25" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"32\", \"64\"], [\"32\", \"64\"], [\"32\", \"64\"], [\"32\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"128\", \"64\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "permute021_27" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
      const int64_t x_dims[] = {reshape_26_0_dim_0, reshape_26_0_dim_1, reshape_26_0_dim_2};
      permute021_27_constant_folding(
          concatenate_25_0,
          permute021_27_0,
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
        ss << "\"" << "permute021_27" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"4\", \"32\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"4\", \"64\", \"32\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_28" << " (" << iters << " iterations)" << std::endl;
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
      W4,
      W5,
      W6,
      W7
    };


      int64_t W4_shape_0[] = {
        W4_dim_0, W4_dim_1
      };
      int64_t W5_shape_1[] = {
        W5_dim_0, W5_dim_1
      };
      int64_t W6_shape_2[] = {
        W6_dim_0, W6_dim_1
      };
      int64_t W7_shape_3[] = {
        W7_dim_0, W7_dim_1
      };

    const int64_t *real_input_shapes[] = {
      W4_shape_0, W5_shape_1, W6_shape_2, W7_shape_3
    };



    const int64_t *all_input_shapes[] = {
      W4_shape_0, W5_shape_1, W6_shape_2, W7_shape_3
    };

    int64_t *concatenate_28_0_shape[] = {
      &concatenate_28_0_dim_0, &W4_dim_1
    };

    int64_t concat_dim_sizes[] = {
      W4_dim_0, W5_dim_0, W6_dim_0, W7_dim_0
    };

    bool input_masks[] = {
      true, true, true, true
    };

    concatenate_25_constant_folding(
        concatenate_28_0,
        concatenate_28_0_shape,
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
        ss << "\"" << "concatenate_28" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"32\", \"64\"], [\"32\", \"64\"], [\"32\", \"64\"], [\"32\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"128\", \"64\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "permute021_30" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
      const int64_t x_dims[] = {reshape_29_0_dim_0, reshape_29_0_dim_1, reshape_29_0_dim_2};
      permute021_27_constant_folding(
          concatenate_28_0,
          permute021_30_0,
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
        ss << "\"" << "permute021_30" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"4\", \"32\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"4\", \"64\", \"32\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_31" << " (" << iters << " iterations)" << std::endl;
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
      B0,
      B1,
      B2,
      B3
    };


      int64_t B0_shape_0[] = {
        B0_dim_0
      };
      int64_t B1_shape_1[] = {
        B1_dim_0
      };
      int64_t B2_shape_2[] = {
        B2_dim_0
      };
      int64_t B3_shape_3[] = {
        B3_dim_0
      };

    const int64_t *real_input_shapes[] = {
      B0_shape_0, B1_shape_1, B2_shape_2, B3_shape_3
    };



    const int64_t *all_input_shapes[] = {
      B0_shape_0, B1_shape_1, B2_shape_2, B3_shape_3
    };

    int64_t *concatenate_31_0_shape[] = {
      &concatenate_31_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      B0_dim_0, B1_dim_0, B2_dim_0, B3_dim_0
    };

    bool input_masks[] = {
      true, true, true, true
    };

    concatenate_31_constant_folding(
        concatenate_31_0,
        concatenate_31_0_shape,
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
        ss << "\"" << "concatenate_31" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"32\"], [\"32\"], [\"32\"], [\"32\"]]"
           << ", \"output_sizes\": " << "[[\"128\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_35" << " (" << iters << " iterations)" << std::endl;
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
      B4,
      B5,
      B6,
      B7
    };


      int64_t B4_shape_0[] = {
        B4_dim_0
      };
      int64_t B5_shape_1[] = {
        B5_dim_0
      };
      int64_t B6_shape_2[] = {
        B6_dim_0
      };
      int64_t B7_shape_3[] = {
        B7_dim_0
      };

    const int64_t *real_input_shapes[] = {
      B4_shape_0, B5_shape_1, B6_shape_2, B7_shape_3
    };



    const int64_t *all_input_shapes[] = {
      B4_shape_0, B5_shape_1, B6_shape_2, B7_shape_3
    };

    int64_t *concatenate_35_0_shape[] = {
      &concatenate_35_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      B4_dim_0, B5_dim_0, B6_dim_0, B7_dim_0
    };

    bool input_masks[] = {
      true, true, true, true
    };

    concatenate_31_constant_folding(
        concatenate_35_0,
        concatenate_35_0_shape,
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
        ss << "\"" << "concatenate_35" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"32\"], [\"32\"], [\"32\"], [\"32\"]]"
           << ", \"output_sizes\": " << "[[\"128\"]]"
        
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
          131072,
          0 * (1 + 0),
          0 * (1 + 0),
          0,
          5,
          0,
          constants,
          allocator
      );
    }

  private:
   void* W {nullptr};
   void* W0 {nullptr};
   void* B0 {nullptr};
   void* W1 {nullptr};
   void* B1 {nullptr};
   void* W2 {nullptr};
   void* B2 {nullptr};
   void* W3 {nullptr};
   void* B3 {nullptr};
   void* W4 {nullptr};
   void* B4 {nullptr};
   void* W5 {nullptr};
   void* B5 {nullptr};
   void* W6 {nullptr};
   void* B6 {nullptr};
   void* W7 {nullptr};
   void* B7 {nullptr};
   void* permute021_5_0 {nullptr};
   void* concatenate_25_0 {nullptr};
   void* permute021_27_0 {nullptr};
   void* concatenate_28_0 {nullptr};
   void* permute021_30_0 {nullptr};
   void* concatenate_31_0 {nullptr};
   void* concatenate_35_0 {nullptr};
   int64_t W_dim_0 { 4 };
   int64_t W_dim_1 { 32 };
   int64_t W_dim_2 { 64 };
   int64_t W0_dim_0 { 32 };
   int64_t W0_dim_1 { 64 };
   int64_t B0_dim_0 { 32 };
   int64_t W1_dim_0 { 32 };
   int64_t W1_dim_1 { 64 };
   int64_t B1_dim_0 { 32 };
   int64_t W2_dim_0 { 32 };
   int64_t W2_dim_1 { 64 };
   int64_t B2_dim_0 { 32 };
   int64_t W3_dim_0 { 32 };
   int64_t W3_dim_1 { 64 };
   int64_t B3_dim_0 { 32 };
   int64_t W4_dim_0 { 32 };
   int64_t W4_dim_1 { 64 };
   int64_t B4_dim_0 { 32 };
   int64_t W5_dim_0 { 32 };
   int64_t W5_dim_1 { 64 };
   int64_t B5_dim_0 { 32 };
   int64_t W6_dim_0 { 32 };
   int64_t W6_dim_1 { 64 };
   int64_t B6_dim_0 { 32 };
   int64_t W7_dim_0 { 32 };
   int64_t W7_dim_1 { 64 };
   int64_t B7_dim_0 { 32 };
   int64_t concatenate_25_0_dim_0 { 128 };
   int64_t reshape_26_0_dim_0 { 4 };
   int64_t reshape_26_0_dim_2 { 64 };
   int64_t reshape_26_0_dim_1 { 32 };
   int64_t concatenate_28_0_dim_0 { 128 };
   int64_t reshape_29_0_dim_0 { 4 };
   int64_t reshape_29_0_dim_2 { 64 };
   int64_t reshape_29_0_dim_1 { 32 };
   int64_t concatenate_31_0_dim_0 { 128 };
   int64_t concatenate_35_0_dim_0 { 128 };


};
} // namespace ait