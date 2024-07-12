
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


void concatenate_23_constant_folding(
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

void permute021_25_constant_folding(
  const void* /* input */,
  void* /* output */,
  int64_t /* rank */,
  const int64_t* /* x_dims */,
  cudaStream_t /* stream */
);

void concatenate_26_constant_folding(
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

void permute021_28_constant_folding(
  const void* /* input */,
  void* /* output */,
  int64_t /* rank */,
  const int64_t* /* x_dims */,
  cudaStream_t /* stream */
);

void concatenate_29_constant_folding(
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
    b_0 = reinterpret_cast<decltype(b_0)>(constants + 16384);
     constant_name_to_ptr_["b_0"] = const_cast<const void**>(reinterpret_cast<void**>(&b_0));
    w_1 = reinterpret_cast<decltype(w_1)>(constants + 16640);
     constant_name_to_ptr_["w_1"] = const_cast<const void**>(reinterpret_cast<void**>(&w_1));
    b_1 = reinterpret_cast<decltype(b_1)>(constants + 33024);
     constant_name_to_ptr_["b_1"] = const_cast<const void**>(reinterpret_cast<void**>(&b_1));
    w_3 = reinterpret_cast<decltype(w_3)>(constants + 33280);
     constant_name_to_ptr_["w_3"] = const_cast<const void**>(reinterpret_cast<void**>(&w_3));
    b_3 = reinterpret_cast<decltype(b_3)>(constants + 51712);
     constant_name_to_ptr_["b_3"] = const_cast<const void**>(reinterpret_cast<void**>(&b_3));
    w_4 = reinterpret_cast<decltype(w_4)>(constants + 51968);
     constant_name_to_ptr_["w_4"] = const_cast<const void**>(reinterpret_cast<void**>(&w_4));
    b_4 = reinterpret_cast<decltype(b_4)>(constants + 70400);
     constant_name_to_ptr_["b_4"] = const_cast<const void**>(reinterpret_cast<void**>(&b_4));
    auto* blob_ptr = static_cast<uint8_t*>(blob_.get());
        concatenate_23_0 = reinterpret_cast<decltype(concatenate_23_0)>(blob_ptr + 0);
    concatenate_26_0 = reinterpret_cast<decltype(concatenate_26_0)>(blob_ptr + 0);
    
         params_[0].shape_ptrs = {ParamDim(2, 2, &reshape_24_0_dim_0), ParamDim(72, 72, &reshape_24_0_dim_2), ParamDim(128, 128, &reshape_24_0_dim_1)};
     params_[1].shape_ptrs = {ParamDim(2, 2, &reshape_27_0_dim_0), ParamDim(64, 64, &reshape_27_0_dim_2), ParamDim(128, 128, &reshape_27_0_dim_1)};
     params_[2].shape_ptrs = {ParamDim(256, 256, &concatenate_29_0_dim_0)};
     params_[3].shape_ptrs = {ParamDim(256, 256, &concatenate_33_0_dim_0)};

      
      
    }

    ~ConstantFolder() {
      
      
    }

    void SetUpInputsOutputs() {
             permute021_25_0 = static_cast<decltype(permute021_25_0)>(params_[0].ptr);

if (permute021_25_0 == nullptr) {
    throw std::runtime_error("Constant permute021_25_0 was not set! Set the value with set_constant.");
}
    
     permute021_28_0 = static_cast<decltype(permute021_28_0)>(params_[1].ptr);

if (permute021_28_0 == nullptr) {
    throw std::runtime_error("Constant permute021_28_0 was not set! Set the value with set_constant.");
}
    
     concatenate_29_0 = static_cast<decltype(concatenate_29_0)>(params_[2].ptr);

if (concatenate_29_0 == nullptr) {
    throw std::runtime_error("Constant concatenate_29_0 was not set! Set the value with set_constant.");
}
    
     concatenate_33_0 = static_cast<decltype(concatenate_33_0)>(params_[3].ptr);

if (concatenate_33_0 == nullptr) {
    throw std::runtime_error("Constant concatenate_33_0 was not set! Set the value with set_constant.");
}
    
    }

    void ResetConstants(uint8_t* constants) {
        /*
         * This can be called if we want to use a different piece of memory
         * for the constants to be consumed.
         */
            w_0 = reinterpret_cast<decltype(w_0)>(constants + 0);
    b_0 = reinterpret_cast<decltype(b_0)>(constants + 16384);
    w_1 = reinterpret_cast<decltype(w_1)>(constants + 16640);
    b_1 = reinterpret_cast<decltype(b_1)>(constants + 33024);
    w_3 = reinterpret_cast<decltype(w_3)>(constants + 33280);
    b_3 = reinterpret_cast<decltype(b_3)>(constants + 51712);
    w_4 = reinterpret_cast<decltype(w_4)>(constants + 51968);
    b_4 = reinterpret_cast<decltype(b_4)>(constants + 70400);
    }

    void DeviceToDeviceCopies(StreamType stream) {
  
    }


    ///////////////////////////////////////////////////////////////////////////
    // default RunImpl implemenation
    void RunImpl(StreamType stream) {
        
  
  
  {

    const void *inputs[] = {
      w_3,
      w_4
    };


      int64_t w_3_shape_0[] = {
        w_3_dim_0, w_3_dim_1
      };
      int64_t w_4_shape_1[] = {
        w_4_dim_0, w_4_dim_1
      };

    const int64_t *real_input_shapes[] = {
      w_3_shape_0, w_4_shape_1
    };



    const int64_t *all_input_shapes[] = {
      w_3_shape_0, w_4_shape_1
    };

    int64_t *concatenate_23_0_shape[] = {
      &concatenate_23_0_dim_0, &w_3_dim_1
    };

    int64_t concat_dim_sizes[] = {
      w_3_dim_0, w_4_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_23_constant_folding(
        concatenate_23_0,
        concatenate_23_0_shape,
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
      const int64_t x_dims[] = {reshape_24_0_dim_0, reshape_24_0_dim_1, reshape_24_0_dim_2};
      permute021_25_constant_folding(
          concatenate_23_0,
          permute021_25_0,
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

    int64_t *concatenate_26_0_shape[] = {
      &concatenate_26_0_dim_0, &w_0_dim_1
    };

    int64_t concat_dim_sizes[] = {
      w_0_dim_0, w_1_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_26_constant_folding(
        concatenate_26_0,
        concatenate_26_0_shape,
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
      const int64_t x_dims[] = {reshape_27_0_dim_0, reshape_27_0_dim_1, reshape_27_0_dim_2};
      permute021_28_constant_folding(
          concatenate_26_0,
          permute021_28_0,
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

    int64_t *concatenate_29_0_shape[] = {
      &concatenate_29_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      b_0_dim_0, b_1_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_29_constant_folding(
        concatenate_29_0,
        concatenate_29_0_shape,
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
      b_3,
      b_4
    };


      int64_t b_3_shape_0[] = {
        b_3_dim_0
      };
      int64_t b_4_shape_1[] = {
        b_4_dim_0
      };

    const int64_t *real_input_shapes[] = {
      b_3_shape_0, b_4_shape_1
    };



    const int64_t *all_input_shapes[] = {
      b_3_shape_0, b_4_shape_1
    };

    int64_t *concatenate_33_0_shape[] = {
      &concatenate_33_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      b_3_dim_0, b_4_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_29_constant_folding(
        concatenate_33_0,
        concatenate_33_0_shape,
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
        std::cout << "Profiling: " << "concatenate_23" << " (" << iters << " iterations)" << std::endl;
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
      w_3,
      w_4
    };


      int64_t w_3_shape_0[] = {
        w_3_dim_0, w_3_dim_1
      };
      int64_t w_4_shape_1[] = {
        w_4_dim_0, w_4_dim_1
      };

    const int64_t *real_input_shapes[] = {
      w_3_shape_0, w_4_shape_1
    };



    const int64_t *all_input_shapes[] = {
      w_3_shape_0, w_4_shape_1
    };

    int64_t *concatenate_23_0_shape[] = {
      &concatenate_23_0_dim_0, &w_3_dim_1
    };

    int64_t concat_dim_sizes[] = {
      w_3_dim_0, w_4_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_23_constant_folding(
        concatenate_23_0,
        concatenate_23_0_shape,
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
        ss << "\"" << "concatenate_23" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"128\", \"72\"], [\"128\", \"72\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"72\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "permute021_25" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
      const int64_t x_dims[] = {reshape_24_0_dim_0, reshape_24_0_dim_1, reshape_24_0_dim_2};
      permute021_25_constant_folding(
          concatenate_23_0,
          permute021_25_0,
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
        ss << "\"" << "permute021_25" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"2\", \"128\", \"72\"]]"
           << ", \"output_sizes\": " << "[[\"2\", \"72\", \"128\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_26" << " (" << iters << " iterations)" << std::endl;
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

    int64_t *concatenate_26_0_shape[] = {
      &concatenate_26_0_dim_0, &w_0_dim_1
    };

    int64_t concat_dim_sizes[] = {
      w_0_dim_0, w_1_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_26_constant_folding(
        concatenate_26_0,
        concatenate_26_0_shape,
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
        ss << "\"" << "concatenate_26" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"128\", \"64\"], [\"128\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"64\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "permute021_28" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
      const int64_t x_dims[] = {reshape_27_0_dim_0, reshape_27_0_dim_1, reshape_27_0_dim_2};
      permute021_28_constant_folding(
          concatenate_26_0,
          permute021_28_0,
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
        ss << "\"" << "permute021_28" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"2\", \"128\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"2\", \"64\", \"128\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_29" << " (" << iters << " iterations)" << std::endl;
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

    int64_t *concatenate_29_0_shape[] = {
      &concatenate_29_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      b_0_dim_0, b_1_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_29_constant_folding(
        concatenate_29_0,
        concatenate_29_0_shape,
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
        ss << "\"" << "concatenate_29" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"128\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"256\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_33" << " (" << iters << " iterations)" << std::endl;
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
      b_3,
      b_4
    };


      int64_t b_3_shape_0[] = {
        b_3_dim_0
      };
      int64_t b_4_shape_1[] = {
        b_4_dim_0
      };

    const int64_t *real_input_shapes[] = {
      b_3_shape_0, b_4_shape_1
    };



    const int64_t *all_input_shapes[] = {
      b_3_shape_0, b_4_shape_1
    };

    int64_t *concatenate_33_0_shape[] = {
      &concatenate_33_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      b_3_dim_0, b_4_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_29_constant_folding(
        concatenate_33_0,
        concatenate_33_0_shape,
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
        ss << "\"" << "concatenate_33" << "\": { \"ms_per_iter\": "
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
          106496,
          0 * (1 + 0),
          0 * (1 + 0),
          0,
          4,
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
   void* w_3 {nullptr};
   void* b_3 {nullptr};
   void* w_4 {nullptr};
   void* b_4 {nullptr};
   void* concatenate_23_0 {nullptr};
   void* permute021_25_0 {nullptr};
   void* concatenate_26_0 {nullptr};
   void* permute021_28_0 {nullptr};
   void* concatenate_29_0 {nullptr};
   void* concatenate_33_0 {nullptr};
   int64_t w_0_dim_0 { 128 };
   int64_t w_0_dim_1 { 64 };
   int64_t b_0_dim_0 { 128 };
   int64_t w_1_dim_0 { 128 };
   int64_t w_1_dim_1 { 64 };
   int64_t b_1_dim_0 { 128 };
   int64_t w_3_dim_0 { 128 };
   int64_t w_3_dim_1 { 72 };
   int64_t b_3_dim_0 { 128 };
   int64_t w_4_dim_0 { 128 };
   int64_t w_4_dim_1 { 72 };
   int64_t b_4_dim_0 { 128 };
   int64_t concatenate_23_0_dim_0 { 256 };
   int64_t reshape_24_0_dim_0 { 2 };
   int64_t reshape_24_0_dim_2 { 72 };
   int64_t reshape_24_0_dim_1 { 128 };
   int64_t concatenate_26_0_dim_0 { 256 };
   int64_t reshape_27_0_dim_0 { 2 };
   int64_t reshape_27_0_dim_2 { 64 };
   int64_t reshape_27_0_dim_1 { 128 };
   int64_t concatenate_29_0_dim_0 { 256 };
   int64_t concatenate_33_0_dim_0 { 256 };


};
} // namespace ait