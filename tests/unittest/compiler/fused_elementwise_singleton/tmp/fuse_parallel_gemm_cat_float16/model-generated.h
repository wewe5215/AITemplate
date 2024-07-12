
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


void split_0(
    void *[] /*outputs*/,
    int64_t **[] /*output_shapes*/,
    const bool [] /*output_masks*/,
    const void * /*input*/,
    const int64_t * /*input_shape*/,
    int64_t /*real_num_splits*/,
    int64_t /*all_num_splits*/,
    int64_t [] /*split_sizes*/,
    int64_t /*split_dim*/,
    int64_t /*rank*/,
    cudaStream_t stream
);

void gemm_rcr_bias_1(
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

void concatenate_5(
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
        W0 = reinterpret_cast<decltype(W0)>(constants + 0);
     constant_name_to_ptr_["W0"] = const_cast<const void**>(reinterpret_cast<void**>(&W0));
    B0 = reinterpret_cast<decltype(B0)>(constants + 16384);
     constant_name_to_ptr_["B0"] = const_cast<const void**>(reinterpret_cast<void**>(&B0));
    W1 = reinterpret_cast<decltype(W1)>(constants + 16640);
     constant_name_to_ptr_["W1"] = const_cast<const void**>(reinterpret_cast<void**>(&W1));
    B1 = reinterpret_cast<decltype(B1)>(constants + 33024);
     constant_name_to_ptr_["B1"] = const_cast<const void**>(reinterpret_cast<void**>(&B1));
    W2 = reinterpret_cast<decltype(W2)>(constants + 33280);
     constant_name_to_ptr_["W2"] = const_cast<const void**>(reinterpret_cast<void**>(&W2));
    B2 = reinterpret_cast<decltype(B2)>(constants + 49664);
     constant_name_to_ptr_["B2"] = const_cast<const void**>(reinterpret_cast<void**>(&B2));
    W3 = reinterpret_cast<decltype(W3)>(constants + 49920);
     constant_name_to_ptr_["W3"] = const_cast<const void**>(reinterpret_cast<void**>(&W3));
    B3 = reinterpret_cast<decltype(B3)>(constants + 66304);
     constant_name_to_ptr_["B3"] = const_cast<const void**>(reinterpret_cast<void**>(&B3));
    auto* blob_ptr = static_cast<uint8_t*>(blob_.get());
        split_0_0 = reinterpret_cast<decltype(split_0_0)>(blob_ptr + 393216);
    split_0_1 = reinterpret_cast<decltype(split_0_1)>(blob_ptr + 458752);
    split_0_2 = reinterpret_cast<decltype(split_0_2)>(blob_ptr + 262144);
    split_0_3 = reinterpret_cast<decltype(split_0_3)>(blob_ptr + 327680);
    
         params_[0].shape_ptrs = {ParamDim(256, 512, &input_batch), ParamDim(256, 256, &X_dim_1)};
     params_[2].shape_ptrs = {ParamDim(256, 512, &input_batch), ParamDim(128, 128, &W0_dim_0)};
     params_[3].shape_ptrs = {ParamDim(256, 512, &input_batch), ParamDim(128, 128, &W1_dim_0)};
     params_[4].shape_ptrs = {ParamDim(256, 512, &input_batch), ParamDim(128, 128, &W2_dim_0)};
     params_[5].shape_ptrs = {ParamDim(256, 512, &input_batch), ParamDim(128, 128, &W3_dim_0)};
     params_[1].shape_ptrs = {ParamDim(256, 512, &input_batch), ParamDim(512, 512, &output0_dim_1)};

      
      
    }

    ~Model() {
      
      
    }

    void SetUpInputsOutputs() {
             X = static_cast<decltype(X)>(params_[0].ptr);

if (X == nullptr) {
    throw std::runtime_error("Constant X was not set! Set the value with set_constant.");
}
    
     output1 = static_cast<decltype(output1)>(params_[2].ptr);

if (output1 == nullptr) {
    throw std::runtime_error("Constant output1 was not set! Set the value with set_constant.");
}
    
     output2 = static_cast<decltype(output2)>(params_[3].ptr);

if (output2 == nullptr) {
    throw std::runtime_error("Constant output2 was not set! Set the value with set_constant.");
}
    
     output3 = static_cast<decltype(output3)>(params_[4].ptr);

if (output3 == nullptr) {
    throw std::runtime_error("Constant output3 was not set! Set the value with set_constant.");
}
    
     output4 = static_cast<decltype(output4)>(params_[5].ptr);

if (output4 == nullptr) {
    throw std::runtime_error("Constant output4 was not set! Set the value with set_constant.");
}
    
     output0 = static_cast<decltype(output0)>(params_[1].ptr);

if (output0 == nullptr) {
    throw std::runtime_error("Constant output0 was not set! Set the value with set_constant.");
}
    
    }

    void ResetConstants(uint8_t* constants) {
        /*
         * This can be called if we want to use a different piece of memory
         * for the constants to be consumed.
         */
            W0 = reinterpret_cast<decltype(W0)>(constants + 0);
    B0 = reinterpret_cast<decltype(B0)>(constants + 16384);
    W1 = reinterpret_cast<decltype(W1)>(constants + 16640);
    B1 = reinterpret_cast<decltype(B1)>(constants + 33024);
    W2 = reinterpret_cast<decltype(W2)>(constants + 33280);
    B2 = reinterpret_cast<decltype(B2)>(constants + 49664);
    W3 = reinterpret_cast<decltype(W3)>(constants + 49920);
    B3 = reinterpret_cast<decltype(B3)>(constants + 66304);
    }

    void DeviceToDeviceCopies(StreamType stream) {
  
    }


    ///////////////////////////////////////////////////////////////////////////
    // default RunImpl implemenation
    void RunImpl(StreamType stream) {
        
  
  
  {

    void *outputs[] = {
      split_0_2,
      split_0_3,
      split_0_0,
      split_0_1
    };


      int64_t *split_0_2_shape[] = {
        &input_batch, &split_0_2_dim_1
      };
      int64_t *split_0_3_shape[] = {
        &input_batch, &split_0_3_dim_1
      };
      int64_t *split_0_0_shape[] = {
        &input_batch, &split_0_0_dim_1
      };
      int64_t *split_0_1_shape[] = {
        &input_batch, &split_0_1_dim_1
      };

    int64_t **output_shapes[] = {
      split_0_2_shape, split_0_3_shape, split_0_0_shape, split_0_1_shape
    };

    const int64_t X_shape[] = {
      input_batch, X_dim_1
    };

    int64_t split_sizes[] = {
      64, 64, 64, 64
    };

    bool output_masks[] = {
      true, true, true, true
    };

    split_0(
        outputs,
        output_shapes,
        output_masks,
        X,
        X_shape,
        4/*real_num_splits*/,
        4/*all_num_splits*/,
        split_sizes,
        1/*split_dim*/,
        2/*rank*/,
        stream
    );
  }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_1(

        split_0_2,
        W0,

        B0,

        output1,
        global_workspace_,
        1,

        &input_batch,

        &split_0_2_dim_1,


        &W0_dim_0,

        &W0_dim_1,


        &input_batch,

        &W0_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_1(

        split_0_3,
        W1,

        B1,

        output2,
        global_workspace_,
        1,

        &input_batch,

        &split_0_3_dim_1,


        &W1_dim_0,

        &W1_dim_1,


        &input_batch,

        &W1_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_1(

        split_0_0,
        W2,

        B2,

        output3,
        global_workspace_,
        1,

        &input_batch,

        &split_0_0_dim_1,


        &W2_dim_0,

        &W2_dim_1,


        &input_batch,

        &W2_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_1(

        split_0_1,
        W3,

        B3,

        output4,
        global_workspace_,
        1,

        &input_batch,

        &split_0_1_dim_1,


        &W3_dim_0,

        &W3_dim_1,


        &input_batch,

        &W3_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    const void *inputs[] = {
      output1,
      output2,
      output3,
      output4
    };


      int64_t output1_shape_0[] = {
        input_batch, W0_dim_0
      };
      int64_t output2_shape_1[] = {
        input_batch, W1_dim_0
      };
      int64_t output3_shape_2[] = {
        input_batch, W2_dim_0
      };
      int64_t output4_shape_3[] = {
        input_batch, W3_dim_0
      };

    const int64_t *real_input_shapes[] = {
      output1_shape_0, output2_shape_1, output3_shape_2, output4_shape_3
    };



    const int64_t *all_input_shapes[] = {
      output1_shape_0, output2_shape_1, output3_shape_2, output4_shape_3
    };

    int64_t *output0_shape[] = {
      &input_batch, &output0_dim_1
    };

    int64_t concat_dim_sizes[] = {
      W0_dim_0, W1_dim_0, W2_dim_0, W3_dim_0
    };

    bool input_masks[] = {
      true, true, true, true
    };

    concatenate_5(
        output0,
        output0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        1/*concat_dim*/,
        2/*rank*/,
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
        std::cout << "Profiling: " << "split_0" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
  {

    void *outputs[] = {
      split_0_2,
      split_0_3,
      split_0_0,
      split_0_1
    };


      int64_t *split_0_2_shape[] = {
        &input_batch, &split_0_2_dim_1
      };
      int64_t *split_0_3_shape[] = {
        &input_batch, &split_0_3_dim_1
      };
      int64_t *split_0_0_shape[] = {
        &input_batch, &split_0_0_dim_1
      };
      int64_t *split_0_1_shape[] = {
        &input_batch, &split_0_1_dim_1
      };

    int64_t **output_shapes[] = {
      split_0_2_shape, split_0_3_shape, split_0_0_shape, split_0_1_shape
    };

    const int64_t X_shape[] = {
      input_batch, X_dim_1
    };

    int64_t split_sizes[] = {
      64, 64, 64, 64
    };

    bool output_masks[] = {
      true, true, true, true
    };

    split_0(
        outputs,
        output_shapes,
        output_masks,
        X,
        X_shape,
        4/*real_num_splits*/,
        4/*all_num_splits*/,
        split_sizes,
        1/*split_dim*/,
        2/*rank*/,
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
        ss << "\"" << "split_0" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"input_batch\", \"256\"]]"
           << ", \"output_sizes\": " << "[[\"input_batch\", \"64\"], [\"input_batch\", \"64\"], [\"input_batch\", \"64\"], [\"input_batch\", \"64\"]]"
        
          << ", \"split_sizes\": " << "\"[64, 64, 64, 64]]\""
        
          << ", \"dim\": " << "\"1]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_1" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_1(

        split_0_2,
        W0,

        B0,

        output1,
        global_workspace_,
        1,

        &input_batch,

        &split_0_2_dim_1,


        &W0_dim_0,

        &W0_dim_1,


        &input_batch,

        &W0_dim_0,

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
        ss << "\"" << "gemm_rcr_bias_1" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"input_batch\", \"64\"], [\"128\", \"64\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"input_batch\", \"128\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_2" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_1(

        split_0_3,
        W1,

        B1,

        output2,
        global_workspace_,
        1,

        &input_batch,

        &split_0_3_dim_1,


        &W1_dim_0,

        &W1_dim_1,


        &input_batch,

        &W1_dim_0,

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
        ss << "\"" << "gemm_rcr_bias_2" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"input_batch\", \"64\"], [\"128\", \"64\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"input_batch\", \"128\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_3" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_1(

        split_0_0,
        W2,

        B2,

        output3,
        global_workspace_,
        1,

        &input_batch,

        &split_0_0_dim_1,


        &W2_dim_0,

        &W2_dim_1,


        &input_batch,

        &W2_dim_0,

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
        ss << "\"" << "gemm_rcr_bias_3" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"input_batch\", \"64\"], [\"128\", \"64\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"input_batch\", \"128\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_4" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_1(

        split_0_1,
        W3,

        B3,

        output4,
        global_workspace_,
        1,

        &input_batch,

        &split_0_1_dim_1,


        &W3_dim_0,

        &W3_dim_1,


        &input_batch,

        &W3_dim_0,

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
        ss << "\"" << "gemm_rcr_bias_4" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"input_batch\", \"64\"], [\"128\", \"64\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"input_batch\", \"128\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_5" << " (" << iters << " iterations)" << std::endl;
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
      output1,
      output2,
      output3,
      output4
    };


      int64_t output1_shape_0[] = {
        input_batch, W0_dim_0
      };
      int64_t output2_shape_1[] = {
        input_batch, W1_dim_0
      };
      int64_t output3_shape_2[] = {
        input_batch, W2_dim_0
      };
      int64_t output4_shape_3[] = {
        input_batch, W3_dim_0
      };

    const int64_t *real_input_shapes[] = {
      output1_shape_0, output2_shape_1, output3_shape_2, output4_shape_3
    };



    const int64_t *all_input_shapes[] = {
      output1_shape_0, output2_shape_1, output3_shape_2, output4_shape_3
    };

    int64_t *output0_shape[] = {
      &input_batch, &output0_dim_1
    };

    int64_t concat_dim_sizes[] = {
      W0_dim_0, W1_dim_0, W2_dim_0, W3_dim_0
    };

    bool input_masks[] = {
      true, true, true, true
    };

    concatenate_5(
        output0,
        output0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        1/*concat_dim*/,
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
        ss << "\"" << "concatenate_5" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"input_batch\", \"128\"], [\"input_batch\", \"128\"], [\"input_batch\", \"128\"], [\"input_batch\", \"128\"]]"
           << ", \"output_sizes\": " << "[[\"input_batch\", \"512\"]]"
        
          << ", \"dim\": " << "\"1\""
        
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
          1048576,
          0 * (1 + 0),
          0 * (1 + 0),
          1,
          5,
          0,
          constants,
          allocator
      );
    }

  private:
   void* X {nullptr};
   void* W0 {nullptr};
   void* B0 {nullptr};
   void* W1 {nullptr};
   void* B1 {nullptr};
   void* W2 {nullptr};
   void* B2 {nullptr};
   void* W3 {nullptr};
   void* B3 {nullptr};
   void* split_0_0 {nullptr};
   void* split_0_1 {nullptr};
   void* split_0_2 {nullptr};
   void* split_0_3 {nullptr};
   void* output1 {nullptr};
   void* output2 {nullptr};
   void* output3 {nullptr};
   void* output4 {nullptr};
   void* output0 {nullptr};
   int64_t input_batch { 0 };
   int64_t X_dim_1 { 256 };
   int64_t W0_dim_0 { 128 };
   int64_t W0_dim_1 { 64 };
   int64_t B0_dim_0 { 128 };
   int64_t W1_dim_0 { 128 };
   int64_t W1_dim_1 { 64 };
   int64_t B1_dim_0 { 128 };
   int64_t W2_dim_0 { 128 };
   int64_t W2_dim_1 { 64 };
   int64_t B2_dim_0 { 128 };
   int64_t W3_dim_0 { 128 };
   int64_t W3_dim_1 { 64 };
   int64_t B3_dim_0 { 128 };
   int64_t split_0_0_dim_1 { 64 };
   int64_t split_0_1_dim_1 { 64 };
   int64_t split_0_2_dim_1 { 64 };
   int64_t split_0_3_dim_1 { 64 };
   int64_t output0_dim_1 { 512 };


};
} // namespace ait