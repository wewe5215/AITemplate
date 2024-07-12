
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


void invoke_fused_elementwise_14(void* output0,void* output1,void* output2, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void concatenate_2(
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

void invoke_fused_elementwise_18(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_17(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void concatenate_8(
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

void invoke_fused_elementwise_19(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void softmax_10(void* input,
               void* output,
               int multiprocessor_count,
               cudaStream_t stream);
    

void concatenate_11(
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

void invoke_fused_elementwise_20(void* output0, const void* input0,const void* input1,const void* input2,const void* input3,   int64_t n_elements, cudaStream_t stream);
    

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
    
    auto* blob_ptr = static_cast<uint8_t*>(blob_.get());
        elementwise_0_0 = reinterpret_cast<decltype(elementwise_0_0)>(blob_ptr + 2457600);
    concatenate_2_0 = reinterpret_cast<decltype(concatenate_2_0)>(blob_ptr + 0);
    elementwise_3_0 = reinterpret_cast<decltype(elementwise_3_0)>(blob_ptr + 2867200);
    concatenate_6_0 = reinterpret_cast<decltype(concatenate_6_0)>(blob_ptr + 819200);
    concatenate_8_0 = reinterpret_cast<decltype(concatenate_8_0)>(blob_ptr + 1638400);
    softmax_10_0 = reinterpret_cast<decltype(softmax_10_0)>(blob_ptr + 3276800);
    concatenate_11_0 = reinterpret_cast<decltype(concatenate_11_0)>(blob_ptr + 2457600);
    
         params_[0].shape_ptrs = {ParamDim(32, 32, &input_x_dim_0), ParamDim(64, 64, &input_x_dim_1), ParamDim(100, 100, &input_x_dim_2)};
     params_[1].shape_ptrs = {ParamDim(32, 32, &input_p_dim_0), ParamDim(64, 64, &input_p_dim_1), ParamDim(100, 100, &input_p_dim_2)};
     params_[2].shape_ptrs = {ParamDim(32, 32, &input_z_dim_0), ParamDim(64, 64, &input_z_dim_1), ParamDim(100, 100, &input_z_dim_2)};
     params_[3].shape_ptrs = {ParamDim(64, 64, &concatenate_8_0_dim_0), ParamDim(64, 64, &input_x_dim_1), ParamDim(100, 100, &input_x_dim_2)};

      
      
    }

    ~Model() {
      
      
    }

    void SetUpInputsOutputs() {
             input_x = static_cast<decltype(input_x)>(params_[0].ptr);

if (input_x == nullptr) {
    throw std::runtime_error("Constant input_x was not set! Set the value with set_constant.");
}
    
     input_p = static_cast<decltype(input_p)>(params_[1].ptr);

if (input_p == nullptr) {
    throw std::runtime_error("Constant input_p was not set! Set the value with set_constant.");
}
    
     input_z = static_cast<decltype(input_z)>(params_[2].ptr);

if (input_z == nullptr) {
    throw std::runtime_error("Constant input_z was not set! Set the value with set_constant.");
}
    
     output = static_cast<decltype(output)>(params_[3].ptr);

if (output == nullptr) {
    throw std::runtime_error("Constant output was not set! Set the value with set_constant.");
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
        int64_t fused_elementwise_14_n_elements = 32 * 64 * 100;
        invoke_fused_elementwise_14(elementwise_0_0,concatenate_2_0,elementwise_3_0, input_x,   fused_elementwise_14_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    const void *inputs[] = {
      input_p
    };


      int64_t input_p_shape_0[] = {
        input_p_dim_0, input_p_dim_1, input_p_dim_2
      };

    const int64_t *real_input_shapes[] = {
      input_p_shape_0
    };


      int64_t orig_elementwise_1_0_shape[] = {
        32, 64, 100
      };

    const int64_t *all_input_shapes[] = {
      orig_elementwise_1_0_shape, input_p_shape_0
    };

    int64_t *concatenate_2_0_shape[] = {
      &concatenate_2_0_dim_0, &input_x_dim_1, &input_x_dim_2
    };

    int64_t concat_dim_sizes[] = {
      32, input_p_dim_0
    };

    bool input_masks[] = {
      false, true
    };

    concatenate_2(
        concatenate_2_0,
        concatenate_2_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        3/*rank*/,
        1/*num_real_inputs*/,
        2/*num_all_inputs*/,
        stream
    );
  }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_18_n_elements = 32 * 64 * 100;
        invoke_fused_elementwise_18(concatenate_6_0, input_z,   fused_elementwise_18_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_17_n_elements = 32 * 64 * 100;
        invoke_fused_elementwise_17(concatenate_6_0, input_x,   fused_elementwise_17_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    const void *inputs[] = {
      elementwise_0_0
    };


      int64_t elementwise_0_0_shape_0[] = {
        input_x_dim_0, input_x_dim_1, input_x_dim_2
      };

    const int64_t *real_input_shapes[] = {
      elementwise_0_0_shape_0
    };


      int64_t orig_elementwise_7_0_shape[] = {
        32, 64, 100
      };

    const int64_t *all_input_shapes[] = {
      elementwise_0_0_shape_0, orig_elementwise_7_0_shape
    };

    int64_t *concatenate_8_0_shape[] = {
      &concatenate_8_0_dim_0, &input_x_dim_1, &input_x_dim_2
    };

    int64_t concat_dim_sizes[] = {
      input_x_dim_0, 32
    };

    bool input_masks[] = {
      true, false
    };

    concatenate_8(
        concatenate_8_0,
        concatenate_8_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        3/*rank*/,
        1/*num_real_inputs*/,
        2/*num_all_inputs*/,
        stream
    );
  }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_19_n_elements = 32 * 64 * 100;
        invoke_fused_elementwise_19(concatenate_8_0, elementwise_0_0,   fused_elementwise_19_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    softmax_10(
       elementwise_3_0,
       softmax_10_0,
       device_properties_.multiProcessorCount,
       stream
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    const void *inputs[] = {
      softmax_10_0,
      softmax_10_0
    };


      int64_t softmax_10_0_shape_0[] = {
        input_x_dim_0, input_x_dim_1, input_x_dim_2
      };
      int64_t softmax_10_0_shape_1[] = {
        input_x_dim_0, input_x_dim_1, input_x_dim_2
      };

    const int64_t *real_input_shapes[] = {
      softmax_10_0_shape_0, softmax_10_0_shape_1
    };



    const int64_t *all_input_shapes[] = {
      softmax_10_0_shape_0, softmax_10_0_shape_1
    };

    int64_t *concatenate_11_0_shape[] = {
      &concatenate_11_0_dim_0, &input_x_dim_1, &input_x_dim_2
    };

    int64_t concat_dim_sizes[] = {
      input_x_dim_0, input_x_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_11(
        concatenate_11_0,
        concatenate_11_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        3/*rank*/,
        2/*num_real_inputs*/,
        2/*num_all_inputs*/,
        stream
    );
  }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_20_n_elements = 64 * 64 * 100;
        invoke_fused_elementwise_20(output, concatenate_8_0,concatenate_2_0,concatenate_11_0,concatenate_6_0,   fused_elementwise_20_n_elements, stream);
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
        std::cout << "Profiling: " << "fused_elementwise_14" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_14_n_elements = 32 * 64 * 100;
        invoke_fused_elementwise_14(elementwise_0_0,concatenate_2_0,elementwise_3_0, input_x,   fused_elementwise_14_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_14" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"32\", \"64\", \"100\"]]"
           << ", \"output_sizes\": " << "[[\"32\", \"64\", \"100\"], [\"32\", \"64\", \"100\"], [\"32\", \"64\", \"100\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.RELU: 18>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_2" << " (" << iters << " iterations)" << std::endl;
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
      input_p
    };


      int64_t input_p_shape_0[] = {
        input_p_dim_0, input_p_dim_1, input_p_dim_2
      };

    const int64_t *real_input_shapes[] = {
      input_p_shape_0
    };


      int64_t orig_elementwise_1_0_shape[] = {
        32, 64, 100
      };

    const int64_t *all_input_shapes[] = {
      orig_elementwise_1_0_shape, input_p_shape_0
    };

    int64_t *concatenate_2_0_shape[] = {
      &concatenate_2_0_dim_0, &input_x_dim_1, &input_x_dim_2
    };

    int64_t concat_dim_sizes[] = {
      32, input_p_dim_0
    };

    bool input_masks[] = {
      false, true
    };

    concatenate_2(
        concatenate_2_0,
        concatenate_2_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        3/*rank*/,
        1/*num_real_inputs*/,
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
        ss << "\"" << "concatenate_2" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"32\", \"64\", \"100\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"64\", \"100\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_18" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_18_n_elements = 32 * 64 * 100;
        invoke_fused_elementwise_18(concatenate_6_0, input_z,   fused_elementwise_18_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_18" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"32\", \"64\", \"100\"]]"
           << ", \"output_sizes\": " << "[[\"32\", \"64\", \"100\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.RELU: 18>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_17" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_17_n_elements = 32 * 64 * 100;
        invoke_fused_elementwise_17(concatenate_6_0, input_x,   fused_elementwise_17_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_17" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"32\", \"64\", \"100\"]]"
           << ", \"output_sizes\": " << "[[\"32\", \"64\", \"100\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.GELU: 23>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_8" << " (" << iters << " iterations)" << std::endl;
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
      elementwise_0_0
    };


      int64_t elementwise_0_0_shape_0[] = {
        input_x_dim_0, input_x_dim_1, input_x_dim_2
      };

    const int64_t *real_input_shapes[] = {
      elementwise_0_0_shape_0
    };


      int64_t orig_elementwise_7_0_shape[] = {
        32, 64, 100
      };

    const int64_t *all_input_shapes[] = {
      elementwise_0_0_shape_0, orig_elementwise_7_0_shape
    };

    int64_t *concatenate_8_0_shape[] = {
      &concatenate_8_0_dim_0, &input_x_dim_1, &input_x_dim_2
    };

    int64_t concat_dim_sizes[] = {
      input_x_dim_0, 32
    };

    bool input_masks[] = {
      true, false
    };

    concatenate_8(
        concatenate_8_0,
        concatenate_8_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        3/*rank*/,
        1/*num_real_inputs*/,
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
        ss << "\"" << "concatenate_8" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"32\", \"64\", \"100\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"64\", \"100\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_19" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_19_n_elements = 32 * 64 * 100;
        invoke_fused_elementwise_19(concatenate_8_0, elementwise_0_0,   fused_elementwise_19_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_19" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"32\", \"64\", \"100\"]]"
           << ", \"output_sizes\": " << "[[\"32\", \"64\", \"100\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.TANH: 5>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "softmax_10" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    softmax_10(
       elementwise_3_0,
       softmax_10_0,
       device_properties_.multiProcessorCount,
       stream
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
        ss << "\"" << "softmax_10" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"32\", \"64\", \"100\"]]"
           << ", \"output_sizes\": " << "[[\"32\", \"64\", \"100\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_11" << " (" << iters << " iterations)" << std::endl;
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
      softmax_10_0,
      softmax_10_0
    };


      int64_t softmax_10_0_shape_0[] = {
        input_x_dim_0, input_x_dim_1, input_x_dim_2
      };
      int64_t softmax_10_0_shape_1[] = {
        input_x_dim_0, input_x_dim_1, input_x_dim_2
      };

    const int64_t *real_input_shapes[] = {
      softmax_10_0_shape_0, softmax_10_0_shape_1
    };



    const int64_t *all_input_shapes[] = {
      softmax_10_0_shape_0, softmax_10_0_shape_1
    };

    int64_t *concatenate_11_0_shape[] = {
      &concatenate_11_0_dim_0, &input_x_dim_1, &input_x_dim_2
    };

    int64_t concat_dim_sizes[] = {
      input_x_dim_0, input_x_dim_0
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_11(
        concatenate_11_0,
        concatenate_11_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        3/*rank*/,
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
        ss << "\"" << "concatenate_11" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"32\", \"64\", \"100\"], [\"32\", \"64\", \"100\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"64\", \"100\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_20" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_20_n_elements = 64 * 64 * 100;
        invoke_fused_elementwise_20(output, concatenate_8_0,concatenate_2_0,concatenate_11_0,concatenate_6_0,   fused_elementwise_20_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_20" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"64\", \"64\", \"100\"], [\"64\", \"64\", \"100\"], [\"64\", \"64\", \"100\"], [\"64\", \"64\", \"100\"]]"
           << ", \"output_sizes\": " << "[[\"64\", \"64\", \"100\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.ADD: 1>, <FuncEnum.ADD: 1>, <FuncEnum.ADD: 1>]\""
        
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
          4096000,
          0 * (1 + 0),
          0 * (1 + 0),
          3,
          1,
          0,
          constants,
          allocator
      );
    }

  private:
   void* input_x {nullptr};
   void* input_p {nullptr};
   void* input_z {nullptr};
   void* elementwise_0_0 {nullptr};
   void* concatenate_2_0 {nullptr};
   void* elementwise_3_0 {nullptr};
   void* concatenate_6_0 {nullptr};
   void* concatenate_8_0 {nullptr};
   void* softmax_10_0 {nullptr};
   void* concatenate_11_0 {nullptr};
   void* output {nullptr};
   int64_t input_x_dim_0 { 32 };
   int64_t input_x_dim_1 { 64 };
   int64_t input_x_dim_2 { 100 };
   int64_t input_p_dim_0 { 32 };
   int64_t input_p_dim_1 { 64 };
   int64_t input_p_dim_2 { 100 };
   int64_t input_z_dim_0 { 32 };
   int64_t input_z_dim_1 { 64 };
   int64_t input_z_dim_2 { 100 };
   int64_t concatenate_2_0_dim_0 { 64 };
   int64_t concatenate_6_0_dim_0 { 64 };
   int64_t concatenate_8_0_dim_0 { 64 };
   int64_t concatenate_11_0_dim_0 { 64 };


};
} // namespace ait