
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


void gemm_rcr_17(
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

void split_18(
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

void invoke_fused_elementwise_0(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_1(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_2(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void gemm_rcr_bias_11(
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

void split_12(
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

void invoke_fused_elementwise_3(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

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
        concatenate_7_0 = reinterpret_cast<decltype(concatenate_7_0)>(constants + 558400);
    concatenate_16_0 = reinterpret_cast<decltype(concatenate_16_0)>(constants + 902464);
    gemm_rcr_17_0 = reinterpret_cast<decltype(gemm_rcr_17_0)>(blob_ptr + 0);
    split_18_0 = reinterpret_cast<decltype(split_18_0)>(blob_ptr + 851968);
    split_18_1 = reinterpret_cast<decltype(split_18_1)>(blob_ptr + 1900544);
    split_18_2 = reinterpret_cast<decltype(split_18_2)>(blob_ptr + 2162688);
    concatenate_10_0 = reinterpret_cast<decltype(concatenate_10_0)>(constants + 1115456);
    gemm_rcr_bias_11_0 = reinterpret_cast<decltype(gemm_rcr_bias_11_0)>(blob_ptr + 0);
    split_12_0 = reinterpret_cast<decltype(split_12_0)>(blob_ptr + 1376256);
    elementwise_13_0 = reinterpret_cast<decltype(elementwise_13_0)>(blob_ptr + 4259840);
    elementwise_14_0 = reinterpret_cast<decltype(elementwise_14_0)>(blob_ptr + 4849664);
    
         params_[0].shape_ptrs = {ParamDim(1024, 1024, &X_dim_0), ParamDim(256, 256, &X_dim_1)};
     params_[6].shape_ptrs = {ParamDim(1024, 1024, &X_dim_0), ParamDim(256, 256, &W5_dim_0)};
     params_[5].shape_ptrs = {ParamDim(1024, 1024, &X_dim_0), ParamDim(128, 128, &W4_dim_0)};
     params_[4].shape_ptrs = {ParamDim(1024, 1024, &X_dim_0), ParamDim(32, 32, &W3_dim_0)};
     params_[1].shape_ptrs = {ParamDim(1024, 1024, &X_dim_0), ParamDim(512, 512, &W0_dim_0)};
     params_[2].shape_ptrs = {ParamDim(1024, 1024, &X_dim_0), ParamDim(128, 128, &W1_dim_0)};
     params_[3].shape_ptrs = {ParamDim(1024, 1024, &X_dim_0), ParamDim(32, 32, &W2_dim_0)};

      
      
    }

    ~Model() {
      
      
    }

    void SetUpInputsOutputs() {
             X = static_cast<decltype(X)>(params_[0].ptr);

if (X == nullptr) {
    throw std::runtime_error("Constant X was not set! Set the value with set_constant.");
}
    
     output5 = static_cast<decltype(output5)>(params_[6].ptr);

if (output5 == nullptr) {
    throw std::runtime_error("Constant output5 was not set! Set the value with set_constant.");
}
    
     output4 = static_cast<decltype(output4)>(params_[5].ptr);

if (output4 == nullptr) {
    throw std::runtime_error("Constant output4 was not set! Set the value with set_constant.");
}
    
     output3 = static_cast<decltype(output3)>(params_[4].ptr);

if (output3 == nullptr) {
    throw std::runtime_error("Constant output3 was not set! Set the value with set_constant.");
}
    
     output0 = static_cast<decltype(output0)>(params_[1].ptr);

if (output0 == nullptr) {
    throw std::runtime_error("Constant output0 was not set! Set the value with set_constant.");
}
    
     output1 = static_cast<decltype(output1)>(params_[2].ptr);

if (output1 == nullptr) {
    throw std::runtime_error("Constant output1 was not set! Set the value with set_constant.");
}
    
     output2 = static_cast<decltype(output2)>(params_[3].ptr);

if (output2 == nullptr) {
    throw std::runtime_error("Constant output2 was not set! Set the value with set_constant.");
}
    
    }

    void ResetConstants(uint8_t* constants) {
        /*
         * This can be called if we want to use a different piece of memory
         * for the constants to be consumed.
         */
            concatenate_7_0 = reinterpret_cast<decltype(concatenate_7_0)>(constants + 558400);
    concatenate_16_0 = reinterpret_cast<decltype(concatenate_16_0)>(constants + 902464);
    concatenate_10_0 = reinterpret_cast<decltype(concatenate_10_0)>(constants + 1115456);
    }

    void DeviceToDeviceCopies(StreamType stream) {
  
    }


    ///////////////////////////////////////////////////////////////////////////
    // default RunImpl implemenation
    void RunImpl(StreamType stream) {
        
  
  
    {
    

    gemm_rcr_17(

        X,
        concatenate_16_0,

        gemm_rcr_17_0,
        global_workspace_,
        1,

        &X_dim_0,

        &X_dim_1,


        &concatenate_16_0_dim_0,

        &W3_dim_1,


        &X_dim_0,

        &concatenate_16_0_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    void *outputs[] = {
      split_18_2,
      split_18_1,
      split_18_0
    };


      int64_t *split_18_2_shape[] = {
        &X_dim_0, &split_18_2_dim_1
      };
      int64_t *split_18_1_shape[] = {
        &X_dim_0, &split_18_1_dim_1
      };
      int64_t *split_18_0_shape[] = {
        &X_dim_0, &split_18_0_dim_1
      };

    int64_t **output_shapes[] = {
      split_18_2_shape, split_18_1_shape, split_18_0_shape
    };

    const int64_t gemm_rcr_17_0_shape[] = {
      X_dim_0, concatenate_16_0_dim_0
    };

    int64_t split_sizes[] = {
      32, 128, 256
    };

    bool output_masks[] = {
      true, true, true
    };

    split_18(
        outputs,
        output_shapes,
        output_masks,
        gemm_rcr_17_0,
        gemm_rcr_17_0_shape,
        3/*real_num_splits*/,
        3/*all_num_splits*/,
        split_sizes,
        1/*split_dim*/,
        2/*rank*/,
        stream
    );
  }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_0_n_elements = 1024 * 256;
        invoke_fused_elementwise_0(output5, split_18_0,   fused_elementwise_0_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_1_n_elements = 1024 * 128;
        invoke_fused_elementwise_1(output4, split_18_1,   fused_elementwise_1_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_2_n_elements = 1024 * 32;
        invoke_fused_elementwise_2(output3, split_18_2,   fused_elementwise_2_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_11(

        X,
        concatenate_7_0,

        concatenate_10_0,

        gemm_rcr_bias_11_0,
        global_workspace_,
        1,

        &X_dim_0,

        &X_dim_1,


        &concatenate_7_0_dim_0,

        &W0_dim_1,


        &X_dim_0,

        &concatenate_7_0_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    void *outputs[] = {
      split_12_0,
      elementwise_13_0,
      elementwise_14_0
    };


      int64_t *split_12_0_shape[] = {
        &X_dim_0, &split_12_0_dim_1
      };
      int64_t *elementwise_13_0_shape[] = {
        &X_dim_0, &elementwise_13_0_dim_1
      };
      int64_t *elementwise_14_0_shape[] = {
        &X_dim_0, &elementwise_14_0_dim_1
      };

    int64_t **output_shapes[] = {
      split_12_0_shape, elementwise_13_0_shape, elementwise_14_0_shape
    };

    const int64_t gemm_rcr_bias_11_0_shape[] = {
      X_dim_0, concatenate_7_0_dim_0
    };

    int64_t split_sizes[] = {
      512, 128, 32
    };

    bool output_masks[] = {
      true, true, true
    };

    split_12(
        outputs,
        output_shapes,
        output_masks,
        gemm_rcr_bias_11_0,
        gemm_rcr_bias_11_0_shape,
        3/*real_num_splits*/,
        3/*all_num_splits*/,
        split_sizes,
        1/*split_dim*/,
        2/*rank*/,
        stream
    );
  }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_3_n_elements = 1024 * 512;
        invoke_fused_elementwise_3(output0, split_12_0,   fused_elementwise_3_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_1_n_elements = 1024 * 128;
        invoke_fused_elementwise_1(output1, elementwise_13_0,   fused_elementwise_1_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_2_n_elements = 1024 * 32;
        invoke_fused_elementwise_2(output2, elementwise_14_0,   fused_elementwise_2_n_elements, stream);
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
        std::cout << "Profiling: " << "gemm_rcr_17" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_17(

        X,
        concatenate_16_0,

        gemm_rcr_17_0,
        global_workspace_,
        1,

        &X_dim_0,

        &X_dim_1,


        &concatenate_16_0_dim_0,

        &W3_dim_1,


        &X_dim_0,

        &concatenate_16_0_dim_0,

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
        ss << "\"" << "gemm_rcr_17" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1024\", \"256\"], [\"416\", \"256\"]]"
           << ", \"output_sizes\": " << "[[\"1024\", \"416\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "split_18" << " (" << iters << " iterations)" << std::endl;
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
      split_18_2,
      split_18_1,
      split_18_0
    };


      int64_t *split_18_2_shape[] = {
        &X_dim_0, &split_18_2_dim_1
      };
      int64_t *split_18_1_shape[] = {
        &X_dim_0, &split_18_1_dim_1
      };
      int64_t *split_18_0_shape[] = {
        &X_dim_0, &split_18_0_dim_1
      };

    int64_t **output_shapes[] = {
      split_18_2_shape, split_18_1_shape, split_18_0_shape
    };

    const int64_t gemm_rcr_17_0_shape[] = {
      X_dim_0, concatenate_16_0_dim_0
    };

    int64_t split_sizes[] = {
      32, 128, 256
    };

    bool output_masks[] = {
      true, true, true
    };

    split_18(
        outputs,
        output_shapes,
        output_masks,
        gemm_rcr_17_0,
        gemm_rcr_17_0_shape,
        3/*real_num_splits*/,
        3/*all_num_splits*/,
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
        ss << "\"" << "split_18" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1024\", \"416\"]]"
           << ", \"output_sizes\": " << "[[\"1024\", \"32\"], [\"1024\", \"128\"], [\"1024\", \"256\"]]"
        
          << ", \"split_sizes\": " << "\"[32, 128, 256]]\""
        
          << ", \"dim\": " << "\"1]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_0" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_0_n_elements = 1024 * 256;
        invoke_fused_elementwise_0(output5, split_18_0,   fused_elementwise_0_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_0" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1024\", \"256\"]]"
           << ", \"output_sizes\": " << "[[\"1024\", \"256\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.RELU: 18>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_1" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_1_n_elements = 1024 * 128;
        invoke_fused_elementwise_1(output4, split_18_1,   fused_elementwise_1_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_1" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1024\", \"128\"]]"
           << ", \"output_sizes\": " << "[[\"1024\", \"128\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.RELU: 18>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_2" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_2_n_elements = 1024 * 32;
        invoke_fused_elementwise_2(output3, split_18_2,   fused_elementwise_2_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_2" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1024\", \"32\"]]"
           << ", \"output_sizes\": " << "[[\"1024\", \"32\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.RELU: 18>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_11" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_11(

        X,
        concatenate_7_0,

        concatenate_10_0,

        gemm_rcr_bias_11_0,
        global_workspace_,
        1,

        &X_dim_0,

        &X_dim_1,


        &concatenate_7_0_dim_0,

        &W0_dim_1,


        &X_dim_0,

        &concatenate_7_0_dim_0,

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
        ss << "\"" << "gemm_rcr_bias_11" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1024\", \"256\"], [\"672\", \"256\"], [\"672\"]]"
           << ", \"output_sizes\": " << "[[\"1024\", \"672\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "split_12" << " (" << iters << " iterations)" << std::endl;
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
      split_12_0,
      elementwise_13_0,
      elementwise_14_0
    };


      int64_t *split_12_0_shape[] = {
        &X_dim_0, &split_12_0_dim_1
      };
      int64_t *elementwise_13_0_shape[] = {
        &X_dim_0, &elementwise_13_0_dim_1
      };
      int64_t *elementwise_14_0_shape[] = {
        &X_dim_0, &elementwise_14_0_dim_1
      };

    int64_t **output_shapes[] = {
      split_12_0_shape, elementwise_13_0_shape, elementwise_14_0_shape
    };

    const int64_t gemm_rcr_bias_11_0_shape[] = {
      X_dim_0, concatenate_7_0_dim_0
    };

    int64_t split_sizes[] = {
      512, 128, 32
    };

    bool output_masks[] = {
      true, true, true
    };

    split_12(
        outputs,
        output_shapes,
        output_masks,
        gemm_rcr_bias_11_0,
        gemm_rcr_bias_11_0_shape,
        3/*real_num_splits*/,
        3/*all_num_splits*/,
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
        ss << "\"" << "split_12" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1024\", \"672\"]]"
           << ", \"output_sizes\": " << "[[\"1024\", \"512\"], [\"1024\", \"128\"], [\"1024\", \"32\"]]"
        
          << ", \"split_sizes\": " << "\"[512, 128, 32]]\""
        
          << ", \"dim\": " << "\"1]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_3" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_3_n_elements = 1024 * 512;
        invoke_fused_elementwise_3(output0, split_12_0,   fused_elementwise_3_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_3" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1024\", \"512\"]]"
           << ", \"output_sizes\": " << "[[\"1024\", \"512\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.RELU: 18>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_4" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_1_n_elements = 1024 * 128;
        invoke_fused_elementwise_1(output1, elementwise_13_0,   fused_elementwise_1_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_4" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1024\", \"128\"]]"
           << ", \"output_sizes\": " << "[[\"1024\", \"128\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.RELU: 18>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_5" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_2_n_elements = 1024 * 32;
        invoke_fused_elementwise_2(output2, elementwise_14_0,   fused_elementwise_2_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_5" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1024\", \"32\"]]"
           << ", \"output_sizes\": " << "[[\"1024\", \"32\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.RELU: 18>]\""
        
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
          4915200,
          0 * (1 + 0),
          0 * (1 + 0),
          1,
          6,
          0,
          constants,
          allocator
      );
    }

  private:
   void* X {nullptr};
   void* concatenate_7_0 {nullptr};
   void* concatenate_16_0 {nullptr};
   void* gemm_rcr_17_0 {nullptr};
   void* split_18_0 {nullptr};
   void* output5 {nullptr};
   void* split_18_1 {nullptr};
   void* output4 {nullptr};
   void* split_18_2 {nullptr};
   void* output3 {nullptr};
   void* concatenate_10_0 {nullptr};
   void* gemm_rcr_bias_11_0 {nullptr};
   void* split_12_0 {nullptr};
   void* output0 {nullptr};
   void* elementwise_13_0 {nullptr};
   void* output1 {nullptr};
   void* elementwise_14_0 {nullptr};
   void* output2 {nullptr};
   int64_t W0_dim_0 { 512 };
   int64_t W0_dim_1 { 256 };
   int64_t W1_dim_0 { 128 };
   int64_t W1_dim_1 { 256 };
   int64_t W2_dim_0 { 32 };
   int64_t W2_dim_1 { 256 };
   int64_t B0_dim_0 { 512 };
   int64_t B1_dim_0 { 128 };
   int64_t B2_dim_0 { 32 };
   int64_t W3_dim_0 { 32 };
   int64_t W3_dim_1 { 256 };
   int64_t W4_dim_0 { 128 };
   int64_t W4_dim_1 { 256 };
   int64_t W5_dim_0 { 256 };
   int64_t W5_dim_1 { 256 };
   int64_t X_dim_0 { 1024 };
   int64_t X_dim_1 { 256 };
   int64_t concatenate_7_0_dim_0 { 672 };
   int64_t concatenate_16_0_dim_0 { 416 };
   int64_t split_18_0_dim_1 { 256 };
   int64_t split_18_1_dim_1 { 128 };
   int64_t split_18_2_dim_1 { 32 };
   int64_t concatenate_10_0_dim_0 { 672 };
   int64_t split_12_0_dim_1 { 512 };
   int64_t elementwise_13_0_dim_1 { 128 };
   int64_t elementwise_14_0_dim_1 { 32 };


};
} // namespace ait