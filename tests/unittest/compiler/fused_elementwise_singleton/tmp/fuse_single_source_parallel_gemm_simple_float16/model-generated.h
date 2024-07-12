
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


void gemm_rcr_7(
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

void split_8(
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
        concatenate_6_0 = reinterpret_cast<decltype(concatenate_6_0)>(constants + 212992);
    gemm_rcr_7_0 = reinterpret_cast<decltype(gemm_rcr_7_0)>(blob_ptr + 0);
    split_8_0 = reinterpret_cast<decltype(split_8_0)>(blob_ptr + 851968);
    elementwise_9_0 = reinterpret_cast<decltype(elementwise_9_0)>(blob_ptr + 1900544);
    elementwise_10_0 = reinterpret_cast<decltype(elementwise_10_0)>(blob_ptr + 2424832);
    
         params_[0].shape_ptrs = {ParamDim(1024, 1024, &X_dim_0), ParamDim(256, 256, &X_dim_1)};
     params_[1].shape_ptrs = {ParamDim(1024, 1024, &X_dim_0), ParamDim(256, 256, &W0_dim_0)};
     params_[3].shape_ptrs = {ParamDim(1024, 1024, &X_dim_0), ParamDim(128, 128, &W2_dim_0)};
     params_[2].shape_ptrs = {ParamDim(1024, 1024, &X_dim_0), ParamDim(32, 32, &W1_dim_0)};

      
      
    }

    ~Model() {
      
      
    }

    void SetUpInputsOutputs() {
             X = static_cast<decltype(X)>(params_[0].ptr);

if (X == nullptr) {
    throw std::runtime_error("Constant X was not set! Set the value with set_constant.");
}
    
     output0 = static_cast<decltype(output0)>(params_[1].ptr);

if (output0 == nullptr) {
    throw std::runtime_error("Constant output0 was not set! Set the value with set_constant.");
}
    
     output2 = static_cast<decltype(output2)>(params_[3].ptr);

if (output2 == nullptr) {
    throw std::runtime_error("Constant output2 was not set! Set the value with set_constant.");
}
    
     output1 = static_cast<decltype(output1)>(params_[2].ptr);

if (output1 == nullptr) {
    throw std::runtime_error("Constant output1 was not set! Set the value with set_constant.");
}
    
    }

    void ResetConstants(uint8_t* constants) {
        /*
         * This can be called if we want to use a different piece of memory
         * for the constants to be consumed.
         */
            concatenate_6_0 = reinterpret_cast<decltype(concatenate_6_0)>(constants + 212992);
    }

    void DeviceToDeviceCopies(StreamType stream) {
  
    }


    ///////////////////////////////////////////////////////////////////////////
    // default RunImpl implemenation
    void RunImpl(StreamType stream) {
        
  
  
    {
    

    gemm_rcr_7(

        X,
        concatenate_6_0,

        gemm_rcr_7_0,
        global_workspace_,
        1,

        &X_dim_0,

        &X_dim_1,


        &concatenate_6_0_dim_0,

        &W0_dim_1,


        &X_dim_0,

        &concatenate_6_0_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    void *outputs[] = {
      split_8_0,
      elementwise_10_0,
      elementwise_9_0
    };


      int64_t *split_8_0_shape[] = {
        &X_dim_0, &split_8_0_dim_1
      };
      int64_t *elementwise_10_0_shape[] = {
        &X_dim_0, &elementwise_10_0_dim_1
      };
      int64_t *elementwise_9_0_shape[] = {
        &X_dim_0, &elementwise_9_0_dim_1
      };

    int64_t **output_shapes[] = {
      split_8_0_shape, elementwise_10_0_shape, elementwise_9_0_shape
    };

    const int64_t gemm_rcr_7_0_shape[] = {
      X_dim_0, concatenate_6_0_dim_0
    };

    int64_t split_sizes[] = {
      256, 32, 128
    };

    bool output_masks[] = {
      true, true, true
    };

    split_8(
        outputs,
        output_shapes,
        output_masks,
        gemm_rcr_7_0,
        gemm_rcr_7_0_shape,
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
        invoke_fused_elementwise_0(output0, split_8_0,   fused_elementwise_0_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_1_n_elements = 1024 * 128;
        invoke_fused_elementwise_1(output2, elementwise_9_0,   fused_elementwise_1_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_2_n_elements = 1024 * 32;
        invoke_fused_elementwise_2(output1, elementwise_10_0,   fused_elementwise_2_n_elements, stream);
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
        std::cout << "Profiling: " << "gemm_rcr_7" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_7(

        X,
        concatenate_6_0,

        gemm_rcr_7_0,
        global_workspace_,
        1,

        &X_dim_0,

        &X_dim_1,


        &concatenate_6_0_dim_0,

        &W0_dim_1,


        &X_dim_0,

        &concatenate_6_0_dim_0,

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
        ss << "\"" << "gemm_rcr_7" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1024\", \"256\"], [\"416\", \"256\"]]"
           << ", \"output_sizes\": " << "[[\"1024\", \"416\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "split_8" << " (" << iters << " iterations)" << std::endl;
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
      split_8_0,
      elementwise_10_0,
      elementwise_9_0
    };


      int64_t *split_8_0_shape[] = {
        &X_dim_0, &split_8_0_dim_1
      };
      int64_t *elementwise_10_0_shape[] = {
        &X_dim_0, &elementwise_10_0_dim_1
      };
      int64_t *elementwise_9_0_shape[] = {
        &X_dim_0, &elementwise_9_0_dim_1
      };

    int64_t **output_shapes[] = {
      split_8_0_shape, elementwise_10_0_shape, elementwise_9_0_shape
    };

    const int64_t gemm_rcr_7_0_shape[] = {
      X_dim_0, concatenate_6_0_dim_0
    };

    int64_t split_sizes[] = {
      256, 32, 128
    };

    bool output_masks[] = {
      true, true, true
    };

    split_8(
        outputs,
        output_shapes,
        output_masks,
        gemm_rcr_7_0,
        gemm_rcr_7_0_shape,
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
        ss << "\"" << "split_8" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1024\", \"416\"]]"
           << ", \"output_sizes\": " << "[[\"1024\", \"256\"], [\"1024\", \"32\"], [\"1024\", \"128\"]]"
        
          << ", \"split_sizes\": " << "\"[256, 32, 128]]\""
        
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
        invoke_fused_elementwise_0(output0, split_8_0,   fused_elementwise_0_n_elements, stream);
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
        invoke_fused_elementwise_1(output2, elementwise_9_0,   fused_elementwise_1_n_elements, stream);
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
        invoke_fused_elementwise_2(output1, elementwise_10_0,   fused_elementwise_2_n_elements, stream);
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
          2490368,
          0 * (1 + 0),
          0 * (1 + 0),
          1,
          3,
          0,
          constants,
          allocator
      );
    }

  private:
   void* X {nullptr};
   void* concatenate_6_0 {nullptr};
   void* gemm_rcr_7_0 {nullptr};
   void* split_8_0 {nullptr};
   void* output0 {nullptr};
   void* elementwise_9_0 {nullptr};
   void* output2 {nullptr};
   void* elementwise_10_0 {nullptr};
   void* output1 {nullptr};
   int64_t W0_dim_0 { 256 };
   int64_t W0_dim_1 { 256 };
   int64_t W1_dim_0 { 32 };
   int64_t W1_dim_1 { 256 };
   int64_t W2_dim_0 { 128 };
   int64_t W2_dim_1 { 256 };
   int64_t X_dim_0 { 1024 };
   int64_t X_dim_1 { 256 };
   int64_t concatenate_6_0_dim_0 { 416 };
   int64_t split_8_0_dim_1 { 256 };
   int64_t elementwise_9_0_dim_1 { 128 };
   int64_t elementwise_10_0_dim_1 { 32 };


};
} // namespace ait