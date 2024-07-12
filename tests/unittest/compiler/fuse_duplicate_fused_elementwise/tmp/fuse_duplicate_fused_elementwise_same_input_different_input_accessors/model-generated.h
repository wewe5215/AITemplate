
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


void invoke_fused_elementwise_7(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_8(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void softmax_4(void* input,
               void* output,
               int multiprocessor_count,
               cudaStream_t stream);
    

void invoke_fused_elementwise_9(void* output0, const void* input0,const void* input1,   int64_t n_elements, cudaStream_t stream);
    

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
        elementwise_2_0 = reinterpret_cast<decltype(elementwise_2_0)>(blob_ptr + 409600);
    elementwise_3_0 = reinterpret_cast<decltype(elementwise_3_0)>(blob_ptr + 614400);
    softmax_4_0 = reinterpret_cast<decltype(softmax_4_0)>(blob_ptr + 0);
    softmax_5_0 = reinterpret_cast<decltype(softmax_5_0)>(blob_ptr + 204800);
    
         params_[0].shape_ptrs = {ParamDim(32, 32, &input_x_dim_0), ParamDim(64, 64, &input_x_dim_1), ParamDim(100, 100, &input_x_dim_2)};
     params_[1].shape_ptrs = {ParamDim(32, 32, &dynamic_slice_0_0_dim_0), ParamDim(64, 64, &dynamic_slice_0_0_dim_1), ParamDim(50, 50, &dynamic_slice_0_0_dim_2)};

      
      
    }

    ~Model() {
      
      
    }

    void SetUpInputsOutputs() {
             input_x = static_cast<decltype(input_x)>(params_[0].ptr);

if (input_x == nullptr) {
    throw std::runtime_error("Constant input_x was not set! Set the value with set_constant.");
}
    
     output = static_cast<decltype(output)>(params_[1].ptr);

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
        int64_t fused_elementwise_7_n_elements = 32 * 64 * 50;
        invoke_fused_elementwise_7(elementwise_2_0, input_x,   fused_elementwise_7_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_8_n_elements = 32 * 64 * 50;
        invoke_fused_elementwise_8(elementwise_3_0, input_x,   fused_elementwise_8_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    softmax_4(
       elementwise_2_0,
       softmax_4_0,
       device_properties_.multiProcessorCount,
       stream
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    softmax_4(
       elementwise_3_0,
       softmax_5_0,
       device_properties_.multiProcessorCount,
       stream
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_9_n_elements = 32 * 64 * 50;
        invoke_fused_elementwise_9(output, softmax_4_0,softmax_5_0,   fused_elementwise_9_n_elements, stream);
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
        std::cout << "Profiling: " << "fused_elementwise_7" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_7_n_elements = 32 * 64 * 50;
        invoke_fused_elementwise_7(elementwise_2_0, input_x,   fused_elementwise_7_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_7" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"32\", \"64\", \"50\"]]"
           << ", \"output_sizes\": " << "[[\"32\", \"64\", \"50\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.SIGMOID: 15>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_8" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_8_n_elements = 32 * 64 * 50;
        invoke_fused_elementwise_8(elementwise_3_0, input_x,   fused_elementwise_8_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_8" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"32\", \"64\", \"50\"]]"
           << ", \"output_sizes\": " << "[[\"32\", \"64\", \"50\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.SIGMOID: 15>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "softmax_4" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    softmax_4(
       elementwise_2_0,
       softmax_4_0,
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
        ss << "\"" << "softmax_4" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"32\", \"64\", \"50\"]]"
           << ", \"output_sizes\": " << "[[\"32\", \"64\", \"50\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "softmax_5" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    softmax_4(
       elementwise_3_0,
       softmax_5_0,
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
        ss << "\"" << "softmax_5" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"32\", \"64\", \"50\"]]"
           << ", \"output_sizes\": " << "[[\"32\", \"64\", \"50\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_9" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_9_n_elements = 32 * 64 * 50;
        invoke_fused_elementwise_9(output, softmax_4_0,softmax_5_0,   fused_elementwise_9_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_9" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"32\", \"64\", \"50\"], [\"32\", \"64\", \"50\"]]"
           << ", \"output_sizes\": " << "[[\"32\", \"64\", \"50\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.ADD: 1>]\""
        
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
          819200,
          0 * (1 + 0),
          0 * (1 + 0),
          1,
          1,
          0,
          constants,
          allocator
      );
    }

  private:
   void* input_x {nullptr};
   void* elementwise_2_0 {nullptr};
   void* elementwise_3_0 {nullptr};
   void* softmax_4_0 {nullptr};
   void* softmax_5_0 {nullptr};
   void* output {nullptr};
   int64_t input_x_dim_0 { 32 };
   int64_t input_x_dim_1 { 64 };
   int64_t input_x_dim_2 { 100 };
   int64_t dynamic_slice_0_0_dim_0 { 32 };
   int64_t dynamic_slice_0_0_dim_1 { 64 };
   int64_t dynamic_slice_0_0_dim_2 { 50 };
   int64_t dynamic_slice_1_0_dim_0 { 32 };
   int64_t dynamic_slice_1_0_dim_1 { 64 };
   int64_t dynamic_slice_1_0_dim_2 { 50 };


};
} // namespace ait