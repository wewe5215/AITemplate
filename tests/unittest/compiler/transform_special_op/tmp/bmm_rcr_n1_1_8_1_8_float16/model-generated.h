
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


void bmm_rcr_n1_4(
  void*,
  void*,
  void*,
  
  int64_t*,
  
  int64_t*,
  
  int64_t*,
  
  
  int64_t*,
  
  int64_t*,
  
  int64_t*,
  
  
  int64_t*,
  
  int64_t*,
  
  int64_t*,
  
  float,
  bool,
  cudaStream_t
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
        bmm_rcr_tensor = reinterpret_cast<decltype(bmm_rcr_tensor)>(blob_ptr + 192);
    
         params_[0].shape_ptrs = {ParamDim(1, 1, &input_0_dim_0), ParamDim(8, 8, &input_0_dim_1), ParamDim(8, 8, &input_0_dim_2)};
     params_[1].shape_ptrs = {ParamDim(1, 1, &input_1_dim_0), ParamDim(1, 1, &input_1_dim_1), ParamDim(8, 8, &input_1_dim_2)};
     params_[2].shape_ptrs = {ParamDim(1, 1, &input_1_dim_0), ParamDim(8, 8, &input_0_dim_1), ParamDim(1, 1, &input_1_dim_1)};

      
      
    }

    ~Model() {
      
      
    }

    void SetUpInputsOutputs() {
             input_0 = static_cast<decltype(input_0)>(params_[0].ptr);

if (input_0 == nullptr) {
    throw std::runtime_error("Constant input_0 was not set! Set the value with set_constant.");
}
    
     input_1 = static_cast<decltype(input_1)>(params_[1].ptr);

if (input_1 == nullptr) {
    throw std::runtime_error("Constant input_1 was not set! Set the value with set_constant.");
}
    
     output_0 = static_cast<decltype(output_0)>(params_[2].ptr);

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
    

    bmm_rcr_n1_4(
        input_0,
        input_1,
        bmm_rcr_tensor,

        &input_0_dim_0,

        &input_0_dim_1,

        &input_0_dim_2,


         &input_1_dim_0,

         &input_1_dim_1,

         &input_1_dim_2,


        &input_1_dim_0,

        &input_0_dim_1,

        &input_1_dim_1,

        1.0,
        false,
        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_3_n_elements = 1 * 8 * 1;
        invoke_fused_elementwise_3(output_0, bmm_rcr_tensor,   fused_elementwise_3_n_elements, stream);
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
        std::cout << "Profiling: " << "bmm_rcr_n1_4" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    bmm_rcr_n1_4(
        input_0,
        input_1,
        bmm_rcr_tensor,

        &input_0_dim_0,

        &input_0_dim_1,

        &input_0_dim_2,


         &input_1_dim_0,

         &input_1_dim_1,

         &input_1_dim_2,


        &input_1_dim_0,

        &input_0_dim_1,

        &input_1_dim_1,

        1.0,
        false,
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
        ss << "\"" << "bmm_rcr_n1_4" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"8\", \"8\"], [\"1\", \"1\", \"8\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"8\", \"1\"]]"
        
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
        int64_t fused_elementwise_3_n_elements = 1 * 8 * 1;
        invoke_fused_elementwise_3(output_0, bmm_rcr_tensor,   fused_elementwise_3_n_elements, stream);
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
           << ", \"input_sizes\": " << "[[\"1\", \"8\", \"1\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"8\", \"1\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.COS: 6>]\""
        
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
          256,
          0 * (1 + 0),
          0 * (1 + 0),
          2,
          1,
          0,
          constants,
          allocator
      );
    }

  private:
   void* input_0 {nullptr};
   void* input_1 {nullptr};
   void* bmm_rcr_tensor {nullptr};
   void* output_0 {nullptr};
   int64_t input_0_dim_0 { 1 };
   int64_t input_0_dim_1 { 8 };
   int64_t input_0_dim_2 { 8 };
   int64_t input_1_dim_0 { 1 };
   int64_t input_1_dim_1 { 1 };
   int64_t input_1_dim_2 { 8 };


};
} // namespace ait