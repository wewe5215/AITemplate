
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


void ait_unsqueeze_0(
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*
);

void ait_reshape_3(
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*
);

void softmax_2(void* input,
               void* output,
               int64_t* in_0,
               int64_t* in_1,
               int64_t* in_2,
               int multiprocessor_count,
               cudaStream_t stream);
    

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
    
    
         params_[0].shape_ptrs = {ParamDim(1, 10, &x_dim_0), ParamDim(32, 32, &x_dim_1), ParamDim(51, 51, &x_dim_2)};
     params_[1].shape_ptrs = {ParamDim(1, 10, &x_dim_0), ParamDim(32, 32, &x_dim_1), ParamDim(1, 1, &unsqueeze_0_0_dim_3), ParamDim(51, 51, &x_dim_2)};

      
      
    }

    ~Model() {
      
      
    }

    void SetUpInputsOutputs() {
             x = static_cast<decltype(x)>(params_[0].ptr);

if (x == nullptr) {
    throw std::runtime_error("Constant x was not set! Set the value with set_constant.");
}
    
     unsqueeze_0_0 = x;
     reshape_3_0 = unsqueeze_0_0;
     z = static_cast<decltype(z)>(params_[1].ptr);

if (z == nullptr) {
    throw std::runtime_error("Constant z was not set! Set the value with set_constant.");
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
        
  
  
    ait_unsqueeze_0(
        &x_dim_0,
        &x_dim_1,
        &x_dim_2,
        &x_dim_0,
        &x_dim_1,
        &x_dim_2,
        &unsqueeze_0_0_dim_3
    );
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    ait_reshape_3(
        &x_dim_0,
        &x_dim_1,
        &x_dim_2,
        &unsqueeze_0_0_dim_3,
        &x_dim_0,
        &x_dim_1,
        &unsqueeze_0_0_dim_3,
        &x_dim_2
    );
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    softmax_2(
       reshape_3_0,
       z,
       &x_dim_0,
       &x_dim_1,
       &unsqueeze_0_0_dim_3,
       device_properties_.multiProcessorCount,
       stream
    );
    
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
        std::cout << "Profiling: " << "unsqueeze_0" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    ait_unsqueeze_0(
        &x_dim_0,
        &x_dim_1,
        &x_dim_2,
        &x_dim_0,
        &x_dim_1,
        &x_dim_2,
        &unsqueeze_0_0_dim_3
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
        ss << "\"" << "unsqueeze_0" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"x_dim_0\", \"32\", \"51\"]]"
           << ", \"output_sizes\": " << "[[\"x_dim_0\", \"32\", \"51\", \"1\"]]"
        
          << ", \"dim\": " << "\"3\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "reshape_3" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    ait_reshape_3(
        &x_dim_0,
        &x_dim_1,
        &x_dim_2,
        &unsqueeze_0_0_dim_3,
        &x_dim_0,
        &x_dim_1,
        &unsqueeze_0_0_dim_3,
        &x_dim_2
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
        ss << "\"" << "reshape_3" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"x_dim_0\", \"32\", \"51\", \"1\"]]"
           << ", \"output_sizes\": " << "[[\"x_dim_0\", \"32\", \"1\", \"51\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "softmax_2" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    softmax_2(
       reshape_3_0,
       z,
       &x_dim_0,
       &x_dim_1,
       &unsqueeze_0_0_dim_3,
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
        ss << "\"" << "softmax_2" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"x_dim_0\", \"32\", \"1\", \"51\"]]"
           << ", \"output_sizes\": " << "[[\"x_dim_0\", \"32\", \"1\", \"51\"]]"
        
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
          65280,
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
   void* x {nullptr};
   void* unsqueeze_0_0 {nullptr};
   void* reshape_3_0 {nullptr};
   void* z {nullptr};
   int64_t x_dim_0 { 0 };
   int64_t x_dim_1 { 32 };
   int64_t x_dim_2 { 51 };
   int64_t unsqueeze_0_0_dim_3 { 1 };


};
} // namespace ait