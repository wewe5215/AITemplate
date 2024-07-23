
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


void ait_unsqueeze_0_constant_folding(
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*
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
         constant_name_to_ptr_["conv1d_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&conv1d_weight));
    auto* blob_ptr = static_cast<uint8_t*>(blob_.get());
    
    
         params_[1].shape_ptrs = {ParamDim(512, 512, &conv1d_weight_dim_0), ParamDim(3, 3, &conv1d_weight_dim_1), ParamDim(80, 80, &conv1d_weight_dim_2)};
     params_[0].shape_ptrs = {ParamDim(512, 512, &conv1d_weight_dim_0), ParamDim(3, 3, &conv1d_weight_dim_1), ParamDim(1, 1, &unsqueeze_0_0_dim_2), ParamDim(80, 80, &conv1d_weight_dim_2)};

      
      
    }

    ~ConstantFolder() {
      
      
    }

    void SetUpInputsOutputs() {
        
if (conv1d_weight == nullptr) {
    throw std::runtime_error("Constant conv1d_weight was not set! Set the value with set_constant.");
}
    

if (params_[0].ptr == nullptr) {
    throw std::runtime_error("Constant unsqueeze_0_0 was not set! Set the value with set_constant.");
}
    
     unsqueeze_0_0 = conv1d_weight;
    }

    void ResetConstants(uint8_t* constants) {
        /*
         * This can be called if we want to use a different piece of memory
         * for the constants to be consumed.
         */
        
    }

    void DeviceToDeviceCopies(StreamType stream) {
  DEVICE_CHECK(DeviceToDeviceCopy(params_[0].ptr, conv1d_weight, 1*512*3*1*80 * 2, stream));
    }


    ///////////////////////////////////////////////////////////////////////////
    // default RunImpl implemenation
    void RunImpl(StreamType stream) {
        
  
  
    ait_unsqueeze_0_constant_folding(
        &conv1d_weight_dim_0,
        &conv1d_weight_dim_1,
        &conv1d_weight_dim_2,
        &conv1d_weight_dim_0,
        &conv1d_weight_dim_1,
        &unsqueeze_0_0_dim_2,
        &conv1d_weight_dim_2
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
            
    ait_unsqueeze_0_constant_folding(
        &conv1d_weight_dim_0,
        &conv1d_weight_dim_1,
        &conv1d_weight_dim_2,
        &conv1d_weight_dim_0,
        &conv1d_weight_dim_1,
        &unsqueeze_0_0_dim_2,
        &conv1d_weight_dim_2
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
           << ", \"input_sizes\": " << "[[\"512\", \"3\", \"80\"]]"
           << ", \"output_sizes\": " << "[[\"512\", \"3\", \"1\", \"80\"]]"
        
          << ", \"dim\": " << "\"2\""
        
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
          245760,
          0 * (1 + 0),
          0 * (1 + 0),
          0,
          1,
          1,
          constants,
          allocator
      );
    }

  private:
   void* conv1d_weight {nullptr};
   void* unsqueeze_0_0 {nullptr};
   int64_t conv1d_weight_dim_0 { 512 };
   int64_t conv1d_weight_dim_1 { 3 };
   int64_t conv1d_weight_dim_2 { 80 };
   int64_t unsqueeze_0_0_dim_2 { 1 };


};
} // namespace ait