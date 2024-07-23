
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


void ait_unsqueeze_1(
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*
);

void conv2d_bias_2(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  cudaStream_t
);

void ait_squeeze_3(
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
         constant_name_to_ptr_["conv1d_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&conv1d_bias));
    auto* blob_ptr = static_cast<uint8_t*>(blob_.get());
        unsqueeze_0_0 = reinterpret_cast<decltype(unsqueeze_0_0)>(constants + 245760);
    
         params_[2].shape_ptrs = {ParamDim(512, 512, &conv1d_weight_dim_0), ParamDim(3, 3, &conv1d_weight_dim_1), ParamDim(80, 80, &conv1d_weight_dim_2)};
     params_[0].shape_ptrs = {ParamDim(4, 4, &input_0_dim_0), ParamDim(28, 28, &input_0_dim_1), ParamDim(80, 80, &input_0_dim_2)};
     params_[3].shape_ptrs = {ParamDim(512, 512, &conv1d_bias_dim_0)};
     params_[1].shape_ptrs = {ParamDim(4, 4, &input_0_dim_0), ParamDim(28, 28, &conv2d_bias_2_0_dim_1), ParamDim(512, 512, &conv2d_bias_2_0_dim_3)};

      
      
    }

    ~Model() {
      
      
    }

    void SetUpInputsOutputs() {
             input_0 = static_cast<decltype(input_0)>(params_[0].ptr);

if (input_0 == nullptr) {
    throw std::runtime_error("Constant input_0 was not set! Set the value with set_constant.");
}
    

if (conv1d_bias == nullptr) {
    throw std::runtime_error("Constant conv1d_bias was not set! Set the value with set_constant.");
}
    
     unsqueeze_1_0 = input_0;
     conv2d_bias_2_0 = static_cast<decltype(conv2d_bias_2_0)>(params_[1].ptr);
     output_0 = conv2d_bias_2_0;

if (output_0 == nullptr) {
    throw std::runtime_error("Constant output_0 was not set! Set the value with set_constant.");
}
    
    }

    void ResetConstants(uint8_t* constants) {
        /*
         * This can be called if we want to use a different piece of memory
         * for the constants to be consumed.
         */
            unsqueeze_0_0 = reinterpret_cast<decltype(unsqueeze_0_0)>(constants + 245760);
    }

    void DeviceToDeviceCopies(StreamType stream) {
  
    }


    ///////////////////////////////////////////////////////////////////////////
    // default RunImpl implemenation
    void RunImpl(StreamType stream) {
        
  
  
    ait_unsqueeze_1(
        &input_0_dim_0,
        &input_0_dim_1,
        &input_0_dim_2,
        &input_0_dim_0,
        &input_0_dim_1,
        &unsqueeze_1_0_dim_2,
        &input_0_dim_2
    );
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    conv2d_bias_2(
        unsqueeze_1_0,
        unsqueeze_0_0,
        conv2d_bias_2_0,

        conv1d_bias,

        global_workspace_,
        &input_0_dim_0,
        &conv1d_weight_dim_0,
        &input_0_dim_2,
        &conv1d_weight_dim_1,
        &unsqueeze_0_0_dim_2,
        &input_0_dim_1,
        &unsqueeze_1_0_dim_2,
        &input_0_dim_0,
        &conv2d_bias_2_0_dim_1,
        &conv2d_bias_2_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        0,
        stream
    );
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    ait_squeeze_3(
        &input_0_dim_0,
        &conv2d_bias_2_0_dim_1,
        &conv2d_bias_2_0_dim_2,
        &conv2d_bias_2_0_dim_3,
        &input_0_dim_0,
        &conv2d_bias_2_0_dim_1,
        &conv2d_bias_2_0_dim_3
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
        std::cout << "Profiling: " << "unsqueeze_1" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    ait_unsqueeze_1(
        &input_0_dim_0,
        &input_0_dim_1,
        &input_0_dim_2,
        &input_0_dim_0,
        &input_0_dim_1,
        &unsqueeze_1_0_dim_2,
        &input_0_dim_2
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
        ss << "\"" << "unsqueeze_1" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"4\", \"28\", \"80\"]]"
           << ", \"output_sizes\": " << "[[\"4\", \"28\", \"1\", \"80\"]]"
        
          << ", \"dim\": " << "\"2\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_2" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    conv2d_bias_2(
        unsqueeze_1_0,
        unsqueeze_0_0,
        conv2d_bias_2_0,

        conv1d_bias,

        global_workspace_,
        &input_0_dim_0,
        &conv1d_weight_dim_0,
        &input_0_dim_2,
        &conv1d_weight_dim_1,
        &unsqueeze_0_0_dim_2,
        &input_0_dim_1,
        &unsqueeze_1_0_dim_2,
        &input_0_dim_0,
        &conv2d_bias_2_0_dim_1,
        &conv2d_bias_2_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        0,
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
        ss << "\"" << "conv2d_bias_2" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"4\", \"28\", \"1\", \"80\"], [\"512\", \"3\", \"1\", \"80\"], [\"512\"]]"
           << ", \"output_sizes\": " << "[[\"4\", \"28\", \"1\", \"512\"]]"
        
          << ", \"dilate\": " << "\"(1, 1)\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"(1, 0)\""
        
          << ", \"stride\": " << "\"(1, 1)\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "squeeze_3" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    ait_squeeze_3(
        &input_0_dim_0,
        &conv2d_bias_2_0_dim_1,
        &conv2d_bias_2_0_dim_2,
        &conv2d_bias_2_0_dim_3,
        &input_0_dim_0,
        &conv2d_bias_2_0_dim_1,
        &conv2d_bias_2_0_dim_3
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
        ss << "\"" << "squeeze_3" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"4\", \"28\", \"1\", \"512\"]]"
           << ", \"output_sizes\": " << "[[\"4\", \"28\", \"512\"]]"
        
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

    static std::unique_ptr<Model> Create(
      AITemplateAllocator& allocator,
      uint8_t* constants
    ) {
      return std::make_unique<Model>(
          132608,
          0 * (1 + 0),
          0 * (1 + 0),
          1,
          1,
          2,
          constants,
          allocator
      );
    }

  private:
   void* input_0 {nullptr};
   void* conv1d_bias {nullptr};
   void* unsqueeze_0_0 {nullptr};
   void* unsqueeze_1_0 {nullptr};
   void* conv2d_bias_2_0 {nullptr};
   void* output_0 {nullptr};
   int64_t conv1d_weight_dim_0 { 512 };
   int64_t conv1d_weight_dim_1 { 3 };
   int64_t conv1d_weight_dim_2 { 80 };
   int64_t input_0_dim_0 { 4 };
   int64_t input_0_dim_1 { 28 };
   int64_t input_0_dim_2 { 80 };
   int64_t conv1d_bias_dim_0 { 512 };
   int64_t unsqueeze_0_0_dim_2 { 1 };
   int64_t unsqueeze_1_0_dim_2 { 1 };
   int64_t conv2d_bias_2_0_dim_1 { 28 };
   int64_t conv2d_bias_2_0_dim_2 { 1 };
   int64_t conv2d_bias_2_0_dim_3 { 512 };


};
} // namespace ait