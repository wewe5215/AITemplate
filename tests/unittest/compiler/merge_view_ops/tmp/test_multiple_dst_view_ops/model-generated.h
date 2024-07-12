
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


void ait_reshape_5(
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*
);

void ait_reshape_6(
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*
);

void reduce_sum_3(
  void*          /*dst_ptr*/,
  void*          /*src_ptr*/,
  int            /*reduction_axis*/,
  const int64_t* /*shape*/,
  const int      /*rank*/,
  uint8_t*       /*workspace*/,
  cudaStream_t
);

void reduce_sum_4(
  void * /*output*/,
  void * /*input*/,
  int /*reduction_axis*/,
  int64_t *[] /*output_shape*/,
  const int64_t * /*input_shape*/,
  int /*rank*/,
  bool /*keep_dim*/,
  cudaStream_t /*stream*/
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
    
    auto* blob_ptr = static_cast<uint8_t*>(blob_.get());
    
    
         params_[0].shape_ptrs = {ParamDim(2, 2, &x0_dim_0), ParamDim(4, 4, &x0_dim_1), ParamDim(8, 8, &x0_dim_2)};
     params_[1].shape_ptrs = {ParamDim(8, 8, &reshape_0_0_dim_0), ParamDim(8, 8, &reshape_0_0_dim_1)};
     params_[2].shape_ptrs = {ParamDim(8, 8, &reshape_0_0_dim_0), ParamDim(8, 8, &reshape_0_0_dim_1)};

      
      
    }

    ~Model() {
      
      
    }

    void SetUpInputsOutputs() {
             x0 = static_cast<decltype(x0)>(params_[0].ptr);

if (x0 == nullptr) {
    throw std::runtime_error("Constant x0 was not set! Set the value with set_constant.");
}
    
     reshape_5_0 = x0;
     reshape_6_0 = x0;
     y0 = static_cast<decltype(y0)>(params_[1].ptr);

if (y0 == nullptr) {
    throw std::runtime_error("Constant y0 was not set! Set the value with set_constant.");
}
    
     y1 = static_cast<decltype(y1)>(params_[2].ptr);

if (y1 == nullptr) {
    throw std::runtime_error("Constant y1 was not set! Set the value with set_constant.");
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
        
  
  
    ait_reshape_5(
        &x0_dim_0,
        &x0_dim_1,
        &x0_dim_2,
        &reshape_0_0_dim_0,
        &unsqueeze_1_0_dim_1,
        &reshape_0_0_dim_1
    );
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    ait_reshape_6(
        &x0_dim_0,
        &x0_dim_1,
        &x0_dim_2,
        &reshape_0_0_dim_0,
        &reshape_0_0_dim_1,
        &unsqueeze_2_0_dim_2
    );
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
      int64_t shape[] = { reshape_0_0_dim_0,unsqueeze_1_0_dim_1,reshape_0_0_dim_1 };
      reduce_sum_3(
          y0,
          reshape_5_0,
          1,
          shape,
          3,
          global_workspace_,
          stream
      );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
      const int64_t reshape_6_0_shape[] = {
        reshape_0_0_dim_0, reshape_0_0_dim_1, unsqueeze_2_0_dim_2
      };

      int64_t *y1_shape[] = {
        &reshape_0_0_dim_0, &reshape_0_0_dim_1
      };

      reduce_sum_4(
          y1,
          reshape_6_0,
          2, /*reduction_axis*/
          y1_shape,
          reshape_6_0_shape,
          3, /*rank*/
          false, /*keep_dim*/
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
        std::cout << "Profiling: " << "reshape_5" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    ait_reshape_5(
        &x0_dim_0,
        &x0_dim_1,
        &x0_dim_2,
        &reshape_0_0_dim_0,
        &unsqueeze_1_0_dim_1,
        &reshape_0_0_dim_1
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
        ss << "\"" << "reshape_5" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"2\", \"4\", \"8\"]]"
           << ", \"output_sizes\": " << "[[\"8\", \"1\", \"8\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "reshape_6" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    ait_reshape_6(
        &x0_dim_0,
        &x0_dim_1,
        &x0_dim_2,
        &reshape_0_0_dim_0,
        &reshape_0_0_dim_1,
        &unsqueeze_2_0_dim_2
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
        ss << "\"" << "reshape_6" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"2\", \"4\", \"8\"]]"
           << ", \"output_sizes\": " << "[[\"8\", \"8\", \"1\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "reduce_sum_3" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
      int64_t shape[] = { reshape_0_0_dim_0,unsqueeze_1_0_dim_1,reshape_0_0_dim_1 };
      reduce_sum_3(
          y0,
          reshape_5_0,
          1,
          shape,
          3,
          global_workspace_,
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
        ss << "\"" << "reduce_sum_3" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"8\", \"1\", \"8\"]]"
           << ", \"output_sizes\": " << "[[\"8\", \"8\"]]"
        
          << ", \"dim\": " << "\"[1]\""
        
          << ", \"dtype\": " << "\"None\""
        
          << ", \"keepdim\": " << "\"False\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "reduce_sum_4" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
      const int64_t reshape_6_0_shape[] = {
        reshape_0_0_dim_0, reshape_0_0_dim_1, unsqueeze_2_0_dim_2
      };

      int64_t *y1_shape[] = {
        &reshape_0_0_dim_0, &reshape_0_0_dim_1
      };

      reduce_sum_4(
          y1,
          reshape_6_0,
          2, /*reduction_axis*/
          y1_shape,
          reshape_6_0_shape,
          3, /*rank*/
          false, /*keep_dim*/
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
        ss << "\"" << "reduce_sum_4" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"8\", \"8\", \"1\"]]"
           << ", \"output_sizes\": " << "[[\"8\", \"8\"]]"
        
          << ", \"dim\": " << "\"[2]\""
        
          << ", \"dtype\": " << "\"None\""
        
          << ", \"keepdim\": " << "\"False\""
        
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
          768,
          0 * (1 + 0),
          0 * (1 + 0),
          1,
          2,
          0,
          constants,
          allocator
      );
    }

  private:
   void* x0 {nullptr};
   void* reshape_5_0 {nullptr};
   void* reshape_6_0 {nullptr};
   void* y0 {nullptr};
   void* y1 {nullptr};
   int64_t x0_dim_0 { 2 };
   int64_t x0_dim_1 { 4 };
   int64_t x0_dim_2 { 8 };
   int64_t reshape_0_0_dim_0 { 8 };
   int64_t unsqueeze_1_0_dim_1 { 1 };
   int64_t reshape_0_0_dim_1 { 8 };
   int64_t unsqueeze_2_0_dim_2 { 1 };


};
} // namespace ait