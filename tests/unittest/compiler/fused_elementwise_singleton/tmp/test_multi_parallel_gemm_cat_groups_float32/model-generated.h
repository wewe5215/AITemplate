
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


void invoke_fused_elementwise_21(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_22(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_23(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_24(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_25(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_26(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_27(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_28(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void perm102_bmm_rrr_bias_53(
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

  cudaStream_t
);

void perm102_bmm_rrr_bias_57(
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

  cudaStream_t
);

void perm102_bmm_rrr_bias_61(
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

  cudaStream_t
);

void perm102_bmm_rrr_bias_65(
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
    
    auto* blob_ptr = static_cast<uint8_t*>(blob_.get());
        concatenate_31_0 = reinterpret_cast<decltype(concatenate_31_0)>(blob_ptr + 1310720);
    concatenate_33_0 = reinterpret_cast<decltype(concatenate_33_0)>(blob_ptr + 1802240);
    concatenate_35_0 = reinterpret_cast<decltype(concatenate_35_0)>(blob_ptr + 1949696);
    concatenate_37_0 = reinterpret_cast<decltype(concatenate_37_0)>(blob_ptr + 2080768);
    permute021_41_0 = reinterpret_cast<decltype(permute021_41_0)>(constants + 455680);
    permute021_44_0 = reinterpret_cast<decltype(permute021_44_0)>(constants + 701440);
    permute021_47_0 = reinterpret_cast<decltype(permute021_47_0)>(constants + 775168);
    permute021_50_0 = reinterpret_cast<decltype(permute021_50_0)>(constants + 840704);
    concatenate_51_0 = reinterpret_cast<decltype(concatenate_51_0)>(constants + 906240);
    concatenate_55_0 = reinterpret_cast<decltype(concatenate_55_0)>(constants + 907264);
    concatenate_59_0 = reinterpret_cast<decltype(concatenate_59_0)>(constants + 909312);
    concatenate_63_0 = reinterpret_cast<decltype(concatenate_63_0)>(constants + 910336);
    
         params_[0].shape_ptrs = {ParamDim(256, 256, &x_0_dim_0), ParamDim(64, 64, &x_0_dim_1)};
     params_[1].shape_ptrs = {ParamDim(256, 256, &x_0_dim_0), ParamDim(64, 64, &x_1_dim_1)};
     params_[2].shape_ptrs = {ParamDim(256, 256, &x_0_dim_0), ParamDim(120, 120, &x_2_dim_1)};
     params_[3].shape_ptrs = {ParamDim(256, 256, &x_0_dim_0), ParamDim(120, 120, &x_3_dim_1)};
     params_[4].shape_ptrs = {ParamDim(256, 256, &x_0_dim_0), ParamDim(120, 120, &x_4_dim_1)};
     params_[5].shape_ptrs = {ParamDim(256, 256, &x_0_dim_0), ParamDim(120, 120, &x_5_dim_1)};
     params_[6].shape_ptrs = {ParamDim(256, 256, &x_0_dim_0), ParamDim(72, 72, &x_6_dim_1)};
     params_[7].shape_ptrs = {ParamDim(256, 256, &x_0_dim_0), ParamDim(72, 72, &x_7_dim_1)};
     params_[8].shape_ptrs = {ParamDim(256, 256, &x_0_dim_0), ParamDim(64, 64, &x_8_dim_1)};
     params_[9].shape_ptrs = {ParamDim(256, 256, &x_0_dim_0), ParamDim(64, 64, &x_9_dim_1)};
     params_[10].shape_ptrs = {ParamDim(256, 256, &x_0_dim_0), ParamDim(1280, 1280, &y_dim_1)};

      
      
    }

    ~Model() {
      
      
    }

    void SetUpInputsOutputs() {
             x_0 = static_cast<decltype(x_0)>(params_[0].ptr);

if (x_0 == nullptr) {
    throw std::runtime_error("Constant x_0 was not set! Set the value with set_constant.");
}
    
     x_1 = static_cast<decltype(x_1)>(params_[1].ptr);

if (x_1 == nullptr) {
    throw std::runtime_error("Constant x_1 was not set! Set the value with set_constant.");
}
    
     x_2 = static_cast<decltype(x_2)>(params_[2].ptr);

if (x_2 == nullptr) {
    throw std::runtime_error("Constant x_2 was not set! Set the value with set_constant.");
}
    
     x_3 = static_cast<decltype(x_3)>(params_[3].ptr);

if (x_3 == nullptr) {
    throw std::runtime_error("Constant x_3 was not set! Set the value with set_constant.");
}
    
     x_4 = static_cast<decltype(x_4)>(params_[4].ptr);

if (x_4 == nullptr) {
    throw std::runtime_error("Constant x_4 was not set! Set the value with set_constant.");
}
    
     x_5 = static_cast<decltype(x_5)>(params_[5].ptr);

if (x_5 == nullptr) {
    throw std::runtime_error("Constant x_5 was not set! Set the value with set_constant.");
}
    
     x_6 = static_cast<decltype(x_6)>(params_[6].ptr);

if (x_6 == nullptr) {
    throw std::runtime_error("Constant x_6 was not set! Set the value with set_constant.");
}
    
     x_7 = static_cast<decltype(x_7)>(params_[7].ptr);

if (x_7 == nullptr) {
    throw std::runtime_error("Constant x_7 was not set! Set the value with set_constant.");
}
    
     x_8 = static_cast<decltype(x_8)>(params_[8].ptr);

if (x_8 == nullptr) {
    throw std::runtime_error("Constant x_8 was not set! Set the value with set_constant.");
}
    
     x_9 = static_cast<decltype(x_9)>(params_[9].ptr);

if (x_9 == nullptr) {
    throw std::runtime_error("Constant x_9 was not set! Set the value with set_constant.");
}
    
     y = static_cast<decltype(y)>(params_[10].ptr);

if (y == nullptr) {
    throw std::runtime_error("Constant y was not set! Set the value with set_constant.");
}
    
    }

    void ResetConstants(uint8_t* constants) {
        /*
         * This can be called if we want to use a different piece of memory
         * for the constants to be consumed.
         */
            permute021_41_0 = reinterpret_cast<decltype(permute021_41_0)>(constants + 455680);
    permute021_44_0 = reinterpret_cast<decltype(permute021_44_0)>(constants + 701440);
    permute021_47_0 = reinterpret_cast<decltype(permute021_47_0)>(constants + 775168);
    permute021_50_0 = reinterpret_cast<decltype(permute021_50_0)>(constants + 840704);
    concatenate_51_0 = reinterpret_cast<decltype(concatenate_51_0)>(constants + 906240);
    concatenate_55_0 = reinterpret_cast<decltype(concatenate_55_0)>(constants + 907264);
    concatenate_59_0 = reinterpret_cast<decltype(concatenate_59_0)>(constants + 909312);
    concatenate_63_0 = reinterpret_cast<decltype(concatenate_63_0)>(constants + 910336);
    }

    void DeviceToDeviceCopies(StreamType stream) {
  
    }


    ///////////////////////////////////////////////////////////////////////////
    // default RunImpl implemenation
    void RunImpl(StreamType stream) {
        
  
  
    {
        int64_t fused_elementwise_21_n_elements = 256 * 120;
        invoke_fused_elementwise_21(concatenate_31_0, x_2,   fused_elementwise_21_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_22_n_elements = 256 * 120;
        invoke_fused_elementwise_22(concatenate_31_0, x_3,   fused_elementwise_22_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_23_n_elements = 256 * 120;
        invoke_fused_elementwise_23(concatenate_31_0, x_4,   fused_elementwise_23_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_24_n_elements = 256 * 120;
        invoke_fused_elementwise_24(concatenate_31_0, x_5,   fused_elementwise_24_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_25_n_elements = 256 * 72;
        invoke_fused_elementwise_25(concatenate_33_0, x_6,   fused_elementwise_25_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_26_n_elements = 256 * 72;
        invoke_fused_elementwise_26(concatenate_33_0, x_7,   fused_elementwise_26_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_27_n_elements = 256 * 64;
        invoke_fused_elementwise_27(concatenate_35_0, x_0,   fused_elementwise_27_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_28_n_elements = 256 * 64;
        invoke_fused_elementwise_28(concatenate_35_0, x_1,   fused_elementwise_28_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_27_n_elements = 256 * 64;
        invoke_fused_elementwise_27(concatenate_37_0, x_8,   fused_elementwise_27_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_28_n_elements = 256 * 64;
        invoke_fused_elementwise_28(concatenate_37_0, x_9,   fused_elementwise_28_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    perm102_bmm_rrr_bias_53(

        concatenate_35_0,
        permute021_47_0,


        concatenate_51_0,

        y,
        global_workspace_,

        &reshape_36_0_dim_0,

        &reshape_36_0_dim_1,

        &reshape_36_0_dim_2,


        &reshape_46_0_dim_0,

        &reshape_46_0_dim_2,

        &reshape_46_0_dim_1,


        &reshape_36_0_dim_0,

        &reshape_36_0_dim_1,

        &reshape_46_0_dim_1,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    perm102_bmm_rrr_bias_57(

        concatenate_31_0,
        permute021_41_0,


        concatenate_55_0,

        y,
        global_workspace_,

        &reshape_32_0_dim_0,

        &reshape_32_0_dim_1,

        &reshape_32_0_dim_2,


        &reshape_40_0_dim_0,

        &reshape_40_0_dim_2,

        &reshape_40_0_dim_1,


        &reshape_32_0_dim_0,

        &reshape_32_0_dim_1,

        &reshape_40_0_dim_1,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    perm102_bmm_rrr_bias_61(

        concatenate_33_0,
        permute021_44_0,


        concatenate_59_0,

        y,
        global_workspace_,

        &reshape_34_0_dim_0,

        &reshape_34_0_dim_1,

        &reshape_34_0_dim_2,


        &reshape_43_0_dim_0,

        &reshape_43_0_dim_2,

        &reshape_43_0_dim_1,


        &reshape_34_0_dim_0,

        &reshape_34_0_dim_1,

        &reshape_43_0_dim_1,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    perm102_bmm_rrr_bias_65(

        concatenate_37_0,
        permute021_50_0,


        concatenate_63_0,

        y,
        global_workspace_,

        &reshape_38_0_dim_0,

        &reshape_38_0_dim_1,

        &reshape_38_0_dim_2,


        &reshape_49_0_dim_0,

        &reshape_49_0_dim_2,

        &reshape_49_0_dim_1,


        &reshape_38_0_dim_0,

        &reshape_38_0_dim_1,

        &reshape_49_0_dim_1,

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
        std::cout << "Profiling: " << "fused_elementwise_21" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_21_n_elements = 256 * 120;
        invoke_fused_elementwise_21(concatenate_31_0, x_2,   fused_elementwise_21_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_21" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"120\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"120\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.TANH: 5>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_22" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_22_n_elements = 256 * 120;
        invoke_fused_elementwise_22(concatenate_31_0, x_3,   fused_elementwise_22_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_22" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"120\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"120\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.TANH: 5>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_23" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_23_n_elements = 256 * 120;
        invoke_fused_elementwise_23(concatenate_31_0, x_4,   fused_elementwise_23_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_23" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"120\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"120\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.TANH: 5>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_24" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_24_n_elements = 256 * 120;
        invoke_fused_elementwise_24(concatenate_31_0, x_5,   fused_elementwise_24_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_24" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"120\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"120\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.TANH: 5>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_25" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_25_n_elements = 256 * 72;
        invoke_fused_elementwise_25(concatenate_33_0, x_6,   fused_elementwise_25_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_25" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"72\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"72\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.TANH: 5>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_26" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_26_n_elements = 256 * 72;
        invoke_fused_elementwise_26(concatenate_33_0, x_7,   fused_elementwise_26_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_26" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"72\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"72\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.TANH: 5>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_27" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_27_n_elements = 256 * 64;
        invoke_fused_elementwise_27(concatenate_35_0, x_0,   fused_elementwise_27_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_27" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"64\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.TANH: 5>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_28" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_28_n_elements = 256 * 64;
        invoke_fused_elementwise_28(concatenate_35_0, x_1,   fused_elementwise_28_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_28" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"64\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.TANH: 5>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_29" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_27_n_elements = 256 * 64;
        invoke_fused_elementwise_27(concatenate_37_0, x_8,   fused_elementwise_27_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_29" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"64\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.TANH: 5>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_30" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_28_n_elements = 256 * 64;
        invoke_fused_elementwise_28(concatenate_37_0, x_9,   fused_elementwise_28_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_30" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"64\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.TANH: 5>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "perm102_bmm_rrr_bias_53" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    perm102_bmm_rrr_bias_53(

        concatenate_35_0,
        permute021_47_0,


        concatenate_51_0,

        y,
        global_workspace_,

        &reshape_36_0_dim_0,

        &reshape_36_0_dim_1,

        &reshape_36_0_dim_2,


        &reshape_46_0_dim_0,

        &reshape_46_0_dim_2,

        &reshape_46_0_dim_1,


        &reshape_36_0_dim_0,

        &reshape_36_0_dim_1,

        &reshape_46_0_dim_1,

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
        ss << "\"" << "perm102_bmm_rrr_bias_53" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"2\", \"64\"], [\"2\", \"64\", \"128\"], [\"2\", \"128\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"2\", \"128\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "perm102_bmm_rrr_bias_57" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    perm102_bmm_rrr_bias_57(

        concatenate_31_0,
        permute021_41_0,


        concatenate_55_0,

        y,
        global_workspace_,

        &reshape_32_0_dim_0,

        &reshape_32_0_dim_1,

        &reshape_32_0_dim_2,


        &reshape_40_0_dim_0,

        &reshape_40_0_dim_2,

        &reshape_40_0_dim_1,


        &reshape_32_0_dim_0,

        &reshape_32_0_dim_1,

        &reshape_40_0_dim_1,

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
        ss << "\"" << "perm102_bmm_rrr_bias_57" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"4\", \"120\"], [\"4\", \"120\", \"128\"], [\"4\", \"128\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"4\", \"128\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "perm102_bmm_rrr_bias_61" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    perm102_bmm_rrr_bias_61(

        concatenate_33_0,
        permute021_44_0,


        concatenate_59_0,

        y,
        global_workspace_,

        &reshape_34_0_dim_0,

        &reshape_34_0_dim_1,

        &reshape_34_0_dim_2,


        &reshape_43_0_dim_0,

        &reshape_43_0_dim_2,

        &reshape_43_0_dim_1,


        &reshape_34_0_dim_0,

        &reshape_34_0_dim_1,

        &reshape_43_0_dim_1,

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
        ss << "\"" << "perm102_bmm_rrr_bias_61" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"2\", \"72\"], [\"2\", \"72\", \"128\"], [\"2\", \"128\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"2\", \"128\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "perm102_bmm_rrr_bias_65" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    perm102_bmm_rrr_bias_65(

        concatenate_37_0,
        permute021_50_0,


        concatenate_63_0,

        y,
        global_workspace_,

        &reshape_38_0_dim_0,

        &reshape_38_0_dim_1,

        &reshape_38_0_dim_2,


        &reshape_49_0_dim_0,

        &reshape_49_0_dim_2,

        &reshape_49_0_dim_1,


        &reshape_38_0_dim_0,

        &reshape_38_0_dim_1,

        &reshape_49_0_dim_1,

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
        ss << "\"" << "perm102_bmm_rrr_bias_65" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"2\", \"64\"], [\"2\", \"64\", \"128\"], [\"2\", \"128\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"2\", \"128\"]]"
        
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
          2211840,
          0 * (1 + 0),
          0 * (1 + 0),
          10,
          1,
          0,
          constants,
          allocator
      );
    }

  private:
   void* x_0 {nullptr};
   void* x_1 {nullptr};
   void* x_2 {nullptr};
   void* x_3 {nullptr};
   void* x_4 {nullptr};
   void* x_5 {nullptr};
   void* x_6 {nullptr};
   void* x_7 {nullptr};
   void* x_8 {nullptr};
   void* x_9 {nullptr};
   void* concatenate_31_0 {nullptr};
   void* concatenate_33_0 {nullptr};
   void* concatenate_35_0 {nullptr};
   void* concatenate_37_0 {nullptr};
   void* permute021_41_0 {nullptr};
   void* permute021_44_0 {nullptr};
   void* permute021_47_0 {nullptr};
   void* permute021_50_0 {nullptr};
   void* concatenate_51_0 {nullptr};
   void* concatenate_55_0 {nullptr};
   void* concatenate_59_0 {nullptr};
   void* concatenate_63_0 {nullptr};
   void* y {nullptr};
   int64_t w_0_dim_0 { 128 };
   int64_t w_0_dim_1 { 64 };
   int64_t b_0_dim_0 { 128 };
   int64_t w_1_dim_0 { 128 };
   int64_t w_1_dim_1 { 64 };
   int64_t b_1_dim_0 { 128 };
   int64_t w_2_dim_0 { 128 };
   int64_t w_2_dim_1 { 120 };
   int64_t b_2_dim_0 { 128 };
   int64_t w_3_dim_0 { 128 };
   int64_t w_3_dim_1 { 120 };
   int64_t b_3_dim_0 { 128 };
   int64_t w_4_dim_0 { 128 };
   int64_t w_4_dim_1 { 120 };
   int64_t b_4_dim_0 { 128 };
   int64_t w_5_dim_0 { 128 };
   int64_t w_5_dim_1 { 120 };
   int64_t b_5_dim_0 { 128 };
   int64_t w_6_dim_0 { 128 };
   int64_t w_6_dim_1 { 72 };
   int64_t b_6_dim_0 { 128 };
   int64_t w_7_dim_0 { 128 };
   int64_t w_7_dim_1 { 72 };
   int64_t b_7_dim_0 { 128 };
   int64_t w_8_dim_0 { 128 };
   int64_t w_8_dim_1 { 64 };
   int64_t b_8_dim_0 { 128 };
   int64_t w_9_dim_0 { 128 };
   int64_t w_9_dim_1 { 64 };
   int64_t b_9_dim_0 { 128 };
   int64_t x_0_dim_0 { 256 };
   int64_t x_0_dim_1 { 64 };
   int64_t x_1_dim_1 { 64 };
   int64_t x_2_dim_1 { 120 };
   int64_t x_3_dim_1 { 120 };
   int64_t x_4_dim_1 { 120 };
   int64_t x_5_dim_1 { 120 };
   int64_t x_6_dim_1 { 72 };
   int64_t x_7_dim_1 { 72 };
   int64_t x_8_dim_1 { 64 };
   int64_t x_9_dim_1 { 64 };
   int64_t concatenate_31_0_dim_1 { 480 };
   int64_t concatenate_33_0_dim_1 { 144 };
   int64_t concatenate_35_0_dim_1 { 128 };
   int64_t concatenate_37_0_dim_1 { 128 };
   int64_t reshape_40_0_dim_0 { 4 };
   int64_t reshape_40_0_dim_2 { 120 };
   int64_t reshape_40_0_dim_1 { 128 };
   int64_t reshape_43_0_dim_0 { 2 };
   int64_t reshape_43_0_dim_2 { 72 };
   int64_t reshape_43_0_dim_1 { 128 };
   int64_t reshape_46_0_dim_0 { 2 };
   int64_t reshape_46_0_dim_2 { 64 };
   int64_t reshape_46_0_dim_1 { 128 };
   int64_t reshape_49_0_dim_0 { 2 };
   int64_t reshape_49_0_dim_2 { 64 };
   int64_t reshape_49_0_dim_1 { 128 };
   int64_t concatenate_51_0_dim_0 { 256 };
   int64_t concatenate_55_0_dim_0 { 512 };
   int64_t concatenate_59_0_dim_0 { 256 };
   int64_t concatenate_63_0_dim_0 { 256 };
   int64_t y_dim_1 { 1280 };
   int64_t reshape_36_0_dim_0 { 256 };
   int64_t reshape_36_0_dim_1 { 2 };
   int64_t reshape_36_0_dim_2 { 64 };
   int64_t reshape_52_0_dim_0 { 2 };
   int64_t reshape_52_0_dim_1 { 128 };
   int64_t reshape_32_0_dim_0 { 256 };
   int64_t reshape_32_0_dim_1 { 4 };
   int64_t reshape_32_0_dim_2 { 120 };
   int64_t reshape_56_0_dim_0 { 4 };
   int64_t reshape_56_0_dim_1 { 128 };
   int64_t reshape_34_0_dim_0 { 256 };
   int64_t reshape_34_0_dim_1 { 2 };
   int64_t reshape_34_0_dim_2 { 72 };
   int64_t reshape_60_0_dim_0 { 2 };
   int64_t reshape_60_0_dim_1 { 128 };
   int64_t reshape_38_0_dim_0 { 256 };
   int64_t reshape_38_0_dim_1 { 2 };
   int64_t reshape_38_0_dim_2 { 64 };
   int64_t reshape_64_0_dim_0 { 2 };
   int64_t reshape_64_0_dim_1 { 128 };


};
} // namespace ait