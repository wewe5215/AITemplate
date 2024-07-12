
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


void invoke_fused_elementwise_13(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_14(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_15(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_16(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_17(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_18(void* output0, const void* input0,   int64_t n_elements, cudaStream_t stream);
    

void perm102_bmm_rrr_bias_31(
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

void gemm_rcr_bias_8(
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

void perm102_bmm_rrr_bias_35(
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
        w_2 = reinterpret_cast<decltype(w_2)>(constants + 70656);
     constant_name_to_ptr_["w_2"] = const_cast<const void**>(reinterpret_cast<void**>(&w_2));
    b_2 = reinterpret_cast<decltype(b_2)>(constants + 101376);
     constant_name_to_ptr_["b_2"] = const_cast<const void**>(reinterpret_cast<void**>(&b_2));
    w_5 = reinterpret_cast<decltype(w_5)>(constants + 101632);
     constant_name_to_ptr_["w_5"] = const_cast<const void**>(reinterpret_cast<void**>(&w_5));
    b_5 = reinterpret_cast<decltype(b_5)>(constants + 118016);
     constant_name_to_ptr_["b_5"] = const_cast<const void**>(reinterpret_cast<void**>(&b_5));
    auto* blob_ptr = static_cast<uint8_t*>(blob_.get());
        elementwise_0_0 = reinterpret_cast<decltype(elementwise_0_0)>(blob_ptr + 532480);
    concatenate_19_0 = reinterpret_cast<decltype(concatenate_19_0)>(blob_ptr + 393216);
    concatenate_21_0 = reinterpret_cast<decltype(concatenate_21_0)>(blob_ptr + 466944);
    elementwise_5_0 = reinterpret_cast<decltype(elementwise_5_0)>(blob_ptr + 593920);
    permute021_25_0 = reinterpret_cast<decltype(permute021_25_0)>(constants + 118272);
    permute021_28_0 = reinterpret_cast<decltype(permute021_28_0)>(constants + 155136);
    concatenate_29_0 = reinterpret_cast<decltype(concatenate_29_0)>(constants + 187904);
    concatenate_33_0 = reinterpret_cast<decltype(concatenate_33_0)>(constants + 188416);
    
         params_[0].shape_ptrs = {ParamDim(256, 256, &x_0_dim_0), ParamDim(64, 64, &x_0_dim_1)};
     params_[1].shape_ptrs = {ParamDim(256, 256, &x_0_dim_0), ParamDim(64, 64, &x_1_dim_1)};
     params_[2].shape_ptrs = {ParamDim(256, 256, &x_0_dim_0), ParamDim(120, 120, &x_2_dim_1)};
     params_[3].shape_ptrs = {ParamDim(256, 256, &x_0_dim_0), ParamDim(72, 72, &x_3_dim_1)};
     params_[4].shape_ptrs = {ParamDim(256, 256, &x_0_dim_0), ParamDim(72, 72, &x_4_dim_1)};
     params_[5].shape_ptrs = {ParamDim(256, 256, &x_0_dim_0), ParamDim(64, 64, &x_5_dim_1)};
     params_[6].shape_ptrs = {ParamDim(256, 256, &x_0_dim_0), ParamDim(768, 768, &y_dim_1)};

      
      
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
    
     y = static_cast<decltype(y)>(params_[6].ptr);

if (y == nullptr) {
    throw std::runtime_error("Constant y was not set! Set the value with set_constant.");
}
    
    }

    void ResetConstants(uint8_t* constants) {
        /*
         * This can be called if we want to use a different piece of memory
         * for the constants to be consumed.
         */
            w_2 = reinterpret_cast<decltype(w_2)>(constants + 70656);
    b_2 = reinterpret_cast<decltype(b_2)>(constants + 101376);
    w_5 = reinterpret_cast<decltype(w_5)>(constants + 101632);
    b_5 = reinterpret_cast<decltype(b_5)>(constants + 118016);
    permute021_25_0 = reinterpret_cast<decltype(permute021_25_0)>(constants + 118272);
    permute021_28_0 = reinterpret_cast<decltype(permute021_28_0)>(constants + 155136);
    concatenate_29_0 = reinterpret_cast<decltype(concatenate_29_0)>(constants + 187904);
    concatenate_33_0 = reinterpret_cast<decltype(concatenate_33_0)>(constants + 188416);
    }

    void DeviceToDeviceCopies(StreamType stream) {
  
    }


    ///////////////////////////////////////////////////////////////////////////
    // default RunImpl implemenation
    void RunImpl(StreamType stream) {
        
  
  
    {
        int64_t fused_elementwise_13_n_elements = 256 * 120;
        invoke_fused_elementwise_13(elementwise_0_0, x_2,   fused_elementwise_13_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_14_n_elements = 256 * 72;
        invoke_fused_elementwise_14(concatenate_19_0, x_3,   fused_elementwise_14_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_15_n_elements = 256 * 72;
        invoke_fused_elementwise_15(concatenate_19_0, x_4,   fused_elementwise_15_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_16_n_elements = 256 * 64;
        invoke_fused_elementwise_16(concatenate_21_0, x_0,   fused_elementwise_16_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_17_n_elements = 256 * 64;
        invoke_fused_elementwise_17(concatenate_21_0, x_1,   fused_elementwise_17_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_18_n_elements = 256 * 64;
        invoke_fused_elementwise_18(elementwise_5_0, x_5,   fused_elementwise_18_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    perm102_bmm_rrr_bias_31(

        concatenate_21_0,
        permute021_28_0,


        concatenate_29_0,

        y,
        global_workspace_,

        &reshape_22_0_dim_0,

        &reshape_22_0_dim_1,

        &reshape_22_0_dim_2,


        &reshape_27_0_dim_0,

        &reshape_27_0_dim_2,

        &reshape_27_0_dim_1,


        &reshape_22_0_dim_0,

        &reshape_22_0_dim_1,

        &reshape_27_0_dim_1,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_8(

        elementwise_0_0,
        w_2,

        b_2,

        y,
        global_workspace_,
        1,

        &x_0_dim_0,

        &x_2_dim_1,


        &w_2_dim_0,

        &w_2_dim_1,


        &x_0_dim_0,

        &w_2_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    perm102_bmm_rrr_bias_35(

        concatenate_19_0,
        permute021_25_0,


        concatenate_33_0,

        y,
        global_workspace_,

        &reshape_20_0_dim_0,

        &reshape_20_0_dim_1,

        &reshape_20_0_dim_2,


        &reshape_24_0_dim_0,

        &reshape_24_0_dim_2,

        &reshape_24_0_dim_1,


        &reshape_20_0_dim_0,

        &reshape_20_0_dim_1,

        &reshape_24_0_dim_1,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_11(

        elementwise_5_0,
        w_5,

        b_5,

        y,
        global_workspace_,
        1,

        &x_0_dim_0,

        &x_5_dim_1,


        &w_5_dim_0,

        &w_5_dim_1,


        &x_0_dim_0,

        &w_5_dim_0,

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
        std::cout << "Profiling: " << "fused_elementwise_13" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_13_n_elements = 256 * 120;
        invoke_fused_elementwise_13(elementwise_0_0, x_2,   fused_elementwise_13_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_13" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"120\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"120\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.TANH: 5>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
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
        int64_t fused_elementwise_14_n_elements = 256 * 72;
        invoke_fused_elementwise_14(concatenate_19_0, x_3,   fused_elementwise_14_n_elements, stream);
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
           << ", \"input_sizes\": " << "[[\"256\", \"72\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"72\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.TANH: 5>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_15" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_15_n_elements = 256 * 72;
        invoke_fused_elementwise_15(concatenate_19_0, x_4,   fused_elementwise_15_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_15" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"72\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"72\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.TANH: 5>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_16" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_16_n_elements = 256 * 64;
        invoke_fused_elementwise_16(concatenate_21_0, x_0,   fused_elementwise_16_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_16" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"64\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.TANH: 5>]\""
        
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
        int64_t fused_elementwise_17_n_elements = 256 * 64;
        invoke_fused_elementwise_17(concatenate_21_0, x_1,   fused_elementwise_17_n_elements, stream);
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
           << ", \"input_sizes\": " << "[[\"256\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"64\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.TANH: 5>]\""
        
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
        int64_t fused_elementwise_18_n_elements = 256 * 64;
        invoke_fused_elementwise_18(elementwise_5_0, x_5,   fused_elementwise_18_n_elements, stream);
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
           << ", \"input_sizes\": " << "[[\"256\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"64\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.TANH: 5>]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "perm102_bmm_rrr_bias_31" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    perm102_bmm_rrr_bias_31(

        concatenate_21_0,
        permute021_28_0,


        concatenate_29_0,

        y,
        global_workspace_,

        &reshape_22_0_dim_0,

        &reshape_22_0_dim_1,

        &reshape_22_0_dim_2,


        &reshape_27_0_dim_0,

        &reshape_27_0_dim_2,

        &reshape_27_0_dim_1,


        &reshape_22_0_dim_0,

        &reshape_22_0_dim_1,

        &reshape_27_0_dim_1,

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
        ss << "\"" << "perm102_bmm_rrr_bias_31" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"2\", \"64\"], [\"2\", \"64\", \"128\"], [\"2\", \"128\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"2\", \"128\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_8" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    gemm_rcr_bias_8(

        elementwise_0_0,
        w_2,

        b_2,

        y,
        global_workspace_,
        1,

        &x_0_dim_0,

        &x_2_dim_1,


        &w_2_dim_0,

        &w_2_dim_1,


        &x_0_dim_0,

        &w_2_dim_0,

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
        ss << "\"" << "gemm_rcr_bias_8" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"120\"], [\"128\", \"120\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"128\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "perm102_bmm_rrr_bias_35" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    perm102_bmm_rrr_bias_35(

        concatenate_19_0,
        permute021_25_0,


        concatenate_33_0,

        y,
        global_workspace_,

        &reshape_20_0_dim_0,

        &reshape_20_0_dim_1,

        &reshape_20_0_dim_2,


        &reshape_24_0_dim_0,

        &reshape_24_0_dim_2,

        &reshape_24_0_dim_1,


        &reshape_20_0_dim_0,

        &reshape_20_0_dim_1,

        &reshape_24_0_dim_1,

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
        ss << "\"" << "perm102_bmm_rrr_bias_35" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"256\", \"2\", \"72\"], [\"2\", \"72\", \"128\"], [\"2\", \"128\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"2\", \"128\"]]"
        
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

        elementwise_5_0,
        w_5,

        b_5,

        y,
        global_workspace_,
        1,

        &x_0_dim_0,

        &x_5_dim_1,


        &w_5_dim_0,

        &w_5_dim_1,


        &x_0_dim_0,

        &w_5_dim_0,

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
           << ", \"input_sizes\": " << "[[\"256\", \"64\"], [\"128\", \"64\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"256\", \"128\"]]"
        
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
          626688,
          0 * (1 + 0),
          0 * (1 + 0),
          6,
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
   void* w_2 {nullptr};
   void* b_2 {nullptr};
   void* x_3 {nullptr};
   void* x_4 {nullptr};
   void* x_5 {nullptr};
   void* w_5 {nullptr};
   void* b_5 {nullptr};
   void* elementwise_0_0 {nullptr};
   void* concatenate_19_0 {nullptr};
   void* concatenate_21_0 {nullptr};
   void* elementwise_5_0 {nullptr};
   void* permute021_25_0 {nullptr};
   void* permute021_28_0 {nullptr};
   void* concatenate_29_0 {nullptr};
   void* concatenate_33_0 {nullptr};
   void* y {nullptr};
   int64_t w_0_dim_0 { 128 };
   int64_t w_0_dim_1 { 64 };
   int64_t b_0_dim_0 { 128 };
   int64_t w_1_dim_0 { 128 };
   int64_t w_1_dim_1 { 64 };
   int64_t b_1_dim_0 { 128 };
   int64_t w_3_dim_0 { 128 };
   int64_t w_3_dim_1 { 72 };
   int64_t b_3_dim_0 { 128 };
   int64_t w_4_dim_0 { 128 };
   int64_t w_4_dim_1 { 72 };
   int64_t b_4_dim_0 { 128 };
   int64_t x_0_dim_0 { 256 };
   int64_t x_0_dim_1 { 64 };
   int64_t x_1_dim_1 { 64 };
   int64_t x_2_dim_1 { 120 };
   int64_t w_2_dim_0 { 128 };
   int64_t w_2_dim_1 { 120 };
   int64_t b_2_dim_0 { 128 };
   int64_t x_3_dim_1 { 72 };
   int64_t x_4_dim_1 { 72 };
   int64_t x_5_dim_1 { 64 };
   int64_t w_5_dim_0 { 128 };
   int64_t w_5_dim_1 { 64 };
   int64_t b_5_dim_0 { 128 };
   int64_t concatenate_19_0_dim_1 { 144 };
   int64_t concatenate_21_0_dim_1 { 128 };
   int64_t reshape_24_0_dim_0 { 2 };
   int64_t reshape_24_0_dim_2 { 72 };
   int64_t reshape_24_0_dim_1 { 128 };
   int64_t reshape_27_0_dim_0 { 2 };
   int64_t reshape_27_0_dim_2 { 64 };
   int64_t reshape_27_0_dim_1 { 128 };
   int64_t concatenate_29_0_dim_0 { 256 };
   int64_t concatenate_33_0_dim_0 { 256 };
   int64_t y_dim_1 { 768 };
   int64_t reshape_22_0_dim_0 { 256 };
   int64_t reshape_22_0_dim_1 { 2 };
   int64_t reshape_22_0_dim_2 { 64 };
   int64_t reshape_30_0_dim_0 { 2 };
   int64_t reshape_30_0_dim_1 { 128 };
   int64_t reshape_20_0_dim_0 { 256 };
   int64_t reshape_20_0_dim_1 { 2 };
   int64_t reshape_20_0_dim_2 { 72 };
   int64_t reshape_34_0_dim_0 { 2 };
   int64_t reshape_34_0_dim_1 { 128 };


};
} // namespace ait