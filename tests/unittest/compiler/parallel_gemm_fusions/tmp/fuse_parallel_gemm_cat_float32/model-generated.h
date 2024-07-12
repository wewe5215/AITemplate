
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


void perm102_bmm_rrr_bias_33(
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

void perm102_bmm_rcr_3(
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

void perm102_bmm_rcr_bias_8(
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

void perm102_bmm_rrr_6(
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

void perm102_bmm_rrr_bias_9(
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

void perm102_bmm_rrr_bias_37(
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
        W = reinterpret_cast<decltype(W)>(constants + 99328);
     constant_name_to_ptr_["W"] = const_cast<const void**>(reinterpret_cast<void**>(&W));
    B = reinterpret_cast<decltype(B)>(constants + 132096);
     constant_name_to_ptr_["B"] = const_cast<const void**>(reinterpret_cast<void**>(&B));
    auto* blob_ptr = static_cast<uint8_t*>(blob_.get());
        permute021_5_0 = reinterpret_cast<decltype(permute021_5_0)>(constants + 132608);
    permute021_27_0 = reinterpret_cast<decltype(permute021_27_0)>(constants + 165376);
    permute021_30_0 = reinterpret_cast<decltype(permute021_30_0)>(constants + 198144);
    concatenate_31_0 = reinterpret_cast<decltype(concatenate_31_0)>(constants + 230912);
    concatenate_35_0 = reinterpret_cast<decltype(concatenate_35_0)>(constants + 231424);
    
         params_[0].shape_ptrs = {ParamDim(128, 256, &input_batch), ParamDim(256, 256, &X1_dim_1)};
     params_[1].shape_ptrs = {ParamDim(128, 256, &input_batch), ParamDim(256, 256, &X2_dim_1)};
     params_[2].shape_ptrs = {ParamDim(128, 256, &input_batch), ParamDim(768, 768, &output0_dim_1)};

      
      
    }

    ~Model() {
      
      
    }

    void SetUpInputsOutputs() {
             X1 = static_cast<decltype(X1)>(params_[0].ptr);

if (X1 == nullptr) {
    throw std::runtime_error("Constant X1 was not set! Set the value with set_constant.");
}
    
     X2 = static_cast<decltype(X2)>(params_[1].ptr);

if (X2 == nullptr) {
    throw std::runtime_error("Constant X2 was not set! Set the value with set_constant.");
}
    
     output0 = static_cast<decltype(output0)>(params_[2].ptr);

if (output0 == nullptr) {
    throw std::runtime_error("Constant output0 was not set! Set the value with set_constant.");
}
    
    }

    void ResetConstants(uint8_t* constants) {
        /*
         * This can be called if we want to use a different piece of memory
         * for the constants to be consumed.
         */
            W = reinterpret_cast<decltype(W)>(constants + 99328);
    B = reinterpret_cast<decltype(B)>(constants + 132096);
    permute021_5_0 = reinterpret_cast<decltype(permute021_5_0)>(constants + 132608);
    permute021_27_0 = reinterpret_cast<decltype(permute021_27_0)>(constants + 165376);
    permute021_30_0 = reinterpret_cast<decltype(permute021_30_0)>(constants + 198144);
    concatenate_31_0 = reinterpret_cast<decltype(concatenate_31_0)>(constants + 230912);
    concatenate_35_0 = reinterpret_cast<decltype(concatenate_35_0)>(constants + 231424);
    }

    void DeviceToDeviceCopies(StreamType stream) {
  
    }


    ///////////////////////////////////////////////////////////////////////////
    // default RunImpl implemenation
    void RunImpl(StreamType stream) {
        
  
  
    {
    

    perm102_bmm_rrr_bias_33(

        X1,
        permute021_27_0,


        concatenate_31_0,

        output0,
        global_workspace_,

        &input_batch,

        &reshape_22_0_dim_1,

        &reshape_22_0_dim_2,


        &reshape_26_0_dim_0,

        &reshape_26_0_dim_2,

        &reshape_26_0_dim_1,


        &input_batch,

        &reshape_22_0_dim_1,

        &reshape_26_0_dim_1,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    perm102_bmm_rcr_3(

        X1,
        W,


        output0,
        global_workspace_,

        &input_batch,

        &reshape_0_0_dim_1,

        &reshape_0_0_dim_2,


        &W_dim_0,

        &W_dim_1,

        &W_dim_2,


        &input_batch,

        &reshape_0_0_dim_1,

        &W_dim_1,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    perm102_bmm_rcr_bias_8(

        X1,
        W,


        B,

        output0,
        global_workspace_,

        &input_batch,

        &reshape_0_0_dim_1,

        &reshape_0_0_dim_2,


        &W_dim_0,

        &W_dim_1,

        &W_dim_2,


        &input_batch,

        &reshape_0_0_dim_1,

        &W_dim_1,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    perm102_bmm_rrr_6(

        X1,
        permute021_5_0,


        output0,
        global_workspace_,

        &input_batch,

        &reshape_0_0_dim_1,

        &reshape_0_0_dim_2,


        &W_dim_0,

        &W_dim_2,

        &W_dim_1,


        &input_batch,

        &reshape_0_0_dim_1,

        &W_dim_1,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    perm102_bmm_rrr_bias_9(

        X1,
        permute021_5_0,


        B,

        output0,
        global_workspace_,

        &input_batch,

        &reshape_0_0_dim_1,

        &reshape_0_0_dim_2,


        &W_dim_0,

        &W_dim_2,

        &W_dim_1,


        &input_batch,

        &reshape_0_0_dim_1,

        &W_dim_1,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    perm102_bmm_rrr_bias_37(

        X2,
        permute021_30_0,


        concatenate_35_0,

        output0,
        global_workspace_,

        &input_batch,

        &reshape_24_0_dim_1,

        &reshape_24_0_dim_2,


        &reshape_29_0_dim_0,

        &reshape_29_0_dim_2,

        &reshape_29_0_dim_1,


        &input_batch,

        &reshape_24_0_dim_1,

        &reshape_29_0_dim_1,

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
        std::cout << "Profiling: " << "perm102_bmm_rrr_bias_33" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    perm102_bmm_rrr_bias_33(

        X1,
        permute021_27_0,


        concatenate_31_0,

        output0,
        global_workspace_,

        &input_batch,

        &reshape_22_0_dim_1,

        &reshape_22_0_dim_2,


        &reshape_26_0_dim_0,

        &reshape_26_0_dim_2,

        &reshape_26_0_dim_1,


        &input_batch,

        &reshape_22_0_dim_1,

        &reshape_26_0_dim_1,

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
        ss << "\"" << "perm102_bmm_rrr_bias_33" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"input_batch\", \"4\", \"64\"], [\"4\", \"64\", \"32\"], [\"4\", \"32\"]]"
           << ", \"output_sizes\": " << "[[\"input_batch\", \"4\", \"32\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "perm102_bmm_rcr_3" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    perm102_bmm_rcr_3(

        X1,
        W,


        output0,
        global_workspace_,

        &input_batch,

        &reshape_0_0_dim_1,

        &reshape_0_0_dim_2,


        &W_dim_0,

        &W_dim_1,

        &W_dim_2,


        &input_batch,

        &reshape_0_0_dim_1,

        &W_dim_1,

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
        ss << "\"" << "perm102_bmm_rcr_3" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"input_batch\", \"4\", \"64\"], [\"4\", \"32\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"input_batch\", \"4\", \"32\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "perm102_bmm_rcr_bias_8" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    perm102_bmm_rcr_bias_8(

        X1,
        W,


        B,

        output0,
        global_workspace_,

        &input_batch,

        &reshape_0_0_dim_1,

        &reshape_0_0_dim_2,


        &W_dim_0,

        &W_dim_1,

        &W_dim_2,


        &input_batch,

        &reshape_0_0_dim_1,

        &W_dim_1,

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
        ss << "\"" << "perm102_bmm_rcr_bias_8" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"input_batch\", \"4\", \"64\"], [\"4\", \"32\", \"64\"], [\"4\", \"32\"]]"
           << ", \"output_sizes\": " << "[[\"input_batch\", \"4\", \"32\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "perm102_bmm_rrr_6" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    perm102_bmm_rrr_6(

        X1,
        permute021_5_0,


        output0,
        global_workspace_,

        &input_batch,

        &reshape_0_0_dim_1,

        &reshape_0_0_dim_2,


        &W_dim_0,

        &W_dim_2,

        &W_dim_1,


        &input_batch,

        &reshape_0_0_dim_1,

        &W_dim_1,

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
        ss << "\"" << "perm102_bmm_rrr_6" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"input_batch\", \"4\", \"64\"], [\"4\", \"64\", \"32\"]]"
           << ", \"output_sizes\": " << "[[\"input_batch\", \"4\", \"32\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "perm102_bmm_rrr_bias_9" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    perm102_bmm_rrr_bias_9(

        X1,
        permute021_5_0,


        B,

        output0,
        global_workspace_,

        &input_batch,

        &reshape_0_0_dim_1,

        &reshape_0_0_dim_2,


        &W_dim_0,

        &W_dim_2,

        &W_dim_1,


        &input_batch,

        &reshape_0_0_dim_1,

        &W_dim_1,

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
        ss << "\"" << "perm102_bmm_rrr_bias_9" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"input_batch\", \"4\", \"64\"], [\"4\", \"64\", \"32\"], [\"4\", \"32\"]]"
           << ", \"output_sizes\": " << "[[\"input_batch\", \"4\", \"32\"]]"
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "perm102_bmm_rrr_bias_37" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
    

    perm102_bmm_rrr_bias_37(

        X2,
        permute021_30_0,


        concatenate_35_0,

        output0,
        global_workspace_,

        &input_batch,

        &reshape_24_0_dim_1,

        &reshape_24_0_dim_2,


        &reshape_29_0_dim_0,

        &reshape_29_0_dim_2,

        &reshape_29_0_dim_1,


        &input_batch,

        &reshape_24_0_dim_1,

        &reshape_29_0_dim_1,

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
        ss << "\"" << "perm102_bmm_rrr_bias_37" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"input_batch\", \"4\", \"64\"], [\"4\", \"64\", \"32\"], [\"4\", \"32\"]]"
           << ", \"output_sizes\": " << "[[\"input_batch\", \"4\", \"32\"]]"
        
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
          1048576,
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
   void* X1 {nullptr};
   void* W {nullptr};
   void* B {nullptr};
   void* X2 {nullptr};
   void* permute021_5_0 {nullptr};
   void* permute021_27_0 {nullptr};
   void* permute021_30_0 {nullptr};
   void* concatenate_31_0 {nullptr};
   void* concatenate_35_0 {nullptr};
   void* output0 {nullptr};
   int64_t W_dim_0 { 4 };
   int64_t W_dim_1 { 32 };
   int64_t W_dim_2 { 64 };
   int64_t W0_dim_0 { 32 };
   int64_t W0_dim_1 { 64 };
   int64_t B0_dim_0 { 32 };
   int64_t W1_dim_0 { 32 };
   int64_t W1_dim_1 { 64 };
   int64_t B1_dim_0 { 32 };
   int64_t W2_dim_0 { 32 };
   int64_t W2_dim_1 { 64 };
   int64_t B2_dim_0 { 32 };
   int64_t W3_dim_0 { 32 };
   int64_t W3_dim_1 { 64 };
   int64_t B3_dim_0 { 32 };
   int64_t W4_dim_0 { 32 };
   int64_t W4_dim_1 { 64 };
   int64_t B4_dim_0 { 32 };
   int64_t W5_dim_0 { 32 };
   int64_t W5_dim_1 { 64 };
   int64_t B5_dim_0 { 32 };
   int64_t W6_dim_0 { 32 };
   int64_t W6_dim_1 { 64 };
   int64_t B6_dim_0 { 32 };
   int64_t W7_dim_0 { 32 };
   int64_t W7_dim_1 { 64 };
   int64_t B7_dim_0 { 32 };
   int64_t input_batch { 0 };
   int64_t X1_dim_1 { 256 };
   int64_t B_dim_0 { 4 };
   int64_t B_dim_1 { 32 };
   int64_t X2_dim_1 { 256 };
   int64_t reshape_26_0_dim_0 { 4 };
   int64_t reshape_26_0_dim_2 { 64 };
   int64_t reshape_26_0_dim_1 { 32 };
   int64_t reshape_29_0_dim_0 { 4 };
   int64_t reshape_29_0_dim_2 { 64 };
   int64_t reshape_29_0_dim_1 { 32 };
   int64_t concatenate_31_0_dim_0 { 128 };
   int64_t concatenate_35_0_dim_0 { 128 };
   int64_t output0_dim_1 { 768 };
   int64_t reshape_22_0_dim_1 { 4 };
   int64_t reshape_22_0_dim_2 { 64 };
   int64_t reshape_32_0_dim_0 { 4 };
   int64_t reshape_32_0_dim_1 { 32 };
   int64_t reshape_0_0_dim_1 { 4 };
   int64_t reshape_0_0_dim_2 { 64 };
   int64_t reshape_24_0_dim_1 { 4 };
   int64_t reshape_24_0_dim_2 { 64 };
   int64_t reshape_36_0_dim_0 { 4 };
   int64_t reshape_36_0_dim_1 { 32 };


};
} // namespace ait