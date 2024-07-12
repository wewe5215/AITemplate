
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


void concatenate_1(
    void * /*output*/,
    int64_t *[] /*output_shape*/,
    const void *[] /*inputs*/,
    const int64_t *[], /* real_input_shapes, representing shapes of those inputs
                                 whose masks are False, i.e. inputs that will be
                                 copied to the output tensor by concat.*/
    const int64_t *[], /* all_input_shapes, including both kinds of inputs,
                                 i.e. not matter input_mask being True or False */
    const bool [] /*input_masks*/,
    const int64_t [] /*concat_dim_sizes*/,
    int64_t /*concat_dim*/,
    int64_t /*rank*/,
    int64_t /*num_real_inputs*/,
    int64_t /*num_all_inputs*/,
    cudaStream_t
);

void invoke_fused_elementwise_6(void* output0, const void* input0,const void* input1, int64_t batch_0,   int64_t n_elements, cudaStream_t stream);
    

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
  void*          /*dst_ptr*/,
  void*          /*src_ptr*/,
  int            /*reduction_axis*/,
  const int64_t* /*shape*/,
  const int      /*rank*/,
  uint8_t*       /*workspace*/,
  cudaStream_t
);

void invoke_fused_elementwise_7(void* output0, const void* input0,const void* input1, int64_t batch_0,   int64_t n_elements, cudaStream_t stream);
    

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
        concatenate_1_0 = reinterpret_cast<decltype(concatenate_1_0)>(blob_ptr + 0);
    elementwise_2_0 = reinterpret_cast<decltype(elementwise_2_0)>(blob_ptr + 229376);
    reduce_sum_3_0 = reinterpret_cast<decltype(reduce_sum_3_0)>(blob_ptr + 131072);
    reduce_sum_4_0 = reinterpret_cast<decltype(reduce_sum_4_0)>(blob_ptr + 147456);
    
         params_[0].shape_ptrs = {ParamDim(1, 1024, &batch_0), ParamDim(2, 2, &x0_dim_1), ParamDim(8, 8, &x0_dim_2)};
     params_[1].shape_ptrs = {ParamDim(1, 1024, &batch_0), ParamDim(4, 4, &x1_dim_1), ParamDim(8, 8, &x1_dim_2)};
     params_[2].shape_ptrs = {ParamDim(1, 1024, &batch_0), ParamDim(6, 6, &x2_dim_1), ParamDim(8, 8, &x2_dim_2)};
     params_[3].shape_ptrs = {ParamDim(1, 1024, &batch_0), ParamDim(8, 8, &x0_dim_2)};

      
      
    }

    ~Model() {
      
      
    }

    void SetUpInputsOutputs() {
             x0 = static_cast<decltype(x0)>(params_[0].ptr);

if (x0 == nullptr) {
    throw std::runtime_error("Constant x0 was not set! Set the value with set_constant.");
}
    
     x1 = static_cast<decltype(x1)>(params_[1].ptr);

if (x1 == nullptr) {
    throw std::runtime_error("Constant x1 was not set! Set the value with set_constant.");
}
    
     x2 = static_cast<decltype(x2)>(params_[2].ptr);

if (x2 == nullptr) {
    throw std::runtime_error("Constant x2 was not set! Set the value with set_constant.");
}
    
     output0 = static_cast<decltype(output0)>(params_[3].ptr);

if (output0 == nullptr) {
    throw std::runtime_error("Constant output0 was not set! Set the value with set_constant.");
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

    const void *inputs[] = {
      x0,
      x0,
      x1
    };


      int64_t x0_shape_0[] = {
        batch_0, x0_dim_1, x0_dim_2
      };
      int64_t x0_shape_1[] = {
        batch_0, x0_dim_1, x0_dim_2
      };
      int64_t x1_shape_2[] = {
        batch_0, x1_dim_1, x1_dim_2
      };

    const int64_t *real_input_shapes[] = {
      x0_shape_0, x0_shape_1, x1_shape_2
    };



    const int64_t *all_input_shapes[] = {
      x0_shape_0, x0_shape_1, x1_shape_2
    };

    int64_t *concatenate_1_0_shape[] = {
      &batch_0, &concatenate_1_0_dim_1, &x0_dim_2
    };

    int64_t concat_dim_sizes[] = {
      x0_dim_1, x0_dim_1, x1_dim_1
    };

    bool input_masks[] = {
      true, true, true
    };

    concatenate_1(
        concatenate_1_0,
        concatenate_1_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        1/*concat_dim*/,
        3/*rank*/,
        3/*num_real_inputs*/,
        3/*num_all_inputs*/,
        stream
    );
  }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_6_n_elements = batch_0 * 6 * 8;
        invoke_fused_elementwise_6(elementwise_2_0, concatenate_1_0,x2, batch_0,   fused_elementwise_6_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
      int64_t shape[] = { batch_0,concatenate_1_0_dim_1,x0_dim_2 };
      reduce_sum_3(
          reduce_sum_3_0,
          concatenate_1_0,
          1,
          shape,
          3,
          global_workspace_,
          stream
      );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
      int64_t shape[] = { batch_0,concatenate_0_0_dim_1,x0_dim_2 };
      reduce_sum_4(
          reduce_sum_4_0,
          elementwise_2_0,
          1,
          shape,
          3,
          global_workspace_,
          stream
      );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int64_t fused_elementwise_7_n_elements = batch_0 * 8;
        invoke_fused_elementwise_7(output0, reduce_sum_4_0,reduce_sum_3_0, batch_0,   fused_elementwise_7_n_elements, stream);
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
        std::cout << "Profiling: " << "concatenate_1" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
  {

    const void *inputs[] = {
      x0,
      x0,
      x1
    };


      int64_t x0_shape_0[] = {
        batch_0, x0_dim_1, x0_dim_2
      };
      int64_t x0_shape_1[] = {
        batch_0, x0_dim_1, x0_dim_2
      };
      int64_t x1_shape_2[] = {
        batch_0, x1_dim_1, x1_dim_2
      };

    const int64_t *real_input_shapes[] = {
      x0_shape_0, x0_shape_1, x1_shape_2
    };



    const int64_t *all_input_shapes[] = {
      x0_shape_0, x0_shape_1, x1_shape_2
    };

    int64_t *concatenate_1_0_shape[] = {
      &batch_0, &concatenate_1_0_dim_1, &x0_dim_2
    };

    int64_t concat_dim_sizes[] = {
      x0_dim_1, x0_dim_1, x1_dim_1
    };

    bool input_masks[] = {
      true, true, true
    };

    concatenate_1(
        concatenate_1_0,
        concatenate_1_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        1/*concat_dim*/,
        3/*rank*/,
        3/*num_real_inputs*/,
        3/*num_all_inputs*/,
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
        ss << "\"" << "concatenate_1" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"batch_0\", \"2\", \"8\"], [\"batch_0\", \"2\", \"8\"], [\"batch_0\", \"4\", \"8\"]]"
           << ", \"output_sizes\": " << "[[\"batch_0\", \"8\", \"8\"]]"
        
          << ", \"dim\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "fused_elementwise_6" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
    {
        int64_t fused_elementwise_6_n_elements = batch_0 * 6 * 8;
        invoke_fused_elementwise_6(elementwise_2_0, concatenate_1_0,x2, batch_0,   fused_elementwise_6_n_elements, stream);
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
        ss << "\"" << "fused_elementwise_6" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"batch_0\", \"6\", \"8\"], [\"batch_0\", \"6\", \"8\"]]"
           << ", \"output_sizes\": " << "[[\"batch_0\", \"6\", \"8\"]]"
        
          << ", \"func\": " << "\"[<FuncEnum.ADD: 1>]\""
        
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
      int64_t shape[] = { batch_0,concatenate_1_0_dim_1,x0_dim_2 };
      reduce_sum_3(
          reduce_sum_3_0,
          concatenate_1_0,
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
           << ", \"input_sizes\": " << "[[\"batch_0\", \"8\", \"8\"]]"
           << ", \"output_sizes\": " << "[[\"batch_0\", \"8\"]]"
        
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
      int64_t shape[] = { batch_0,concatenate_0_0_dim_1,x0_dim_2 };
      reduce_sum_4(
          reduce_sum_4_0,
          elementwise_2_0,
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
        ss << "\"" << "reduce_sum_4" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"batch_0\", \"6\", \"8\"]]"
           << ", \"output_sizes\": " << "[[\"batch_0\", \"8\"]]"
        
          << ", \"dim\": " << "\"[1]\""
        
          << ", \"dtype\": " << "\"None\""
        
          << ", \"keepdim\": " << "\"False\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
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
        int64_t fused_elementwise_7_n_elements = batch_0 * 8;
        invoke_fused_elementwise_7(output0, reduce_sum_4_0,reduce_sum_3_0, batch_0,   fused_elementwise_7_n_elements, stream);
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
           << ", \"input_sizes\": " << "[[\"batch_0\", \"8\"], [\"batch_0\", \"8\"]]"
           << ", \"output_sizes\": " << "[[\"batch_0\", \"8\"]]"
        
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
          327680,
          0 * (1 + 0),
          0 * (1 + 0),
          3,
          1,
          0,
          constants,
          allocator
      );
    }

  private:
   void* x0 {nullptr};
   void* x1 {nullptr};
   void* x2 {nullptr};
   void* concatenate_1_0 {nullptr};
   void* elementwise_2_0 {nullptr};
   void* reduce_sum_3_0 {nullptr};
   void* reduce_sum_4_0 {nullptr};
   void* output0 {nullptr};
   int64_t batch_0 { 0 };
   int64_t x0_dim_1 { 2 };
   int64_t x0_dim_2 { 8 };
   int64_t x1_dim_1 { 4 };
   int64_t x1_dim_2 { 8 };
   int64_t x2_dim_1 { 6 };
   int64_t x2_dim_2 { 8 };
   int64_t concatenate_1_0_dim_1 { 8 };
   int64_t concatenate_0_0_dim_1 { 6 };
   int64_t dynamic_slice_0_concatenate_0_0_dim_1 { 6 };


};
} // namespace ait