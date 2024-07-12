
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


void concatenate_3(
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

void split_4(
    void *[] /*outputs*/,
    int64_t **[] /*output_shapes*/,
    const bool [] /*output_masks*/,
    const void * /*input*/,
    const int64_t * /*input_shape*/,
    int64_t /*real_num_splits*/,
    int64_t /*all_num_splits*/,
    int64_t [] /*split_sizes*/,
    int64_t /*split_dim*/,
    int64_t /*rank*/,
    cudaStream_t stream
);

void concatenate_5(
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
        split_4_0 = reinterpret_cast<decltype(split_4_0)>(blob_ptr + 10485760);
    split_4_1 = reinterpret_cast<decltype(split_4_1)>(blob_ptr + 11796480);
    
         params_[0].shape_ptrs = {ParamDim(1024, 1024, &input0_dim_0), ParamDim(10, 10, &input0_dim_1), ParamDim(128, 128, &input0_dim_2)};
     params_[1].shape_ptrs = {ParamDim(1024, 1024, &input1_dim_0), ParamDim(10, 10, &input1_dim_1), ParamDim(128, 128, &input1_dim_2)};
     params_[2].shape_ptrs = {ParamDim(1024, 1024, &input2_dim_0), ParamDim(10, 10, &input2_dim_1), ParamDim(128, 128, &input2_dim_2)};
     params_[4].shape_ptrs = {ParamDim(1024, 1024, &input2_dim_0), ParamDim(30, 30, &output1_dim_1), ParamDim(128, 128, &input2_dim_2)};
     params_[3].shape_ptrs = {ParamDim(1024, 1024, &input0_dim_0), ParamDim(20, 20, &output0_dim_1), ParamDim(64, 64, &split_4_0_dim_2)};

      
      
    }

    ~Model() {
      
      
    }

    void SetUpInputsOutputs() {
             input0 = static_cast<decltype(input0)>(params_[0].ptr);

if (input0 == nullptr) {
    throw std::runtime_error("Constant input0 was not set! Set the value with set_constant.");
}
    
     input1 = static_cast<decltype(input1)>(params_[1].ptr);

if (input1 == nullptr) {
    throw std::runtime_error("Constant input1 was not set! Set the value with set_constant.");
}
    
     input2 = static_cast<decltype(input2)>(params_[2].ptr);

if (input2 == nullptr) {
    throw std::runtime_error("Constant input2 was not set! Set the value with set_constant.");
}
    
     output1 = static_cast<decltype(output1)>(params_[4].ptr);

if (output1 == nullptr) {
    throw std::runtime_error("Constant output1 was not set! Set the value with set_constant.");
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
      input2,
      input1,
      input1
    };


      int64_t input2_shape_0[] = {
        input2_dim_0, input2_dim_1, input2_dim_2
      };
      int64_t input1_shape_1[] = {
        input1_dim_0, input1_dim_1, input1_dim_2
      };
      int64_t input1_shape_2[] = {
        input1_dim_0, input1_dim_1, input1_dim_2
      };

    const int64_t *real_input_shapes[] = {
      input2_shape_0, input1_shape_1, input1_shape_2
    };



    const int64_t *all_input_shapes[] = {
      input2_shape_0, input1_shape_1, input1_shape_2
    };

    int64_t *output1_shape[] = {
      &input2_dim_0, &output1_dim_1, &input2_dim_2
    };

    int64_t concat_dim_sizes[] = {
      input2_dim_1, input1_dim_1, input1_dim_1
    };

    bool input_masks[] = {
      true, true, true
    };

    concatenate_3(
        output1,
        output1_shape,
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

    void *outputs[] = {
      split_4_0,
      split_4_1
    };


      int64_t *split_4_0_shape[] = {
        &input0_dim_0, &concatenate_1_0_dim_1, &split_4_0_dim_2
      };
      int64_t *split_4_1_shape[] = {
        &input0_dim_0, &concatenate_1_0_dim_1, &split_4_1_dim_2
      };

    int64_t **output_shapes[] = {
      split_4_0_shape, split_4_1_shape
    };

    const int64_t input0_shape[] = {
      input0_dim_0, input0_dim_1, input0_dim_2
    };

    int64_t split_sizes[] = {
      64, 64
    };

    bool output_masks[] = {
      true, true
    };

    split_4(
        outputs,
        output_shapes,
        output_masks,
        input0,
        input0_shape,
        2/*real_num_splits*/,
        2/*all_num_splits*/,
        split_sizes,
        2/*split_dim*/,
        3/*rank*/,
        stream
    );
  }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    const void *inputs[] = {
      split_4_0,
      split_4_1
    };


      int64_t split_4_0_shape_0[] = {
        input0_dim_0, concatenate_1_0_dim_1, split_4_0_dim_2
      };
      int64_t split_4_1_shape_1[] = {
        input0_dim_0, concatenate_1_0_dim_1, split_4_1_dim_2
      };

    const int64_t *real_input_shapes[] = {
      split_4_0_shape_0, split_4_1_shape_1
    };



    const int64_t *all_input_shapes[] = {
      split_4_0_shape_0, split_4_1_shape_1
    };

    int64_t *output0_shape[] = {
      &input0_dim_0, &output0_dim_1, &split_4_0_dim_2
    };

    int64_t concat_dim_sizes[] = {
      concatenate_1_0_dim_1, concatenate_1_0_dim_1
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_5(
        output0,
        output0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        1/*concat_dim*/,
        3/*rank*/,
        2/*num_real_inputs*/,
        2/*num_all_inputs*/,
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
        std::cout << "Profiling: " << "concatenate_3" << " (" << iters << " iterations)" << std::endl;
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
      input2,
      input1,
      input1
    };


      int64_t input2_shape_0[] = {
        input2_dim_0, input2_dim_1, input2_dim_2
      };
      int64_t input1_shape_1[] = {
        input1_dim_0, input1_dim_1, input1_dim_2
      };
      int64_t input1_shape_2[] = {
        input1_dim_0, input1_dim_1, input1_dim_2
      };

    const int64_t *real_input_shapes[] = {
      input2_shape_0, input1_shape_1, input1_shape_2
    };



    const int64_t *all_input_shapes[] = {
      input2_shape_0, input1_shape_1, input1_shape_2
    };

    int64_t *output1_shape[] = {
      &input2_dim_0, &output1_dim_1, &input2_dim_2
    };

    int64_t concat_dim_sizes[] = {
      input2_dim_1, input1_dim_1, input1_dim_1
    };

    bool input_masks[] = {
      true, true, true
    };

    concatenate_3(
        output1,
        output1_shape,
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
        ss << "\"" << "concatenate_3" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1024\", \"10\", \"128\"], [\"1024\", \"10\", \"128\"], [\"1024\", \"10\", \"128\"]]"
           << ", \"output_sizes\": " << "[[\"1024\", \"30\", \"128\"]]"
        
          << ", \"dim\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "split_4" << " (" << iters << " iterations)" << std::endl;
        std::vector<std::pair<EventType, EventType>> call_events(iters);
        for (auto& [call_start, call_end] : call_events) {
          CreateEvent(&call_start);
          CreateEvent(&call_end);
        }
        for (auto& [call_start, call_end]: call_events) {
          DeviceMemset(L2CacheSlab, 0x73, L2SizeInBytes);
          EventRecord(call_start, stream);
            
  {

    void *outputs[] = {
      split_4_0,
      split_4_1
    };


      int64_t *split_4_0_shape[] = {
        &input0_dim_0, &concatenate_1_0_dim_1, &split_4_0_dim_2
      };
      int64_t *split_4_1_shape[] = {
        &input0_dim_0, &concatenate_1_0_dim_1, &split_4_1_dim_2
      };

    int64_t **output_shapes[] = {
      split_4_0_shape, split_4_1_shape
    };

    const int64_t input0_shape[] = {
      input0_dim_0, input0_dim_1, input0_dim_2
    };

    int64_t split_sizes[] = {
      64, 64
    };

    bool output_masks[] = {
      true, true
    };

    split_4(
        outputs,
        output_shapes,
        output_masks,
        input0,
        input0_shape,
        2/*real_num_splits*/,
        2/*all_num_splits*/,
        split_sizes,
        2/*split_dim*/,
        3/*rank*/,
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
        ss << "\"" << "split_4" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1024\", \"10\", \"128\"]]"
           << ", \"output_sizes\": " << "[[\"1024\", \"10\", \"64\"], [\"1024\", \"10\", \"64\"]]"
        
          << ", \"split_sizes\": " << "\"[64, 64]]\""
        
          << ", \"dim\": " << "\"2]\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_5" << " (" << iters << " iterations)" << std::endl;
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
      split_4_0,
      split_4_1
    };


      int64_t split_4_0_shape_0[] = {
        input0_dim_0, concatenate_1_0_dim_1, split_4_0_dim_2
      };
      int64_t split_4_1_shape_1[] = {
        input0_dim_0, concatenate_1_0_dim_1, split_4_1_dim_2
      };

    const int64_t *real_input_shapes[] = {
      split_4_0_shape_0, split_4_1_shape_1
    };



    const int64_t *all_input_shapes[] = {
      split_4_0_shape_0, split_4_1_shape_1
    };

    int64_t *output0_shape[] = {
      &input0_dim_0, &output0_dim_1, &split_4_0_dim_2
    };

    int64_t concat_dim_sizes[] = {
      concatenate_1_0_dim_1, concatenate_1_0_dim_1
    };

    bool input_masks[] = {
      true, true
    };

    concatenate_5(
        output0,
        output0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        1/*concat_dim*/,
        3/*rank*/,
        2/*num_real_inputs*/,
        2/*num_all_inputs*/,
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
        ss << "\"" << "concatenate_5" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1024\", \"10\", \"64\"], [\"1024\", \"10\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"1024\", \"20\", \"64\"]]"
        
          << ", \"dim\": " << "\"1\""
        
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
          13107200,
          0 * (1 + 0),
          0 * (1 + 0),
          3,
          2,
          0,
          constants,
          allocator
      );
    }

  private:
   void* input0 {nullptr};
   void* input1 {nullptr};
   void* input2 {nullptr};
   void* output1 {nullptr};
   void* split_4_0 {nullptr};
   void* split_4_1 {nullptr};
   void* output0 {nullptr};
   int64_t input0_dim_0 { 1024 };
   int64_t input0_dim_1 { 10 };
   int64_t input0_dim_2 { 128 };
   int64_t input1_dim_0 { 1024 };
   int64_t input1_dim_1 { 10 };
   int64_t input1_dim_2 { 128 };
   int64_t input2_dim_0 { 1024 };
   int64_t input2_dim_1 { 10 };
   int64_t input2_dim_2 { 128 };
   int64_t output1_dim_1 { 30 };
   int64_t concatenate_1_0_dim_1 { 10 };
   int64_t split_4_0_dim_2 { 64 };
   int64_t split_4_1_dim_2 { 64 };
   int64_t output0_dim_1 { 20 };


};
} // namespace ait