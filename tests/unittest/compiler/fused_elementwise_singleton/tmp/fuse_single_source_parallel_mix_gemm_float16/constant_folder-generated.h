
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


void concatenate_7_constant_folding(
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

void concatenate_16_constant_folding(
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

void concatenate_10_constant_folding(
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
        W0 = reinterpret_cast<decltype(W0)>(constants + 0);
     constant_name_to_ptr_["W0"] = const_cast<const void**>(reinterpret_cast<void**>(&W0));
    W1 = reinterpret_cast<decltype(W1)>(constants + 262144);
     constant_name_to_ptr_["W1"] = const_cast<const void**>(reinterpret_cast<void**>(&W1));
    W2 = reinterpret_cast<decltype(W2)>(constants + 327680);
     constant_name_to_ptr_["W2"] = const_cast<const void**>(reinterpret_cast<void**>(&W2));
    B0 = reinterpret_cast<decltype(B0)>(constants + 344064);
     constant_name_to_ptr_["B0"] = const_cast<const void**>(reinterpret_cast<void**>(&B0));
    B1 = reinterpret_cast<decltype(B1)>(constants + 345088);
     constant_name_to_ptr_["B1"] = const_cast<const void**>(reinterpret_cast<void**>(&B1));
    B2 = reinterpret_cast<decltype(B2)>(constants + 345344);
     constant_name_to_ptr_["B2"] = const_cast<const void**>(reinterpret_cast<void**>(&B2));
    W3 = reinterpret_cast<decltype(W3)>(constants + 345408);
     constant_name_to_ptr_["W3"] = const_cast<const void**>(reinterpret_cast<void**>(&W3));
    W4 = reinterpret_cast<decltype(W4)>(constants + 361792);
     constant_name_to_ptr_["W4"] = const_cast<const void**>(reinterpret_cast<void**>(&W4));
    W5 = reinterpret_cast<decltype(W5)>(constants + 427328);
     constant_name_to_ptr_["W5"] = const_cast<const void**>(reinterpret_cast<void**>(&W5));
    auto* blob_ptr = static_cast<uint8_t*>(blob_.get());
    
    
         params_[0].shape_ptrs = {ParamDim(672, 672, &concatenate_7_0_dim_0), ParamDim(256, 256, &W0_dim_1)};
     params_[1].shape_ptrs = {ParamDim(416, 416, &concatenate_16_0_dim_0), ParamDim(256, 256, &W3_dim_1)};
     params_[2].shape_ptrs = {ParamDim(672, 672, &concatenate_10_0_dim_0)};

      
      
    }

    ~ConstantFolder() {
      
      
    }

    void SetUpInputsOutputs() {
             concatenate_7_0 = static_cast<decltype(concatenate_7_0)>(params_[0].ptr);

if (concatenate_7_0 == nullptr) {
    throw std::runtime_error("Constant concatenate_7_0 was not set! Set the value with set_constant.");
}
    
     concatenate_16_0 = static_cast<decltype(concatenate_16_0)>(params_[1].ptr);

if (concatenate_16_0 == nullptr) {
    throw std::runtime_error("Constant concatenate_16_0 was not set! Set the value with set_constant.");
}
    
     concatenate_10_0 = static_cast<decltype(concatenate_10_0)>(params_[2].ptr);

if (concatenate_10_0 == nullptr) {
    throw std::runtime_error("Constant concatenate_10_0 was not set! Set the value with set_constant.");
}
    
    }

    void ResetConstants(uint8_t* constants) {
        /*
         * This can be called if we want to use a different piece of memory
         * for the constants to be consumed.
         */
            W0 = reinterpret_cast<decltype(W0)>(constants + 0);
    W1 = reinterpret_cast<decltype(W1)>(constants + 262144);
    W2 = reinterpret_cast<decltype(W2)>(constants + 327680);
    B0 = reinterpret_cast<decltype(B0)>(constants + 344064);
    B1 = reinterpret_cast<decltype(B1)>(constants + 345088);
    B2 = reinterpret_cast<decltype(B2)>(constants + 345344);
    W3 = reinterpret_cast<decltype(W3)>(constants + 345408);
    W4 = reinterpret_cast<decltype(W4)>(constants + 361792);
    W5 = reinterpret_cast<decltype(W5)>(constants + 427328);
    }

    void DeviceToDeviceCopies(StreamType stream) {
  
    }


    ///////////////////////////////////////////////////////////////////////////
    // default RunImpl implemenation
    void RunImpl(StreamType stream) {
        
  
  
  {

    const void *inputs[] = {
      W0,
      W1,
      W2
    };


      int64_t W0_shape_0[] = {
        W0_dim_0, W0_dim_1
      };
      int64_t W1_shape_1[] = {
        W1_dim_0, W1_dim_1
      };
      int64_t W2_shape_2[] = {
        W2_dim_0, W2_dim_1
      };

    const int64_t *real_input_shapes[] = {
      W0_shape_0, W1_shape_1, W2_shape_2
    };



    const int64_t *all_input_shapes[] = {
      W0_shape_0, W1_shape_1, W2_shape_2
    };

    int64_t *concatenate_7_0_shape[] = {
      &concatenate_7_0_dim_0, &W0_dim_1
    };

    int64_t concat_dim_sizes[] = {
      W0_dim_0, W1_dim_0, W2_dim_0
    };

    bool input_masks[] = {
      true, true, true
    };

    concatenate_7_constant_folding(
        concatenate_7_0,
        concatenate_7_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        2/*rank*/,
        3/*num_real_inputs*/,
        3/*num_all_inputs*/,
        stream
    );
  }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    const void *inputs[] = {
      W3,
      W4,
      W5
    };


      int64_t W3_shape_0[] = {
        W3_dim_0, W3_dim_1
      };
      int64_t W4_shape_1[] = {
        W4_dim_0, W4_dim_1
      };
      int64_t W5_shape_2[] = {
        W5_dim_0, W5_dim_1
      };

    const int64_t *real_input_shapes[] = {
      W3_shape_0, W4_shape_1, W5_shape_2
    };



    const int64_t *all_input_shapes[] = {
      W3_shape_0, W4_shape_1, W5_shape_2
    };

    int64_t *concatenate_16_0_shape[] = {
      &concatenate_16_0_dim_0, &W3_dim_1
    };

    int64_t concat_dim_sizes[] = {
      W3_dim_0, W4_dim_0, W5_dim_0
    };

    bool input_masks[] = {
      true, true, true
    };

    concatenate_16_constant_folding(
        concatenate_16_0,
        concatenate_16_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        2/*rank*/,
        3/*num_real_inputs*/,
        3/*num_all_inputs*/,
        stream
    );
  }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
  {

    const void *inputs[] = {
      B0,
      B1,
      B2
    };


      int64_t B0_shape_0[] = {
        B0_dim_0
      };
      int64_t B1_shape_1[] = {
        B1_dim_0
      };
      int64_t B2_shape_2[] = {
        B2_dim_0
      };

    const int64_t *real_input_shapes[] = {
      B0_shape_0, B1_shape_1, B2_shape_2
    };



    const int64_t *all_input_shapes[] = {
      B0_shape_0, B1_shape_1, B2_shape_2
    };

    int64_t *concatenate_10_0_shape[] = {
      &concatenate_10_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      B0_dim_0, B1_dim_0, B2_dim_0
    };

    bool input_masks[] = {
      true, true, true
    };

    concatenate_10_constant_folding(
        concatenate_10_0,
        concatenate_10_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        1/*rank*/,
        3/*num_real_inputs*/,
        3/*num_all_inputs*/,
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
        std::cout << "Profiling: " << "concatenate_7" << " (" << iters << " iterations)" << std::endl;
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
      W0,
      W1,
      W2
    };


      int64_t W0_shape_0[] = {
        W0_dim_0, W0_dim_1
      };
      int64_t W1_shape_1[] = {
        W1_dim_0, W1_dim_1
      };
      int64_t W2_shape_2[] = {
        W2_dim_0, W2_dim_1
      };

    const int64_t *real_input_shapes[] = {
      W0_shape_0, W1_shape_1, W2_shape_2
    };



    const int64_t *all_input_shapes[] = {
      W0_shape_0, W1_shape_1, W2_shape_2
    };

    int64_t *concatenate_7_0_shape[] = {
      &concatenate_7_0_dim_0, &W0_dim_1
    };

    int64_t concat_dim_sizes[] = {
      W0_dim_0, W1_dim_0, W2_dim_0
    };

    bool input_masks[] = {
      true, true, true
    };

    concatenate_7_constant_folding(
        concatenate_7_0,
        concatenate_7_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        2/*rank*/,
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
        ss << "\"" << "concatenate_7" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"512\", \"256\"], [\"128\", \"256\"], [\"32\", \"256\"]]"
           << ", \"output_sizes\": " << "[[\"672\", \"256\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_16" << " (" << iters << " iterations)" << std::endl;
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
      W3,
      W4,
      W5
    };


      int64_t W3_shape_0[] = {
        W3_dim_0, W3_dim_1
      };
      int64_t W4_shape_1[] = {
        W4_dim_0, W4_dim_1
      };
      int64_t W5_shape_2[] = {
        W5_dim_0, W5_dim_1
      };

    const int64_t *real_input_shapes[] = {
      W3_shape_0, W4_shape_1, W5_shape_2
    };



    const int64_t *all_input_shapes[] = {
      W3_shape_0, W4_shape_1, W5_shape_2
    };

    int64_t *concatenate_16_0_shape[] = {
      &concatenate_16_0_dim_0, &W3_dim_1
    };

    int64_t concat_dim_sizes[] = {
      W3_dim_0, W4_dim_0, W5_dim_0
    };

    bool input_masks[] = {
      true, true, true
    };

    concatenate_16_constant_folding(
        concatenate_16_0,
        concatenate_16_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        2/*rank*/,
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
        ss << "\"" << "concatenate_16" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"32\", \"256\"], [\"128\", \"256\"], [\"256\", \"256\"]]"
           << ", \"output_sizes\": " << "[[\"416\", \"256\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "concatenate_10" << " (" << iters << " iterations)" << std::endl;
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
      B0,
      B1,
      B2
    };


      int64_t B0_shape_0[] = {
        B0_dim_0
      };
      int64_t B1_shape_1[] = {
        B1_dim_0
      };
      int64_t B2_shape_2[] = {
        B2_dim_0
      };

    const int64_t *real_input_shapes[] = {
      B0_shape_0, B1_shape_1, B2_shape_2
    };



    const int64_t *all_input_shapes[] = {
      B0_shape_0, B1_shape_1, B2_shape_2
    };

    int64_t *concatenate_10_0_shape[] = {
      &concatenate_10_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      B0_dim_0, B1_dim_0, B2_dim_0
    };

    bool input_masks[] = {
      true, true, true
    };

    concatenate_10_constant_folding(
        concatenate_10_0,
        concatenate_10_0_shape,
        inputs,
        real_input_shapes,
        all_input_shapes,
        input_masks,
        concat_dim_sizes,
        0/*concat_dim*/,
        1/*rank*/,
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
        ss << "\"" << "concatenate_10" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"512\"], [\"128\"], [\"32\"]]"
           << ", \"output_sizes\": " << "[[\"672\"]]"
        
          << ", \"dim\": " << "\"0\""
        
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
          558400,
          0 * (1 + 0),
          0 * (1 + 0),
          0,
          3,
          0,
          constants,
          allocator
      );
    }

  private:
   void* W0 {nullptr};
   void* W1 {nullptr};
   void* W2 {nullptr};
   void* B0 {nullptr};
   void* B1 {nullptr};
   void* B2 {nullptr};
   void* W3 {nullptr};
   void* W4 {nullptr};
   void* W5 {nullptr};
   void* concatenate_7_0 {nullptr};
   void* concatenate_16_0 {nullptr};
   void* concatenate_10_0 {nullptr};
   int64_t W0_dim_0 { 512 };
   int64_t W0_dim_1 { 256 };
   int64_t W1_dim_0 { 128 };
   int64_t W1_dim_1 { 256 };
   int64_t W2_dim_0 { 32 };
   int64_t W2_dim_1 { 256 };
   int64_t B0_dim_0 { 512 };
   int64_t B1_dim_0 { 128 };
   int64_t B2_dim_0 { 32 };
   int64_t W3_dim_0 { 32 };
   int64_t W3_dim_1 { 256 };
   int64_t W4_dim_0 { 128 };
   int64_t W4_dim_1 { 256 };
   int64_t W5_dim_0 { 256 };
   int64_t W5_dim_1 { 256 };
   int64_t concatenate_7_0_dim_0 { 672 };
   int64_t concatenate_16_0_dim_0 { 416 };
   int64_t concatenate_10_0_dim_0 { 672 };


};
} // namespace ait