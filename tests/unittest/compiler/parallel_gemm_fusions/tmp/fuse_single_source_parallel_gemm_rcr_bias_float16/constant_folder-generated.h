
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


void concatenate_6_constant_folding(
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
    W1 = reinterpret_cast<decltype(W1)>(constants + 18432);
     constant_name_to_ptr_["W1"] = const_cast<const void**>(reinterpret_cast<void**>(&W1));
    W2 = reinterpret_cast<decltype(W2)>(constants + 30720);
     constant_name_to_ptr_["W2"] = const_cast<const void**>(reinterpret_cast<void**>(&W2));
    B0 = reinterpret_cast<decltype(B0)>(constants + 161792);
     constant_name_to_ptr_["B0"] = const_cast<const void**>(reinterpret_cast<void**>(&B0));
    B1 = reinterpret_cast<decltype(B1)>(constants + 161920);
     constant_name_to_ptr_["B1"] = const_cast<const void**>(reinterpret_cast<void**>(&B1));
    B2 = reinterpret_cast<decltype(B2)>(constants + 161984);
     constant_name_to_ptr_["B2"] = const_cast<const void**>(reinterpret_cast<void**>(&B2));
    auto* blob_ptr = static_cast<uint8_t*>(blob_.get());
    
    
         params_[0].shape_ptrs = {ParamDim(316, 316, &concatenate_6_0_dim_0), ParamDim(256, 256, &W0_dim_1)};
     params_[1].shape_ptrs = {ParamDim(316, 316, &concatenate_7_0_dim_0)};

      
      
    }

    ~ConstantFolder() {
      
      
    }

    void SetUpInputsOutputs() {
             concatenate_6_0 = static_cast<decltype(concatenate_6_0)>(params_[0].ptr);

if (concatenate_6_0 == nullptr) {
    throw std::runtime_error("Constant concatenate_6_0 was not set! Set the value with set_constant.");
}
    
     concatenate_7_0 = static_cast<decltype(concatenate_7_0)>(params_[1].ptr);

if (concatenate_7_0 == nullptr) {
    throw std::runtime_error("Constant concatenate_7_0 was not set! Set the value with set_constant.");
}
    
    }

    void ResetConstants(uint8_t* constants) {
        /*
         * This can be called if we want to use a different piece of memory
         * for the constants to be consumed.
         */
            W0 = reinterpret_cast<decltype(W0)>(constants + 0);
    W1 = reinterpret_cast<decltype(W1)>(constants + 18432);
    W2 = reinterpret_cast<decltype(W2)>(constants + 30720);
    B0 = reinterpret_cast<decltype(B0)>(constants + 161792);
    B1 = reinterpret_cast<decltype(B1)>(constants + 161920);
    B2 = reinterpret_cast<decltype(B2)>(constants + 161984);
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

    int64_t *concatenate_6_0_shape[] = {
      &concatenate_6_0_dim_0, &W0_dim_1
    };

    int64_t concat_dim_sizes[] = {
      W0_dim_0, W1_dim_0, W2_dim_0
    };

    bool input_masks[] = {
      true, true, true
    };

    concatenate_6_constant_folding(
        concatenate_6_0,
        concatenate_6_0_shape,
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

    int64_t *concatenate_7_0_shape[] = {
      &concatenate_7_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      B0_dim_0, B1_dim_0, B2_dim_0
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
        std::cout << "Profiling: " << "concatenate_6" << " (" << iters << " iterations)" << std::endl;
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

    int64_t *concatenate_6_0_shape[] = {
      &concatenate_6_0_dim_0, &W0_dim_1
    };

    int64_t concat_dim_sizes[] = {
      W0_dim_0, W1_dim_0, W2_dim_0
    };

    bool input_masks[] = {
      true, true, true
    };

    concatenate_6_constant_folding(
        concatenate_6_0,
        concatenate_6_0_shape,
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
        ss << "\"" << "concatenate_6" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"36\", \"256\"], [\"24\", \"256\"], [\"256\", \"256\"]]"
           << ", \"output_sizes\": " << "[[\"316\", \"256\"]]"
        
          << ", \"dim\": " << "\"0\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
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

    int64_t *concatenate_7_0_shape[] = {
      &concatenate_7_0_dim_0
    };

    int64_t concat_dim_sizes[] = {
      B0_dim_0, B1_dim_0, B2_dim_0
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
        ss << "\"" << "concatenate_7" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"36\"], [\"24\"], [\"256\"]]"
           << ", \"output_sizes\": " << "[[\"316\"]]"
        
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
          162432,
          0 * (1 + 0),
          0 * (1 + 0),
          0,
          2,
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
   void* concatenate_6_0 {nullptr};
   void* concatenate_7_0 {nullptr};
   int64_t W0_dim_0 { 36 };
   int64_t W0_dim_1 { 256 };
   int64_t W1_dim_0 { 24 };
   int64_t W1_dim_1 { 256 };
   int64_t W2_dim_0 { 256 };
   int64_t W2_dim_1 { 256 };
   int64_t B0_dim_0 { 36 };
   int64_t B1_dim_0 { 24 };
   int64_t B2_dim_0 { 256 };
   int64_t concatenate_6_0_dim_0 { 316 };
   int64_t concatenate_7_0_dim_0 { 316 };


};
} // namespace ait