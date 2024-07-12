
#include "model_container.h"
#include "owned_constants.h"

namespace ait {
namespace {



// Contains the metadata for each constant.
constexpr std::array<ConstantInfo, 6> owned_constants = {
  ConstantInfo{"W0", 0, 0, 18432},ConstantInfo{"W1", 18432, 18432, 12288},ConstantInfo{"W2", 30720, 30720, 131072},ConstantInfo{"B0", 161792, 161792, 72},ConstantInfo{"B1", 161864, 161920, 48},ConstantInfo{"B2", 161912, 161984, 512}
};
} // namespace

ModelContainerBase::ModelContainerBase(
    size_t num_inputs,
    size_t num_outputs,
    size_t num_bound_constants,
    size_t num_unbound_constants,
    size_t params_size,
    AITemplateAllocator& allocator)
    : constants_size_(params_size),
      constants_primary_(RAII_DeviceMalloc(constants_size_, allocator)),
      constants_secondary_(nullptr),
      use_constants_primary_buffer_(true),
      buffer_state_(BufferState::CLEAN),
      bound_constant_size_(num_bound_constants),
      bound_constant_dtypes_(num_bound_constants),
      num_params_(num_inputs + num_outputs + num_unbound_constants),
      param_names_(num_params_),
      param_dtypes_(num_params_),
      max_param_shapes_(num_params_),
      max_param_numel_(num_params_),
      max_param_storage_bytes_(num_params_) {
     bound_constant_name_to_idx_["W0"] = 0;
     bound_constant_name_to_idx_["W1"] = 1;
     bound_constant_name_to_idx_["W2"] = 2;
     bound_constant_name_to_idx_["B0"] = 3;
     bound_constant_name_to_idx_["B1"] = 4;
     bound_constant_name_to_idx_["B2"] = 5;

     param_names_[0] = "X";
     param_names_[3] = "output2";
     param_names_[1] = "output0";
     param_names_[2] = "output1";
     param_dtypes_[0] = AITemplateDtype::kHalf;
     param_dtypes_[3] = AITemplateDtype::kHalf;
     param_dtypes_[1] = AITemplateDtype::kHalf;
     param_dtypes_[2] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[0] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[1] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[2] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[3] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[4] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[5] = AITemplateDtype::kHalf;
     bound_constant_size_[0] = 18432;
     bound_constant_size_[1] = 12288;
     bound_constant_size_[2] = 131072;
     bound_constant_size_[3] = 72;
     bound_constant_size_[4] = 48;
     bound_constant_size_[5] = 512;
     max_param_shapes_[0] = {1024, 256};
     max_param_shapes_[3] = {1024, 256};
     max_param_shapes_[1] = {1024, 36};
     max_param_shapes_[2] = {1024, 24};
  for (size_t i = 0; i < num_params_; ++i) {
    max_param_numel_[i] = std::accumulate(
      max_param_shapes_[i].begin(),
      max_param_shapes_[i].end(),
      1,
      std::multiplies<int64_t>()
    );
    max_param_storage_bytes_[i] = max_param_numel_[i] * AITemplateDtypeSizeBytes(param_dtypes_[i]);
  }

    constant_folding_outputs_offsets_.resize(2);
         constant_folding_outputs_offsets_[0] = 162496;
     constant_folding_outputs_offsets_[1] = 324288;
    

    bound_constant_offsets_.resize(6);
         bound_constant_offsets_[0] = 0;
     bound_constant_offsets_[1] = 18432;
     bound_constant_offsets_[2] = 30720;
     bound_constant_offsets_[3] = 161792;
     bound_constant_offsets_[4] = 161920;
     bound_constant_offsets_[5] = 161984;
    
constant_folding_optional_inputs_.insert("W0");
constant_folding_optional_inputs_.insert("W1");
constant_folding_optional_inputs_.insert("W2");
constant_folding_optional_inputs_.insert("B0");
constant_folding_optional_inputs_.insert("B1");
constant_folding_optional_inputs_.insert("B2");


  const auto binary_constants_bin_size = static_cast<size_t>(_binary_constants_bin_end - _binary_constants_bin_start);
  const uint8_t* const binary_constants_bin_start = _binary_constants_bin_start;


  auto* constants_ptr = static_cast<uint8_t*>(constants_primary_.get());
  for (auto& constant_info : owned_constants) {
    auto* dst = constants_ptr + constant_info.internal_offset;
    if (constant_info.data_offset + constant_info.num_bytes > binary_constants_bin_size) {
      throw std::runtime_error(std::string("Copying constant ") + constant_info.name + " would overflow constant buffer");
    }
    DEVICE_CHECK(CopyToDevice(dst, binary_constants_bin_start + constant_info.data_offset, constant_info.num_bytes));
  }
}

ModelContainer* CreateModelContainer(size_t num_runtimes, AITemplateAllocator& allocator) {
  // num_runtimes, num_inputs, num_outputs, num_bound_constants, num_unbound_constants, params_size, allocator
  return new ModelContainer(num_runtimes, 1, 3, 6, 0, 324928, allocator);
}
} // namespace ait