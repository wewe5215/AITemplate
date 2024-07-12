
#include "model_container.h"
#include "owned_constants.h"

namespace ait {
namespace {



// Contains the metadata for each constant.
constexpr std::array<ConstantInfo, 8> owned_constants = {
  ConstantInfo{"W0", 0, 0, 16384},ConstantInfo{"B0", 16384, 16384, 256},ConstantInfo{"W1", 16640, 16640, 16384},ConstantInfo{"B1", 33024, 33024, 256},ConstantInfo{"W2", 33280, 33280, 16384},ConstantInfo{"B2", 49664, 49664, 256},ConstantInfo{"W3", 49920, 49920, 16384},ConstantInfo{"B3", 66304, 66304, 256}
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
     bound_constant_name_to_idx_["B0"] = 1;
     bound_constant_name_to_idx_["W1"] = 2;
     bound_constant_name_to_idx_["B1"] = 3;
     bound_constant_name_to_idx_["W2"] = 4;
     bound_constant_name_to_idx_["B2"] = 5;
     bound_constant_name_to_idx_["W3"] = 6;
     bound_constant_name_to_idx_["B3"] = 7;

     param_names_[0] = "X";
     param_names_[2] = "output1";
     param_names_[3] = "output2";
     param_names_[4] = "output3";
     param_names_[5] = "output4";
     param_names_[1] = "output0";
     param_dtypes_[0] = AITemplateDtype::kHalf;
     param_dtypes_[2] = AITemplateDtype::kHalf;
     param_dtypes_[3] = AITemplateDtype::kHalf;
     param_dtypes_[4] = AITemplateDtype::kHalf;
     param_dtypes_[5] = AITemplateDtype::kHalf;
     param_dtypes_[1] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[0] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[1] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[2] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[3] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[4] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[5] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[6] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[7] = AITemplateDtype::kHalf;
     bound_constant_size_[0] = 16384;
     bound_constant_size_[1] = 256;
     bound_constant_size_[2] = 16384;
     bound_constant_size_[3] = 256;
     bound_constant_size_[4] = 16384;
     bound_constant_size_[5] = 256;
     bound_constant_size_[6] = 16384;
     bound_constant_size_[7] = 256;
     max_param_shapes_[0] = {512, 256};
     max_param_shapes_[2] = {512, 128};
     max_param_shapes_[3] = {512, 128};
     max_param_shapes_[4] = {512, 128};
     max_param_shapes_[5] = {512, 128};
     max_param_shapes_[1] = {512, 512};
  for (size_t i = 0; i < num_params_; ++i) {
    max_param_numel_[i] = std::accumulate(
      max_param_shapes_[i].begin(),
      max_param_shapes_[i].end(),
      1,
      std::multiplies<int64_t>()
    );
    max_param_storage_bytes_[i] = max_param_numel_[i] * AITemplateDtypeSizeBytes(param_dtypes_[i]);
  }

    bound_constant_offsets_.resize(8);
         bound_constant_offsets_[0] = 0;
     bound_constant_offsets_[1] = 16384;
     bound_constant_offsets_[2] = 16640;
     bound_constant_offsets_[3] = 33024;
     bound_constant_offsets_[4] = 33280;
     bound_constant_offsets_[5] = 49664;
     bound_constant_offsets_[6] = 49920;
     bound_constant_offsets_[7] = 66304;
    



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
  return new ModelContainer(num_runtimes, 1, 5, 8, 0, 66560, allocator);
}
} // namespace ait