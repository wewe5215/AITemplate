
#include "model_container.h"
#include "owned_constants.h"

namespace ait {
namespace {



// Contains the metadata for each constant.
constexpr std::array<ConstantInfo, 12> owned_constants = {
  ConstantInfo{"w_0", 0, 0, 16384},ConstantInfo{"b_0", 16384, 16384, 256},ConstantInfo{"w_1", 16640, 16640, 16384},ConstantInfo{"b_1", 33024, 33024, 256},ConstantInfo{"w_3", 33280, 33280, 18432},ConstantInfo{"b_3", 51712, 51712, 256},ConstantInfo{"w_4", 51968, 51968, 18432},ConstantInfo{"b_4", 70400, 70400, 256},ConstantInfo{"w_2", 70656, 70656, 30720},ConstantInfo{"b_2", 101376, 101376, 256},ConstantInfo{"w_5", 101632, 101632, 16384},ConstantInfo{"b_5", 118016, 118016, 256}
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
     bound_constant_name_to_idx_["w_0"] = 0;
     bound_constant_name_to_idx_["b_0"] = 1;
     bound_constant_name_to_idx_["w_1"] = 2;
     bound_constant_name_to_idx_["b_1"] = 3;
     bound_constant_name_to_idx_["w_3"] = 4;
     bound_constant_name_to_idx_["b_3"] = 5;
     bound_constant_name_to_idx_["w_4"] = 6;
     bound_constant_name_to_idx_["b_4"] = 7;
     bound_constant_name_to_idx_["w_2"] = 8;
     bound_constant_name_to_idx_["b_2"] = 9;
     bound_constant_name_to_idx_["w_5"] = 10;
     bound_constant_name_to_idx_["b_5"] = 11;

     param_names_[0] = "x_0";
     param_names_[1] = "x_1";
     param_names_[2] = "x_2";
     param_names_[3] = "x_3";
     param_names_[4] = "x_4";
     param_names_[5] = "x_5";
     param_names_[6] = "y";
     param_dtypes_[0] = AITemplateDtype::kHalf;
     param_dtypes_[1] = AITemplateDtype::kHalf;
     param_dtypes_[2] = AITemplateDtype::kHalf;
     param_dtypes_[3] = AITemplateDtype::kHalf;
     param_dtypes_[4] = AITemplateDtype::kHalf;
     param_dtypes_[5] = AITemplateDtype::kHalf;
     param_dtypes_[6] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[0] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[1] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[2] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[3] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[4] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[5] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[6] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[7] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[8] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[9] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[10] = AITemplateDtype::kHalf;
     bound_constant_dtypes_[11] = AITemplateDtype::kHalf;
     bound_constant_size_[0] = 16384;
     bound_constant_size_[1] = 256;
     bound_constant_size_[2] = 16384;
     bound_constant_size_[3] = 256;
     bound_constant_size_[4] = 18432;
     bound_constant_size_[5] = 256;
     bound_constant_size_[6] = 18432;
     bound_constant_size_[7] = 256;
     bound_constant_size_[8] = 30720;
     bound_constant_size_[9] = 256;
     bound_constant_size_[10] = 16384;
     bound_constant_size_[11] = 256;
     max_param_shapes_[0] = {256, 64};
     max_param_shapes_[1] = {256, 64};
     max_param_shapes_[2] = {256, 120};
     max_param_shapes_[3] = {256, 72};
     max_param_shapes_[4] = {256, 72};
     max_param_shapes_[5] = {256, 64};
     max_param_shapes_[6] = {256, 768};
  for (size_t i = 0; i < num_params_; ++i) {
    max_param_numel_[i] = std::accumulate(
      max_param_shapes_[i].begin(),
      max_param_shapes_[i].end(),
      1,
      std::multiplies<int64_t>()
    );
    max_param_storage_bytes_[i] = max_param_numel_[i] * AITemplateDtypeSizeBytes(param_dtypes_[i]);
  }

    constant_folding_outputs_offsets_.resize(4);
         constant_folding_outputs_offsets_[0] = 118272;
     constant_folding_outputs_offsets_[1] = 155136;
     constant_folding_outputs_offsets_[2] = 187904;
     constant_folding_outputs_offsets_[3] = 188416;
    

    bound_constant_offsets_.resize(12);
         bound_constant_offsets_[0] = 0;
     bound_constant_offsets_[1] = 16384;
     bound_constant_offsets_[2] = 16640;
     bound_constant_offsets_[3] = 33024;
     bound_constant_offsets_[4] = 33280;
     bound_constant_offsets_[5] = 51712;
     bound_constant_offsets_[6] = 51968;
     bound_constant_offsets_[7] = 70400;
     bound_constant_offsets_[8] = 70656;
     bound_constant_offsets_[9] = 101376;
     bound_constant_offsets_[10] = 101632;
     bound_constant_offsets_[11] = 118016;
    
constant_folding_optional_inputs_.insert("w_0");
constant_folding_optional_inputs_.insert("b_0");
constant_folding_optional_inputs_.insert("w_1");
constant_folding_optional_inputs_.insert("b_1");
constant_folding_optional_inputs_.insert("w_3");
constant_folding_optional_inputs_.insert("b_3");
constant_folding_optional_inputs_.insert("w_4");
constant_folding_optional_inputs_.insert("b_4");


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
  return new ModelContainer(num_runtimes, 6, 1, 12, 0, 188928, allocator);
}
} // namespace ait