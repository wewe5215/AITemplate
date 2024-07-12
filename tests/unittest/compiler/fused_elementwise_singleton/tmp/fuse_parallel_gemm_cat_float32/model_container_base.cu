
#include "model_container.h"
#include "owned_constants.h"

namespace ait {
namespace {



// Contains the metadata for each constant.
constexpr std::array<ConstantInfo, 19> owned_constants = {
  ConstantInfo{"W", 0, 0, 32768},ConstantInfo{"W0", 32768, 32768, 8192},ConstantInfo{"B0", 40960, 40960, 128},ConstantInfo{"W1", 41088, 41088, 8192},ConstantInfo{"B1", 49280, 49280, 128},ConstantInfo{"W2", 49408, 49408, 8192},ConstantInfo{"B2", 57600, 57600, 128},ConstantInfo{"W3", 57728, 57728, 8192},ConstantInfo{"B3", 65920, 65920, 128},ConstantInfo{"W4", 66048, 66048, 8192},ConstantInfo{"B4", 74240, 74240, 128},ConstantInfo{"W5", 74368, 74368, 8192},ConstantInfo{"B5", 82560, 82560, 128},ConstantInfo{"W6", 82688, 82688, 8192},ConstantInfo{"B6", 90880, 90880, 128},ConstantInfo{"W7", 91008, 91008, 8192},ConstantInfo{"B7", 99200, 99200, 128},ConstantInfo{"W", 99328, 99328, 32768},ConstantInfo{"B", 132096, 132096, 512}
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
     bound_constant_name_to_idx_["W"] = 0;
     bound_constant_name_to_idx_["W0"] = 1;
     bound_constant_name_to_idx_["B0"] = 2;
     bound_constant_name_to_idx_["W1"] = 3;
     bound_constant_name_to_idx_["B1"] = 4;
     bound_constant_name_to_idx_["W2"] = 5;
     bound_constant_name_to_idx_["B2"] = 6;
     bound_constant_name_to_idx_["W3"] = 7;
     bound_constant_name_to_idx_["B3"] = 8;
     bound_constant_name_to_idx_["W4"] = 9;
     bound_constant_name_to_idx_["B4"] = 10;
     bound_constant_name_to_idx_["W5"] = 11;
     bound_constant_name_to_idx_["B5"] = 12;
     bound_constant_name_to_idx_["W6"] = 13;
     bound_constant_name_to_idx_["B6"] = 14;
     bound_constant_name_to_idx_["W7"] = 15;
     bound_constant_name_to_idx_["B7"] = 16;
     bound_constant_name_to_idx_["W"] = 17;
     bound_constant_name_to_idx_["B"] = 18;

     param_names_[0] = "X1";
     param_names_[1] = "X2";
     param_names_[2] = "output0";
     param_dtypes_[0] = AITemplateDtype::kFloat;
     param_dtypes_[1] = AITemplateDtype::kFloat;
     param_dtypes_[2] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[0] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[1] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[2] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[3] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[4] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[5] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[6] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[7] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[8] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[9] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[10] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[11] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[12] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[13] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[14] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[15] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[16] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[17] = AITemplateDtype::kFloat;
     bound_constant_dtypes_[18] = AITemplateDtype::kFloat;
     bound_constant_size_[0] = 32768;
     bound_constant_size_[1] = 8192;
     bound_constant_size_[2] = 128;
     bound_constant_size_[3] = 8192;
     bound_constant_size_[4] = 128;
     bound_constant_size_[5] = 8192;
     bound_constant_size_[6] = 128;
     bound_constant_size_[7] = 8192;
     bound_constant_size_[8] = 128;
     bound_constant_size_[9] = 8192;
     bound_constant_size_[10] = 128;
     bound_constant_size_[11] = 8192;
     bound_constant_size_[12] = 128;
     bound_constant_size_[13] = 8192;
     bound_constant_size_[14] = 128;
     bound_constant_size_[15] = 8192;
     bound_constant_size_[16] = 128;
     bound_constant_size_[17] = 32768;
     bound_constant_size_[18] = 512;
     max_param_shapes_[0] = {256, 256};
     max_param_shapes_[1] = {256, 256};
     max_param_shapes_[2] = {256, 768};
  for (size_t i = 0; i < num_params_; ++i) {
    max_param_numel_[i] = std::accumulate(
      max_param_shapes_[i].begin(),
      max_param_shapes_[i].end(),
      1,
      std::multiplies<int64_t>()
    );
    max_param_storage_bytes_[i] = max_param_numel_[i] * AITemplateDtypeSizeBytes(param_dtypes_[i]);
  }

    constant_folding_outputs_offsets_.resize(5);
         constant_folding_outputs_offsets_[0] = 132608;
     constant_folding_outputs_offsets_[1] = 165376;
     constant_folding_outputs_offsets_[2] = 198144;
     constant_folding_outputs_offsets_[3] = 230912;
     constant_folding_outputs_offsets_[4] = 231424;
    

    bound_constant_offsets_.resize(19);
         bound_constant_offsets_[0] = 0;
     bound_constant_offsets_[1] = 32768;
     bound_constant_offsets_[2] = 40960;
     bound_constant_offsets_[3] = 41088;
     bound_constant_offsets_[4] = 49280;
     bound_constant_offsets_[5] = 49408;
     bound_constant_offsets_[6] = 57600;
     bound_constant_offsets_[7] = 57728;
     bound_constant_offsets_[8] = 65920;
     bound_constant_offsets_[9] = 66048;
     bound_constant_offsets_[10] = 74240;
     bound_constant_offsets_[11] = 74368;
     bound_constant_offsets_[12] = 82560;
     bound_constant_offsets_[13] = 82688;
     bound_constant_offsets_[14] = 90880;
     bound_constant_offsets_[15] = 91008;
     bound_constant_offsets_[16] = 99200;
     bound_constant_offsets_[17] = 99328;
     bound_constant_offsets_[18] = 132096;
    
constant_folding_optional_inputs_.insert("W");
constant_folding_optional_inputs_.insert("W0");
constant_folding_optional_inputs_.insert("B0");
constant_folding_optional_inputs_.insert("W1");
constant_folding_optional_inputs_.insert("B1");
constant_folding_optional_inputs_.insert("W2");
constant_folding_optional_inputs_.insert("B2");
constant_folding_optional_inputs_.insert("W3");
constant_folding_optional_inputs_.insert("B3");
constant_folding_optional_inputs_.insert("W4");
constant_folding_optional_inputs_.insert("B4");
constant_folding_optional_inputs_.insert("W5");
constant_folding_optional_inputs_.insert("B5");
constant_folding_optional_inputs_.insert("W6");
constant_folding_optional_inputs_.insert("B6");
constant_folding_optional_inputs_.insert("W7");
constant_folding_optional_inputs_.insert("B7");


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
  return new ModelContainer(num_runtimes, 2, 1, 19, 0, 231936, allocator);
}
} // namespace ait