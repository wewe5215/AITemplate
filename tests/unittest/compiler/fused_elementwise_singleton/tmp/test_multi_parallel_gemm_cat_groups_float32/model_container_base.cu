
#include "model_container.h"
#include "owned_constants.h"

namespace ait {
namespace {



// Contains the metadata for each constant.
constexpr std::array<ConstantInfo, 20> owned_constants = {
  ConstantInfo{"w_0", 0, 0, 32768},ConstantInfo{"b_0", 32768, 32768, 512},ConstantInfo{"w_1", 33280, 33280, 32768},ConstantInfo{"b_1", 66048, 66048, 512},ConstantInfo{"w_2", 66560, 66560, 61440},ConstantInfo{"b_2", 128000, 128000, 512},ConstantInfo{"w_3", 128512, 128512, 61440},ConstantInfo{"b_3", 189952, 189952, 512},ConstantInfo{"w_4", 190464, 190464, 61440},ConstantInfo{"b_4", 251904, 251904, 512},ConstantInfo{"w_5", 252416, 252416, 61440},ConstantInfo{"b_5", 313856, 313856, 512},ConstantInfo{"w_6", 314368, 314368, 36864},ConstantInfo{"b_6", 351232, 351232, 512},ConstantInfo{"w_7", 351744, 351744, 36864},ConstantInfo{"b_7", 388608, 388608, 512},ConstantInfo{"w_8", 389120, 389120, 32768},ConstantInfo{"b_8", 421888, 421888, 512},ConstantInfo{"w_9", 422400, 422400, 32768},ConstantInfo{"b_9", 455168, 455168, 512}
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
     bound_constant_name_to_idx_["w_2"] = 4;
     bound_constant_name_to_idx_["b_2"] = 5;
     bound_constant_name_to_idx_["w_3"] = 6;
     bound_constant_name_to_idx_["b_3"] = 7;
     bound_constant_name_to_idx_["w_4"] = 8;
     bound_constant_name_to_idx_["b_4"] = 9;
     bound_constant_name_to_idx_["w_5"] = 10;
     bound_constant_name_to_idx_["b_5"] = 11;
     bound_constant_name_to_idx_["w_6"] = 12;
     bound_constant_name_to_idx_["b_6"] = 13;
     bound_constant_name_to_idx_["w_7"] = 14;
     bound_constant_name_to_idx_["b_7"] = 15;
     bound_constant_name_to_idx_["w_8"] = 16;
     bound_constant_name_to_idx_["b_8"] = 17;
     bound_constant_name_to_idx_["w_9"] = 18;
     bound_constant_name_to_idx_["b_9"] = 19;

     param_names_[0] = "x_0";
     param_names_[1] = "x_1";
     param_names_[2] = "x_2";
     param_names_[3] = "x_3";
     param_names_[4] = "x_4";
     param_names_[5] = "x_5";
     param_names_[6] = "x_6";
     param_names_[7] = "x_7";
     param_names_[8] = "x_8";
     param_names_[9] = "x_9";
     param_names_[10] = "y";
     param_dtypes_[0] = AITemplateDtype::kFloat;
     param_dtypes_[1] = AITemplateDtype::kFloat;
     param_dtypes_[2] = AITemplateDtype::kFloat;
     param_dtypes_[3] = AITemplateDtype::kFloat;
     param_dtypes_[4] = AITemplateDtype::kFloat;
     param_dtypes_[5] = AITemplateDtype::kFloat;
     param_dtypes_[6] = AITemplateDtype::kFloat;
     param_dtypes_[7] = AITemplateDtype::kFloat;
     param_dtypes_[8] = AITemplateDtype::kFloat;
     param_dtypes_[9] = AITemplateDtype::kFloat;
     param_dtypes_[10] = AITemplateDtype::kFloat;
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
     bound_constant_dtypes_[19] = AITemplateDtype::kFloat;
     bound_constant_size_[0] = 32768;
     bound_constant_size_[1] = 512;
     bound_constant_size_[2] = 32768;
     bound_constant_size_[3] = 512;
     bound_constant_size_[4] = 61440;
     bound_constant_size_[5] = 512;
     bound_constant_size_[6] = 61440;
     bound_constant_size_[7] = 512;
     bound_constant_size_[8] = 61440;
     bound_constant_size_[9] = 512;
     bound_constant_size_[10] = 61440;
     bound_constant_size_[11] = 512;
     bound_constant_size_[12] = 36864;
     bound_constant_size_[13] = 512;
     bound_constant_size_[14] = 36864;
     bound_constant_size_[15] = 512;
     bound_constant_size_[16] = 32768;
     bound_constant_size_[17] = 512;
     bound_constant_size_[18] = 32768;
     bound_constant_size_[19] = 512;
     max_param_shapes_[0] = {256, 64};
     max_param_shapes_[1] = {256, 64};
     max_param_shapes_[2] = {256, 120};
     max_param_shapes_[3] = {256, 120};
     max_param_shapes_[4] = {256, 120};
     max_param_shapes_[5] = {256, 120};
     max_param_shapes_[6] = {256, 72};
     max_param_shapes_[7] = {256, 72};
     max_param_shapes_[8] = {256, 64};
     max_param_shapes_[9] = {256, 64};
     max_param_shapes_[10] = {256, 1280};
  for (size_t i = 0; i < num_params_; ++i) {
    max_param_numel_[i] = std::accumulate(
      max_param_shapes_[i].begin(),
      max_param_shapes_[i].end(),
      1,
      std::multiplies<int64_t>()
    );
    max_param_storage_bytes_[i] = max_param_numel_[i] * AITemplateDtypeSizeBytes(param_dtypes_[i]);
  }

    constant_folding_outputs_offsets_.resize(8);
         constant_folding_outputs_offsets_[0] = 455680;
     constant_folding_outputs_offsets_[1] = 701440;
     constant_folding_outputs_offsets_[2] = 775168;
     constant_folding_outputs_offsets_[3] = 840704;
     constant_folding_outputs_offsets_[4] = 906240;
     constant_folding_outputs_offsets_[5] = 907264;
     constant_folding_outputs_offsets_[6] = 909312;
     constant_folding_outputs_offsets_[7] = 910336;
    

    bound_constant_offsets_.resize(20);
         bound_constant_offsets_[0] = 0;
     bound_constant_offsets_[1] = 32768;
     bound_constant_offsets_[2] = 33280;
     bound_constant_offsets_[3] = 66048;
     bound_constant_offsets_[4] = 66560;
     bound_constant_offsets_[5] = 128000;
     bound_constant_offsets_[6] = 128512;
     bound_constant_offsets_[7] = 189952;
     bound_constant_offsets_[8] = 190464;
     bound_constant_offsets_[9] = 251904;
     bound_constant_offsets_[10] = 252416;
     bound_constant_offsets_[11] = 313856;
     bound_constant_offsets_[12] = 314368;
     bound_constant_offsets_[13] = 351232;
     bound_constant_offsets_[14] = 351744;
     bound_constant_offsets_[15] = 388608;
     bound_constant_offsets_[16] = 389120;
     bound_constant_offsets_[17] = 421888;
     bound_constant_offsets_[18] = 422400;
     bound_constant_offsets_[19] = 455168;
    
constant_folding_optional_inputs_.insert("w_0");
constant_folding_optional_inputs_.insert("b_0");
constant_folding_optional_inputs_.insert("w_1");
constant_folding_optional_inputs_.insert("b_1");
constant_folding_optional_inputs_.insert("w_2");
constant_folding_optional_inputs_.insert("b_2");
constant_folding_optional_inputs_.insert("w_3");
constant_folding_optional_inputs_.insert("b_3");
constant_folding_optional_inputs_.insert("w_4");
constant_folding_optional_inputs_.insert("b_4");
constant_folding_optional_inputs_.insert("w_5");
constant_folding_optional_inputs_.insert("b_5");
constant_folding_optional_inputs_.insert("w_6");
constant_folding_optional_inputs_.insert("b_6");
constant_folding_optional_inputs_.insert("w_7");
constant_folding_optional_inputs_.insert("b_7");
constant_folding_optional_inputs_.insert("w_8");
constant_folding_optional_inputs_.insert("b_8");
constant_folding_optional_inputs_.insert("w_9");
constant_folding_optional_inputs_.insert("b_9");


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
  return new ModelContainer(num_runtimes, 10, 1, 20, 0, 911360, allocator);
}
} // namespace ait