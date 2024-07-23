
#include <cuda_runtime.h>

void ait_unsqueeze_0_constant_folding (
    int64_t* in_0,
    int64_t* in_1,
    int64_t* in_2,
    int64_t* out_0,
    int64_t* out_1,
    int64_t* out_2,
    int64_t* out_3
) {
  *out_0 = *in_0;
  *out_1 = *in_1;
  *out_3 = *in_2;
}