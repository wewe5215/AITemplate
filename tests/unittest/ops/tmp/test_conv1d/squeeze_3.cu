
#include <cuda_runtime.h>

void ait_squeeze_3 (
    int64_t* in_0,
    int64_t* in_1,
    int64_t* in_2,
    int64_t* in_3,
    int64_t* out_0,
    int64_t* out_1,
    int64_t* out_2
) {
  *out_0 = *in_0;
  *out_1 = *in_1;
  *out_2 = *in_3;
}