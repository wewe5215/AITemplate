
#include <cuda_runtime.h>

void ait_reshape_58 (
    int64_t* in_0,
    int64_t* in_1,
    int64_t* in_2,
    int64_t* out_0,
    int64_t* out_1
) {
  int64_t IN_0 = *in_0;
  int64_t IN_1 = *in_1;
  int64_t IN_2 = *in_2;

  int64_t OUT_0 = *out_0;
  int64_t OUT_1 = *out_1;

  int64_t prod = 1;
  prod *= IN_0;
  prod *= IN_1;
  prod *= IN_2;

  int64_t out_prod = 1;

  out_prod *= OUT_1;

  *out_0 = prod / out_prod;
}