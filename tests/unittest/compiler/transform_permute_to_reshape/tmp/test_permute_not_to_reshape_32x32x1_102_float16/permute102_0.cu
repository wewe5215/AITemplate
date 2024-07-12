

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/util/host_tensor.h"



#define TILE_SIZE 32
#define ITEMS_PER_THREAD 4
#define DIRECT_BLOCK_Y 4
#define DIRECT_BLOCK_Z 2

namespace {
using bfloat16 = __nv_bfloat16;

template<typename T>
__global__ void permute102_tiled_kernel(T* output,
                                        const T *input,
                                        const int M,
                                        const int N,
                                        const int D,
                                        const int n) {
  __shared__ T shbuf[TILE_SIZE * TILE_SIZE];

  const int nD = n * D;
  const int ND = N * D;
  const int MD = M * D;
  const int bxn = blockIdx.x * n;
  const int DT = D * TILE_SIZE;
  int x, y, i, tid, threadIdxY;

  if (threadIdx.x < nD) {
    x = blockIdx.x * nD + threadIdx.x;
    if (x < ND) {
      threadIdxY = threadIdx.y;
      if ((blockIdx.y + 1) * TILE_SIZE <= M) {
        #pragma unroll
        for (i = 0; i < ITEMS_PER_THREAD; ++i) {
          y = blockIdx.y * TILE_SIZE + threadIdxY;
          shbuf[threadIdxY * TILE_SIZE + (D * threadIdxY + threadIdx.x) % TILE_SIZE] =
            input[y * ND + x];
          threadIdxY += TILE_SIZE / ITEMS_PER_THREAD;
        }
      } else {
        #pragma unroll
        for (i = 0; i < ITEMS_PER_THREAD; ++i) {
          y = blockIdx.y * TILE_SIZE + threadIdxY;
          if (y >= M) break;
          shbuf[threadIdxY * TILE_SIZE + (D * threadIdxY + threadIdx.x) % TILE_SIZE] =
            input[y * ND + x];
          threadIdxY += TILE_SIZE / ITEMS_PER_THREAD;
        }
      }
    }
  }

  __syncthreads();

  threadIdxY = threadIdx.y;
  if ((blockIdx.x + 1) * n <= N) {
    if ((blockIdx.y + 1) * TILE_SIZE * D <= MD) {
      #pragma unroll
      for (i = 0; i < ITEMS_PER_THREAD; i++) {
        tid = threadIdxY * TILE_SIZE + threadIdx.x;
        x = tid % DT;
        y = tid / DT;
        output[(bxn + y) * MD + blockIdx.y * DT + x] =
          shbuf[(x / D) * TILE_SIZE + (D * y + x) % TILE_SIZE];
        threadIdxY += TILE_SIZE / ITEMS_PER_THREAD;
        if (threadIdxY >= nD) break;
      }
    } else {
      #pragma unroll
      for (i = 0; i < ITEMS_PER_THREAD; i++) {
        tid = threadIdxY * TILE_SIZE + threadIdx.x;
        x = tid % DT;
        y = tid / DT;
        if (blockIdx.y * DT + x < MD) {
          output[(bxn + y) * MD + blockIdx.y * DT + x] =
            shbuf[(x / D) * TILE_SIZE + (D * y + x) % TILE_SIZE];
        }
        threadIdxY += TILE_SIZE / ITEMS_PER_THREAD;
        if (threadIdxY >= nD) break;
      }
    }
  } else {
    if ((blockIdx.y + 1) * TILE_SIZE * D <= MD) {
      #pragma unroll
      for (i = 0; i < ITEMS_PER_THREAD; i++) {
        tid = threadIdxY * TILE_SIZE + threadIdx.x;
        x = tid % DT;
        y = tid / DT;
        if (bxn + y < N) {
          output[(bxn + y) * MD + blockIdx.y * DT + x] =
            shbuf[(x / D) * TILE_SIZE + (D * y + x) % TILE_SIZE];
        }
        threadIdxY += TILE_SIZE / ITEMS_PER_THREAD;
        if (threadIdxY >= nD) break;
      }
    } else {
      #pragma unroll
      for (i = 0; i < ITEMS_PER_THREAD; i++) {
        tid = threadIdxY * TILE_SIZE + threadIdx.x;
        x = tid % DT;
        y = tid / DT;
        if (bxn + y < N && blockIdx.y * DT + x < MD) {
          output[(bxn + y) * MD + blockIdx.y * DT + x] =
            shbuf[(x / D) * TILE_SIZE + (D * y + x) % TILE_SIZE];
        }
        threadIdxY += TILE_SIZE / ITEMS_PER_THREAD;
        if (threadIdxY >= nD) break;
      }
    }
  }
}

template <typename T>
__global__ void permute102_direct_kernel(T* output,
                                         const T *input,
                                         const int M,
                                         const int N,
                                         const int D) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < D && y < N) {
    int bound = min(M, (blockIdx.z + 1) * TILE_SIZE);
    for (int z = blockIdx.z * TILE_SIZE + threadIdx.z; z < bound; z += DIRECT_BLOCK_Z) {
      output[y * M * D + z * D + x] = input[z * N * D + y * D + x];
    }
  }
}

template <typename T>
void permute102_launcher(const void* in_ptr,
                         void* out_ptr,
                         int x_dim0,
                         int x_dim1,
                         int x_dim2,
                         cudaStream_t stream) {
  const int M = x_dim0;
  const int N = x_dim1;
  const int D = x_dim2;

  if (D <= 16) {
    // each warp reads n x d coalesced items of input
    const int d = min(TILE_SIZE, D);
    const int n = TILE_SIZE / d;

    dim3 grid((N + n - 1) / n, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE / ITEMS_PER_THREAD);

    permute102_tiled_kernel<T><<<grid, block, 0, stream>>>(
      static_cast<T*>(out_ptr),
      static_cast<const T*>(in_ptr),
      M,
      N,
      D,
      n
    );
  } else {
    dim3 grid((D + 31) / 32, (N + DIRECT_BLOCK_Y - 1) / DIRECT_BLOCK_Y, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(32, DIRECT_BLOCK_Y, DIRECT_BLOCK_Z);  // x = 32, the warp size

    permute102_direct_kernel<T><<<grid, block, 0, stream>>>(
      static_cast<T*>(out_ptr),
      static_cast<const T*>(in_ptr),
      M,
      N,
      D
    );
  }
}
} // namespace

void permute102_0 (
    const void* in_ptr,
    void* out_ptr,
    int64_t x_dim0,
    int64_t x_dim1,
    int64_t x_dim2,
    cudaStream_t stream
) {
  if (x_dim0 == 0 || x_dim1 == 0 || x_dim2 == 0) {
    // empty input: nothing to do
    return;
  }
  if (!in_ptr) {
    throw std::runtime_error("in_ptr is NULL!");
  }
  if (!out_ptr) {
    throw std::runtime_error("out_ptr is NULL!");
  }
  

  if (x_dim2 % 8 == 0) {
    permute102_launcher<float4>(
        in_ptr,
        out_ptr,
        x_dim0,
        x_dim1,
        x_dim2 / 8,
        stream
    );
  } else if (x_dim2 % 4 == 0) {
    permute102_launcher<float2>(
        in_ptr,
        out_ptr,
        x_dim0,
        x_dim1,
        x_dim2 / 4,
        stream
    );
  } else if (x_dim2 % 2 == 0) {
    permute102_launcher<float>(
        in_ptr,
        out_ptr,
        x_dim0,
        x_dim1,
        x_dim2 / 2,
        stream
    );
  } else {
    permute102_launcher<half>(
        in_ptr,
        out_ptr,
        x_dim0,
        x_dim1,
        x_dim2,
        stream
    );
  }

  return;
}
