#ifndef CONFIG_CUH
#define CONFIG_CUH

#include <cuda_runtime.h>
#include "common.hpp"

__device__ inline uchar3* d_ms_buffer = nullptr;        /* multisampling buffer */
__device__ inline unsigned int* d_randoms = nullptr;    /* array of random seeds */

#endif //CONFIG_CUH
