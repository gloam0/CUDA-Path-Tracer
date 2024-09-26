#ifndef CONFIG_CUH
#define CONFIG_CUH

#include <cuda_runtime.h>
#include "common.hpp"

__device__ inline unsigned int* d_randoms = nullptr;    /* array of random seeds */
__device__ inline float4* d_render_mode_buff = nullptr;

__device__ inline input_state d_input_state;

namespace render {
    __device__ constexpr color3 background_color = color3{0.7f, 0.9f, 1.f};
}

#endif //CONFIG_CUH
