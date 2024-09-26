#ifndef RENDER_H
#define RENDER_H

#include <cuda_runtime.h>

#include "scene.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////
/// Initialize curand with a seed.
__global__ void init_curand(curandState *curand_state, unsigned long long seed);

////////////////////////////////////////////////////////////////////////////////////////////////
/// Render a frame in the CUDA-OpenGL-GLFW pipeline
void render_frame(scene* d_scene, cudaGraphicsResource* pbo_resource, int frame_count);

////////////////////////////////////////////////////////////////////////////////////////////////
/// Displays all border pixels as white, else black. Used for testing image-window alignment.
__global__ void check_borders(uchar4* out);

/// Render a scene with multi-sampling to d_ms_buffer.
__global__ void render_scene(scene* s, int frame_count);

/// Reduce multisamples in d_ms_buffer to an output array / PBO.
__global__ void reduce_multisamples(uchar4* out);

/// Trace a ray from camera center through the viewport location (vp_x, vp_y).
__device__ color3 trace_ray(float vp_x, float vp_y, scene* s, unsigned int* seed);

////////////////////////////////////////////////////////////////////////////////////////////////
#endif //RENDER_H
