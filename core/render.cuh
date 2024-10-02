#ifndef RENDER_H
#define RENDER_H

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <GLFW/glfw3.h>

#include "scene.hpp"
#include "exr_utils.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
/// Initialize curand with a seed.
__global__ void init_curand(curandState *curand_state, unsigned long long seed);

////////////////////////////////////////////////////////////////////////////////////////////////
/// Render a frame in the CUDA-OpenGL-GLFW pipeline
void render_frame(scene* d_scene, cudaGraphicsResource* pbo_resource,
    cudaTextureObject_t env_tex, double fps, int render_mode_frame_count, GLuint gl_pbo);

void render_messagebar(GLuint prev_buffer, const char* message);
////////////////////////////////////////////////////////////////////////////////////////////////
/// Displays all border pixels as white, else black. Used for testing image-window alignment.
__global__ void check_borders(uchar4* out);

/// Render a scene with multi-sampling to d_ms_buffer.
__global__ void render_scene(uchar4* out, scene* s, cudaTextureObject_t env_tex,
    int render_mode_frame_count);

/// Trace a ray from camera center through the viewport location (vp_x, vp_y).
__device__ color3 trace_ray(float vp_x, float vp_y, scene* s, cudaTextureObject_t env_tex,
    unsigned int* seed);

////////////////////////////////////////////////////////////////////////////////////////////////
#endif //RENDER_H
