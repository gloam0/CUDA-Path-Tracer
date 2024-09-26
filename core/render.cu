#include "render.cuh"

#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h>

#include "common.hpp"
#include "cuda_utils.cuh"
#include "input.hpp"
#include "camera.cuh"
#include "hit.cuh"
#include "material.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void init_curand(curandState *curand_state, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int idx = i + j * (blockDim.x * gridDim.x) + k * (blockDim.x * gridDim.x * blockDim.y * gridDim.y);

    curand_init(seed, idx, 0, &curand_state[idx]);
}
////////////////////////////////////////////////////////////////////////////////////////////////
void render_frame(scene* d_scene, cudaGraphicsResource* pbo_resource, int frame_count) {
    /* Map the shared pbo to allow writing and get a pointer to it */
    uchar4* cuda_pbo;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&cuda_pbo, &num_bytes, pbo_resource);

    /* Render the current frame with multisampling and reduce multisamples */
    render_scene<<<grid_size, block_size>>>(d_scene, frame_count);
    CHECK_ERR(cudaGetLastError());
    reduce_multisamples<<<grid_size, block_size>>>(cuda_pbo);
    CHECK_ERR(cudaGetLastError());

    /* Unmap the shared pbo to finish writing and synchronize */
    cudaGraphicsUnmapResources(1, &pbo_resource, 0);

    /* Update the texture
     *    GL_PIXEL_UNPACK_BUFFER is bound (to the pbo) and will be used to update
     *    the texture, hence *pixels is used as an offset which we set to 0. */
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img::w, img::h, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    /* Draw the texture to the fullscreen quad */
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(-1, -1);
    glTexCoord2f(1, 0); glVertex2f(1, -1);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(-1, 1);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    /* Swap back/front buffers, handled by GLFW */
    glfwSwapBuffers(glfwGetCurrentContext());

    /* Handle events that occurred this frame (see callbacks in input.h) */
    glfwPollEvents();
}
////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void check_borders(uchar4 *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < img::w && j < img::h) {
        int index = j * img::w + i;

        unsigned char c = (i == 0) || (i == img::w-1) || (j == 0) || (j == img::h-1);
        out[index] = make_uchar4(c * 255, c * 255, c * 255, 255);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void render_scene(scene* s, int frame_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= img::w || j >= img::h || k >= multisampling_rate) return;

    int thread_index = (j * img::w * multisampling_rate) + (i * multisampling_rate) + k;
    unsigned int seed = d_randoms[thread_index];

    /* Get a random subpixel in the pixel associated with this thread */
    auto sub_x = i + xorshift32_f_norm(&seed);
    auto sub_y = j + xorshift32_f_norm(&seed);

    /* Trace this ray and write its resulting color to the multisampling buffer */
    float3 col = trace_ray(sub_x, sub_y, s, &seed);
    d_ms_buffer[thread_index] = make_uchar3(
        (unsigned char)(255.999 * col.x),
        (unsigned char)(255.999 * col.y),
        (unsigned char)(255.999 * col.z)
    );

    /* Store random state */
    d_randoms[thread_index] = xorshift32_i(&seed);
}
////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void reduce_multisamples(uchar4* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= img::w || j >= img::h || k > 0) return;

    /* Accumulate and average multisamples for pixel (i, j) */
    auto thread_idx = j * img::w + i;
    auto ms_buff_offs = multisampling_rate * thread_idx;
    uint3 accum_color{0,0,0};
    for (int c = 0; c < multisampling_rate; c++) {
        uchar3 ms_color = d_ms_buffer[ms_buff_offs + c];
        accum_color.x += ms_color.x;
        accum_color.y += ms_color.y;
        accum_color.z += ms_color.z;
    }

    out[thread_idx] = make_uchar4(
        (unsigned char)(accum_color.x / multisampling_rate),
        (unsigned char)(accum_color.y / multisampling_rate),
        (unsigned char)(accum_color.z / multisampling_rate),
        255
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////
__device__ color3 trace_ray(float vp_x, float vp_y, scene* s, unsigned int* seed) {
    ray3 r = make_ray(vp_x, vp_y);      /* Get ray through (vp_x, vp_y) */
    color3 att = color3{1.f,1.f,1.f};   /* Attenuation */

    hit best_hit;
    float best_t;
    hit this_hit;
    int depth = 0;
    do {  /* check scene objects for intersection, track nearest hit */
        best_t = -1.f;
        for (int idx = 0; idx < s->num_objects; idx++){
            if (s->spheres[idx].is_hit(r, this_hit)) {
                if (this_hit.t < best_t || best_t < 0) {
                    best_hit = this_hit;
                    best_hit.mat = &s->materials[idx];
                    best_t = this_hit.t;
                }
            }
        }
        if (best_t > 0.f) { /* scatter r and apply attenuation */
            scatter(&r,&best_hit, best_hit.mat, seed);
            att = elem_product(att, best_hit.mat->albedo);
        } else {            /* No hit, use background color and exit */
            return elem_product(att, render::background_color);
        }
        depth++;
    } while (depth < render::max_scatter_depth);

    return att;
}