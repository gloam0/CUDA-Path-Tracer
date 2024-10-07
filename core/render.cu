#include "render.cuh"

#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h>

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_opengl2.h>

#include "common.hpp"
#include "cuda_utils.cuh"
#include "input.hpp"
#include "camera.cuh"
#include "hit.cuh"
#include "material.cuh"
#include "frame_to_image.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void init_curand(curandState *curand_state, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int idx = i + j * (blockDim.x * gridDim.x) + k * (blockDim.x * gridDim.x * blockDim.y * gridDim.y);

    curand_init(seed, idx, 0, &curand_state[idx]);
}
////////////////////////////////////////////////////////////////////////////////////////////////
void render_frame(
        scene* d_scene,
        cudaGraphicsResource* pbo_resource,
        cudaTextureObject_t env_tex,
        double fps,
        int render_mode_frame_count,
        GLuint gl_pbo)
{
    static int img_count = 0;

    /* Map the shared pbo to allow writing and get a pointer to it */
    uchar4* cuda_pbo;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&cuda_pbo, &num_bytes, pbo_resource);

    /* Render the current frame with multisampling and reduce multisamples */
    render_scene<<<grid_size, block_size>>>(cuda_pbo, d_scene, env_tex, render_mode_frame_count);
    CHECK_ERR(cudaGetLastError());

    if (h_input_state.save_this_frame) {
        save_frame_as_image(cuda_pbo, 100, ("./frame" + std::to_string(img_count++) + ".jpg").c_str());
        h_input_state.save_this_frame = false;
    }

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

    /* Update and render messagebar */
    char m[64];
    snprintf(m, sizeof(m), "FPS:%5d    Frames:%6d", (int)fps, render_mode_frame_count);
    render_messagebar(gl_pbo, m);

    /* Swap back/front buffers, handled by GLFW */
    glfwSwapBuffers(glfwGetCurrentContext());

    /* Handle events that occurred this frame (see callbacks in input.h) */
    glfwPollEvents();
}

void render_messagebar(GLuint prev_buffer, const char* message) {
    /* Unbind the PBO to prevent it from affecting ImGui's rendering */
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    /* Start imgui frame*/
    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    /* Define message bar */
    float barHeight = 32.0f; // Adjust as needed
    ImGui::SetNextWindowPos(ImVec2(0, img::h - barHeight));
    ImGui::SetNextWindowSize(ImVec2(16.f * 13, (float)barHeight));
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize
        | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar
        | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs;

    /* Create message bar and render*/
    ImGui::Begin("messagebar", nullptr, window_flags);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.f, 1.f, 1.f, 1.f));
    ImGui::Text(message);
    ImGui::PopStyleColor();
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

    /* Rebind the previous buffer */
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, prev_buffer);
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
__global__ void render_scene(
        uchar4* out,
        scene* s,
        cudaTextureObject_t env_tex,
        int render_mode_frame_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= img::w || j >= img::h ) return;

    int thread_index = j * img::w + i;
    unsigned int seed = d_randoms[thread_index];

    /* Get a random subpixel in the pixel associated with this thread */
    auto sub_x = i + xorshift32_f_norm(&seed);
    auto sub_y = j + xorshift32_f_norm(&seed);

    /* Trace this ray and write its resulting color to the multisampling buffer */
    float3 col = trace_ray(sub_x, sub_y, s, env_tex, &seed);

    if (d_input_state.free_mode) {
        /* Free mode: write frame directly to out buffer */
        /* Apply tone mapping and color correction */
        color3 mapped_col = reinhard_tone_map(col);
        mapped_col = gamma_correct(mapped_col);

        out[thread_index] = make_uchar4(
            (unsigned char)(255.999f * mapped_col.x),
            (unsigned char)(255.999f * mapped_col.y),
            (unsigned char)(255.999f * mapped_col.z),
            255
        );
    } else {
        /* Render mode */
        /* clear render_mode_buffer on first frame in render mode */
        if (d_input_state.render_mode_first_frame)
            d_render_mode_buff[thread_index] = make_float4(0.f, 0.f, 0.f, 0.f);

        /* Accumulate color */
        d_render_mode_buff[thread_index] += make_float4(col.x, col.y, col.z, 0.f);
        float inv_frame_count = 1.f / render_mode_frame_count;

        /* Get averaged color and apply tone mapping and gamma correction */
        col = make_float3(d_render_mode_buff[thread_index].x * inv_frame_count,
                          d_render_mode_buff[thread_index].y * inv_frame_count,
                          d_render_mode_buff[thread_index].z * inv_frame_count);
        col = reinhard_tone_map(col);
        col = gamma_correct(col);

        /* Write current color average to out buffer */
        out[thread_index] = make_uchar4(
            (unsigned char)(255.999f * col.x),
            (unsigned char)(255.999f * col.y),
            (unsigned char)(255.999f * col.z),
            255
        );
    }

    /* Store random state */
    d_randoms[thread_index] = xorshift32_i(&seed);
}
////////////////////////////////////////////////////////////////////////////////////////////////
__device__ color3 trace_ray(
        float vp_x,
        float vp_y,
        scene* s,
        cudaTextureObject_t env_tex,
        unsigned int* seed)
{
    ray3 r = make_ray_pinhole(vp_x, vp_y);      /* Get ray through (vp_x, vp_y) */
    //ray3 r = make_ray_thin_lens(vp_x, vp_y, seed);
    color3 curr_attenuation = color3{1.f,1.f,1.f};   /* Attenuation */

    hit best_hit;
    float best_t;
    hit this_hit;
    int depth = 0;
    do {  /* check scene objects for intersection, track nearest hit */
        best_t = -1.f;
        for (int idx = 0; idx < s->geoms_info.num_instances; idx++) {
            geometry_instance g = s->geoms_info.instances[idx];
            if (is_hit_dispatch(r, this_hit, g, s->geoms)) {
                if (this_hit.t < best_t || best_t < 0.f) {
                    best_hit = this_hit;
                    best_hit.mat_id = s->geoms_info.instances[idx].material_id;
                    best_t = this_hit.t;
                }
            }
        }
        if (best_t > 0.f) {
            /* scatter r and apply attenuation */
            curr_attenuation = elem_product(curr_attenuation,
                                            scatter(&r, &best_hit, s->mats_info.instances[best_hit.mat_id], s->mats, seed));
        } else {
            /* No hit, sample HDR or use background color and exit */
            if (!d_use_hdr) return elem_product(curr_attenuation, render::background_color);

            /* Convert to spherical coordinates */
            float3 dir = r.direction;
            float theta = acosf(dir.y);
            float phi = atan2f(dir.z, dir.x);
            if (phi < 0.f) phi += 2.f * M_PIf;

            /* Normalize angles for texture coordinates */
            float u = phi / (2.f * M_PIf);
            float v = theta / M_PIf;

            /* Sample the environment map */
            float4 env_col = tex2D<float4>(env_tex, u, v);
            color3 env_color = {env_col.x, env_col.y, env_col.z};

            /* Return the sampled environment color scaled by current attenuation */
            return elem_product(curr_attenuation, env_color);
        }
        depth++;
    } while (depth < render::max_scatter_depth
        && (curr_attenuation.x > 0 || curr_attenuation.y > 0 || curr_attenuation.z > 0));

    return curr_attenuation;
}