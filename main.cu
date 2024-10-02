#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <atomic>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "common.hpp"
#include "cuda_utils.cuh"
#include "exr_utils.cuh"

#include "logger.hpp"
#include "timer.hpp"

#include "camera.cuh"
#include "scene.hpp"
#include "render.cuh"
#include "init.cuh"
#include "input.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////
int main() {
    ////////////////////////////////////////////////////////////////////////////////////////
    // Init
    Logger logger("./log.txt");
    Timer timer;

    /* Log some device info */
    int device;
    cudaDeviceProp props;
    CHECK_ERR(cudaGetDevice(&device));
    CHECK_ERR(cudaGetDeviceProperties(&props, device));

    logger << props.name
        << "\nTotal global memory:      " << props.totalGlobalMem
        << "\nShared memory per block:  " << props.sharedMemPerBlock
        << "\nReserved shmem per block: " << props.reservedSharedMemPerBlock
        << "\nShared memory per SM:     " << props.sharedMemPerMultiprocessor
        << "\nSM count:                 " << props.multiProcessorCount
        << "\nWarp size:                " << props.warpSize << '\n' << std::endl;

    /* Create main window and initialize OpenGL */
    GLFWwindow* main_window = nullptr;
    initialize_GLFW(&main_window);
    initialize_OpenGL();
    initialize_ImGui(main_window);

    /* Initialize pixel buffer object and texture */
    GLuint gl_pbo, gl_tex;
    cudaGraphicsResource* pbo_resource;
    get_pbo_tex(&gl_pbo, &gl_tex, &pbo_resource);

    /* Pre-generate randoms */
    init_d_randoms();

    /* Create buffer render mode color accumulation */
    init_render_mode_buffer();

    /* Create camera */
    Camera camera;
    camera.init();

    /* Load EXR environment map */
    hdr_map map;
    bool success;
    if (render::use_hdr) {
        create_env_map_threaded(exr_paths::indoor_1, &map, &success);
        /* prevent the window from being detected as unresponsive */
        while (!env_map_loaded) glfwPollEvents();
        if (!success) logger << "HDR load failed, using render::background_color" << std::endl;
    }

    /* Create and setup input handler */
    InputHandler::init(main_window, &camera);
    InputHandler::register_callbacks(main_window);

    /* Create the scene and copy it to device memory */
    scene* h_scene = create_scene();
    scene* d_scene = copy_scene_to_device(h_scene);
    free_scene(h_scene);

    /* set buffer swap interval (enable/disable vsync) */
    glfwSwapInterval(render::vsync);

    logger << Logger::get_local_time() << " Initialization completed in: "
           << timer.lap<milliseconds>() << "ms" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////
    // Render
    int frame_count = 0;
    int render_mode_frame_count = 1;
    double fps = 0.;
    timer.start_fps();

    while (!glfwWindowShouldClose(main_window)) {
        render_frame(d_scene, pbo_resource, map.tex_obj, fps, render_mode_frame_count, gl_pbo);

        fps = timer.get_current_fps();
        if (frame_count % 400 == 0) {
            logger << Logger::get_local_time() << " FPS: " << fps << std::endl;
        }

        double frame_time = timer.get_frame_time();
        timer.frame_now();
        InputHandler::poll_and_handle_events(frame_time);

        /* If render mode was entered this frame, reset the render mode frame counter */
        if (h_input_state.render_mode_first_frame) {
            render_mode_frame_count = 0;
            h_input_state.render_mode_first_frame = false;
        }

        camera.frame_now(frame_time);
        frame_count++;
        render_mode_frame_count++;
    }

    ////////////////////////////////////////////////////////////////////////////////////////
    // Clean up
    if (pbo_resource) cudaGraphicsUnregisterResource(pbo_resource);
    if (gl_pbo) glDeleteBuffers(1, &gl_pbo);
    if (main_window) {
        glfwDestroyWindow(main_window);
        glfwTerminate();
    }

    logger << Logger::get_local_time() << " Ran for: "
           << timer.duration<milliseconds>() << "ms" << std::endl;
    return 0;
}
