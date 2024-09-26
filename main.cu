#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "common.hpp"

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

    /* Initialize pixel buffer object and texture */
    GLuint pbo, tex;
    cudaGraphicsResource* pbo_resource;
    get_pbo_tex(&pbo, &tex, &pbo_resource);

    /* Allocate multisampling buffer */
    init_multisampling_buffer();

    /* Pre-generate randoms */
    init_d_randoms();

    /* Create camera */
    Camera camera;
    camera.init();

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
    timer.start_fps();

    while (!glfwWindowShouldClose(main_window)) {
        render_frame(d_scene, pbo_resource, frame_count);

        if (frame_count % 400 == 0) {
            logger << Logger::get_local_time() << " FPS: " << timer.get_current_fps() << std::endl;
        }

        double frame_time = timer.get_frame_time();
        timer.frame_now();
        InputHandler::poll_and_handle_events(frame_time);
        camera.frame_now(frame_time);
        frame_count++;
    }

    ////////////////////////////////////////////////////////////////////////////////////////
    // Clean up
    if (pbo_resource) cudaGraphicsUnregisterResource(pbo_resource);
    if (pbo) glDeleteBuffers(1, &pbo);
    if (main_window) {
        glfwDestroyWindow(main_window);
        glfwTerminate();
    }

    logger << Logger::get_local_time() << " Ran for: "
           << timer.duration<milliseconds>() << "ms" << std::endl;
    return 0;
}
