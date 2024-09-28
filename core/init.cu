#include "init.cuh"

#include <common.cuh>
#include <iostream>

#include <cuda_gl_interop.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include "common.hpp"
#include "cuda_utils.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
void glfw_error_callback(int error, const char* description) { fprintf(stderr, "Error: %s\n", description); }
////////////////////////////////////////////////////////////////////////////////////////////////
void initialize_GLFW(GLFWwindow** window) {
    if (!glfwInit()) {
        std::cerr << "glfwInit() failed." << std::endl;
        exit(-1);
    }

    glfwSetErrorCallback(glfw_error_callback);

    *window = glfwCreateWindow(img::w, img::h, "", nullptr, nullptr);
    if (!*window) {
        std::cerr << "glfwCreateWindow() failed." << std::endl;
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(*window);
}
////////////////////////////////////////////////////////////////////////////////////////////////
void initialize_OpenGL() {
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "glewInit() failed: " << glewGetErrorString(err) << std::endl;
        glfwTerminate();
        exit(-1);
    }

    /* Map world directly to NDC and disable depth testing
     * as we are just rendering frames to a 2D quad */
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);
}
////////////////////////////////////////////////////////////////////////////////////////////////
void get_pbo_tex(GLuint* pbo, GLuint* tex, cudaGraphicsResource** cuda_pbo) {
    glGenBuffers(1, pbo);                         /* Request a buffer object name in pbo */
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);   /* Declare pbo as a texture data source */
    glBufferData(                                 /* Initialize the pbo's data */
        GL_PIXEL_UNPACK_BUFFER,                       /* - texture data source */
        img::w * img::h * sizeof(uchar4),             /* - size of the window (px) */
        NULL,                                         /* - no init data */
        GL_STREAM_DRAW);                              /* - data used as source for GL drawing */

    glGenTextures(1, tex);                        /* Request a texture name in tex */
    glBindTexture(GL_TEXTURE_2D, *tex);           /* Declare tex as a 2D texture target */
    glTexParameteri(GL_TEXTURE_2D,                /* Change default of GL_NEAREST_MIPMAP_LINEAR */
        GL_TEXTURE_MIN_FILTER, GL_NEAREST);       /* to GL_NEAREST - we aren't generating mipmaps */
    glTexParameteri(GL_TEXTURE_2D,                /* Change default of GL_LINEAR to GL_NEAREST */
        GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,       /* Specifications for the 2D texture  */
        img::w, img::h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    /* Register the OpenGL pbo with CUDA as write-only */
    CHECK_ERR(cudaGraphicsGLRegisterBuffer(cuda_pbo, *pbo, cudaGraphicsMapFlagsWriteDiscard));
}

void init_render_mode_buffer() {
    float4* tmp_ptr;
    CHECK_ERR(cudaMalloc(&tmp_ptr, img::w * img::h * sizeof(float4)));
    CHECK_ERR(cudaMemcpyToSymbol(d_render_mode_buff, &tmp_ptr, sizeof(float4*), 0, cudaMemcpyHostToDevice));
}

////////////////////////////////////////////////////////////////////////////////////////////////
void init_d_randoms() {
    unsigned int* tmp_ptr;
    curandGenerator_t curand_gen;
    CHECK_ERR(cudaMalloc(&tmp_ptr, num_threads * sizeof(unsigned int)));
    CHECK_ERR(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_ERR(curandSetPseudoRandomGeneratorSeed(curand_gen, 3663747ULL));
    CHECK_ERR(curandGenerate(curand_gen, tmp_ptr, num_threads));
    CHECK_ERR(cudaMemcpyToSymbol(d_randoms, &tmp_ptr, sizeof(unsigned int*), 0, cudaMemcpyHostToDevice));
    CHECK_ERR(curandDestroyGenerator(curand_gen));
}
////////////////////////////////////////////////////////////////////////////////////////////////
