#ifndef INIT_H
#define INIT_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>

////////////////////////////////////////////////////////////////////////////////////////////////
void glfw_error_callback(int error, const char* description);
void initialize_GLFW(GLFWwindow** window);
void initialize_OpenGL();
void get_pbo_tex(GLuint *pbo, GLuint *tex, cudaGraphicsResource** cuda_pbo);
void init_render_mode_buffer();
void init_d_randoms();
////////////////////////////////////////////////////////////////////////////////////////////////

#endif //INIT_H
