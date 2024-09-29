#include "input.hpp"

#include "cuda_utils.cuh"
#include "camera.cuh"

namespace InputHandler {
    void init(GLFWwindow* window, Camera* cam) {
        // Initialize state
        h_input_state.free_mode = state::free_mode;
        mouse_captured = true;
        camera = cam;

        // Create and initialize memory on device
        CHECK_ERR(cudaMemcpyToSymbol(d_input_state, &h_input_state, sizeof(input_state)));

        /* Hide cursor and place it at the center of the window */
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        if (glfwRawMouseMotionSupported())
            glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
        glfwSetCursorPos(window, window_center.x, window_center.y);
    }

    void register_callbacks(GLFWwindow* window) {
        glfwSetKeyCallback(window, key_callback);
        glfwSetScrollCallback(window, scroll_callback);
        glfwSetCursorPosCallback(window, cursor_pos_callback);
    }

    void poll_and_handle_events(double frame_time) {
        last_frame_time = frame_time;
        glfwPollEvents();

        CHECK_ERR(cudaMemcpyToSymbol(d_input_state, &h_input_state, sizeof(input_state)));
    }

    void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (action == GLFW_PRESS) {
            switch (key) {
                case GLFW_KEY_SPACE:
                    h_input_state.free_mode = !h_input_state.free_mode;
                    /* set flag, to be unset once render_frame_counter is reset in main */
                    if (!h_input_state.free_mode)
                        h_input_state.render_mode_first_frame = true;
                break;
                case GLFW_KEY_M:
                    if (mouse_captured) {
                        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                        if (glfwRawMouseMotionSupported())
                            glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
                    } else {
                        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                        if (glfwRawMouseMotionSupported())
                            glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
                    }
                    mouse_captured = !mouse_captured;
                break;
                case GLFW_KEY_P:
                    h_input_state.save_this_frame = true;
                break;
                case GLFW_KEY_W:
                    Camera::wasd_state.W = true;
                break;
                case GLFW_KEY_A:
                    Camera::wasd_state.A = true;
                break;
                case GLFW_KEY_S:
                    Camera::wasd_state.S = true;
                break;
                case GLFW_KEY_D:
                    Camera::wasd_state.D = true;
                break;
                default:
                    break;
            }
        } else if (action == GLFW_RELEASE) {
            switch (key) {
                case GLFW_KEY_W:
                    Camera::wasd_state.W = false;
                break;
                case GLFW_KEY_A:
                    Camera::wasd_state.A = false;
                break;
                case GLFW_KEY_S:
                    Camera::wasd_state.S = false;
                break;
                case GLFW_KEY_D:
                    Camera::wasd_state.D = false;
                break;
                default:
                    break;
            }
        }
    }

    void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        if (!h_input_state.free_mode) return;
        camera->scroll_zoom(float(yoffset) * view::scroll_sensitivity);
    }

    void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos) {
        if (!h_input_state.free_mode) return;
        if (!mouse_captured) return;
        if (xpos == window_center.x && ypos == window_center.y) return;

        camera->mouse_rotate(float2{-float(xpos - window_center.x), -float(ypos - window_center.y)});
        glfwSetCursorPos(window, window_center.x, window_center.y);
    }
}

