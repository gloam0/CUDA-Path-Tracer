#ifndef INPUT_H
#define INPUT_H

#include <GLFW/glfw3.h>

#include "camera.cuh"
#include "common.hpp"
#include "timer.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////
struct input_state {
    bool free_mode;
};

__constant__ static input_state d_input_state;
inline input_state h_input_state;
////////////////////////////////////////////////////////////////////////////////////////////////
namespace InputHandler {
    inline bool mouse_captured;

    void init(GLFWwindow* window, Camera* cam);
    void register_callbacks(GLFWwindow* window);
    void poll_and_handle_events(double frame_time);
    void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos);

    constexpr inline double2 window_center = {img::w * 0.5, img::h * 0.5};

    inline Timer* timer;
    inline double last_frame_time;
    inline Camera* camera;
}
////////////////////////////////////////////////////////////////////////////////////////////////

#endif //INPUT_H
