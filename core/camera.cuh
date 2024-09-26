#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "common.hpp"
#include "math_utils.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
struct CameraParams {
    point3 c_location;
    vec3 c_direction;
    quaternion c_orientation;
    double focal_length;

    float c_yaw;
    float c_pitch;

    float c_long;
    float c_lat;

    float c_dlong;
    float c_dlat;

    vec3 vp_w_vec;
    vec3 vp_h_vec;
    vec3 vp_delta_px_w;
    vec3 vp_delta_px_h;

    vec3 vp_start;
};
////////////////////////////////////////////////////////////////////////////////////////////////
__device__ __constant__ inline CameraParams d_camera_params;
////////////////////////////////////////////////////////////////////////////////////////////////
/// Cast a ray from the camera's location through the viewport at (vp_i, vp_j).
/// @param vp_i Rightward distance from viewport center.
/// @param vp_j Downward distance from viewport center.
__device__ ray3 make_ray(float vp_i, float vp_j);
////////////////////////////////////////////////////////////////////////////////////////////////
class Camera {
private:
    ////////////////////////////////////////////////////////////////////////////////////////
    /// Represent the current state of keys used for camera movement.
    struct wasdState {
        bool W = false;
        bool A = false;
        bool S = false;
        bool D = false;
    };
    ////////////////////////////////////////////////////////////////////////////////////////
public:
    /// Derive camera state from configuration constants (see common.hpp).
    void init();
    /// Update camera state based on current h_camera_params.c_orientation quaternion
    /// and copy derived state to device.
    void update();
    /// Call after each frame for time-dependent state updates.
    /// @param frame_time Time since last frame (microseconds).
    void frame_now(double frame_time);
    ////////////////////////////////////////////////////////////////////////////////////////
    /// Zoom the camera by updating the focal length.
    /// @param d_focal_length Change in focal length.
    void scroll_zoom(float d_focal_length);
    /// Rotate the camera based on a mouse movement.
    /// @param mouse_delta 2D vector representing the mouse movement.
    void mouse_rotate(const float2& mouse_delta);
    ////////////////////////////////////////////////////////////////////////////////////////
    CameraParams h_camera_params;
    static wasdState wasd_state;
};
////////////////////////////////////////////////////////////////////////////////////////////////

#endif //CAMERA_CUH
