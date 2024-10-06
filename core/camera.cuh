#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "common.hpp"
#include "math_utils.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
/// Represent the state of a camera
struct CameraParams {
    point3 c_location;          /* Camera location */
    vec3 c_direction;           /* Camera 'look at' vector */
    quaternion c_orientation;   /* Quaternion describing camera's orientation */
    double c_focal_length;      /* Length from camera center to viewport */

    float c_aperture_diameter;  /* Aperture size; f-number is focal length / aperture diameter */
    float c_focus_distance;     /* Distance to point of 'perfect focus' */
    vec3 c_focal_plane_center;  /* Vector to center of focal plane */

    float c_yaw;                /* Camera yaw */
    float c_pitch;              /* Camera pitch */

    float c_dlong;              /* Camera velocity in direction of 'look at' vector */
    float c_dlat;               /* Camera velocity perpendicular to 'look at' vector (in XZ plane)*/

    vec3 vp_w_vec;              /* Viewport width vector */
    vec3 vp_h_vec;              /* Viewport height vector */
    vec3 vp_delta_px_w;         /* Horizontal change per pixel in viewport */
    vec3 vp_delta_px_h;         /* Vertical change per pixel in viewport */

    vec3 vp_start;              /* Vector from camera center to upper-left pixel in viewport */
};
////////////////////////////////////////////////////////////////////////////////////////////////
__device__ __constant__ inline CameraParams d_camera_params;
////////////////////////////////////////////////////////////////////////////////////////////////
/// Cast a ray from the camera's location through the viewport at (vp_i, vp_j) using
/// a pinhole camera model.
/// @param vp_i Rightward distance from viewport center.
/// @param vp_j Downward distance from viewport center.
__device__ ray3 make_ray_pinhole(float vp_i, float vp_j);

/// Cast a ray from the camera's location through the viewport at (vp_i, vp_j) using
/// a thin lens camera model.
/// @param vp_i Rightward distance from viewport center.
/// @param vp_j Downward distance from viewport center.
/// @param seed Random seec.
__device__ ray3 make_ray_thin_lens(float vp_i, float vp_j, unsigned int* seed);
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
    static wasdState wasd_state;
    ////////////////////////////////////////////////////////////////////////////////////////
private:
    /// Update camera state based on current h_camera_params.c_orientation quaternion
    /// and copy derived state to device.
    void update();
    ////////////////////////////////////////////////////////////////////////////////////////
    CameraParams h_camera_params;
};
////////////////////////////////////////////////////////////////////////////////////////////////

#endif //CAMERA_CUH
