#include "camera.cuh"

#include <cuda_runtime.h>

#include "common.hpp"
#include "cuda_utils.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
__device__ ray3 make_ray(float vp_i, float vp_j) {
    vec3 vp_point = d_camera_params.vp_start + vp_i * d_camera_params.vp_delta_px_w
                                             + vp_j * d_camera_params.vp_delta_px_h;
    vec3 direction = normalize(vp_point - d_camera_params.c_location);
    return ray3{
        d_camera_params.c_location,
        direction
    };
}
////////////////////////////////////////////////////////////////////////////////////////////////
void Camera::init() {
    h_camera_params.c_location = view::init_camera_loc;
    h_camera_params.c_direction = normalize(view::init_camera_dir);

    h_camera_params.c_yaw = 0.0f;
    h_camera_params.c_pitch = 0.0f;

    h_camera_params.c_dlat = 0.f;
    h_camera_params.c_dlong = 0.f;

    h_camera_params.c_orientation = quaternion{1.f, 0.f, 0.f, 0.f};

    h_camera_params.focal_length = view::init_focal_length;

    update();
}
////////////////////////////////////////////////////////////////////////////////////////////////
void Camera::update() {
    if (!h_input_state.free_mode) return;

    vec3 c_direction = rotate_v(view::init_camera_dir, h_camera_params.c_orientation);
    vec3 c_up = rotate_v(normalize(view::init_camera_up), h_camera_params.c_orientation);
    vec3 c_right = rotate_v(view::init_camera_right, h_camera_params.c_orientation);

    h_camera_params.c_direction = c_direction;

    h_camera_params.vp_w_vec = view::w * c_right;
    h_camera_params.vp_h_vec = view::h * c_up;

    h_camera_params.vp_delta_px_w = h_camera_params.vp_w_vec / img::w;
    h_camera_params.vp_delta_px_h = h_camera_params.vp_h_vec / img::h;

    h_camera_params.vp_start =
        h_camera_params.c_location                                              /* camera loc */
      + h_camera_params.focal_length * h_camera_params.c_direction              /* vector from camera to vp center */
      - 0.5 * (h_camera_params.vp_w_vec + h_camera_params.vp_h_vec)             /* center to top left vp */
      + 0.5 * (h_camera_params.vp_delta_px_w + h_camera_params.vp_delta_px_h);  /* to px center (1/2 px dimensions) */

    CHECK_ERR(cudaMemcpyToSymbol(d_camera_params, &h_camera_params, sizeof(CameraParams)));
}
////////////////////////////////////////////////////////////////////////////////////////////////
void Camera::frame_now(double frame_time) {
    if (!h_input_state.free_mode) return;

    float dt = static_cast<float>(frame_time) / 1000000.0f;
    float accel = (view::move_accel_scale + view::move_decel_scale) * dt;
    float decel = view::move_decel_scale * dt;

    /* Apply acceleration based on key states */
    if (wasd_state.W)
        h_camera_params.c_dlong += accel;
    if (wasd_state.S)
        h_camera_params.c_dlong -= accel;
    if (wasd_state.A)
        h_camera_params.c_dlat -= accel;
    if (wasd_state.D)
        h_camera_params.c_dlat += accel;

    /* Apply deceleration when keys are not pressed */
    if (h_camera_params.c_dlong > 0.f) {
        h_camera_params.c_dlong -= decel;
        if (h_camera_params.c_dlong < 0.f) h_camera_params.c_dlong = 0.f;
    } else if (h_camera_params.c_dlong < 0.f) {
        h_camera_params.c_dlong += view::move_decel_scale * dt;
        if (h_camera_params.c_dlong > 0.f) h_camera_params.c_dlong = 0.f;
    }

    if (h_camera_params.c_dlat > 0.f) {
        h_camera_params.c_dlat -= decel;
        if (h_camera_params.c_dlat < 0.f) h_camera_params.c_dlat = 0.f;
    } else if (h_camera_params.c_dlat < 0.f) {
        h_camera_params.c_dlat += decel;
        if (h_camera_params.c_dlat > 0.f) h_camera_params.c_dlat = 0.f;
    }

    h_camera_params.c_dlong = clamp(h_camera_params.c_dlong, -view::max_velocity, view::max_velocity);
    h_camera_params.c_dlat = clamp(h_camera_params.c_dlat, -view::max_velocity, view::max_velocity);

    /* Update position based on velocity and delta time */
    vec3 c_forward = h_camera_params.c_direction;
    vec3 c_right = normalize(rotate_v(view::init_camera_right, h_camera_params.c_orientation));
    h_camera_params.c_location +=
        c_forward * float(h_camera_params.c_dlong * dt) +
        c_right * float(h_camera_params.c_dlat * dt);

    update();
}
////////////////////////////////////////////////////////////////////////////////////////////////
void Camera::scroll_zoom(float d_focal_length) {
    if (!h_input_state.free_mode) return;

    h_camera_params.focal_length += d_focal_length;
    h_camera_params.focal_length = clamp(h_camera_params.focal_length,
                                         view::min_focal_length,
                                         view::max_focal_length);
}
////////////////////////////////////////////////////////////////////////////////////////////////
void Camera::mouse_rotate(const float2& mouse_delta) {
    if (!h_input_state.free_mode) return;

    /* Convert mouse movement to rotation angles */
    float delta_yaw = mouse_delta.x * view::look_sensitivity;
    float delta_pitch = mouse_delta.y * view::look_sensitivity;

    h_camera_params.c_yaw += delta_yaw;
    h_camera_params.c_pitch += delta_pitch;

    /* Create rotor quaternions for yaw and pitch rotations */
    quaternion yaw_rotation = make_rotor_quaternion(view::init_camera_up, h_camera_params.c_yaw);
    quaternion pitch_rotation = make_rotor_quaternion(view::init_camera_right, h_camera_params.c_pitch);

    /* Combine rotations */
    h_camera_params.c_orientation = yaw_rotation * pitch_rotation;

    /* Call update to recalculate other parameters */
    update();
}
////////////////////////////////////////////////////////////////////////////////////////////////
Camera::wasdState Camera::wasd_state;
////////////////////////////////////////////////////////////////////////////////////////////////
