#ifndef CONFIG_H
#define CONFIG_H

////////////////////////////////////////////////////////////////////////////////////////////////
// Type aliases and primitives
using color3 = float3;
using vec3 = float3;
using point3 = float3;

struct ray3 {
    point3  origin;
    vec3    direction;
};

////////////////////////////////////////////////////////////////////////////////////////////////
// Image/frame constants
namespace img {
    /* configurable */
    constexpr int       w =             1600;
    constexpr double    desired_ar =    16. / 9.;

    /* derived */
    constexpr int       h =             int(w / desired_ar);
    constexpr double    ar =            double(w) / h;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// Camera constants (viewport, movement)
namespace view {
    /* configurable */
    constexpr double    h =                     2.;
    constexpr double    init_focal_length =     1.;
    constexpr point3    init_camera_loc =       vec3{0, 0, 0};
    constexpr vec3      init_camera_dir =       vec3{0, 0, -1}; /* these vecs should be normalized */
    constexpr vec3      init_camera_up =        vec3{0, 1, 0};
    constexpr vec3      init_camera_right =     vec3{1, 0, 0};
    constexpr float     look_sensitivity =      0.1;

    constexpr float     move_accel_scale =      10.;  /* du/s */
    constexpr float     move_decel_scale =      5.;   /* du/s */

    constexpr float     max_velocity =          7.;   /* u/s */

    constexpr float     scroll_sensitivity =    0.1;
    constexpr float     min_focal_length =      0.01;
    constexpr float     max_focal_length =      10.;

    /* derived */
    constexpr double    aspect_ratio =          double(img::w) / img::h;
    constexpr double    w =                     h * aspect_ratio;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// Rendering constants
namespace render {
    /* configurable */
    constexpr int       max_scatter_depth =     40;
    constexpr int       vsync =                 0;
    constexpr float     self_intersect_eps =    1e-3;

    /* derived */
    constexpr float     self_intersect_eps_sq = self_intersect_eps * self_intersect_eps;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// Input/state constants
namespace state {
    constexpr bool free_mode = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel constants
constexpr int       ms_axis_length =    3;  /* sqrt of desired multisampling rate */
constexpr int       multisampling_rate = ms_axis_length * ms_axis_length;
constexpr dim3      block_size(8, 8, multisampling_rate);
constexpr dim3      grid_size((img::w + block_size.x - 1) / block_size.x,
                              (img::h + block_size.y - 1) / block_size.y,
                              (multisampling_rate + block_size.z - 1) / block_size.z);
constexpr size_t    num_threads = grid_size.x * grid_size.y * grid_size.z * block_size.x * block_size.y * block_size.z;

////////////////////////////////////////////////////////////////////////////////////////////////
#endif //CONFIG_H
