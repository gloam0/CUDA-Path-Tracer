#ifndef CONFIG_H
#define CONFIG_H

#if defined(__CUDACC__) // NVCC
   #define ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define ALIGN(n) __declspec(align(n))
#else
  #error "ALIGN macro (common.hpp): not defined for host compiler."
#endif

////////////////////////////////////////////////////////////////////////////////////////////////
// Type aliases and primitives
using color3 = float3;
using vec3 = float3;
using point3 = float3;

struct ray3 {
    point3  origin;
    vec3    direction;
};

struct input_state {
    bool free_mode = true;
    bool render_mode_first_frame = false;
    bool save_this_frame = false;
};

inline input_state h_input_state;

////////////////////////////////////////////////////////////////////////////////////////////////
// Image/frame constants
namespace img {
    /* configurable */
    constexpr int       w =             2200;
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
    constexpr double    init_aperture_diameter= 0.1;
    constexpr double    init_focus_distance =   8.;
    constexpr point3    init_camera_loc =       vec3{0, 1, 0};
    constexpr vec3      init_camera_dir =       vec3{0, 0, -1}; /* these vecs should be normalized */
    constexpr vec3      init_camera_up =        vec3{0, 1, 0};
    constexpr vec3      init_camera_right =     vec3{1, 0, 0};
    constexpr float     look_sensitivity =      0.002;

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
    constexpr int       max_scatter_depth =     30;
    constexpr int       vsync =                 0;
    constexpr float     eps =                   1e-4;
    inline    bool      use_hdr =               true;

    /* derived */
    constexpr float     eps_sq = eps * eps;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// Input/state constants
namespace state {
    constexpr bool free_mode = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel constants
constexpr dim3      block_size(8, 8);
constexpr dim3      grid_size((img::w + block_size.x - 1) / block_size.x,
                              (img::h + block_size.y - 1) / block_size.y);
constexpr size_t    num_threads = grid_size.x * grid_size.y * grid_size.z * block_size.x * block_size.y * block_size.z;

////////////////////////////////////////////////////////////////////////////////////////////////
#endif //CONFIG_H
