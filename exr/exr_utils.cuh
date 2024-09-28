#ifndef EXR_UTILS_CUH
#define EXR_UTILS_CUH

#include <atomic>

////////////////////////////////////////////////////////////////////////////////////////////////
/// Relative paths to EXR files
namespace exr_paths {
    inline const char* day_sky_1 = "../exr/assets/DaySkyHDRI008B_8K-HDR.exr";
    inline const char* day_sky_2 = "../exr/assets/DaySkyHDRI027B_8K-HDR.exr";
    inline const char* indoor_1  = "../exr/assets/IndoorEnvironmentHDRI005_8K-HDR.exr";
    inline const char* night_1   = "../exr/assets/NightEnvironmentHDRI002_8K-HDR.exr";
    inline const char* night_2   = "../exr/assets/NightEnvironmentHDRI008_8K-HDR.exr";
};

/// Represent
struct hdr_map {
    cudaTextureObject_t tex_obj;
    cudaArray* cuda_array;
    int width;
    int height;
};

inline std::atomic<bool> env_map_loaded = false;
////////////////////////////////////////////////////////////////////////////////////////////////
void create_env_map_threaded(const char* filepath, hdr_map* map, bool* success);
void create_env_map(const char* filepath, hdr_map* map, bool* success);
void free_env_map(hdr_map map);
////////////////////////////////////////////////////////////////////////////////////////////////

#endif //EXR_UTILS_CUH