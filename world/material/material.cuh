#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "common.hpp"
#include "hit.cuh"

#include "diffuse.cuh"
#include "conductor.cuh"
#include "dielectric.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
/// All available material types
enum class material_type {
    DIFFUSE,
    CONDUCTOR,
    DIELECTRIC
};

/// Describes an entry in a unified array referencing various geometry types
struct material_instance {
    material_type type; /* type of this material instance */
    int i;              /* index of this instance in the associated material-specific SoA */
};

/// Describes a struct materials
struct materials_info {
    material_instance* instances;   /* unified array referencing various material types */
    int num_diffuse;
    int num_dielectrics;
    int num_conductors;
    int num_instances;
};

/// Collection of materials of various types
struct materials {
    diffuse_params* diffuses;
    dielectric_params* dielectrics;
    conductor_params* conductors;
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// Calls the appropriate scatter() function for the given material type.
/// @param r    The incoming ray to be scattered. Replaced with outgoing scattered ray.
/// @param h    A struct hit describing the hit.
/// @param m    Information about the material instance supplying the scatter function
/// @param mats All material-specific parameters
/// @param seed Pointer to the current random seed value.
__device__ inline color3 scatter(ray3* r, const hit* h, const material_instance& m, const materials& mats, unsigned int* seed) {
    switch (m.type) {
        default:
        case material_type::DIFFUSE:
            return scatter_diffuse(r, h, mats.diffuses[m.i], seed);
        case material_type::CONDUCTOR:
            return scatter_conductor(r, h, mats.conductors[m.i], seed);
        case material_type::DIELECTRIC:
            if (mats.dielectrics[m.i].render_method == dielectric_render_method::SCHLICK) {
                return scatter_dielectric_shlick(r, h, mats.dielectrics[m.i], seed);
            } else {
                return scatter_dielectric_fresnel(r, h, mats.dielectrics[m.i], seed);
            }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////

#endif //MATERIAL_CUH
