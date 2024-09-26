#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "diffuse.cuh"
#include "common.hpp"
#include "hit.cuh"

enum class material_type {
    Diffuse
};

struct material {
    material_type type;
    color3 albedo;
};

/// Calls the appropriate scatter() function for the given material type.
/// @param r The incoming ray to be scattered. Replaced with outgoing scattered ray.
/// @param h A struct hit describing the hit.
/// @param mat The material of the object that was hit.
/// @param seed Pointer to the current random seed value.
__device__ inline color3 scatter(ray3* r, const hit* h, const material* mat, unsigned int* seed) {
    switch (mat->type) {
        default:
        case material_type::Diffuse:
            scatter_diffuse(r, h, seed);
            break;
    }
    return mat->albedo;
}

#endif //MATERIAL_CUH
