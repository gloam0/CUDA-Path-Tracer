#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "common.hpp"
#include "hit.cuh"

#include "diffuse.cuh"
#include "conductor.cuh"
#include "dielectric.cuh"

enum class material_type {
    Diffuse,
    Conductor,
    Dielectric
};

union mat_properties {
    conductor_params conductor;
    dielectric_params dielectric;
};

struct material {
    material_type type;
    color3 albedo;
    mat_properties props;
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
            return elem_product(mat->albedo, scatter_diffuse(r, h, seed));
        case material_type::Conductor:
            return elem_product(mat->albedo, scatter_conductor(r, h, &mat->props.conductor, seed));
        case material_type::Dielectric:
            if (mat->props.dielectric.render_method == dielectric_render_method::SHLICK) {
                return elem_product(mat->albedo, scatter_dielectric_shlick(r, h, &mat->props.dielectric, seed));
            } else {
                return elem_product(mat->albedo, scatter_dielectric_fresnel(r, h, &mat->props.dielectric, seed));
            }
    }
}

#endif //MATERIAL_CUH
