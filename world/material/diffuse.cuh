#ifndef DIFFUSE_CUH
#define DIFFUSE_CUH

#include "common.hpp"
#include "math_utils.cuh"

struct diffuse_params {
    color3 albedo;
};

/// Material-specific scatter function for a diffuse (Lambertian) material.
__device__ inline color3 scatter_diffuse(ray3* r, const hit* h, const diffuse_params& params, unsigned int* seed) {
    /* Lambertian diffuse - random directional scatter away from surface */
    vec3 scatter_direction = h->normal + random_unit_vector(seed);
    if (near_zero(scatter_direction)) scatter_direction = h->normal;

    /* Incoming ray becomes scattered ray */
    r->origin = h->loc;
    r->direction = normalize(scatter_direction);

    return params.albedo;
}


#endif //DIFFUSE_CUH
