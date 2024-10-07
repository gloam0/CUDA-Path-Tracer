#ifndef PLANE_CUH
#define PLANE_CUH

#include <cuda_runtime.h>

#include "common.hpp"
#include "math_utils.cuh"
#include "hit.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
/// SoA describing a collection of planes.
struct plane_params_soa {
    vec3* normals;
    float* dists;
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Determine if the ray hits this plane.
/// @param ray      the ray to test for intersection
/// @param hit_info struct to be populated with hit information
/// @param normal   normal describing plane orientation
/// @param dist     distance of closest plane point to origin, plane equation term 'D'
/// @return         whether a hit was detected
__device__ inline bool is_hit_plane(const ray3& ray, hit& hit_info, const vec3& normal, float dist) {
    /*---------------------------- intersections ---------------------------------*/
    /*
     * ray.direction (D) is assumed to be a unit vector.
     *
     * Ray R(t) = O + tD
     * Plane Equation: P(<x, y, z>) = normal . <x, y, z> + dist = 0
     * P(R(t)) = normal . (O + tD) + dist = 0
     * => normal . O + normal . tD + dist = 0
     * => t = -(normal . O + dist) / (normal . D); where normal . D != 0
     */
    float den = dot(normal, ray.direction);
    if (fabs(den) < render::eps) return false;

    float num = -(dot(normal, ray.origin) + dist);
    float t = num / den;
    if (t < render::eps) return false;

    hit_info.t = t;
    hit_info.loc = ray.origin + t * ray.direction;

    /* Treat both sides of the plane as outward / front faces */
    hit_info.normal = den > 0 ? -normal : normal;
    hit_info.is_front_face = true;

    return true;
}
////////////////////////////////////////////////////////////////////////////////////////////////

#endif //PLANE_CUH
