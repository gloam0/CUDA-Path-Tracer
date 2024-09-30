#ifndef PLANE_CUH
#define PLANE_CUH

#include <cuda_runtime.h>

#include "common.hpp"
#include "math_utils.cuh"
#include "hit.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
class plane {
public:
    ////////////////////////////////////////////////////////////////////////////////////////
    __host__ __device__ plane() : normal{0, 1, 0}, dist(0) {}
    __host__ __device__ plane(vec3 normal, float dist) : normal(normal), dist(dist) {}
    ////////////////////////////////////////////////////////////////////////////////////////
    /// @brief Determine if the ray hits this plane.
    /// @param ray      the ray to test for intersection
    /// @param hit_info struct to be populated with hit information
    /// @return         whether a hit was detected
    __device__ bool is_hit(const ray3& ray, hit& hit_info) const {
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
        if (fabs(den) < render::self_intersect_eps) return false;

        float num = -(dot(normal, ray.origin) + dist);
        float t = num / den;
        if (t < render::self_intersect_eps) return false;

        point3 intersect = ray.origin + t * ray.direction;

        hit_info.t = t;
        hit_info.loc = intersect;

        /* Treat both sides of the plane as outward / front faces */
        hit_info.normal = den > 0 ? -normal : normal;
        hit_info.is_front_face = true;

        return true;
    }

    float3 normal;
    float dist;
};
////////////////////////////////////////////////////////////////////////////////////////////////


#endif //PLANE_CUH
