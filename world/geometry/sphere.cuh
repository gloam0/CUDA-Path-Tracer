#ifndef SPHERE_CUH
#define SPHERE_CUH

#include <cuda_runtime.h>

#include "common.hpp"
#include "math_utils.cuh"
#include "hit.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
class sphere {
public:
    ////////////////////////////////////////////////////////////////////////////////////////
    __host__ __device__ sphere() : center{0,0,0}, radius(0) {}
    __host__ __device__ sphere(point3 center, float radius) : center(center), radius(radius) {}
    ////////////////////////////////////////////////////////////////////////////////////////
    /// @brief Determine if the ray hits this sphere.
    /// @param ray      the ray to test for intersection
    /// @param hit_info struct to be populated with hit information
    /// @return         whether a hit was detected
    __device__ bool is_hit(const ray3& ray, hit& hit_info) const {
        /*---------------------------- intersections ---------------------------------*/
        /*
         * ray.direction (D) is assumed to be a unit vector.
         *
         * Ray R(t) = O + tD
         * Sphere surface point S(P, C) = (P - C) . (P - C) = r^2
         * let P = R(t): (O + tD - C) . (O + tD - C) = r^2
         * expands to quadratic w.r.t. t:
         * (D. D)t^2 + 2(O - C)t + ((O - C) . (O - C) - r^2) = 0;
         * => (a = D . D = 1), b = 2 * D . (O - C), c = (O - C) . (O - C) - r^2 = 0
         */
        vec3 oc = ray.origin - center;
        float b = 2.f * dot(ray.direction, oc);
        float c = dot(oc, oc) - radius * radius;

        float discriminant = b * b - 4.f * c;
        if (discriminant < 0.f) return false;

        /* solve for roots */
        float sqrtd = sqrtf(discriminant);
        float t0 = (-b - sqrtd) * 0.5f;
        float t1 = (-b + sqrtd) * 0.5f;
        /*-------------------------------- hit_pt ------------------------------------*/
        /* ensure t0 is min */
        if (t0 > t1) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        /* avoid negative or near-zero results */
        if (t0 < render::self_intersect_eps) {
            if (t1 > render::self_intersect_eps) {
                t0 = t1;
            } else {
                return false;
            }
        }
        /* Update hit_info */
        point3 hit_pt = ray.origin + t0 * ray.direction;
        hit_info.t = t0;
        hit_info.loc = hit_pt;
        /*-------------------------------- normal ------------------------------------*/
        hit_info.normal = (hit_pt - center) / radius;
        return true;
    }

    point3 center;
    float radius;
};
////////////////////////////////////////////////////////////////////////////////////////////////

#endif //SPHERE_CUH
