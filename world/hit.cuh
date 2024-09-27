#ifndef HIT_CUH
#define HIT_CUH

#include "common.hpp"

struct material;

/// Provides necessary information about a ray-object intersection.
struct hit {
    point3      loc;    /* location of hit */
    vec3        normal; /* normal from surface at hit location */
    float       t;      /* t value of hit */
    material*   mat;    /* material associated with object hit */
    bool        is_front_face;

    __device__ void set_face_normal(const ray3& r, const vec3& outward_normal) {
        is_front_face = dot(r.direction, outward_normal) < 0.0f;
        normal = is_front_face ? outward_normal : -outward_normal;
    }
};

#endif //HIT_CUH
