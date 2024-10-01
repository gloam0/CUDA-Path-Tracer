#ifndef HIT_CUH
#define HIT_CUH

#include "common.hpp"
#include "math_utils.cuh"

struct material;

/// Provides necessary information about a ray-object intersection.
struct hit {
    point3      loc;    /* location of hit */
    vec3        normal; /* normal from surface at hit location */
    float       t;      /* t value of hit */
    int         mat_id; /* ID of material hit (the material's index in the unified array of material_instances */
    bool        is_front_face;  /* did we hit the front face of the geometry? */

    /// Record the outward normal regardless of whether we hit the front face or back face
    __device__ void set_face_normal(const ray3& r, const vec3& outward_normal) {
        is_front_face = dot(r.direction, outward_normal) < 0.0f;
        normal = is_front_face ? outward_normal : -outward_normal;
    }
};

#endif //HIT_CUH
