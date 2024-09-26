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
};

#endif //HIT_CUH
