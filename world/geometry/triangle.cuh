#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include "common.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
/// Vertices of a triangle.
struct triangle_vertices {
    point3 v1;
    point3 v2;
    point3 v3;
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// SoA describing a collection of triangles.
struct triangle_params_soa {
    triangle_vertices* vertices;
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Determine if the ray hits this sphere.
/// @param ray      the ray to test for intersection
/// @param hit_info struct to be populated with hit information
/// @param triangle struct triangle_vertices representing vertices of the triangle
/// @return         whether a hit was detected
__device__ inline bool is_hit_triangle(const ray3& ray, hit& hit_info, const triangle_vertices& triangle) {
    /*---------------------------- intersections ---------------------------------*/
    /*
     * ray.direction (D) is assumed to be a unit vector.
     *
     * https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
     */
    vec3 edge1 = triangle.v2 - triangle.v1;
    vec3 edge2 = triangle.v3 - triangle.v1;
    vec3 ray_cross_e2 = cross(ray.direction, edge2);
    float det = dot(edge1, ray_cross_e2);

    /* ray is parallel or near-parallel to the plane of this triangle */
    if (det > -render::eps && det < render::eps) return false;

    float inv_det = 1.f / det;
    vec3 s = ray.origin - triangle.v1;
    float u = inv_det * dot(s, ray_cross_e2);
    if (u < 0 || u > 1) return false;

    vec3 s_cross_e1 = cross(s, edge1);
    float v = inv_det * dot(ray.direction, s_cross_e1);
    if (v < 0 || u + v > 1) return false;

    /*-------------------------------- hit_pt ------------------------------------*/
    float t = inv_det * dot(edge2, s_cross_e1);
    if (t < render::eps) return false;

    point3 hit_pt = ray.origin + t * ray.direction;
    hit_info.t = t;
    hit_info.loc = hit_pt;

    /*-------------------------------- normal ------------------------------------*/
    vec3 normal = normalize(cross(edge1, edge2));
    /* Treat both sides as front face for now. Reconsider this when adding support for meshes. */
    hit_info.normal = dot(normal, ray.direction) > 0 ? -normal : normal;
    hit_info.is_front_face = true;

    return true;
}
////////////////////////////////////////////////////////////////////////////////////////////////

#endif //TRIANGLE_CUH
