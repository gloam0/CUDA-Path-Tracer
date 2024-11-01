#ifndef GEOMETRY_CUH
#define GEOMETRY_CUH

#include "sphere.cuh"
#include "plane.cuh"
#include "triangle.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
/// All available geometry types
enum class geometry_type {
    SPHERE,
    PLANE,
    TRIANGLE
};

/// Describes an entry in a unified array referencing various geometry types
struct geometry_instance {
    geometry_type type;  /* type of this geometry instance */
    int i;               /* index of this instance in the associated geometry-specific SoA */
    int material_id;     /* index of this geometry's material in an associated unified array of materials_instance */
};

/// Describe a struct geometries
struct geometries_info {
    geometry_instance* instances;  /* unified array referencing various geometry types */
    int num_spheres;
    int num_planes;
    int num_triangles;
    int num_instances;
};

/// Collection of geometries of various types
struct geometries {
    sphere_params_soa spheres;
    plane_params_soa planes;
    triangle_params_soa triangles;
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// Call the appropriate geometry-specific intersection testing function based on the type of
/// the referenced geometry instance.
/// @param ray      Ray to be tested for intersection
/// @param hit_info Hit struct to be updated with information about the intersection, if one occurred.
/// @param g        Information about the geometry instance to be tested
/// @param geoms    All geometry-specific parameters.
__device__ inline bool is_hit_dispatch(
    const ray3& ray,
    hit& hit_info,
    const geometry_instance& g,
    const geometries& geoms)
{
    switch (g.type) {
        case geometry_type::SPHERE:
            return is_hit_sphere(ray, hit_info, geoms.spheres.centers[g.i], geoms.spheres.radii[g.i]);
        case geometry_type::PLANE:
            return is_hit_plane(ray, hit_info, geoms.planes.normals[g.i], geoms.planes.dists[g.i]);
        case geometry_type::TRIANGLE:
            return is_hit_triangle(ray, hit_info, geoms.triangles.vertices[g.i]);
        default:
            return false;
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////

#endif //GEOMETRY_CUH
