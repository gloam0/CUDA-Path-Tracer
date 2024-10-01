#include "scene.hpp"

#include "cuda_utils.cuh"
#include "material.cuh"
#include "predefined_materials.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////
scene* create_scene() {
    scene* s = new scene;

    /* Create spheres */
    s->geoms_info.num_spheres = 3;
    s->geoms.spheres.centers = new point3[s->geoms_info.num_spheres] {
        point3{0.f, 10.f, -13.f},
        point3{16.f, 5.f, -8.f},
        point3{24.f, 2.f, -5.f}
    };
    s->geoms.spheres.radii = new float[s->geoms_info.num_spheres] {
        10.f, 5.f, 2.f
    };

    /* Create planes */
    s->geoms_info.num_planes = 1;
    s->geoms.planes.normals = new vec3[s->geoms_info.num_planes] {
        vec3{0.f, 1.f, 0.f}
    };
    s->geoms.planes.dists = new float[s->geoms_info.num_planes] {
        1.f
    };

    /* Populate instances array */
    s->geoms_info.num_instances = s->geoms_info.num_spheres + s->geoms_info.num_planes;
    s->geoms_info.instances = new geometry_instance[s->geoms_info.num_instances];

    int instance_i = 0;
    for (int i = 0; i < s->geoms_info.num_spheres; i++) {
        s->geoms_info.instances[instance_i++] = geometry_instance{geometry_type::SPHERE, i};
    }

    for (int i = 0; i < s->geoms_info.num_planes; i++) {
        s->geoms_info.instances[instance_i++] = geometry_instance{geometry_type::PLANE, i};
    }

    /* Materials */
    s->sphere_materials = new material[s->geoms_info.num_spheres] {
        material{material_type::Diffuse, color3{0.5f, 0.5f, 0.5f}},
        predefined_materials::GLASS_BK7(),
        material{material_type::Dielectric, color3{1.f,1.f,1.f},
            mat_properties{
                .dielectric = dielectric_params{
                    .ior = 1.5f,
                    .roughness = 0.f,
                    .render_method = dielectric_render_method::SHLICK
                }
            }
        }
    };

    s->plane_materials = new material[s->geoms_info.num_planes] {
        material{material_type::Diffuse, color3{.6, .6, 1.}},
    };

    return s;
}

void free_scene(scene* h_scene) {
    delete[] h_scene->geoms.planes.normals;
    delete[] h_scene->geoms.planes.dists;
    delete[] h_scene->geoms.spheres.centers;
    delete[] h_scene->geoms.spheres.radii;
    delete[] h_scene->geoms_info.instances;
    delete[] h_scene->sphere_materials;
    delete[] h_scene->plane_materials;
    delete h_scene;
}
////////////////////////////////////////////////////////////////////////////////////////////////
scene* copy_scene_to_device(scene* h_scene) {
    /* Create scene device pointer */
    scene* d_scene;
    CHECK_ERR(cudaMalloc(&d_scene, sizeof(scene)));

    /* Allocate and copy sphere SoA */
    sphere_params_soa h_sphere_soa = h_scene->geoms.spheres;
    sphere_params_soa d_sphere_soa;

    CHECK_ERR(cudaMalloc(&d_sphere_soa.centers, sizeof(point3) * h_scene->geoms_info.num_spheres));
    CHECK_ERR(cudaMemcpy(d_sphere_soa.centers, h_sphere_soa.centers,
                        sizeof(point3) * h_scene->geoms_info.num_spheres,
                        cudaMemcpyHostToDevice));

    CHECK_ERR(cudaMalloc(&d_sphere_soa.radii, sizeof(float) * h_scene->geoms_info.num_spheres));
    CHECK_ERR(cudaMemcpy(d_sphere_soa.radii, h_sphere_soa.radii,
                        sizeof(float) * h_scene->geoms_info.num_spheres,
                        cudaMemcpyHostToDevice));

    /* Allocate and copy plane SoA */
    plane_params_soa h_plane_soa = h_scene->geoms.planes;
    plane_params_soa d_plane_soa;

    CHECK_ERR(cudaMalloc(&d_plane_soa.normals, sizeof(vec3) * h_scene->geoms_info.num_planes));
    CHECK_ERR(cudaMemcpy(d_plane_soa.normals, h_plane_soa.normals,
                        sizeof(vec3) * h_scene->geoms_info.num_planes,
                        cudaMemcpyHostToDevice));

    CHECK_ERR(cudaMalloc(&d_plane_soa.dists, sizeof(float) * h_scene->geoms_info.num_planes));
    CHECK_ERR(cudaMemcpy(d_plane_soa.dists, h_plane_soa.dists,
                        sizeof(float) * h_scene->geoms_info.num_planes,
                        cudaMemcpyHostToDevice));

    /* Allocate and copy unified instances array */
    int total_instances = h_scene->geoms_info.num_instances;
    geometry_instance* d_instances;
    CHECK_ERR(cudaMalloc(&d_instances, sizeof(geometry_instance) * total_instances));
    CHECK_ERR(cudaMemcpy(d_instances, h_scene->geoms_info.instances,
                        sizeof(geometry_instance) * total_instances,
                        cudaMemcpyHostToDevice));

    /* Allocate and copy materials */
    material* d_sphere_materials;
    CHECK_ERR(cudaMalloc(&d_sphere_materials, sizeof(material) * h_scene->geoms_info.num_spheres));
    CHECK_ERR(cudaMemcpy(d_sphere_materials, h_scene->sphere_materials,
                        sizeof(material) * h_scene->geoms_info.num_spheres,
                        cudaMemcpyHostToDevice));

    material* d_plane_materials;
    CHECK_ERR(cudaMalloc(&d_plane_materials, sizeof(material) * h_scene->geoms_info.num_planes));
    CHECK_ERR(cudaMemcpy(d_plane_materials, h_scene->plane_materials,
                        sizeof(material) * h_scene->geoms_info.num_planes,
                        cudaMemcpyHostToDevice));

    /* Assemble device structs */
    geometries tmp_geoms;
    tmp_geoms.spheres.centers = d_sphere_soa.centers;
    tmp_geoms.spheres.radii = d_sphere_soa.radii;
    tmp_geoms.planes.normals = d_plane_soa.normals;
    tmp_geoms.planes.dists = d_plane_soa.dists;

    geometries_info tmp_geoms_info;
    tmp_geoms_info.instances = d_instances;
    tmp_geoms_info.num_spheres = h_scene->geoms_info.num_spheres;
    tmp_geoms_info.num_planes = h_scene->geoms_info.num_planes;
    tmp_geoms_info.num_instances = h_scene->geoms_info.num_instances;

    scene tmp_scene;
    tmp_scene.geoms = tmp_geoms;
    tmp_scene.geoms_info = tmp_geoms_info;
    tmp_scene.sphere_materials = d_sphere_materials;
    tmp_scene.plane_materials = d_plane_materials;

    /* Copy assembled struct scene to device */
    CHECK_ERR(cudaMemcpy(d_scene, &tmp_scene, sizeof(scene), cudaMemcpyHostToDevice));

    return d_scene;
}
////////////////////////////////////////////////////////////////////////////////////////////////