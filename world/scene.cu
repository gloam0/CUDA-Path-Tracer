#include "scene.hpp"

#include "cuda_utils.cuh"
#include "material.cuh"
#include "predefined_materials.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////
scene* create_scene() {
    scene* s = new scene;
    s->num_spheres = 7;
    s->num_planes = 2;
    s->spheres = new sphere[s->num_spheres]{
        sphere{point3{0, -1000, -6}, 1000},
        sphere{point3{0, 2, -3}, 2},
        sphere{point3{0, 2, -3}, 1.5},
        sphere{point3{-4.1, 2, -3}, 2},
        sphere{point3{-4.1, 2, -3}, 1.5},
        sphere{point3{4.1, 2, -3}, 2},
        sphere{point3{0, 5, -3}, 2}
    };
    s->sphere_materials = new material[s->num_spheres] {
        material{material_type::Diffuse, color3{0.5f, 0.5f, 0.5f}},
        predefined_materials::GLASS_BK7(),
        material{material_type::Dielectric, color3{1.f,1.f,1.f},
            mat_properties{
                .dielectric = dielectric_params{
                    .ior = 1.000278f / 1.5191f,
                    .roughness = 0.f,
                    .render_method = dielectric_render_method::SHLICK
                }
            }
        },
        predefined_materials::GLASS_BK7(),
        predefined_materials::GOLD(color3{1.f,1.f,1.f}, 0.01f),
        predefined_materials::GOLD(color3{1.f,1.f,1.f}, 0.01f),
        material{material_type::Diffuse, color3{1., 0.1f, 0.14f}}
    };
    s->planes = new plane[s->num_planes] {
        plane{vec3{0, 1, 0}, 0.5},
        plane{vec3{1, 0, 0}, 0.5}
    };
    s->plane_materials = new material[s->num_planes] {
        material{material_type::Diffuse, color3{.6, .6, .6}},
        material{material_type::Diffuse, color3{.6, .1, 1.}},
    };
    return s;
}

void free_scene(scene* h_scene) {
    delete[] h_scene->spheres;
    delete[] h_scene->sphere_materials;
    delete h_scene;
}
////////////////////////////////////////////////////////////////////////////////////////////////
scene* copy_scene_to_device(scene* h_scene) {
    // Allocate memory for a struct scene and its components

    scene* d_scene;
    CHECK_ERR(cudaMalloc(&d_scene, sizeof(scene)));

    sphere* d_spheres;
    material* d_sphere_materials;
    plane* d_planes;
    material* d_plane_materials;

    CHECK_ERR(cudaMalloc(&d_spheres, h_scene->num_spheres * sizeof(sphere)));
    CHECK_ERR(cudaMalloc(&d_sphere_materials, h_scene->num_spheres * sizeof(material)));

    CHECK_ERR(cudaMalloc(&d_planes, h_scene->num_planes * sizeof(plane)));
    CHECK_ERR(cudaMalloc(&d_plane_materials, h_scene->num_planes * sizeof(material)));

    // Copy components from host to device
    CHECK_ERR(cudaMemcpy(d_spheres, h_scene->spheres, h_scene->num_spheres * sizeof(sphere), cudaMemcpyHostToDevice));
    CHECK_ERR(cudaMemcpy(d_sphere_materials, h_scene->sphere_materials, h_scene->num_spheres * sizeof(material), cudaMemcpyHostToDevice));

    CHECK_ERR(cudaMemcpy(d_planes, h_scene->planes, h_scene->num_planes * sizeof(plane), cudaMemcpyHostToDevice));
    CHECK_ERR(cudaMemcpy(d_plane_materials, h_scene->plane_materials, h_scene->num_planes * sizeof(material), cudaMemcpyHostToDevice));


    // Temporary scene for copying device pointers
    scene* tmp_d_scene = new scene;
    tmp_d_scene->spheres = d_spheres;
    tmp_d_scene->sphere_materials = d_sphere_materials;
    tmp_d_scene->num_spheres = h_scene->num_spheres;
    tmp_d_scene->planes = d_planes;
    tmp_d_scene->plane_materials = d_plane_materials;
    tmp_d_scene->num_planes = h_scene->num_planes;

    // Copy temp struct scene with device pointers to device
    CHECK_ERR(cudaMemcpy(d_scene, tmp_d_scene, sizeof(scene), cudaMemcpyHostToDevice));

    delete tmp_d_scene;
    return d_scene;
}
////////////////////////////////////////////////////////////////////////////////////////////////