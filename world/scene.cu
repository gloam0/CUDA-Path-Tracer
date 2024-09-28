#include "scene.hpp"

#include "cuda_utils.cuh"
#include "material.cuh"
#include "predefined_materials.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////
scene* create_scene() {
    scene* s = new scene;
    s->num_objects = 7;
    s->spheres = new sphere[s->num_objects]{
        sphere{point3{0, -1000, -6}, 1000},
        sphere{point3{0, 2, -3}, 2},
        sphere{point3{0, 2, -3}, 1.5},
        sphere{point3{-4.1, 2, -3}, 2},
        sphere{point3{-4.1, 2, -3}, 1.5},
        sphere{point3{4.1, 2, -3}, 2},
        sphere{point3{0, 5, -3}, 2}
    };
    s->materials = new material[s->num_objects] {
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
    return s;
}

void free_scene(scene* h_scene) {
    delete[] h_scene->spheres;
    delete[] h_scene->materials;
    delete h_scene;
}
////////////////////////////////////////////////////////////////////////////////////////////////
scene* copy_scene_to_device(scene* h_scene) {
    scene* d_scene;
    sphere* d_spheres;
    material* d_materials;

    // Allocate memory for a struct scene and its components
    CHECK_ERR(cudaMalloc(&d_scene, sizeof(scene)));
    CHECK_ERR(cudaMalloc(&d_spheres, h_scene->num_objects * sizeof(sphere)));
    CHECK_ERR(cudaMalloc(&d_materials, h_scene->num_objects * sizeof(material)));

    // Copy components from host to device
    CHECK_ERR(cudaMemcpy(d_spheres, h_scene->spheres, h_scene->num_objects * sizeof(sphere), cudaMemcpyHostToDevice));
    CHECK_ERR(cudaMemcpy(d_materials, h_scene->materials, h_scene->num_objects * sizeof(material), cudaMemcpyHostToDevice));

    // Temporary scene for copying device pointers
    scene* tmp_d_scene = new scene;
    tmp_d_scene->spheres = d_spheres;
    tmp_d_scene->materials = d_materials;
    tmp_d_scene->num_objects = h_scene->num_objects;

    // Copy temp struct scene with device pointers to device
    CHECK_ERR(cudaMemcpy(d_scene, tmp_d_scene, sizeof(scene), cudaMemcpyHostToDevice));

    delete tmp_d_scene;
    return d_scene;
}
////////////////////////////////////////////////////////////////////////////////////////////////