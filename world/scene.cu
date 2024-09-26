#include "scene.hpp"

#include "material.cuh"

#include "common.cuh"
#include "cuda_utils.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
scene* create_scene() {
    scene* s = new scene;
    s->num_objects = 10;
    s->spheres = new sphere[s->num_objects]{
        sphere{point3{0, -101, -6}, 100},
        sphere{point3{2, 1, -4}, 0.2},
        sphere{point3{-4, 0.2, -6}, 0.3},
        sphere{point3{-2.3, 0.4, -6}, 0.1},
        sphere{point3{5.4, 3.2, -6}, 0.14},
        sphere{point3{0, 0, -3}, 1},
        sphere{point3{1, 3, -12}, 2.1},
        sphere{point3{-5.4, 6.2, -6}, 3.14},
        sphere{point3{-3, 4, 6}, 1.5},
        sphere{point3{-13, 4, -12}, 6.1}
    };
    s->materials = new material[s->num_objects] {
        material{material_type::Diffuse, color3{0.4, 0.86, 0.97}},
        material{material_type::Diffuse, color3{0.5, 0.5, 0.8}},
        material{material_type::Diffuse, color3{0.5, 0.5, 0.8}},
        material{material_type::Diffuse, color3{0.5, 0.5, 0.8}},
        material{material_type::Diffuse, color3{0.5, 0.5, 0.8}},
        material{material_type::Diffuse, color3{0.5, 0.5, 0.8}},
        material{material_type::Diffuse, color3{0.5, 0.5, 0.8}},
        material{material_type::Diffuse, color3{0.5, 0.5, 0.8}},
        material{material_type::Diffuse, color3{0.5, 0.5, 0.8}},
        material{material_type::Diffuse, color3{1, 0., 0.}}
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