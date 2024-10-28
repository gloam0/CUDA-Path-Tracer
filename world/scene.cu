#include "scene.hpp"

#include "cuda_utils.cuh"
#include "material.cuh"
#include "predefined_materials.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////
scene* create_scene() {
    scene* s = new scene;

    /* ---------------------------------- Geometry ---------------------------------------*/
    /* Create spheres */
    s->geoms_info.num_spheres = 6;
    s->geoms.spheres.centers = new point3[s->geoms_info.num_spheres] {
        point3{0.f, 1.f, -5.f},
        point3{0.f, 1.f, -5.f},
        point3{2.5f, 1.f, -5.f},
        point3{5.f, 1.f, -5.f},
        point3{0.f, 4.f, -7.f},
        point3{0.f, 5.f, 25.f}
    };
    s->geoms.spheres.radii = new float[s->geoms_info.num_spheres] {
        1.f, 0.9f, 1.f, 1.f, 1.5f, 5.f
    };

    /* Create planes */
    s->geoms_info.num_planes = 1;
    s->geoms.planes.normals = new vec3[s->geoms_info.num_planes] {
        vec3{0.f, 1.f, 0.f}
    };
    s->geoms.planes.dists = new float[s->geoms_info.num_planes] {
        0.01f
    };

    /* Create triangles */
    s->geoms_info.num_triangles = 2;
    s->geoms.triangles.vertices = new triangle_vertices[s->geoms_info.num_triangles] {
        triangle_vertices{point3{-2.f, 0.f, 1.f}, point3{-5.f, 2.f, 1.2f}, point3{-3.f, 1.f, 1.4f}},
        triangle_vertices{point3{0.f, 6.f, 1.f}, point3{-4.f, 1.f, 2.f}, point3{-2.f, 2.f, 4.f}}
    };

    /* Populate geometry instances array */
    s->geoms_info.num_instances = s->geoms_info.num_spheres + s->geoms_info.num_planes + s->geoms_info.num_triangles;
    s->geoms_info.instances = new geometry_instance[s->geoms_info.num_instances];

    s->geoms_info.instances[0] = geometry_instance{geometry_type::SPHERE, 0, 1};
    s->geoms_info.instances[1] = geometry_instance{geometry_type::SPHERE, 1, 3};
    s->geoms_info.instances[2] = geometry_instance{geometry_type::SPHERE, 2, 0};
    s->geoms_info.instances[3] = geometry_instance{geometry_type::SPHERE, 3, 5};
    s->geoms_info.instances[4] = geometry_instance{geometry_type::SPHERE, 4, 4};
    s->geoms_info.instances[5] = geometry_instance{geometry_type::SPHERE, 5, 2};

    s->geoms_info.instances[6] = geometry_instance{geometry_type::PLANE, 0, 0};

    s->geoms_info.instances[7] = geometry_instance{geometry_type::TRIANGLE, 0, 0};
    s->geoms_info.instances[8] = geometry_instance{geometry_type::TRIANGLE, 1, 0};

    /* ---------------------------------- Material ---------------------------------------*/
    /* Create diffuse */
    s->mats_info.num_diffuse = 2;
    s->mats.diffuses = new diffuse_params[s->mats_info.num_diffuse] {
        color3{0.7f, 0.7f, 0.7f}, .01f,
        color3{0.7f, 0.7f, 0.7f}, .9f
    };

    /* Create dielectrics */
    s->mats_info.num_dielectrics = 3;
    s->mats.dielectrics = new dielectric_params[s->mats_info.num_dielectrics] {
        predefined_materials::GLASS_BK7(),
        predefined_materials::GLASS_BK7({1.f,1.f,1.f}, 0.2f),
        dielectric_params{color3{1.f,1.f,1.f},
            color3{1.f / 1.5158f, 1.f / 1.5186f, 1.f / 1.5240f},
            color3{0.f, 0.f, 0.f},
            0.0f,
            dielectric_render_method::FRESNEL
        }
    };

    /* Create conductors */
    s->mats_info.num_conductors = 1;
    s->mats.conductors = new conductor_params[s->mats_info.num_conductors] {
        predefined_materials::ALUMINUM()
    };

    /* Populate material instances array */
    s->mats_info.num_instances = s->mats_info.num_diffuse + s->mats_info.num_dielectrics + s->mats_info.num_conductors;
    s->mats_info.instances = new material_instance[s->mats_info.num_instances];

    s->mats_info.instances[0] = material_instance{material_type::DIFFUSE, 0};

    s->mats_info.instances[1] = material_instance{material_type::DIELECTRIC, 0};
    s->mats_info.instances[2] = material_instance{material_type::DIELECTRIC, 1};
    s->mats_info.instances[3] = material_instance{material_type::DIELECTRIC, 2};

    s->mats_info.instances[4] = material_instance{material_type::CONDUCTOR, 0};

    s->mats_info.instances[5] = material_instance{material_type::DIFFUSE, 1};

    return s;
}

////////////////////////////////////////////////////////////////////////////////////////////////
void free_scene(scene* h_scene) {
    delete[] h_scene->geoms.planes.normals;
    delete[] h_scene->geoms.planes.dists;
    delete[] h_scene->geoms.spheres.centers;
    delete[] h_scene->geoms.spheres.radii;
    delete[] h_scene->geoms.triangles.vertices;
    delete[] h_scene->geoms_info.instances;

    delete[] h_scene->mats.diffuses;
    delete[] h_scene->mats.dielectrics;
    delete[] h_scene->mats.conductors;
    delete[] h_scene->mats_info.instances;

    delete h_scene;
}

////////////////////////////////////////////////////////////////////////////////////////////////
scene* copy_scene_to_device(scene* h_scene) {
    /* Create scene device pointer */
    scene* d_scene;
    CHECK_ERR(cudaMalloc(&d_scene, sizeof(scene)));

    /* ---------------------------------- Geometry ---------------------------------------*/
    /* Allocate and copy spheres */
    sphere_params_soa h_sphere_soa = h_scene->geoms.spheres;
    sphere_params_soa d_sphere_soa;

    if (h_scene->geoms_info.num_spheres > 0) {
        CHECK_ERR(cudaMalloc(&d_sphere_soa.centers, sizeof(point3) * h_scene->geoms_info.num_spheres));
        CHECK_ERR(cudaMemcpy(d_sphere_soa.centers, h_sphere_soa.centers,
                            sizeof(point3) * h_scene->geoms_info.num_spheres,
                            cudaMemcpyHostToDevice));

        CHECK_ERR(cudaMalloc(&d_sphere_soa.radii, sizeof(float) * h_scene->geoms_info.num_spheres));
        CHECK_ERR(cudaMemcpy(d_sphere_soa.radii, h_sphere_soa.radii,
                            sizeof(float) * h_scene->geoms_info.num_spheres,
                            cudaMemcpyHostToDevice));
    } else {
        d_sphere_soa.centers = nullptr;
        d_sphere_soa.radii = nullptr;
    }

    /* Allocate and copy planes */
    plane_params_soa h_plane_soa = h_scene->geoms.planes;
    plane_params_soa d_plane_soa;

    if (h_scene->geoms_info.num_planes > 0) {
        CHECK_ERR(cudaMalloc(&d_plane_soa.normals, sizeof(vec3) * h_scene->geoms_info.num_planes));
        CHECK_ERR(cudaMemcpy(d_plane_soa.normals, h_plane_soa.normals,
                            sizeof(vec3) * h_scene->geoms_info.num_planes,
                            cudaMemcpyHostToDevice));

        CHECK_ERR(cudaMalloc(&d_plane_soa.dists, sizeof(float) * h_scene->geoms_info.num_planes));
        CHECK_ERR(cudaMemcpy(d_plane_soa.dists, h_plane_soa.dists,
                            sizeof(float) * h_scene->geoms_info.num_planes,
                            cudaMemcpyHostToDevice));
    } else {
        d_plane_soa.normals = nullptr;
        d_plane_soa.dists = nullptr;
    }

    /* Allocate and copy triangles */
    triangle_params_soa h_triangle_soa = h_scene->geoms.triangles;
    triangle_params_soa d_triangle_soa;

    if (h_scene->geoms_info.num_triangles > 0) {
        CHECK_ERR(cudaMalloc(&d_triangle_soa.vertices, sizeof(triangle_vertices) * h_scene->geoms_info.num_triangles));
        CHECK_ERR(cudaMemcpy(d_triangle_soa.vertices, h_triangle_soa.vertices,
                            sizeof(triangle_vertices) * h_scene->geoms_info.num_triangles,
                            cudaMemcpyHostToDevice));
    } else {
        d_triangle_soa.vertices = nullptr;
    }

    /* Allocate and copy unified geometry instances array */
    int total_instances = h_scene->geoms_info.num_instances;
    geometry_instance* d_instances = nullptr;

    if (total_instances > 0) {
        CHECK_ERR(cudaMalloc(&d_instances, sizeof(geometry_instance) * total_instances));
        CHECK_ERR(cudaMemcpy(d_instances, h_scene->geoms_info.instances,
                            sizeof(geometry_instance) * total_instances,
                            cudaMemcpyHostToDevice));
    }

    /* --------------------------------- Materials ---------------------------------------*/
    /* Allocate and copy diffuse_params */
    diffuse_params* d_diffuses = nullptr;
    if (h_scene->mats_info.num_diffuse > 0) {
        CHECK_ERR(cudaMalloc(&d_diffuses, sizeof(diffuse_params) * h_scene->mats_info.num_diffuse));
        CHECK_ERR(cudaMemcpy(d_diffuses, h_scene->mats.diffuses,
                            sizeof(diffuse_params) * h_scene->mats_info.num_diffuse,
                            cudaMemcpyHostToDevice));
    }

    /* Allocate and copy dielectric_params */
    dielectric_params* d_dielectrics = nullptr;
    if (h_scene->mats_info.num_dielectrics > 0) {
        CHECK_ERR(cudaMalloc(&d_dielectrics, sizeof(dielectric_params) * h_scene->mats_info.num_dielectrics));
        CHECK_ERR(cudaMemcpy(d_dielectrics, h_scene->mats.dielectrics,
                            sizeof(dielectric_params) * h_scene->mats_info.num_dielectrics,
                            cudaMemcpyHostToDevice));
    }

    /* Allocate and copy conductor_params */
    conductor_params* d_conductors = nullptr;
    if (h_scene->mats_info.num_conductors > 0) {
        CHECK_ERR(cudaMalloc(&d_conductors, sizeof(conductor_params) * h_scene->mats_info.num_conductors));
        CHECK_ERR(cudaMemcpy(d_conductors, h_scene->mats.conductors,
                            sizeof(conductor_params) * h_scene->mats_info.num_conductors,
                            cudaMemcpyHostToDevice));
    }

    /* Allocate and copy material instances array */
    int total_material_instances = h_scene->mats_info.num_instances;
    material_instance* d_material_instances = nullptr;
    if (total_material_instances > 0) {
        CHECK_ERR(cudaMalloc(&d_material_instances, sizeof(material_instance) * total_material_instances));
        CHECK_ERR(cudaMemcpy(d_material_instances, h_scene->mats_info.instances,
                            sizeof(material_instance) * total_material_instances,
                            cudaMemcpyHostToDevice));
    }

    /* --------------------------- Assemble Device Scene ---------------------------------*/
    /* Assemble device geometries struct */
    geometries d_geoms;
    d_geoms.spheres.centers = d_sphere_soa.centers;
    d_geoms.spheres.radii = d_sphere_soa.radii;
    d_geoms.planes.normals = d_plane_soa.normals;
    d_geoms.planes.dists = d_plane_soa.dists;
    d_geoms.triangles.vertices = d_triangle_soa.vertices;

    /* Assembly device geometries_info struct */
    geometries_info d_geoms_info;
    d_geoms_info.instances = d_instances;
    d_geoms_info.num_spheres = h_scene->geoms_info.num_spheres;
    d_geoms_info.num_planes = h_scene->geoms_info.num_planes;
    d_geoms_info.num_triangles = h_scene->geoms_info.num_triangles;
    d_geoms_info.num_instances = h_scene->geoms_info.num_instances;

    /* Assemble device materials struct */
    materials d_materials;
    d_materials.diffuses = d_diffuses;
    d_materials.dielectrics = d_dielectrics;
    d_materials.conductors = d_conductors;

    /* Assemble device materials_info struct */
    materials_info d_mats_info;
    d_mats_info.instances = d_material_instances;
    d_mats_info.num_diffuse = h_scene->mats_info.num_diffuse;
    d_mats_info.num_dielectrics = h_scene->mats_info.num_dielectrics;
    d_mats_info.num_conductors = h_scene->mats_info.num_conductors;
    d_mats_info.num_instances = total_material_instances;

    /* Assemble temporary host scene struct with device pointers */
    scene tmp_scene;
    tmp_scene.geoms = d_geoms;
    tmp_scene.geoms_info = d_geoms_info;
    tmp_scene.mats = d_materials;
    tmp_scene.mats_info = d_mats_info;

    /* Copy assembled struct scene to device */
    CHECK_ERR(cudaMemcpy(d_scene, &tmp_scene, sizeof(scene), cudaMemcpyHostToDevice));

    return d_scene;
}
////////////////////////////////////////////////////////////////////////////////////////////////