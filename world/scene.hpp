#ifndef SCENE_H
#define SCENE_H

#include "material.cuh"
#include "geometry.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
/// A struct representing the objects in a scene.
struct scene {
    geometries      geoms;
    geometries_info geoms_info;
    materials       mats;
    materials_info  mats_info;
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// Create a scene and its objects.
/// @return The allocated and initialized struct scene.
scene* create_scene();

/// Free a struct scene and its components.
void free_scene(scene* h_scene);

/// Copy a struct scene to device memory.
/// @return A pointer to the device's struct scene.
scene* copy_scene_to_device(scene* h_scene);
////////////////////////////////////////////////////////////////////////////////////////////////

#endif //SCENE_H
