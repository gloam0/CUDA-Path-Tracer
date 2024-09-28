#include "exr_utils.cuh"

#include <iostream>
#include <thread>

#include <cuda_runtime.h>

#include "cuda_utils.cuh"
#include "common.cuh"

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

////////////////////////////////////////////////////////////////////////////////////////////////
void create_env_map_threaded(const char* filepath, hdr_map* map, bool* success) {
    std::thread tmp(create_env_map, filepath, map, success);
    tmp.detach();
}
////////////////////////////////////////////////////////////////////////////////////////////////
void create_env_map(const char* filepath, hdr_map* map, bool* success) {
    float* exr_data;
    int width, height;
    const char* err = nullptr;

    int ret = LoadEXR(&exr_data, &width, &height, filepath, &err);
    if (ret != TINYEXR_SUCCESS) {
        if (err) {
            std::cerr << "Error loading EXR file: " << err << std::endl;
            FreeEXRErrorMessage(err);
        } else {
            std::cerr << "Error loading EXR file." << std::endl;
        }
        *success = false;
        cudaMemcpyToSymbol(d_use_hdr, success, sizeof(bool));
        return;
    }

    map->width = width;
    map->height = height;

    /* Create array on device and copy EXR data */
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
    CHECK_ERR(cudaMallocArray(&map->cuda_array, &channel_desc, width, height));
    CHECK_ERR(cudaMemcpy2DToArray(map->cuda_array, 0, 0, exr_data, width * sizeof(float4),
        width * sizeof(float4), height, cudaMemcpyHostToDevice));

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = map->cuda_array;

    /* Describe and create a texture object for the EXR data */
    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;  /* Wrap horizontally */
    tex_desc.addressMode[1] = cudaAddressModeWrap;  /* Wrap vertically */
    tex_desc.filterMode = cudaFilterModePoint;      /* Point sampling, faster than linear */
    tex_desc.readMode = cudaReadModeElementType;    /* Read as specified type */
    tex_desc.normalizedCoords = 1;                  /* Use normalized coords */
    CHECK_ERR(cudaCreateTextureObject(&map->tex_obj, &res_desc, &tex_desc, nullptr));

    *success = true;
    env_map_loaded = true;
    cudaMemcpyToSymbol(d_use_hdr, success, sizeof(bool));
}
////////////////////////////////////////////////////////////////////////////////////////////////
void free_env_map(hdr_map map) {
    cudaDestroyTextureObject(map.tex_obj);
    cudaFreeArray(map.cuda_array);
}
////////////////////////////////////////////////////////////////////////////////////////////////
