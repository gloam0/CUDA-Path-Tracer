find_package(ZLIB REQUIRED)  # tinyexr dependency

include_directories(${ZLIB_INCLUDE_DIRS})

add_library(hdr
        exr_utils.cuh
        exr_utils.cu)

target_include_directories(hdr PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(hdr
        PUBLIC
        common
        utils
        PRIVATE
        ${ZLIB_LIBRARIES}
)

set_target_properties(hdr PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)