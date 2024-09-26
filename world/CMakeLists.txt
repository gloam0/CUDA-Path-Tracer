add_library(world
        scene.cu
        scene.hpp
        hit.cuh
)

add_subdirectory(geometry)
add_subdirectory(material)

target_include_directories(world PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(world
        INTERFACE
            common
            utils
        PUBLIC
            geometry
            material
)
